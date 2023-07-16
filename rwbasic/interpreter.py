import dataclasses
import enum
import sys
import typing
from functools import cache, cached_property

from lark import Lark, Token, Transformer, Tree
from lark.exceptions import UnexpectedInput, VisitError
from lark.visitors import Interpreter as LarkInterpreter
from lark.visitors import v_args
from numpy import int32, uint32
from sortedcontainers import SortedKeyList

BasicValue = typing.Union[int32, float, str]


@cache
def load_parser() -> Lark:
    return Lark.open_from_package(
        __package__,
        "grammar.lark",
        propagate_positions=True,
        parser="lalr",
        lexer="basic",
        cache=True,
        start=["program", "promptline", "expression"],
    )


# Binary operators which understand string types
_STRING_BINARY_OPS = {"+", "=", "<>"}


class BasicError(RuntimeError):
    def __init__(self, message: str, *, tree: typing.Optional[Tree] = None):
        super().__init__(message)
        self._tree = tree


class BasicSyntaxError(BasicError):
    """
    Raised when there is a syntax error. The __cause__ attribute will be the underlying lark
    parser exception.
    """


class BasicMistakeError(BasicError):
    """
    Raised when there was some runtime mistake.
    """


class BasicBadProgramError(BasicError):
    """
    Raised when the program itself is wrongly specified.
    """


class InternalParserError(BasicError):
    pass


def _is_integer_basic_value(value: BasicValue) -> typing.TypeGuard[int32]:
    """
    Return True if and only if value is an integer BASIC value.
    """
    return isinstance(value, int32)


def _is_numeric_basic_value(value: BasicValue) -> typing.TypeGuard[typing.Union[int32, float]]:
    """
    Return True if and only if value is a numeric BASIC value.
    """
    return _is_integer_basic_value(value) or isinstance(value, float)


def _is_string_basic_value(value: BasicValue) -> typing.TypeGuard[str]:
    """
    Return True if and only if value is a string BASIC value.
    """
    return isinstance(value, str)


def _is_basic_value(value: BasicValue) -> typing.TypeGuard[BasicValue]:
    """
    Return True if and only if value is a BASIC value.
    """
    return _is_numeric_basic_value(value) or _is_string_basic_value(value)


def _basic_bool(value: bool) -> BasicValue:
    """
    Convert a Python boolean into a BASIC boolean.
    """
    return int32(-1) if value else int32(0)


@dataclasses.dataclass
class _ExecutionLocation:
    # Index of statement within execution line.
    statement_index: int

    # Index of line within the program. None indicates we're executing in the prompt line.
    line_index: typing.Optional[int] = None


@dataclasses.dataclass
class _NumberedLine:
    number: int
    statements: list[Tree]


class _SortedNumberedLineList(SortedKeyList[_NumberedLine]):
    def __init__(self, iterable=None):
        super().__init__(iterable=iterable, key=lambda nl: nl.number)


class _BlockType(enum.Enum):
    FOR_LOOP = enum.auto()


@dataclasses.dataclass
class _BlockState:
    block_type: _BlockType
    body_start_location: typing.Optional[_ExecutionLocation]

    # For loops
    variable_name: typing.Optional[str]
    to_expression: typing.Optional[Tree]
    step_expression: typing.Optional[Tree]


@dataclasses.dataclass
class _InterpreterState:
    # Mapping from variable name to current value.
    variables: dict[str, BasicValue] = dataclasses.field(default_factory=dict)

    # Lines which make up the current program.
    lines: _SortedNumberedLineList = dataclasses.field(default_factory=_SortedNumberedLineList)

    # Current execution location or None if we're not executing at the moment.
    execution_location: typing.Optional[_ExecutionLocation] = None

    # Execution location for next statement or None if we should stop execution.
    next_execution_location: typing.Optional[_ExecutionLocation] = None

    # Live blocks. Innermost block is at the end of the list.
    block_states: list[_BlockState] = dataclasses.field(default_factory=list)

    def reset(self):
        """Reset state to defaults."""
        self.variables = dict()
        self.lines = _SortedNumberedLineList()
        self.execution_location = None
        self.next_execution_location = None
        self.block_states = []


class Interpreter:
    _state: _InterpreterState

    def __init__(self):
        self._state = _InterpreterState()

    @cached_property
    def _parse_tree_interpreter(self) -> "_ParseTreeInterpreter":
        return _ParseTreeInterpreter(self._state)

    @cached_property
    def _expression_transformer(self) -> "_ExpressionTransformer":
        return _ExpressionTransformer(self._state)

    def _parse(self, input_text: str, **kwargs) -> Tree:
        try:
            return load_parser().parse(input_text, **kwargs)
        except UnexpectedInput as lark_exception:
            raise BasicSyntaxError(str(lark_exception)) from lark_exception

    def load_program(self, program: str):
        self.execute("NEW")
        self._parse_tree_interpreter.visit(self._parse(program, start="program"))

    def load_and_run_program(self, program: str):
        self.load_program(program)
        self.execute("RUN")

    def evaluate(self, expression: str) -> typing.Optional[BasicValue]:
        tree = self._parse(expression, start="expression")
        try:
            value = self._expression_transformer.transform(tree)
        except VisitError as e:
            raise e.orig_exc
        assert _is_basic_value(value)
        return value

    def execute(self, prompt_line: str):
        self._parse_tree_interpreter.visit(self._parse(prompt_line, start="promptline"))


class _ParseTreeInterpreter(LarkInterpreter):
    _state: _InterpreterState

    def __init__(self, state: _InterpreterState):
        super().__init__()
        self._state = state

    @cached_property
    def _expression_transformer(self) -> "_ExpressionTransformer":
        return _ExpressionTransformer(self._state)

    def _start_execution(self):
        # We should start executing with some start point within the program.
        assert self._state.execution_location is not None
        assert self._state.execution_location.line_index is not None

        # Keep going until we run out of program.
        while self._state.execution_location is not None:
            current_line_statements = self._state.lines[
                self._state.execution_location.line_index
            ].statements

            if self._state.execution_location.statement_index + 1 < len(current_line_statements):
                # Next statement is in the same line.
                self._state.next_execution_location = _ExecutionLocation(
                    line_index=self._state.execution_location.line_index,
                    statement_index=self._state.execution_location.statement_index + 1,
                )
            else:
                # Move to next line.
                if self._state.execution_location.line_index + 1 >= len(self._state.lines):
                    # No more lines
                    self._state.next_execution_location = None
                else:
                    self._state.next_execution_location = _ExecutionLocation(
                        line_index=self._state.execution_location.line_index + 1,
                        statement_index=0,
                    )

            # Execute the current statement if there are any on this line. We have the check
            # because we might have an execution location pointing to the start of an empty line.
            if len(current_line_statements) > 0:
                current_statement = current_line_statements[
                    self._state.execution_location.statement_index
                ]
                self.visit(current_statement)

                # If we're about to end, make sure we don't have any dangling loops, etc.
                if self._state.next_execution_location is None:
                    if len(self._state.block_states) > 0:
                        raise BasicMistakeError("Missing block close", tree=current_statement)

            # Move to next location.
            self._state.execution_location = self._state.next_execution_location

    def evaluate_expression(self, expression: Tree) -> BasicValue:
        try:
            value = self._expression_transformer.transform(expression)
        except VisitError as e:
            raise e.orig_exc
        assert _is_basic_value(value)
        return value

    def numbered_line_definition(self, tree: Tree):
        line_number = int(tree.children[0])
        if len(self._state.lines) > 0 and line_number <= self._state.lines[-1].number:
            raise BasicBadProgramError("Line numbers must increase in programs", tree=tree)
        self._add_line_definition(line_number, tree.children[1])

    def numbered_line_update(self, tree: Tree):
        line_number = int(tree.children[0])
        self._add_line_definition(line_number, tree.children[1])

    def unnumbered_line_definition(self, tree: Tree):
        # Unnumbered lines in programs get numbered automagically.
        if len(self._state.lines) == 0:
            line_number = 1
        else:
            line_number = self._state.lines[-1].number + 1
        self._add_line_definition(line_number, tree.children[0])

    def _add_line_definition(self, line_number: int, line_statements: Tree):
        if line_number <= 0:
            raise BasicBadProgramError(
                "Line numbers must be greater than zero", tree=line_statements
            )

        # We store the individual statement trees as the line content. We replace any existing
        # lines with the same line number.
        new_line = _NumberedLine(number=line_number, statements=line_statements.children)
        insert_index = self._state.lines.bisect_key_left(new_line.number)
        if (
            len(self._state.lines) > insert_index
            and self._state.lines[insert_index].number == line_number
        ):
            del self._state.lines[insert_index]
        self._state.lines.add(new_line)

    def _execute_inline_statements(self, statements: list[Tree]):
        for statement in statements:
            self.visit(statement)

    def prompt_line_statements(self, tree: Tree):
        # Execute statements within line.
        self._execute_inline_statements(tree.children)

    def print_statement(self, tree: Tree):
        item_index = 1
        separator = " "
        while item_index < len(tree.children):
            item_expression = tree.children[item_index]
            item_index += 1
            if (
                item_index < len(tree.children)
                and isinstance(tree.children[item_index], Token)
                and tree.children[item_index].type == "PRINT_ITEM_SEPARATOR"
            ):
                separator = tree.children[item_index]
                item_index += 1
            else:
                separator = " "

            value = self.evaluate_expression(item_expression)
            sys.stdout.write(f"{value}")
            match separator:
                case "'":
                    sys.stdout.write("\n")
                case ";" | " ":
                    pass  # no-op separator
                case ",":
                    # FIXME: we don't properly support the column alignment separator "," yet.
                    sys.stdout.write(" ")
                case _:  # pragma: no cover
                    raise InternalParserError(
                        f"Unknown print item separator: {separator!r}", tree=tree
                    )

        if separator != ";":
            sys.stdout.write("\n")

    def let_statement(self, tree: Tree):
        variable_name = tree.children[-3]
        value_expression = tree.children[-1]
        value = self.evaluate_expression(value_expression)
        self._set_variable(tree, variable_name, value)

    def _set_variable(self, tree: Tree, variable_name: str, value: BasicValue):
        assert _is_basic_value(value)
        if variable_name.endswith("$") and not _is_string_basic_value(value):
            raise BasicMistakeError("Cannot assign non-string value to string variable", tree=tree)
        if not variable_name.endswith("$") and not _is_numeric_basic_value(value):
            raise BasicMistakeError("Cannot assign non-numeric value to variable", tree=tree)
        if variable_name.endswith("%"):
            # Ensure we only assign numbers to integer variables.
            value = int32(value)
        self._state.variables[variable_name] = value

    def new_statement(self, tree: Tree):
        self._state.reset()

    def run_statement(self, tree: Tree):
        # Nothing to do if there is no program.
        if len(self._state.lines) == 0:
            return

        # Start at first line
        self._state.execution_location = _ExecutionLocation(line_index=0, statement_index=0)
        self._start_execution()

    def end_statement(self, tree: Tree):
        self._state.next_execution_location = None

    def _unpack_for_statement(self, for_statement: Tree):
        var_name = for_statement.children[1]
        from_expr = for_statement.children[3]
        to_expr = for_statement.children[5]
        try:
            step_expr = for_statement.children[7]
        except IndexError:
            step_expr = None
        return var_name, from_expr, to_expr, step_expr

    def inline_if_statement(self, tree: Tree):
        if_header = tree.children[0]
        then_block = tree.children[1]
        try:
            else_block = tree.children[3]
        except IndexError:
            else_block = None

        condition_expr = if_header.children[1]
        condition_value = self.evaluate_expression(condition_expr)

        if not _is_numeric_basic_value(condition_value):
            raise BasicMistakeError("IF conditions must be numeric", tree=condition_expr)

        if condition_value != 0:
            self._execute_inline_statements(then_block.children)
        elif else_block is not None:
            self._execute_inline_statements(else_block.children)

    def inline_for_statement(self, tree: Tree):
        loop_defn = tree.children[0]
        statements = tree.children[1:-1]
        matching_next = tree.children[-1]

        # Extract variable name and loop expressions.
        var_name, from_expr, to_expr, step_expr = self._unpack_for_statement(loop_defn)

        # Check next variable, if defined, matches the for loop one.
        if len(matching_next.children) > 1:
            if matching_next.children[1] != var_name:
                raise BasicMistakeError("Mismatched NEXT", tree=matching_next)

        # Set loop variable.
        from_value = self.evaluate_expression(from_expr)
        if not _is_numeric_basic_value(from_value):
            raise BasicMistakeError("FOR start value must be numeric", tree=tree)
        self._set_variable(tree, var_name, from_value)

        keep_looping = True
        while keep_looping:
            # Execute body.
            self._execute_inline_statements(statements)

            # Advance index
            if step_expr is not None:
                step = self.evaluate_expression(step_expr)
                if not _is_numeric_basic_value(step):
                    raise BasicMistakeError("FOR step value must be numeric", tree=step_expr)
            else:
                step = int32(1)

            index_value = self._state.variables[var_name]
            assert _is_numeric_basic_value(index_value)
            next_index = index_value + step

            # Should we keep looping?
            to_value = self.evaluate_expression(to_expr)
            if not _is_numeric_basic_value(to_value):
                raise BasicMistakeError("FOR end value must be numeric", tree=to_expr)
            if step >= 0 and next_index > to_value:
                keep_looping = False
            elif step < 0 and next_index < to_value:
                keep_looping = False
            else:
                self._set_variable(tree, var_name, next_index)

    def for_statement(self, tree: Tree):
        # Extract variable name and loop expressions.
        var_name, from_expr, to_expr, step_expr = self._unpack_for_statement(tree)

        from_value = self.evaluate_expression(from_expr)
        if not _is_numeric_basic_value(from_value):
            raise BasicMistakeError("FOR from value must be numeric", tree=tree)
        self._set_variable(tree, var_name, from_value)

        # TODO: do we care about repeated variable names?
        self._state.block_states.append(
            _BlockState(
                block_type=_BlockType.FOR_LOOP,
                variable_name=var_name,
                to_expression=to_expr,
                step_expression=step_expr,
                body_start_location=self._state.next_execution_location,
            )
        )

    def next_statement(self, tree: Tree):
        if len(self._state.block_states) == 0:
            raise BasicMistakeError("Unexpected NEXT", tree=tree)

        block_state = self._state.block_states[-1]
        if block_state.block_type != _BlockType.FOR_LOOP:
            raise BasicMistakeError("Inappropriate NEXT for block", tree=tree)

        assert block_state.variable_name is not None
        assert block_state.to_expression is not None

        if len(tree.children) > 1:
            var_name = tree.children[1]
            if var_name != block_state.variable_name:
                raise BasicMistakeError(f"Unexpected NEXT variable: {var_name}", tree=tree)

        to_value = self.evaluate_expression(block_state.to_expression)
        if not _is_numeric_basic_value(to_value):
            raise BasicMistakeError("FOR to value must be numeric", tree=block_state.to_expression)

        if block_state.step_expression is not None:
            step = self.evaluate_expression(block_state.step_expression)
            if not _is_numeric_basic_value(step):
                raise BasicMistakeError(
                    "FOR step value must be numeric", tree=block_state.step_expression
                )
        else:
            step = int32(1)

        index_value = self._state.variables[block_state.variable_name]
        assert _is_numeric_basic_value(index_value)
        next_index = index_value + step

        # Do we loop?
        should_loop = False
        if step >= 0 and next_index <= to_value:
            should_loop = True
        elif step < 0 and next_index >= to_value:
            should_loop = True

        if should_loop:
            self._set_variable(tree, block_state.variable_name, next_index)
            self._state.next_execution_location = block_state.body_start_location
        else:
            self._state.block_states.pop()


class _ExpressionTransformer(Transformer):
    """
    Transformer which evaluates expressions given the current interpreter state.
    """

    _state: _InterpreterState

    def __init__(self, state: _InterpreterState):
        self._state = state

    @v_args(tree=True)
    def numliteralexpr(self, tree: Tree):
        token = tree.children[0]
        match token.type:
            case "BOOLEAN_LITERAL":
                return _basic_bool(token.upper() == "TRUE")
            case "BINARY_LITERAL":
                return int32(int(token[1:], base=2))
            case "HEX_LITERAL":
                return int32(int(token[1:], base=16))
            case "DECIMAL_LITERAL":
                return int32(int(token, base=10))
            case "FLOAT_LITERAL":
                return float(token)
            case _:  # pragma: no cover
                raise InternalParserError("Unexpected literal type", tree=tree)

    @v_args(tree=True)
    def strliteralexpr(self, tree: Tree):
        token = tree.children[0]
        return token[1:-1].replace('""', '"')

    @v_args(tree=True)
    def unaryexpr(self, tree: Tree):
        children = list(tree.children)
        rhs = children.pop()
        while len(children) > 0:
            op = children.pop().upper()
            # All unary operators need numeric input.
            if not _is_numeric_basic_value(rhs):
                raise BasicMistakeError(f"Inappropriate type for unary operation {op}", tree=tree)
            match op:
                case "+":
                    pass  # Do nothing
                case "-":
                    rhs = -rhs
                case "NOT":
                    rhs = int32(rhs ^ uint32(0xFFFFFFFF))
                case _:  # pragma: no cover
                    raise InternalParserError(f"Unknown unary operator: {op}", tree=tree)
        assert _is_numeric_basic_value(rhs)
        return rhs

    @v_args(tree=True)
    def powerexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    @v_args(tree=True)
    def mulexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    @v_args(tree=True)
    def addexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    @v_args(tree=True)
    def compexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    @v_args(tree=True)
    def andexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    @v_args(tree=True)
    def orexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    def _binaryopexpr(self, tree: Tree):
        children = list(tree.children)
        rhs = children.pop()
        while len(children) > 0:
            op = children.pop().upper()
            if op not in _STRING_BINARY_OPS and not _is_numeric_basic_value(rhs):
                raise BasicMistakeError(f"Inappropriate type for operation {op}", tree=tree)
            lhs = children.pop()
            if _is_numeric_basic_value(lhs) != _is_numeric_basic_value(rhs):
                raise BasicMistakeError(f"Cannot mix types for operator {op}", tree=tree)
            match op:
                case "+":
                    rhs = lhs + rhs
                case "-":
                    rhs = lhs - rhs
                case "*":
                    rhs = lhs * rhs
                case "/":
                    rhs = lhs / rhs
                case "DIV":
                    rhs = lhs // rhs
                case "MOD":
                    rhs = lhs % rhs
                case "AND":
                    rhs = lhs & rhs
                case "OR":
                    rhs = lhs | rhs
                case "EOR":
                    rhs = lhs ^ rhs
                case "=":
                    rhs = _basic_bool(lhs == rhs)
                case "<>":
                    rhs = _basic_bool(lhs != rhs)
                case "<":
                    rhs = _basic_bool(lhs < rhs)
                case ">":
                    rhs = _basic_bool(lhs > rhs)
                case "<=":
                    rhs = _basic_bool(lhs <= rhs)
                case ">=":
                    rhs = _basic_bool(lhs >= rhs)
                case "<<":
                    rhs = lhs << rhs
                case ">>":
                    # Arithmetic shift right is a little bit complicated in Python(!)
                    rhs = (lhs >> rhs) | int32(
                        (uint32(0xFFFFFFFF >> rhs)) ^ uint32(0xFFFFFFFF) if lhs & 0x80000000 else 0
                    )
                case ">>>":
                    rhs = lhs >> rhs
                case "^":
                    rhs = lhs**rhs
                case _:  # pragma: no cover
                    raise InternalParserError(f"Unknown binary operator: {op}", tree=tree)
        assert _is_basic_value(rhs), f"Non-numeric output: {rhs!r}"
        return rhs

    @v_args(tree=True)
    def variablerefexpr(self, tree: Tree):
        variable_name = tree.children[0]
        try:
            return self._state.variables[variable_name]
        except KeyError:
            raise BasicMistakeError(f"No such variable: {variable_name}", tree=tree)
