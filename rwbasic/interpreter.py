import dataclasses
import enum
import sys
import typing
from functools import cache, cached_property

from lark import Lark, Token, Transformer, Tree, Visitor
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
    line_index: int


@dataclasses.dataclass
class _NumberedLine:
    number: int
    statements: list[Tree]


class _SortedNumberedLineList(SortedKeyList[_NumberedLine]):
    def __init__(self, iterable=None):
        super().__init__(iterable=iterable, key=lambda nl: nl.number)


class _LoopType(enum.Enum):
    FOR_NEXT = enum.auto()


@dataclasses.dataclass
class _LoopState:
    loop_type: _LoopType

    # Statement defining loop.
    loop_statement: Tree

    # Location of start of loop body.
    body_start_location: _ExecutionLocation


@dataclasses.dataclass
class _ProgramAnalysis:
    # Mapping from IF statement trees to locations to jump to if condition *NOT* met.
    if_jump_targets: dict[Tree, _ExecutionLocation] = dataclasses.field(default_factory=dict)

    # Mapping from ELSE statement trees to location to jump to if we fall into it.
    else_jump_targets: dict[Tree, _ExecutionLocation] = dataclasses.field(default_factory=dict)


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

    # Live loops. Innermost loop is at the end of the list.
    loop_states: list[_LoopState] = dataclasses.field(default_factory=list)

    # Analysis of the current program.
    program_analysis: _ProgramAnalysis = dataclasses.field(default_factory=_ProgramAnalysis)

    def reset(self):
        """Reset state to defaults."""
        self.variables = dict()
        self.lines = _SortedNumberedLineList()
        self.execution_location = None
        self.next_execution_location = None
        self.loop_states = []
        self.program_analysis = _ProgramAnalysis()


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
            raise e.orig_exc from e
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

    def evaluate_expression(self, expression: Tree) -> BasicValue:
        try:
            value = self._expression_transformer.transform(expression)
        except VisitError as e:
            raise e.orig_exc from e
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

        # Analyse program loops, etc.
        analysis_visitor = _ProgramAnalysisVisitor()
        analysis_visitor.analyse_lines(self._state.lines)
        self._state.program_analysis = analysis_visitor.analysis

        # Keep going until we run out of program or the END statement sets the next location to
        # None.
        while self._state.execution_location is not None:
            try:
                current_line_statements = self._state.lines[
                    self._state.execution_location.line_index
                ].statements
            except IndexError:
                # We fell off the bottom of the program, exit the execution loop.
                break

            try:
                current_statement = current_line_statements[
                    self._state.execution_location.statement_index
                ]
            except IndexError:
                # We fell off the line, move to next line and loop.
                self._state.execution_location = _ExecutionLocation(
                    line_index=self._state.execution_location.line_index + 1, statement_index=0
                )
                continue

            # Prepare the expected next location to be the following statement.
            self._state.next_execution_location = _ExecutionLocation(
                line_index=self._state.execution_location.line_index,
                statement_index=self._state.execution_location.statement_index + 1,
            )

            # Execute the current statement.
            self.visit(current_statement)

            # Move to next location which may have been altered by the statement we just executed.
            self._state.execution_location = self._state.next_execution_location

        # We're about to end. Make sure we don't have any dangling loops, etc.
        if len(self._state.loop_states) > 0:
            raise BasicMistakeError("Missing loop closing statement", tree=current_statement)

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

    def _jump(self, location: _ExecutionLocation):
        self._state.next_execution_location = location

    def if_statement(self, tree: Tree):
        condition_expr = tree.children[1]
        condition_value = self.evaluate_expression(condition_expr)
        if not _is_numeric_basic_value(condition_value):
            raise BasicMistakeError("IF conditions must be numeric", tree=condition_expr)

        if condition_value == 0:
            self._jump(self._state.program_analysis.if_jump_targets[tree])

    def else_statement(self, tree: Tree):
        # Skip this statement. If we're executing it, we fell through from the THEN body.
        self._jump(self._state.program_analysis.else_jump_targets[tree])

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

        # Loop over body.
        while True:
            self._execute_inline_statements(statements)
            if not self._process_next(matching_next, loop_defn):
                break

    def for_statement(self, tree: Tree):
        # Extract variable name and loop expressions.
        var_name, from_expr, _, _ = self._unpack_for_statement(tree)

        from_value = self.evaluate_expression(from_expr)
        if not _is_numeric_basic_value(from_value):
            raise BasicMistakeError("FOR from value must be numeric", tree=tree)
        self._set_variable(tree, var_name, from_value)

        # TODO: do we care about repeated variable names?
        assert self._state.next_execution_location is not None
        self._state.loop_states.append(
            _LoopState(
                loop_type=_LoopType.FOR_NEXT,
                loop_statement=tree,
                body_start_location=self._state.next_execution_location,
            )
        )

    def next_statement(self, tree: Tree):
        try:
            loop_state = self._state.loop_states[-1]
        except IndexError:
            raise BasicMistakeError("Unexpected NEXT", tree=tree)

        if loop_state.loop_type != _LoopType.FOR_NEXT:
            raise BasicMistakeError("Inappropriate NEXT for block", tree=tree)

        if self._process_next(tree, loop_state.loop_statement):
            self._jump(loop_state.body_start_location)
        else:
            self._state.loop_states.pop()

    def _process_next(self, next_statement: Tree, for_statement: Tree) -> bool:
        index_var_name, _, to_expr, step_expr = self._unpack_for_statement(for_statement)
        if len(next_statement.children) > 1:
            if next_statement.children[1] != index_var_name:
                raise BasicMistakeError(
                    f"Unexpected NEXT variable: {next_statement.children[1]}", tree=next_statement
                )

        to_value = self.evaluate_expression(to_expr)
        if not _is_numeric_basic_value(to_value):
            raise BasicMistakeError("FOR to value must be numeric", tree=to_expr)

        if step_expr is not None:
            step = self.evaluate_expression(step_expr)
            if not _is_numeric_basic_value(step):
                raise BasicMistakeError("FOR step value must be numeric", tree=step_expr)
        else:
            step = int32(1)

        index_value = self._state.variables[index_var_name]
        assert _is_numeric_basic_value(index_value)
        next_index = index_value + step

        # Do we loop?
        if step >= 0 and next_index <= to_value:
            self._set_variable(next_statement, index_var_name, next_index)
            return True
        elif step < 0 and next_index >= to_value:
            self._set_variable(next_statement, index_var_name, next_index)
            return True
        return False


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

            # TODO: string lexical comparison (e.g. "HI" > "HELLO")

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


@v_args(tree=True)
class _ProgramAnalysisVisitor(Visitor):
    @dataclasses.dataclass
    class _IfState:
        if_statement: Tree
        else_statement: typing.Optional[Tree] = None
        else_location: typing.Optional[_ExecutionLocation] = None

    analysis: _ProgramAnalysis

    _if_statement_stack: list[_IfState]
    _current_location: _ExecutionLocation

    def __init__(self):
        self.analysis = _ProgramAnalysis()
        self._reset()

    def _reset(self):
        self._if_statement_stack = []
        self._current_location = _ExecutionLocation(statement_index=0, line_index=0)

    def analyse_lines(self, lines: typing.Sequence[_NumberedLine]):
        self._reset()
        for line_index, line in enumerate(lines):
            for statement_index, statement in enumerate(line.statements):
                self._current_location = _ExecutionLocation(
                    statement_index=statement_index, line_index=line_index
                )
                self.visit(statement)

        if len(self._if_statement_stack) > 0:
            raise BasicBadProgramError(
                "Unclosed IF", tree=self._if_statement_stack[-1].if_statement
            )

    def if_statement(self, tree: Tree):
        self._if_statement_stack.append(_ProgramAnalysisVisitor._IfState(if_statement=tree))

    def else_statement(self, tree: Tree):
        if len(self._if_statement_stack) == 0:
            raise BasicBadProgramError("ELSE without matching IF", tree=tree)
        if_state = self._if_statement_stack[-1]
        if if_state.else_statement is not None:
            raise BasicBadProgramError("Multiple ELSE within single IF", tree=tree)
        if_state.else_statement = tree

        # We jump to the statement following the else location.
        if_state.else_location = _ExecutionLocation(
            line_index=self._current_location.line_index,
            statement_index=self._current_location.statement_index + 1,
        )

    def endif_statement(self, tree: Tree):
        try:
            if_state = self._if_statement_stack.pop()
        except IndexError:
            raise BasicBadProgramError("ENDIF without matching IF", tree=tree)
        if if_state.else_statement is not None:
            assert if_state.else_location is not None
            self.analysis.else_jump_targets[if_state.else_statement] = self._current_location
            self.analysis.if_jump_targets[if_state.if_statement] = if_state.else_location
        else:
            self.analysis.if_jump_targets[if_state.if_statement] = self._current_location
