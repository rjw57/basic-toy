import dataclasses
import enum
import sys
import typing
from functools import cache

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


@dataclasses.dataclass(eq=True, frozen=True)
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


@dataclasses.dataclass
class _ProgramAnalysis:
    # Mapping from IF statement locations to locations to jump to if condition *NOT* met.
    if_jump_targets: dict[_ExecutionLocation, _ExecutionLocation] = dataclasses.field(
        default_factory=dict
    )

    # Mapping from ELSE statement locations to location to jump to if we want to skip it.
    else_jump_targets: dict[_ExecutionLocation, _ExecutionLocation] = dataclasses.field(
        default_factory=dict
    )

    # Mapping from loop block end statement locations to defining statements and body start
    # locations.
    loop_definitions_and_bodies: dict[
        _ExecutionLocation, tuple[Tree, _ExecutionLocation]
    ] = dataclasses.field(default_factory=dict)


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

    # Analysis of the current program.
    program_analysis: _ProgramAnalysis = dataclasses.field(default_factory=_ProgramAnalysis)

    def reset(self):
        """Reset state to defaults."""
        self.variables = dict()
        self.lines = _SortedNumberedLineList()
        self.execution_location = None
        self.next_execution_location = None
        self.program_analysis = _ProgramAnalysis()


@v_args(tree=True)
class _ExpressionTransformer(Transformer):
    """
    Transformer which evaluates expressions given the current interpreter state.
    """

    _state: _InterpreterState

    def __init__(self, state: _InterpreterState):
        self._state = state

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

    def strliteralexpr(self, tree: Tree):
        token = tree.children[0]
        return token[1:-1].replace('""', '"')

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

    def powerexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    def mulexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    def addexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    def compexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

    def andexpr(self, tree: Tree):
        return self._binaryopexpr(tree)

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

    def variablerefexpr(self, tree: Tree):
        variable_name = tree.children[0]
        try:
            return self._state.variables[variable_name]
        except KeyError:
            raise BasicMistakeError(f"No such variable: {variable_name}", tree=tree)


class _ParseTreeInterpreter(LarkInterpreter):
    _state: _InterpreterState
    _expression_transformer: _ExpressionTransformer

    def __init__(self, state: _InterpreterState):
        super().__init__()
        self._state = state
        self._expression_transformer = _ExpressionTransformer(self._state)

    def evaluate_expression(self, expression: Tree) -> BasicValue:
        """
        Evaluate an expression from its AST node based on the current interpreter state.

        """
        try:
            value = self._expression_transformer.transform(expression)
        except VisitError as e:
            raise e.orig_exc from e
        assert _is_basic_value(value)
        return value

    def _execute_inline_statements(self, statements: list[Tree]):
        """
        Execute a list of statements. Must only be called if we're evaluating statements from the
        prompt line.

        """
        for statement in statements:
            self.visit(statement)

    def _jump(self, location: _ExecutionLocation):
        """
        When RUN-ing a program, jump to a new location.

        """
        self._state.next_execution_location = location

    def _set_variable(self, tree: Tree, variable_name: str, value: BasicValue):
        """
        Set a named variable to a Basic value.

        """
        assert _is_basic_value(value)
        if variable_name.endswith("$") and not _is_string_basic_value(value):
            raise BasicMistakeError("Cannot assign non-string value to string variable", tree=tree)
        if not variable_name.endswith("$") and not _is_numeric_basic_value(value):
            raise BasicMistakeError("Cannot assign non-numeric value to variable", tree=tree)
        if variable_name.endswith("%"):
            # Ensure we only assign numbers to integer variables.
            value = int32(value)
        self._state.variables[variable_name] = value

    # Program definition

    def numbered_line_definition(self, tree: Tree):
        line_number = int(tree.children[0])
        if len(self._state.lines) > 0 and line_number <= self._state.lines[-1].number:
            raise BasicBadProgramError("Line numbers must increase in programs", tree=tree)
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

    # Prompt line statements

    def prompt_line_statements(self, tree: Tree):
        # Execute statements within line.
        self._execute_inline_statements(tree.children)

    def numbered_line_update(self, tree: Tree):
        line_number = int(tree.children[0])
        self._add_line_definition(line_number, tree.children[1])

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

    # IF... THEN... ELSE

    def inline_if_statement(self, tree: Tree):
        if_header, then_block = tree.children[:2]
        try:
            else_block = tree.children[3]
        except IndexError:
            else_block = None

        if self._test_if_condition(if_header):
            self._execute_inline_statements(then_block.children)
        elif else_block is not None:
            self._execute_inline_statements(else_block.children)

    def if_statement(self, tree: Tree):
        # Skip to ELSE or ENDIF if the condition is not met.
        if not self._test_if_condition(tree):
            assert self._state.execution_location is not None
            self._jump(
                self._state.program_analysis.if_jump_targets[self._state.execution_location]
            )

    def else_statement(self, tree: Tree):
        # Skip this statement. If we're executing it, we fell through from the THEN body.
        assert self._state.execution_location is not None
        self._jump(self._state.program_analysis.else_jump_targets[self._state.execution_location])

    def _test_if_condition(self, if_header: Tree) -> bool:
        condition_expr = if_header.children[1]
        condition_value = self.evaluate_expression(condition_expr)
        if not _is_numeric_basic_value(condition_value):
            raise BasicMistakeError("IF conditions must be numeric", tree=condition_expr)
        return condition_value != 0

    # FOR... NEXT loops

    def inline_for_statement(self, tree: Tree):
        for_statement = tree.children[0]
        body_statements = tree.children[1:-1]
        next_statement = tree.children[-1]

        self._begin_for(for_statement)
        while True:
            self._execute_inline_statements(body_statements)
            if not self._process_next(next_statement, for_statement):
                break

    def for_statement(self, tree: Tree):
        self._begin_for(tree)

    def next_statement(self, tree: Tree):
        assert self._state.execution_location is not None
        loop_stmt, jump_loc = self._state.program_analysis.loop_definitions_and_bodies[
            self._state.execution_location
        ]
        if self._process_next(tree, loop_stmt):
            self._jump(jump_loc)

    def _unpack_for_statement(self, for_statement: Tree):
        var_name = for_statement.children[1]
        from_expr = for_statement.children[3]
        to_expr = for_statement.children[5]
        try:
            step_expr = for_statement.children[7]
        except IndexError:
            step_expr = None
        return var_name, from_expr, to_expr, step_expr

    def _begin_for(self, for_statement: Tree):
        # TODO: do we care about repeated variable names?
        var_name, from_expr, _, _ = self._unpack_for_statement(for_statement)
        from_value = self.evaluate_expression(from_expr)
        if not _is_numeric_basic_value(from_value):
            raise BasicMistakeError("FOR start value must be numeric", tree=for_statement)
        self._set_variable(for_statement, var_name, from_value)

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

    # Other statements

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

    def end_statement(self, tree: Tree):
        self._state.next_execution_location = None


@v_args(tree=True)
class _ProgramAnalysisVisitor(Visitor):
    class _ControlFlowBlockType(enum.Enum):
        IF_THEN = enum.auto()
        FOR_NEXT = enum.auto()

    @dataclasses.dataclass
    class _ControlFlowBlockState:
        # What sort of control flow block is this?
        block_type: "_ProgramAnalysisVisitor._ControlFlowBlockType"

        # The statement which initiated the block, (IF, FOR, etc.) and its location
        definition_statement: Tree
        definition_location: _ExecutionLocation

        # The location of the start of the block content.
        entry_location: typing.Optional[_ExecutionLocation] = None

        # The location to jump to to exit the block.
        exit_location: typing.Optional[_ExecutionLocation] = None

        # For if-then blocks only, the location of the start of the ELSE block and the ELSE
        # statement itself.
        else_statement: typing.Optional[Tree] = None
        else_location: typing.Optional[_ExecutionLocation] = None
        else_entry_location: typing.Optional[_ExecutionLocation] = None

    analysis: _ProgramAnalysis

    _control_flow_block_stack: list[_ControlFlowBlockState]
    _current_location: _ExecutionLocation

    def __init__(self):
        self._reset()

    def _reset(self):
        self.analysis = _ProgramAnalysis()
        self._control_flow_block_stack = []
        self._current_location = _ExecutionLocation(statement_index=0, line_index=0)

    def analyse_lines(self, lines: typing.Sequence[_NumberedLine]):
        self._reset()
        for line_index, line in enumerate(lines):
            for statement_index, statement in enumerate(line.statements):
                self._current_location = _ExecutionLocation(
                    statement_index=statement_index, line_index=line_index
                )
                self.visit(statement)

        if len(self._control_flow_block_stack) > 0:
            raise BasicBadProgramError(
                "Unclosed control flow block",
                tree=self._control_flow_block_stack[-1].definition_statement,
            )

    def _location_following_current(self) -> _ExecutionLocation:
        "Helper to return a location pointing to the next statement."
        return _ExecutionLocation(
            line_index=self._current_location.line_index,
            statement_index=self._current_location.statement_index + 1,
        )

    def for_statement(self, tree: Tree):
        self._control_flow_block_stack.append(
            _ProgramAnalysisVisitor._ControlFlowBlockState(
                block_type=_ProgramAnalysisVisitor._ControlFlowBlockType.FOR_NEXT,
                definition_statement=tree,
                definition_location=self._current_location,
                entry_location=self._location_following_current(),
            )
        )

    def next_statement(self, tree: Tree):
        try:
            block_state = self._control_flow_block_stack.pop()
        except IndexError:
            raise BasicBadProgramError("NEXT without matching FOR", tree=tree)

        if block_state.block_type != _ProgramAnalysisVisitor._ControlFlowBlockType.FOR_NEXT:
            raise BasicBadProgramError("NEXT within unclosed non-FOR block", tree=tree)

        index_var_name = block_state.definition_statement.children[1]
        if len(tree.children) > 1:
            if tree.children[1] != index_var_name:
                raise BasicBadProgramError(
                    f"Unexpected NEXT variable name: {tree.children[1]}", tree=tree
                )

        assert block_state.entry_location is not None
        self.analysis.loop_definitions_and_bodies[self._current_location] = (
            block_state.definition_statement,
            block_state.entry_location,
        )

    def if_statement(self, tree: Tree):
        self._control_flow_block_stack.append(
            _ProgramAnalysisVisitor._ControlFlowBlockState(
                block_type=_ProgramAnalysisVisitor._ControlFlowBlockType.IF_THEN,
                definition_statement=tree,
                definition_location=self._current_location,
                entry_location=self._location_following_current(),
            )
        )

    def else_statement(self, tree: Tree):
        if len(self._control_flow_block_stack) == 0:
            raise BasicBadProgramError("ELSE without matching IF", tree=tree)

        block_state = self._control_flow_block_stack[-1]
        if block_state.else_statement is not None:
            raise BasicBadProgramError("Multiple ELSE within single IF", tree=tree)

        if block_state.block_type != _ProgramAnalysisVisitor._ControlFlowBlockType.IF_THEN:
            raise BasicBadProgramError("ELSE within unclosed non-IF block", tree=tree)

        # Record the else statement. We jump to the statement following the else location.
        block_state.else_statement = tree
        block_state.else_location = self._current_location
        block_state.else_entry_location = self._location_following_current()

    def endif_statement(self, tree: Tree):
        try:
            block_state = self._control_flow_block_stack.pop()
        except IndexError:
            raise BasicBadProgramError("ENDIF without matching IF", tree=tree)

        if block_state.block_type != _ProgramAnalysisVisitor._ControlFlowBlockType.IF_THEN:
            raise BasicBadProgramError("ENDIF within unclosed non-IF block", tree=tree)

        if block_state.else_statement is not None:
            assert block_state.else_location is not None
            assert block_state.else_entry_location is not None
            self.analysis.else_jump_targets[
                block_state.else_location
            ] = self._location_following_current()
            self.analysis.if_jump_targets[
                block_state.definition_location
            ] = block_state.else_entry_location
        else:
            self.analysis.if_jump_targets[
                block_state.definition_location
            ] = self._location_following_current()


class Interpreter:
    _state: _InterpreterState
    _parse_tree_interpreter: _ParseTreeInterpreter

    def __init__(self):
        self._state = _InterpreterState()
        self._parse_tree_interpreter = _ParseTreeInterpreter(self._state)

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
        return self._parse_tree_interpreter.evaluate_expression(tree)

    def execute(self, prompt_line: str):
        self._parse_tree_interpreter.visit(self._parse(prompt_line, start="promptline"))
