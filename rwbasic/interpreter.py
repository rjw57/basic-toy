import dataclasses
import importlib.resources
import sys
import typing

import numpy as np
from lark import Lark, Tree
from lark.exceptions import UnexpectedInput
from lark.visitors import Interpreter as LarkInterpreter
from sortedcontainers import SortedKeyList

BasicValue = typing.Union[np.int32, float, str]

_GRAMMAR = importlib.resources.files(__package__).joinpath("grammar.lark").read_text()

# Parser used for input from interactive prompt.
_PROMPT_LINE_PARSER = Lark(
    _GRAMMAR,
    start="promptline",
    parser="lalr",
    propagate_positions=True,
)

# Parser for expressions only.
_EXPRESSION_PARSER = Lark(
    _GRAMMAR,
    start="expression",
    parser="lalr",
    propagate_positions=True,
)

# Parser for programs.
_PROGRAM_PARSER = Lark(
    _GRAMMAR,
    start="program",
    parser="lalr",
    propagate_positions=True,
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


def _is_integer_basic_value(value: BasicValue):
    """
    Return True if and only if value is an integer BASIC value.
    """
    return isinstance(value, np.int32)


def _is_numeric_basic_value(value: BasicValue):
    """
    Return True if and only if value is a numeric BASIC value.
    """
    return _is_integer_basic_value(value) or isinstance(value, float)


def _is_string_basic_value(value: BasicValue):
    """
    Return True if and only if value is a string BASIC value.
    """
    return isinstance(value, str)


def _is_basic_value(value: BasicValue):
    """
    Return True if and only if value is a BASIC value.
    """
    return _is_numeric_basic_value(value) or _is_string_basic_value(value)


def _basic_bool(value: bool) -> BasicValue:
    """
    Convert a Python boolean into a BASIC boolean.
    """
    return np.int32(-1) if value else np.int32(0)


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


@dataclasses.dataclass
class _InterpreterState:
    # Mapping from variable name to current value.
    variables: dict[str, BasicValue] = dataclasses.field(default_factory=dict)

    # Lines which make up the current program.
    lines: _SortedNumberedLineList = dataclasses.field(default_factory=_SortedNumberedLineList)

    # The most recent line passed as an immediate line to execute in the prompt. May be None if
    # there is no such line.
    prompt_line: typing.Optional[list[Tree]] = None

    # Current execution location or None if we're not executing at the moment.
    execution_location: typing.Optional[_ExecutionLocation] = None

    # Execution location for next statement or None if we should stop execution.
    next_execution_location: typing.Optional[_ExecutionLocation] = None

    def reset(self):
        """Reset state to defaults."""
        self.variables = dict()
        self.lines = _SortedNumberedLineList()
        self.prompt_line = None
        self.execution_location = None
        self.next_execution_location = None


class Interpreter:
    _state: _InterpreterState

    def __init__(self):
        self._state = _InterpreterState()

    def _parse(self, parser: Lark, input_text: str) -> Tree:
        try:
            return parser.parse(input_text)
        except UnexpectedInput as lark_exception:
            raise BasicSyntaxError(str(lark_exception)) from lark_exception

    def load_program(self, program: str):
        self.execute("NEW")
        _ParseTreeInterpreter(self._state).visit(self._parse(_PROGRAM_PARSER, program))

    def load_and_run_program(self, program: str):
        self.load_program(program)
        self.execute("RUN")

    def evaluate(self, expression: str) -> typing.Optional[BasicValue]:
        tree = self._parse(_EXPRESSION_PARSER, expression)
        _ParseTreeInterpreter(self._state).visit(tree)
        assert _is_basic_value(tree.data)
        return tree.data

    def execute(self, prompt_line: str):
        _ParseTreeInterpreter(self._state, executing_from_prompt=True).visit(
            self._parse(_PROMPT_LINE_PARSER, prompt_line)
        )


class _ParseTreeInterpreter(LarkInterpreter):
    _state: _InterpreterState
    _executing_from_prompt: bool

    def __init__(self, state: _InterpreterState, *, executing_from_prompt=False):
        super().__init__()
        self._state = state
        self._executing_from_prompt = executing_from_prompt

    def _start_execution(self):
        # We should start executing with some start point.
        assert self._state.execution_location is not None

        # Keep going until we run out of program.
        while self._state.execution_location is not None:
            # Assume the next execution location will be the next statement.
            if self._state.execution_location.line_index is None:
                current_line_statements = self._state.prompt_line
            else:
                current_line_statements = self._state.lines[
                    self._state.execution_location.line_index
                ].statements

            if self._state.execution_location.statement_index + 1 < len(current_line_statements):
                # Next statement is in the same line.
                self._state.next_execution_location = _ExecutionLocation(
                    line_index=self._state.execution_location.line_index,
                    statement_index=self._state.execution_location.statement_index + 1,
                )
            elif self._state.execution_location.line_index is not None:
                # Move to next line.
                if self._state.execution_location.line_index + 1 >= len(self._state.lines):
                    # No more lines
                    self._state.next_execution_location = None
                else:
                    self._state.next_execution_location = _ExecutionLocation(
                        line_index=self._state.execution_location.line_index + 1,
                        statement_index=0,
                    )
            else:
                # At end of prompt line.
                self._state.next_execution_location = None

            # Execute the current statement if there are any on this line. We have the check
            # because we might have an execution location pointing to the start of an empty line.
            if len(current_line_statements) > 0:
                current_statement = current_line_statements[
                    self._state.execution_location.statement_index
                ]
                self.visit(current_statement)

            # Move to next location.
            self._state.execution_location = self._state.next_execution_location

    def numbered_line_definition(self, tree: Tree):
        line_number = int(tree.children[0])
        self._add_line_definition(line_number, tree.children[1])

    def unnumbered_line_definition(self, tree: Tree):
        # Unnumbered lines in programs get numbered automagically.
        assert not self._executing_from_prompt
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
        # If we're not in the interactive prompt, line numbers must strictly increase
        if (
            not self._executing_from_prompt
            and len(self._state.lines) > 0
            and line_number <= self._state.lines[-1].number
        ):
            raise BasicBadProgramError(
                "Line numbers must increase in programs", tree=line_statements
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

    def line_statements(self, tree: Tree):
        if self._executing_from_prompt:
            # If we're executing a single prompt line, record this line as the current prompt line.
            self._state.prompt_line = tree.children

            # If there is at least one statement in the line, start execution.
            if len(tree.children) > 0:
                self._state.execution_location = _ExecutionLocation(statement_index=0)
                self._start_execution()

    def print_statement(self, tree: Tree):
        self.visit_children(tree)
        assert _is_basic_value(tree.children[1].data)
        sys.stdout.write(f"{tree.children[1].data}")
        if len(tree.children) == 2:
            # We don't have a terminal ";"
            sys.stdout.write("\n")
        sys.stdout.flush()

    def let_statement(self, tree: Tree):
        self.visit_children(tree)
        variable_name, value_node = tree.children[-2:]
        value = value_node.data
        assert _is_basic_value(value)

        if variable_name.endswith("$") and not _is_string_basic_value(value):
            raise BasicMistakeError("Cannot assign non-string value to string variable", tree=tree)
        if not variable_name.endswith("$") and not _is_numeric_basic_value(value):
            raise BasicMistakeError("Cannot assign non-numeric value to variable", tree=tree)
        if variable_name.endswith("%"):
            # Ensure we only assign numbers to integer variables.
            value = np.int32(value)
        self._state.variables[variable_name] = value

    def new_statement(self, tree: Tree):
        if not self._executing_from_prompt:
            raise BasicMistakeError("Cannot NEW when not in interactive prompt.", tree=tree)
        self._state.reset()

    def run_statement(self, tree: Tree):
        if not self._executing_from_prompt:
            raise BasicMistakeError("Cannot RUN when not in interactive prompt.", tree=tree)

        # Nothing to do if there is no program.
        if len(self._state.lines) == 0:
            return

        # Start at first line
        self._state.execution_location = _ExecutionLocation(line_index=0, statement_index=0)
        assert self._executing_from_prompt
        self._executing_from_prompt = False
        self._start_execution()
        self._executing_from_prompt = True

    def literalexpr(self, tree: Tree):
        token = tree.children[0]
        match token.type:
            case "BOOLEAN_LITERAL":
                tree.data = _basic_bool(token.upper() == "TRUE")
            case "BINARY_LITERAL":
                tree.data = np.int32(int(token[1:], base=2))
            case "HEX_LITERAL":
                tree.data = np.int32(int(token[1:], base=16))
            case "DECIMAL_LITERAL":
                tree.data = np.int32(int(token, base=10))
            case "FLOAT_LITERAL":
                tree.data = float(token)
            case "STRING_LITERAL":
                tree.data = token[1:-1].replace('""', '"')
            case _:  # pragma: no cover
                raise InternalParserError("Unexpected literal type", tree=tree)

    def unaryexpr(self, tree: Tree):
        self.visit_children(tree)
        children = list(tree.children)
        rhs = children.pop()
        tree.data = rhs.data
        while len(children) > 0:
            op = children.pop().upper()
            # All unary operators need numeric input.
            if not _is_numeric_basic_value(rhs.data):
                raise BasicMistakeError(f"Inappropriate type for unary operation {op}", tree=rhs)
            match op:
                case "+":
                    pass  # Do nothing
                case "-":
                    tree.data = -tree.data
                case "NOT":
                    tree.data = np.int32(tree.data ^ np.uint32(0xFFFFFFFF))
                case _:  # pragma: no cover
                    raise InternalParserError(f"Unknown unary operator: {op}", tree=tree)
        assert _is_numeric_basic_value(tree.data)

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
        self.visit_children(tree)
        children = list(tree.children)
        rhs = children.pop()
        tree.data = rhs.data
        while len(children) > 0:
            op = children.pop().upper()
            if op not in _STRING_BINARY_OPS and not _is_numeric_basic_value(rhs.data):
                raise BasicMistakeError(f"Inappropriate type for operation {op}", tree=rhs)
            lhs = children.pop()
            if _is_numeric_basic_value(lhs.data) != _is_numeric_basic_value(rhs.data):
                raise BasicMistakeError(f"Cannot mix types for operator {op}", tree=tree)
            match op:
                case "+":
                    tree.data = lhs.data + tree.data
                case "-":
                    tree.data = lhs.data - tree.data
                case "*":
                    tree.data = lhs.data * tree.data
                case "/":
                    tree.data = lhs.data / tree.data
                case "DIV":
                    tree.data = lhs.data // tree.data
                case "MOD":
                    tree.data = lhs.data % tree.data
                case "AND":
                    tree.data = lhs.data & tree.data
                case "OR":
                    tree.data = lhs.data | tree.data
                case "EOR":
                    tree.data = lhs.data ^ tree.data
                case "=":
                    tree.data = _basic_bool(lhs.data == tree.data)
                case "<>":
                    tree.data = _basic_bool(lhs.data != tree.data)
                case "<":
                    tree.data = _basic_bool(lhs.data < tree.data)
                case ">":
                    tree.data = _basic_bool(lhs.data > tree.data)
                case "<=":
                    tree.data = _basic_bool(lhs.data <= tree.data)
                case ">=":
                    tree.data = _basic_bool(lhs.data >= tree.data)
                case "<<":
                    tree.data = lhs.data << tree.data
                case ">>":
                    # Arithmetic shift right is a little bit complicated in Python(!)
                    tree.data = (lhs.data >> tree.data) | np.int32(
                        (np.uint32(0xFFFFFFFF >> tree.data)) ^ np.uint32(0xFFFFFFFF)
                        if lhs.data & 0x80000000
                        else 0
                    )
                case ">>>":
                    tree.data = lhs.data >> tree.data
                case "^":
                    tree.data = lhs.data**tree.data
                case _:  # pragma: no cover
                    raise InternalParserError(f"Unknown binary operator: {op}", tree=tree)
            rhs = lhs
        assert _is_basic_value(tree.data), f"Non-numeric output: {tree.data!r}"

    def variablerefexpr(self, tree: Tree):
        variable_name = tree.children[0]
        try:
            tree.data = self._state.variables[variable_name]
        except KeyError:
            raise BasicMistakeError(f"No such variable: {variable_name}", tree=tree)
