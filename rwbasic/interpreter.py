import dataclasses
import importlib.resources
import sys
import typing

import numpy as np
from lark import Lark, Tree
from lark.exceptions import UnexpectedInput
from lark.visitors import Interpreter as LarkInterpreter

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

# Binary operators which understand string types
_STRING_BINARY_OPS = {"+", "=", "<>"}


class BasicError(RuntimeError):
    def __init__(self, message: str, *, tree: typing.Optional[Tree] = None):
        super().__init__(message)
        self._tree = tree


class BasicSyntaxError(BasicError):
    def __init__(self, message: str, *, lark_exception: Exception, **kwargs):
        super().__init__(message, **kwargs)
        self._lark_exception = lark_exception


class BasicMistakeError(BasicError):
    pass


class InternalParserError(BasicError):
    pass


def _is_numeric_basic_value(value: BasicValue):
    """
    Return True if and only if value is a numeric BASIC value.
    """
    return isinstance(value, np.int32) or isinstance(value, float)


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
class _InterpreterState:
    # Mapping from variable name to current value.
    variables: dict[str, BasicValue] = dataclasses.field(default_factory=dict)

    # Mapping from line number to a list of the statements within that line.
    lines: dict[int, list[Tree]] = dataclasses.field(default_factory=dict)

    # List of statements comprising the current line. When running a line from the interactive
    # prompt, this is the line being run. When running a program, this is one of the lines in the
    # lines mapping.
    current_line: list[Tree] = dataclasses.field(default_factory=list)


class Interpreter:
    _state: _InterpreterState
    _parse_tree_interpreter: "_ParseTreeInterpreter"

    def __init__(self):
        self._state = _InterpreterState()
        self._parse_tree_interpreter = _ParseTreeInterpreter(self._state)

    def evaluate(self, expression: str) -> typing.Optional[BasicValue]:
        try:
            tree = _EXPRESSION_PARSER.parse(expression)
        except UnexpectedInput as lark_exception:
            raise BasicSyntaxError(str(lark_exception), lark_exception=lark_exception)
        self._parse_tree_interpreter.visit(tree)
        assert _is_basic_value(tree.data)
        return tree.data

    def execute(self, prompt_line: str):
        try:
            tree = _PROMPT_LINE_PARSER.parse(prompt_line)
        except UnexpectedInput as lark_exception:
            raise BasicSyntaxError(str(lark_exception), lark_exception=lark_exception)
        self._parse_tree_interpreter.visit(tree)


class _ParseTreeInterpreter(LarkInterpreter):
    _state: _InterpreterState

    def __init__(self, state: _InterpreterState):
        super().__init__()
        self._state = state

    def line_definition(self, tree: Tree):
        line_number = int(tree.children[0])
        if line_number <= 0:
            raise BasicMistakeError("Line numbers must be greater than zero", tree=tree)
        # We store the individual statement trees as the line content.
        self._state.lines[line_number] = tree.children[1].children

    def line_statements(self, tree: Tree):
        # Record this line as the current line.
        self._state.current_line = tree.children
        self.visit_children(tree)

    def print_statement(self, tree: Tree):
        self.visit_children(tree)
        assert _is_basic_value(tree.children[1].data)
        sys.stdout.write(f"{tree.children[1].data}")
        if len(tree.children) == 2:
            # We don't have a terminal ";"
            sys.stdout.write("\n")
        sys.stdout.flush()

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
