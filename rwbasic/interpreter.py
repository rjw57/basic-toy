import importlib.resources
import typing

import numpy as np
from lark import Lark, Tree, Visitor
from lark.exceptions import UnexpectedInput

BasicValue = typing.Union[np.int32, float, str]

_IMMEDIATE_LINE_PARSER = Lark(
    importlib.resources.files(__package__).joinpath("grammar.lark").read_text(),
    start="immediateline",
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


class Interpreter:
    def __init__(self):
        self._expression_visitor = _ExpressionVisitor()

    def execute(self, input_line: str) -> typing.Optional[BasicValue]:
        try:
            return self._expression_visitor.visit(_IMMEDIATE_LINE_PARSER.parse(input_line)).data
        except UnexpectedInput as lark_exception:
            raise BasicSyntaxError(str(lark_exception), lark_exception=lark_exception)


class _ExpressionVisitor(Visitor):
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
