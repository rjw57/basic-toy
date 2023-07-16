import typing

from lark import Transformer, Tree
from lark.visitors import v_args
from numpy import int32, uint32

from .exceptions import BasicMistakeError, InternalInterpreterError
from .values import BasicValue, basic_bool, is_basic_value, is_numeric_basic_value

# Binary operators which understand string types
_STRING_BINARY_OPS = {"+", "=", "<>", "<", ">", "<=", ">="}


@v_args(tree=True)
class ExpressionTransformer(Transformer):
    """
    Transformer which evaluates expressions given the current interpreter state.
    """

    _variable_fetcher: typing.Callable[[Tree, str], BasicValue]

    def __init__(self, variable_fetcher: typing.Callable[[Tree, str], BasicValue]):
        self._variable_fetcher = variable_fetcher

    def numliteralexpr(self, tree: Tree):
        token = tree.children[0]
        match token.type:
            case "BOOLEAN_LITERAL":
                return basic_bool(token.upper() == "TRUE")
            case "BINARY_LITERAL":
                return int32(int(token[1:], base=2))
            case "HEX_LITERAL":
                return int32(int(token[1:], base=16))
            case "DECIMAL_LITERAL":
                return int32(int(token, base=10))
            case "FLOAT_LITERAL":
                return float(token)
            case _:  # pragma: no cover
                raise InternalInterpreterError("Unexpected literal type", tree=tree)

    def strliteralexpr(self, tree: Tree):
        token = tree.children[0]
        return token[1:-1].replace('""', '"')

    def unaryexpr(self, tree: Tree):
        children = list(tree.children)
        rhs = children.pop()
        while len(children) > 0:
            op = children.pop().upper()
            # All unary operators need numeric input.
            if not is_numeric_basic_value(rhs):
                raise BasicMistakeError(f"Inappropriate type for unary operation {op}", tree=tree)
            match op:
                case "+":
                    pass  # Do nothing
                case "-":
                    rhs = -rhs
                case "NOT":
                    rhs = int32(rhs ^ uint32(0xFFFFFFFF))
                case _:  # pragma: no cover
                    raise InternalInterpreterError(f"Unknown unary operator: {op}", tree=tree)
        assert is_numeric_basic_value(rhs)
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
            if op not in _STRING_BINARY_OPS and not is_numeric_basic_value(rhs):
                raise BasicMistakeError(f"Inappropriate type for operation {op}", tree=tree)
            lhs = children.pop()
            if is_numeric_basic_value(lhs) != is_numeric_basic_value(rhs):
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
                    rhs = basic_bool(lhs == rhs)
                case "<>":
                    rhs = basic_bool(lhs != rhs)
                case "<":
                    rhs = basic_bool(lhs < rhs)
                case ">":
                    rhs = basic_bool(lhs > rhs)
                case "<=":
                    rhs = basic_bool(lhs <= rhs)
                case ">=":
                    rhs = basic_bool(lhs >= rhs)
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
                    raise InternalInterpreterError(f"Unknown binary operator: {op}", tree=tree)
        assert is_basic_value(rhs), f"Non-numeric output: {rhs!r}"
        return rhs

    def variablerefexpr(self, tree: Tree):
        return self._variable_fetcher(tree, tree.children[0])
