import typing

import numpy as np
from lark import Tree, Visitor

from .parser import EXPRESSION_PARSER

BasicValue = typing.Union[np.int32, float, str]


class BasicError(RuntimeError):
    def __init__(self, message: str, tree: Tree):
        super().__init__(message)
        self.tree = tree


class InternalParserError(BasicError):
    pass


def _is_numeric_value(value: BasicValue):
    return isinstance(value, np.int32) or isinstance(value, float)


def _basic_bool(value: bool) -> BasicValue:
    return np.int32(-1) if value else np.int32(0)


class Interpreter:
    def execute(self, input_line: str) -> typing.Optional[BasicValue]:
        tree = EXPRESSION_PARSER.parse(input_line)
        _ExpressionVisitor().visit(tree)
        return tree.data


class _ExpressionVisitor(Visitor):
    def literalexpr(self, tree: Tree):
        token = tree.children[0]
        match token.type:
            case "BOOLEAN_LITERAL":
                tree.data = _basic_bool(token == "TRUE")
            case "BINARY_LITERAL":
                tree.data = np.int32(int(token[1:], base=2))
            case "HEX_LITERAL":
                tree.data = np.int32(int(token[1:], base=16))
            case "DECIMAL_LITERAL":
                tree.data = np.int32(int(token, base=10))
            case "FLOAT_LITERAL":
                tree.data = float(token)
            case _:
                raise InternalParserError("Unexpected literal type", tree)

    def unaryexpr(self, tree: Tree):
        children = list(tree.children)
        rhs = children.pop()
        tree.data = rhs.data
        while len(children) > 0:
            op = children.pop()
            if not _is_numeric_value(rhs.data):
                raise BasicError(f"Inappropriate type for unary operation {op}", rhs)
            match op:
                case "+":
                    pass  # Do nothing
                case "-":
                    tree.data = -tree.data
                case "NOT":
                    tree.data = np.int32(tree.data ^ np.uint32(0xFFFFFFFF))
                case _:
                    raise InternalParserError(f"Unknown unary operator: {op}", tree)
        assert _is_numeric_value(tree.data)

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
            op = children.pop()
            # TODO: string operations
            if not _is_numeric_value(rhs.data):
                raise BasicError(f"Inappropriate type for operation {op}", rhs)
            lhs = children.pop()
            if not _is_numeric_value(lhs.data):
                raise BasicError(f"Inappropriate type for operation {op}", lhs)
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
                case _:
                    raise InternalParserError(f"Unknown power operator: {op}", tree)
            rhs = lhs
        assert _is_numeric_value(tree.data), f"Non-numeric output: {tree.data!r}"
