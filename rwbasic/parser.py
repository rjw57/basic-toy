"""
Parser for RWBASIC dialect.
"""
import structlog
from lark import Lark

LOG = structlog.get_logger()

GRAMMAR = """
    // Tokens
    %import common (DIGIT, HEXDIGIT, ESCAPED_STRING)

    // Ignore comments and whitespace
    %import common (WS_INLINE)
    %ignore WS_INLINE

    BOOLEAN_LITERAL: "TRUE" | "FALSE"
    BINARY_LITERAL: "%" ("0".."1")+
    DECIMAL_LITERAL: DIGIT+
    HEX_LITERAL: "&" HEXDIGIT+
    _EXPONENT: "E" ["+" | "-"] DIGIT+
    _FIXED_LITERAL: DECIMAL_LITERAL "." DECIMAL_LITERAL?
    FLOAT_LITERAL: DECIMAL_LITERAL _EXPONENT | _FIXED_LITERAL _EXPONENT?

    // Classes of operator grouped by priority. Operators listed first are acted upon before
    // operators listed second and so on.
    UNARYOP: "+" | "-" | "NOT" // TODO: indirection
    POWEROP: "^"
    MULOP: "*" | "/" | "DIV" | "MOD"
    ADDOP: "+" | "-"
    COMPOP: "=" | "<>" | "<" | ">" | "<=" | ">=" | "<<" | ">>" | ">>>"
    ANDOP: "AND"
    OROP: "OR" | "EOR"

    // Expressions
    ?expression: orexpr
    ?orexpr: (andexpr OROP)* andexpr
    ?andexpr: (compexpr ANDOP)* compexpr
    ?compexpr: (addexpr COMPOP)* addexpr
    ?addexpr: (mulexpr ADDOP)* mulexpr
    ?mulexpr: (powerexpr MULOP)* powerexpr
    ?powerexpr: (unaryexpr POWEROP)* unaryexpr
    ?unaryexpr: UNARYOP* atomexpr
    ?atomexpr: "(" expression ")"
             | literalexpr

    literalexpr: BOOLEAN_LITERAL | BINARY_LITERAL | DECIMAL_LITERAL | HEX_LITERAL | FLOAT_LITERAL
"""

EXPRESSION_PARSER = Lark(GRAMMAR, start="expression")
