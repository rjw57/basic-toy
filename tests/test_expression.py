import pytest

from rwbasic.interpreter import BasicMistakeError, BasicSyntaxError, Interpreter


@pytest.fixture
def interpreter():
    return Interpreter()


@pytest.mark.parametrize(
    "expr,value",
    [
        # Decimals
        ("123456", 123456),
        ("0123", 123),
        ("987", 987),
        ("5", 5),
        # Binary
        ("%0", 0),
        ("%1", 1),
        ("%0101", 0b0101),
        ("%10", 0b10),
        # Hex
        ("&0", 0),
        ("&1", 1),
        ("&F", 0xF),
        ("&1f2B", 0x1F2B),
        # Float
        ("1.", 1.0),
        ("98.76", 98.76),
        ("1E-2", 1e-2),
        ("1.0E-2", 1e-2),
        ("1.0E5", 1e5),
        # Boolean
        ("TRUE", -1),
        ("True", -1),
        ("FALSE", 0),
        ("False", 0),
        # String
        ('"Hello, world"', "Hello, world"),
        ('""', ""),
        ('"String with ""quotes"""', 'String with "quotes"'),
        # Unary ops
        ("-123", -123),
        ("--123", 123),
        ("-+789.123", -789.123),
        ("+-1.", -1.0),
        ("NOT 3", -4),
        ("Not 3", -4),
        ("NOT TRUE", 0),
        ("NOT FALSE", -1),
        # Power ops
        ("-2^2", 4),
        ("-(2^2)", -4),
        ("2^2", 4),
        ("4^0.5", 2),
        # Multiplication ops
        ("-2.2/2", -1.1),
        ("-2.2 DIV 2", -2),
        ("2.2 DIV 2", 1),
        ("2.2 Div 2", 1),
        ("4.5*2", 9),
        ("4.5*-4", -18),
        ("7 MOD 2", 1),
        ("7 mod 2", 1),
        # Addition ops
        ("1+2", 3),
        ("-2+-1", -3),
        ("-2--1", -1),
        ("1+2*2+5", 10),
        ("(1+2)*2+5", 11),
        ("1+2*(2+5)", 15),
        # Comparison ops
        ("1=2", 0),
        ("1=1", -1),
        ("1<>2", -1),
        ("1<>1", 0),
        ("1<1", 0),
        ("1<2", -1),
        ("2<1", 0),
        ("1>1", 0),
        ("2>1", -1),
        ("1>2", 0),
        ("1<=1", -1),
        ("2<=1", 0),
        ("1<=2", -1),
        ("1>=1", -1),
        ("2>=1", -1),
        ("1>=2", 0),
        ("3<<2", 3 << 2),
        ("123>>2", 123 >> 2),
        ("123>>>2", 123 >> 2),
        ("-123>>2", -31),
        ("-123>>>2", (-123) >> 2),
        ('"Hello"="Hello"', -1),
        ('"Hello"<>"Hello"', 0),
        ('"Hello"="World"', 0),
        ('"Hello"<>"World"', -1),
        ('"Hello, "+"world"', "Hello, world"),
        # Logical operators
        ("&ff AND &5", 0xFF & 0x5),
        ("&f0 OR &5", 0xF0 | 0x5),
        ("&ff EOR &5", 0xFF ^ 0x5),
    ],
)
def test_constant_expression(interpreter, expr, value):
    assert interpreter.execute(expr) == value


@pytest.mark.parametrize(
    "expr",
    [
        # Inappropriate types
        '-"hello"',
        '"hello"-"world"',
        '"hello"<"world"',
        # Mismatched types
        '5+"one"',
        '"one"+5',
    ],
)
def test_mistake(interpreter, expr):
    with pytest.raises(BasicMistakeError):
        interpreter.execute(expr)


@pytest.mark.parametrize(
    "expr",
    [
        "1.2.3",
    ],
)
def test_parse_error(interpreter, expr):
    with pytest.raises(BasicSyntaxError):
        interpreter.execute(expr)
