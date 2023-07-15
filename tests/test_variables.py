import pytest

from rwbasic.interpreter import BasicMistakeError, BasicValue, Interpreter


@pytest.mark.parametrize(
    "setup,expression,expected_value",
    [
        ("A=1:B=2", "A+B", 3),
        ("A=1:LETA=A+1", "A", 2),
        ('NAME$="Jane":GREETING$="Hello, "', "GREETING$+NAME$", "Hello, Jane"),
        ("I%=1.2", "I%", 1),
        ("I%=1/2", "I%", 0),
        ("I%=2:X=3", "I%/X", 2 / 3),
    ],
)
def test_expected_output(
    interpreter: Interpreter, setup: str, expression: str, expected_value: BasicValue
):
    interpreter.execute(setup)
    assert interpreter.evaluate(expression) == expected_value


@pytest.mark.parametrize(
    "line",
    [
        # Inappropriate types
        'A="hello"',
        'IVAL%="hello"',
        "NAME$=1",
    ],
)
def test_mistake(interpreter: Interpreter, line: str):
    with pytest.raises(BasicMistakeError):
        interpreter.execute(line)


def test_no_such_variable(interpreter: Interpreter):
    with pytest.raises(BasicMistakeError):
        interpreter.evaluate("A$")
