import pytest

from rwbasic.interpreter import BasicMistakeError, BasicSyntaxError, Interpreter


@pytest.mark.parametrize(
    "line",
    [
        "10 PRINT 1234",
        "123 PRINT 1234:REM print a number",
    ],
)
def test_line_definition(interpreter: Interpreter, line: str, capsys):
    """Defining lines should not run them immediately."""
    interpreter.execute(line)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_zero_line_number(interpreter: Interpreter):
    with pytest.raises(BasicMistakeError):
        interpreter.execute("0 PRINT 1234")


def test_negative_line_number(interpreter: Interpreter):
    with pytest.raises(BasicSyntaxError):
        interpreter.execute("-10 PRINT 1234")


@pytest.fixture
def program_lines(interpreter: Interpreter):
    lines = [
        "20 REM some line",
        "10 PRINT 1234",
        "15 PRINT 1:PRINT 2:REM comment to eol",
        "30 PRINT 1:PRINT 2:REM comment to eol ignores next PRINT:PRINT 4",
    ]
    for line in lines:
        interpreter.execute(line)
    return interpreter._state.lines


def test_line_count(program_lines):
    assert len(program_lines) == 4


@pytest.mark.parametrize("line_number", [10, 15, 20, 30])
def test_line_number_present(program_lines, line_number):
    assert line_number in program_lines


@pytest.mark.parametrize(
    "line_number,statement_count",
    [
        (10, 1),
        (15, 3),
        (20, 1),
        (30, 3),
    ],
)
def test_line_number_statement_count(program_lines, line_number, statement_count):
    assert len(program_lines[line_number]) == statement_count
