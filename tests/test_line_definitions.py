import textwrap

import pytest

from rwbasic.interpreter import BasicBadProgramError, BasicSyntaxError, Interpreter


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
    with pytest.raises(BasicBadProgramError):
        interpreter.execute("0 PRINT 1234")


def test_negative_line_number(interpreter: Interpreter):
    with pytest.raises(BasicSyntaxError):
        interpreter.execute("-10 PRINT 1234")


@pytest.fixture
def program_lines(interpreter: Interpreter):
    lines = [
        "20 PRINT 20:REM original line",
        "10 PRINT 1234",
        "20 REM replace the line",
        "15 PRINT 1:PRINT 2:REM comment to eol",
        "30 PRINT 1:PRINT 2:REM comment to eol ignores next PRINT:PRINT 4",
        "100 PRINT 3",
    ]
    for line in lines:
        interpreter.execute(line)
    return interpreter._state.lines


def test_line_count(program_lines):
    assert len(program_lines) == 5


@pytest.mark.parametrize("line_number", [10, 15, 20, 30, 100])
def test_line_number_present(program_lines, line_number):
    assert len(list(program_lines.irange_key(line_number, line_number))) == 1


def test_line_number_statement_count(program_lines):
    assert [len(ln.statements) for ln in program_lines] == [1, 3, 1, 3, 1]


def test_list(program_lines, interpreter: Interpreter, capsys):
    interpreter.execute("LIST")
    captured = capsys.readouterr()
    assert captured.out == textwrap.dedent(
        """\
         10 PRINT 1234
         15 PRINT 1:PRINT 2:REM comment to eol
         20 REM replace the line
         30 PRINT 1:PRINT 2:REM comment to eol ignores next PRINT:PRINT 4
        100 PRINT 3
        """
    )


def test_renumber(program_lines, interpreter: Interpreter, capsys):
    interpreter.execute("RENUMBER:LIST")
    captured = capsys.readouterr()
    assert captured.out == textwrap.dedent(
        """\
        10 PRINT 1234
        20 PRINT 1:PRINT 2:REM comment to eol
        30 REM replace the line
        40 PRINT 1:PRINT 2:REM comment to eol ignores next PRINT:PRINT 4
        50 PRINT 3
        """
    )


def test_empty_list(interpreter: Interpreter, capsys):
    interpreter.execute("LIST")
    captured = capsys.readouterr()
    assert captured.out == ""
