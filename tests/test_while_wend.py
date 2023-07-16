import pytest

from rwbasic.exceptions import BasicBadProgramError, BasicMistakeError, BasicSyntaxError
from rwbasic.interpreter import Interpreter


@pytest.mark.parametrize(
    "line,expected_output",
    [
        ("J%=1:WHILE J%<3:PRINT J%:J%=J%+1:WEND:PRINT 5", "1\n2\n5\n"),
        ("WHILE FALSE:PRINT 1:WEND:PRINT 2", "2\n"),
        ("WHILE FALSE:WEND:PRINT 2", "2\n"),
    ],
)
def test_expected_output(interpreter: Interpreter, line: str, expected_output: str, capsys):
    interpreter.execute(line)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "line",
    [
        # Bad value types.
        'WHILE "FALSE":WEND',
    ],
)
def test_mistake(interpreter: Interpreter, line: str):
    with pytest.raises(BasicMistakeError):
        interpreter.execute(line)


@pytest.mark.parametrize(
    "line",
    [
        # Missing WEND
        "WHILE FALSE:PRINT 1",
        # Missing WHILE
        "PRINT 1:WEND",
    ],
)
def test_syntax_error(interpreter: Interpreter, line: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.execute(line)


@pytest.mark.parametrize(
    "program,expected_output",
    [
        ("I%=1\nWHILE I%<3\nPRINT I%\nI%=I%+1\nWEND", "1\n2\n"),
        ("WHILE FALSE\nPRINT 1\nWEND\nPRINT 2", "2\n"),
    ],
)
def test_program(interpreter: Interpreter, program: str, expected_output: str, capsys):
    interpreter.load_and_run_program(program)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "program",
    [
        # Non-numeric WHILE
        'WHILE "FALSE"\nPRINT 1\nWEND',
    ],
)
def test_program_mistake(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicMistakeError):
        interpreter.execute("RUN")


@pytest.mark.parametrize(
    "program",
    [
        # Missing WEND
        "WHILE FALSE\nPRINT 1",
        # Missing WHILE
        "PRINT 1\nWEND",
        # WEND in IF THEN
        "WHILE FALSE\nIF 1 THEN\nWEND\nENDIF",
    ],
)
def test_bad_program(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicBadProgramError):
        interpreter.execute("RUN")
