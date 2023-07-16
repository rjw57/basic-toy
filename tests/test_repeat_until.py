import pytest

from rwbasic.exceptions import BasicBadProgramError, BasicMistakeError, BasicSyntaxError
from rwbasic.interpreter import Interpreter


@pytest.mark.parametrize(
    "line,expected_output",
    [
        ("J%=1:REPEAT:PRINT J%:J%=J%+1:UNTIL J%=3:PRINT 5", "1\n2\n5\n"),
        ("REPEAT:UNTIL TRUE:PRINT 2", "2\n"),
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
        'REPEAT:UNTIL "FALSE"',
    ],
)
def test_mistake(interpreter: Interpreter, line: str):
    with pytest.raises(BasicMistakeError):
        interpreter.execute(line)


@pytest.mark.parametrize(
    "line",
    [
        # Missing UNTIL
        "REPEAT:PRINT 1",
        # Missing REPEAT
        "PRINT 1:UNTIL FALSE",
    ],
)
def test_syntax_error(interpreter: Interpreter, line: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.execute(line)


@pytest.mark.parametrize(
    "program,expected_output",
    [
        ("I%=1\nREPEAT\nPRINT I%\nI%=I%+1\nUNTIL I%=3", "1\n2\n"),
    ],
)
def test_program(interpreter: Interpreter, program: str, expected_output: str, capsys):
    interpreter.load_and_run_program(program)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "program",
    [
        # Non-numeric UNTIL
        'REPEAT\nPRINT 1\nUNTIL "TRUE"',
    ],
)
def test_program_mistake(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicMistakeError):
        interpreter.execute("RUN")


@pytest.mark.parametrize(
    "program",
    [
        # Missing UNTIL
        "REPEAT\nPRINT 1",
        # Missing REPEAT
        "PRINT 1\nUNTIL TRUE",
        # UNTIL in IF THEN
        "REPEAT\nIF 1 THEN\nUNTIL TRUE\nENDIF",
    ],
)
def test_bad_program(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicBadProgramError):
        interpreter.execute("RUN")
