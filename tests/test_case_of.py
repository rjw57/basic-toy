import pytest

from rwbasic.exceptions import BasicBadProgramError, BasicSyntaxError
from rwbasic.interpreter import Interpreter


@pytest.mark.parametrize(
    "program,expected_output",
    [
        ("CASE 1 OF\nWHEN 2:PRINT 2\nWHEN 3,1:PRINT 1\nOTHERWISE\nPRINT 4\nENDCASE", "1\n"),
        ("CASE 2 OF\nWHEN 2:PRINT 2\nWHEN 3,1:PRINT 1\nOTHERWISE\nPRINT 4\nENDCASE", "2\n"),
        ('CASE "X" OF\nWHEN 2:PRINT 2\nWHEN 3,1:PRINT 1\nOTHERWISE\nPRINT 4\nENDCASE', "4\n"),
        (
            "CASE 3 OF\nWHEN 2:PRINT 2\nWHEN 3,1:PRINT 1\nOTHERWISE\nPRINT 4\nENDCASE\nPRINT 5",
            "1\n5\n",
        ),
        ("CASE 3 OF\nWHEN 2:PRINT 2\nWHEN 1:PRINT 1\nENDCASE\nPRINT 5", "5\n"),
    ],
)
def test_expected_program_output(
    interpreter: Interpreter, program: str, expected_output: str, capsys
):
    interpreter.load_and_run_program(program)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "program",
    [
        "CASE 1 OF:WHEN 2\nPRINT 1\nENDCASE",
        "CASE 1 OF\nWHEN 1,2\nPRINT 1\nPRINT 2:ENDCASE",
    ],
)
def test_syntax_error_on_loading(interpreter: Interpreter, program: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.load_program(program)


@pytest.mark.parametrize(
    "program",
    [
        # CASE with no ENDCASE
        "CASE 1 OF\nWHEN 1\nPRINT 1",
        # multiple OTHERWISE
        "CASE 1 OF\nOTHERWISE\nPRINT 1\nOTHERWISE\nPRINT 2\nENDCASE",
    ],
)
def test_bad_program_on_running(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicBadProgramError):
        interpreter.execute("RUN")
