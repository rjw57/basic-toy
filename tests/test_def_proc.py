import pytest

from rwbasic.exceptions import BasicBadProgramError, BasicMistakeError, BasicSyntaxError
from rwbasic.interpreter import Interpreter


@pytest.mark.parametrize(
    "program,expected_output",
    [
        ("PROCp:PROCp\nDEFPROCp:PRINT1:ENDPROC", "1\n1\n"),
        ("PROCx:END:DEFPROCx:ENDPROC:DEFPROCy:PRINT1:ENDPROC", ""),
        ("PROCpr(4):PROCpr(5):END:DEFPROCpr(N):PRINTN:ENDPROC\nPRINT6", "4\n5\n"),
        # It's OK to skip over procedures.
        ("PROCpr(4):PROCpr(5):DEFPROCpr(N):PRINTN:ENDPROC:PRINT6", "4\n5\n6\n"),
        # Procedure arguments are local
        ("I%=1:PROCx(2):PRINTI%:DEFPROCx(I%):PRINTI%:ENDPROC", "2\n1\n"),
        # Can reference global variables
        (
            'I%=1:J$="x":PRINTI%:PROCx("h"):PRINTI%:PRINTJ$\nDEFPROCx(J$):I%=5:PRINTI%\n'
            'PRINTJ$:J$="y":PRINTJ$:I%=2:ENDPROC',
            "1\n5\nh\ny\n2\nx\n",
        ),
        # Local variables
        ("I%=1:PROCh:PRINTI%:DEFPROCh:LOCALI%:I%=4:PRINTI%:ENDPROC", "4\n1\n"),
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
        "DEFPROC1a:ENDPROC",
        "DEFPROCa(b,c,3):ENDPROC",
    ],
)
def test_syntax_error_on_loading(interpreter: Interpreter, program: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.load_program(program)


@pytest.mark.parametrize(
    "program",
    [
        # ENDPROC outside of PROC
        "PRINT1\nENDPROC",
        # DEFPROC inside loop.
        "FORI%=1TO2\nDEFPROCa:ENDPROC\nNEXT",
        # Nested DEFPROC
        "DEFPROCa:DEFPROCb:ENDPROC:ENDPROC",
        # LOCAL outside of procedure.
        "LOCAL I%",
        "FORI%=1TO2:LOCAL I%:NEXT",
    ],
)
def test_bad_program_on_running(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicBadProgramError):
        interpreter.execute("RUN")


@pytest.mark.parametrize(
    "program",
    [
        # Bad type in procedure call
        'PROCa("hello"):DEFPROCa(b%):ENDPROC',
        # No such procedure
        "PROCa",
        # Incorrect number of arguments.
        "PROCa:DEFPROCa(N):ENDPROC",
        "PROCa(1):DEFPROCa:ENDPROC",
        "PROCa(1,2):DEFPROCa(N):ENDPROC",
    ],
)
def test_mistake_on_running(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicMistakeError):
        interpreter.execute("RUN")


@pytest.mark.parametrize(
    "line,expected_output",
    [
        ("PROCa(5)", "5\n"),
        ("FORI%=1TO3:PROCa(I%):NEXT", "1\n2\n3\n"),
        ("PROCb", "0\n"),
    ],
)
def test_inline_proc_call(interpreter: Interpreter, line: str, expected_output: str, capsys):
    interpreter.load_program("DEFPROCa(I%)\nPRINTI%\nENDPROC\nDEFPROCb\nPRINT0\nENDPROC")
    interpreter.execute(line)
    captured = capsys.readouterr()
    assert captured.out == expected_output
