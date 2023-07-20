import pytest

from rwbasic.exceptions import BasicBadProgramError, BasicMistakeError, BasicSyntaxError
from rwbasic.interpreter import Interpreter


@pytest.mark.parametrize(
    "program,expected_output",
    [
        ("PRINTFNa(4):PRINTFNa(5)\nEND\nDEFFNa(I%)\n=I%", "4\n5\n"),
        ("PRINTFNb()\nEND\nDEFFNb()\n=4", "4\n"),
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
        "DEFFN1a\n=1",
        "DEFFN(b,c,3)\n=5",
    ],
)
def test_syntax_error_on_loading(interpreter: Interpreter, program: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.load_program(program)


@pytest.mark.parametrize(
    "program",
    [
        # =... outside of DEFFN
        "PRINT1\n=4",
        # DEFFN inside loop.
        "FORI%=1TO2\nDEFFNa()\n=1\nNEXT",
        # ENDPROC in function
        "DEFFNa(N)\nENDPROC",
    ],
)
def test_bad_program_on_running(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicBadProgramError):
        interpreter.execute("RUN")


@pytest.mark.parametrize(
    "program",
    [
        # Bad type in function call
        'PRINTFNa("hello"):DEFFNa(b%)\n=b%+1',
        # No such function
        "PRINTFNa(3)",
        # Try to call badly named function
        "PRINTmidstring(3)",
        # Incorrect number of arguments.
        "PRINTFNa():DEFFNa(N)\n=N",
        "PRINTFNa(1):DEFFNa()\n=3",
        "PRINTFNa(1,2):DEFFNa(N)\n=N",
        # Run into DEFFN
        "DEFFNa():=4",
    ],
)
def test_mistake_on_running(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicMistakeError):
        interpreter.execute("RUN")
