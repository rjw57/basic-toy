import pytest

from rwbasic.exceptions import BasicBadProgramError, BasicMistakeError, BasicSyntaxError
from rwbasic.interpreter import Interpreter


@pytest.mark.parametrize(
    "line,expected_output",
    [
        ("FOR I%=1TO3:PRINT I%;:NEXT", "123"),
        ("FOR I%=1TO9STEP3:PRINT I%;:NEXT", "147"),
        ("FOR I%=9TO1STEP-3:PRINT I%;:NEXT", "963"),
        ("FOR I%=9TO1:PRINT I%;:NEXT", "9"),  # For loop always executes at least once
        ("FOR I%=1 TO 3:NEXT:PRINT I%;", "3"),  # Loop variable not incremented beyond bound
    ],
)
def test_expected_output(interpreter: Interpreter, line: str, expected_output: str, capsys):
    interpreter.execute(line)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "line",
    [
        # Incorrectly nested NEXT
        "FOR I%=1 TO 2:FOR J%=3 TO 4:PRINT I%:NEXT I%:NEXT J%",
        # Bad value types.
        'FOR I%="hello" TO 2:PRINT I%:NEXT',
        'FOR I%=1 TO "hello":PRINT I%:NEXT',
        'FOR I%=1 TO 2 STEP "hello":PRINT I%:NEXT',
    ],
)
def test_mistake(interpreter: Interpreter, line: str):
    with pytest.raises(BasicMistakeError):
        interpreter.execute(line)


@pytest.mark.parametrize(
    "line",
    [
        # Missing NEXT
        "FOR I%=1 TO 2:PRINT I%",
        "FOR I%=1 TO 2",
        # Missing nested NEXT
        "FOR I%=1 TO 2:FOR J%=3 TO 4:PRINT I%:NEXT J%",
        # Unexpected NEXT
        "PRINT 5:NEXT",
    ],
)
def test_syntax_error(interpreter: Interpreter, line: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.execute(line)


@pytest.mark.parametrize(
    "program",
    [
        # Multi-line for loops must be defined as the last statement on the line.
        "FORI%=1TO3:FORJ%=I%TO5STEP2\nPRINTI%,J%\nNEXT:NEXT",
    ],
)
def test_syntax_error_on_load(interpreter: Interpreter, program: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.load_program(program)


@pytest.mark.parametrize(
    "program,expected_output",
    [
        (
            "FORI%=1TO3\nFORJ%=I%TO5STEP2\nPRINTI%,J%\nNEXT:NEXT",
            "1 1\n1 3\n1 5\n2 2\n2 4\n3 3\n3 5\n",
        ),
        (
            "FORI%=1TO3\nNEXT:PRINTI%",
            "3\n",
        ),
        (
            "FORI%=10TO5STEP-2\nPRINTI%:NEXT",
            "10\n8\n6\n",
        ),
    ],
)
def test_program(interpreter: Interpreter, program: str, expected_output: str, capsys):
    interpreter.load_and_run_program(program)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "program",
    [
        # Non-numeric bounds
        'FOR I%="one" TO 2\nNEXT I%',
        'FOR I%=1 TO "two"\nNEXT I%',
        'FOR I%=1 TO 10 STEP "two"\nNEXT I%',
    ],
)
def test_program_mistake(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicMistakeError):
        interpreter.execute("RUN")


@pytest.mark.parametrize(
    "program",
    [
        # Missing NEXT
        "FOR I%=1 TO 2\nPRINT I%",
        "FOR I%=1 TO 2",
        # Missing nested NEXT
        "FOR I%=1 TO 2\nFOR J%=3 TO 4\nPRINT I%:NEXT J%",
        # Unexpected NEXT
        "PRINT 5:NEXT",
        # Incorrect NEXT variable
        "FOR I%=1 TO 2\nNEXT J%",
        # NEXT in IF THEN
        "FOR I%=1 TO 2\nIF 1 THEN\nNEXT\nENDIF\nNEXT",
    ],
)
def test_bad_program(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicBadProgramError):
        interpreter.execute("RUN")
