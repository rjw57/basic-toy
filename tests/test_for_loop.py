import pytest

from rwbasic.interpreter import BasicMistakeError, Interpreter


@pytest.mark.parametrize(
    "line,expected_output",
    [
        ("FOR I%=1TO3:PRINT I%;:NEXT", "123"),
        ("FOR I%=1TO9STEP3:PRINT I%;:NEXT", "147"),
        ("FOR I%=9TO1STEP-3:PRINT I%;:NEXT", "963"),
    ],
)
def test_expected_output(interpreter: Interpreter, line: str, expected_output: str, capsys):
    interpreter.execute(line)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "line",
    [
        # Missing NEXT
        "FOR I%=1 TO 2:PRINT I%",
        "FOR I%=1 TO 2",
        # Missing nested NEXT
        "FOR I%=1 TO 2:FOR J%=3 TO 4:PRINT I%:NEXT J%",
        # Incorrectly nested NEXT
        "FOR I%=1 TO 2:FOR J%=3 TO 4:PRINT I%:NEXT I%:NEXT J%",
        # Unexpected NEXT
        "PRINT 5:NEXT",
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
    "program,expected_output",
    [
        (
            "FORI%=1TO3:FORJ%=I%TO5STEP2\nPRINTI%,J%\nNEXT:NEXT",
            "1 1\n1 3\n1 5\n2 2\n2 4\n3 3\n3 5\n",
        )
    ],
)
def test_program(interpreter: Interpreter, program: str, expected_output: str, capsys):
    interpreter.load_and_run_program(program)
    captured = capsys.readouterr()
    assert captured.out == expected_output
