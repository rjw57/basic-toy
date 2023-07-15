import pytest

from rwbasic.interpreter import (
    BasicBadProgramError,
    BasicMistakeError,
    BasicSyntaxError,
    Interpreter,
)


@pytest.mark.parametrize(
    "program,expected_output",
    [
        ('10PRINT"Hello, ";\n20PRINT"world"\n', "Hello, world\n"),
        ('10\n20PRINT"Hello, ";\n30\n40PRINT"world"', "Hello, world\n"),
        ('PRINT"Hello, ";\nPRINT"world"', "Hello, world\n"),
        ("", ""),
        ("PRINT1\nPRINT2\nEND\nPRINT3", "1\n2\n"),
    ],
)
def test_expected_output(interpreter: Interpreter, program: str, expected_output: str, capsys):
    interpreter.load_and_run_program(program)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "program",
    [
        # Bad line number ordering
        "10PRINT1\n5PRINT2",
    ],
)
def test_mistake_when_loading(interpreter: Interpreter, program: str):
    with pytest.raises(BasicBadProgramError):
        interpreter.load_program(program)


@pytest.mark.parametrize(
    "program",
    [
        "10A$",
    ],
)
def test_syntax_error_when_loading(interpreter: Interpreter, program: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.load_program(program)


@pytest.mark.parametrize(
    "program",
    [
        # Interactive statements in a program.
        "10RUN",
        "10NEW",
    ],
)
def test_mistake_when_running(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicMistakeError):
        interpreter.execute("RUN")


def test_empty_program(interpreter: Interpreter):
    """Can RUN with no program."""
    interpreter.execute("RUN")


def test_no_naked_end(interpreter: Interpreter):
    """Cannot have END without being in a program."""
    with pytest.raises(BasicMistakeError):
        interpreter.execute("END")
