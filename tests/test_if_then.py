import pytest

from rwbasic.interpreter import (
    BasicBadProgramError,
    BasicMistakeError,
    BasicSyntaxError,
    Interpreter,
)


@pytest.mark.parametrize(
    "line,expected_output",
    [
        ("IF TRUE THEN PRINT 1:PRINT 2 ELSE PRINT 3:PRINT 4", "1\n2\n"),
        ("IF TRUE THEN PRINT 1 ELSE PRINT 2", "1\n"),
        ("IF FALSE THEN PRINT 1:PRINT 2 ELSE PRINT 3:PRINT 4", "3\n4\n"),
        ("IF FALSE THEN PRINT 1 ELSE PRINT 2", "2\n"),
        ("IF TRUE THEN PRINT 1:PRINT 2", "1\n2\n"),
        ("IF FALSE THEN PRINT 1:PRINT 2", ""),
        ("IF TRUE PRINT 1:PRINT 2 ELSE PRINT 3:PRINT 4", "1\n2\n"),
        ("IF TRUE PRINT 1 ELSE PRINT 2", "1\n"),
        ("IF FALSE PRINT 1:PRINT 2 ELSE PRINT 3:PRINT 4", "3\n4\n"),
        ("IF FALSE PRINT 1 ELSE PRINT 2", "2\n"),
        ("IF TRUE PRINT 1:PRINT 2", "1\n2\n"),
        ("IF FALSE PRINT 1:PRINT 2", ""),
    ],
)
def test_expected_output(interpreter: Interpreter, line: str, expected_output: str, capsys):
    interpreter.execute(line)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "program,expected_output",
    [
        ("IF 1 THEN\nPRINT 1\nPRINT 2\nENDIF", "1\n2\n"),
        ("IF 1 THEN\nPRINT 1\nPRINT 2\nELSE\nPRINT 3\nPRINT 4\nENDIF", "1\n2\n"),
        ("IF 0 THEN\nPRINT 1\nPRINT 2\nENDIF", ""),
        ("IF 0 THEN\nPRINT 1\nPRINT 2\nELSE:PRINT 3\nPRINT 4\nENDIF", "3\n4\n"),
        ("IF 0 THEN\nPRINT 1\nPRINT 2\nELSE\nPRINT 3\nPRINT 4\nENDIF", "3\n4\n"),
        (
            "IF 1 THEN\nPRINT 1\nIF 1 THEN\nPRINT 3\nELSE:PRINT 4\nENDIF\nPRINT 2\nENDIF",
            "1\n3\n2\n",
        ),
    ],
)
def test_expected_program_output(
    interpreter: Interpreter, program: str, expected_output: str, capsys
):
    interpreter.load_and_run_program(program)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "line",
    [
        'IF "HELLO" THEN PRINT 1',
    ],
)
def test_mistake(interpreter: Interpreter, line: str):
    with pytest.raises(BasicMistakeError):
        interpreter.execute(line)


@pytest.mark.parametrize(
    "program",
    [
        "IF 1 THEN:PRINT 1\nENDIF",
        "IF 1 THEN\nPRINT 1:ENDIF",
        "IF 1 THEN\nPRINT 1\nELSE PRINT 2:ENDIF",
        "IF 1 THEN\nPRINT 1:ELSE PRINT 2\nENDIF",
    ],
)
def test_syntax_error_on_loading(interpreter: Interpreter, program: str):
    with pytest.raises(BasicSyntaxError):
        interpreter.load_program(program)


@pytest.mark.parametrize(
    "program",
    [
        # Various ENDIFs without matching IFs.
        "PRINT 1\nENDIF",
        "IF 1 THEN\nPRINT 2\nENDIF\nENDIF",
        "IF 1 THEN\nPRINT 2\nELSE\nIF 2 THEN\nPRINT 3\nENDIF\nENDIF\nENDIF",
        # Various ELSEs without matching IFs.
        "PRINT 1\nELSE",
        "IF 1 THEN\nPRINT 2\nENDIF\nELSE",
        "IF 1 THEN\nPRINT 2\nELSE\nIF 2 THEN\nPRINT 3\nENDIF\nELSE\nENDIF",
        # Repeated ELSEs
        "IF 1 THEN\nPRINT 2\nELSE\nPRINT 3\nELSE\nPRINT 4\nENDIF",
        # Unclosed IF
        "IF 1 THEN\nPRINT 2\nELSE\nIF 1 THEN\nPRINT 3\nENDIF",
        # ENDIF/ELSE in FOR loop
        "IF 1 THEN\nFOR I%=1 TO 2\nENDIF\nNEXT",
        "IF 1 THEN\nFOR I%=1 TO 2\nELSE\nPRINT 4\nNEXT",
    ],
)
def test_bad_program_on_running(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicBadProgramError):
        interpreter.execute("RUN")


@pytest.mark.parametrize(
    "program",
    [
        'IF "notanumber" THEN\nPRINT 1\nENDIF',
    ],
)
def test_mistake_on_running(interpreter: Interpreter, program: str):
    interpreter.load_program(program)
    with pytest.raises(BasicMistakeError):
        interpreter.execute("RUN")
