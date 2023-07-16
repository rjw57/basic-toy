import pytest

from rwbasic.interpreter import BasicMistakeError, Interpreter


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
    "line",
    [
        'IF "HELLO" THEN PRINT 1',
    ],
)
def test_mistake(interpreter: Interpreter, line: str):
    with pytest.raises(BasicMistakeError):
        interpreter.execute(line)
