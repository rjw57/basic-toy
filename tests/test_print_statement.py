import pytest

from rwbasic.interpreter import BasicMistakeError, Interpreter


@pytest.mark.parametrize(
    "line,expected_output",
    [
        ('PRINT "hello"', "hello\n"),
        ('PRINT "hello";', "hello"),
        ('PRINT "hello, ";:print "world"', "hello, world\n"),
        ('PRINT "hello" : REM print a greeting', "hello\n"),
        ("PRINT 1+2*3+4", "11\n"),
        ('PRINT "hello, "+"world";', "hello, world"),
    ],
)
def test_expected_output(interpreter: Interpreter, line: str, expected_output: str, capsys):
    interpreter.execute(line)
    captured = capsys.readouterr()
    assert captured.out == expected_output


@pytest.mark.parametrize(
    "line",
    [
        'PRINT 1+"hello"',
    ],
)
def test_mistake(interpreter: Interpreter, line: str):
    with pytest.raises(BasicMistakeError):
        interpreter.execute(line)
