import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import PromptSession

from rwbasic.repl import ReplSession


@pytest.fixture
def pipe_input():
    with create_pipe_input() as inp:
        yield inp


@pytest.fixture
def repl_session(pipe_input) -> ReplSession:
    prompt_session = PromptSession(input=pipe_input, output=DummyOutput())
    return ReplSession(prompt_session=prompt_session)


def test_no_input(repl_session, pipe_input):
    pipe_input.send_text("\x03")  # == Ctrl-C
    repl_session.start_interactive()


def test_print_immediate(repl_session, pipe_input, capsys):
    pipe_input.send_text('PRINT "Hello"\n')
    pipe_input.send_text("\x03")  # == Ctrl-C
    repl_session.start_interactive()
    captured = capsys.readouterr()
    assert captured.out == "Hello\n"


def test_syntax_error(repl_session, pipe_input, mocker, capsys):
    mock_print = mocker.patch("rwbasic.repl.print_formatted_text")
    pipe_input.send_text("this is not basic\n")
    pipe_input.send_text('print "hello"\n')
    pipe_input.send_text("\x03")  # == Ctrl-C
    repl_session.start_interactive()
    mock_print.assert_called()
    captured = capsys.readouterr()
    assert captured.out == "hello\n"


def test_internal_error(repl_session, pipe_input, mocker):
    """An internal error in the interpreter is handled gracefully."""
    mock_print = mocker.patch("rwbasic.repl.print_formatted_text")
    mock_execute = mocker.patch(
        "rwbasic.interpreter.Interpreter.execute", side_effect=RuntimeError
    )
    pipe_input.send_text("print 1\n")
    pipe_input.send_text("\x03")  # == Ctrl-C
    repl_session.start_interactive()
    mock_execute.assert_called()
    mock_print.assert_called()
