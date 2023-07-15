"""
Read-eval-print loop for rwbasic.
"""
import sys
import typing

from better_exceptions import format_exception
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from .interpreter import BasicError, Interpreter


class ReplSession:
    style = Style.from_dict(
        {
            "error": "red",
        }
    )

    def __init__(self, *, prompt_session: typing.Optional[PromptSession] = None):
        self._prompt_session = prompt_session if prompt_session is not None else PromptSession()
        self._interpreter = Interpreter()

    def run(self):
        while True:
            try:
                prompt_line = self._prompt_session.prompt(">")
                self._interpreter.execute(prompt_line)
            except BasicError as err:
                self._print_error(str(err))
            except (EOFError, KeyboardInterrupt):
                # Exit from REPL on SIGINT or on end of input.
                break
            except Exception:
                self._print_error("Unexpected Python error:")
                for line in format_exception(*sys.exc_info()):
                    sys.stderr.write(line)

    def _print_error(self, error_message: str):
        print_formatted_text(
            FormattedText([("class:error", error_message)]), style=self.style, file=sys.stderr
        )
