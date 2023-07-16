"""
Read-eval-print loop for rwbasic.
"""
import sys
import typing

from better_exceptions import format_exception
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from prompt_toolkit.styles.pygments import style_from_pygments_cls
from pygments.lexers.basic import BBCBasicLexer
from pygments.styles import get_style_by_name

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

    def load_program_from_file(self, program_path: str):
        with open(program_path) as f:
            self._interpreter.load_program(f.read())

    def run(self):
        self._interpreter.execute("RUN")

    def start_interactive(self):
        # TODO: let style by configurable
        style = style_from_pygments_cls(get_style_by_name("solarized-dark"))
        while True:
            try:
                prompt_line = self._prompt_session.prompt(
                    ">",
                    lexer=PygmentsLexer(BBCBasicLexer),
                    style=style,
                    include_default_pygments_style=False,
                    auto_suggest=AutoSuggestFromHistory(),
                )
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
