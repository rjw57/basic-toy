"""
Execute RW-BASIC programs.

Usage:
    {cmd}
    {cmd} (-h | --help)
    {cmd} [-i | --interactive] <program>

Options:
    -h, --help          Show a brief usage summary.
    -i, --interactive   Load program but drop to interactive prompt.

    <program>           Load and run a program from a file.
"""
import os
import sys

import docopt

from .repl import ReplSession


def main():
    opts = docopt.docopt(__doc__.format(cmd=os.path.basename(sys.argv[0])))

    session = ReplSession()
    if opts["<program>"] is not None:
        session.load_program_from_file(opts["<program>"])

    if opts["--interactive"] or opts["<program>"] is None:
        session.start_interactive()
    else:
        session.run()
