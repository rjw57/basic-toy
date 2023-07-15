"""
Execute RW-BASIC programs.

Usage:
    {cmd} (-h | --help)
    {cmd} [<program>]

Options:
    -h, --help      Show a brief usage summary.

    <program>       Load and run a program from a file.
"""
import os
import sys

import docopt

from .interpreter import Interpreter
from .repl import ReplSession


def main():
    opts = docopt.docopt(__doc__.format(cmd=os.path.basename(sys.argv[0])))
    if opts["<program>"] is not None:
        interpreter = Interpreter()
        with open(opts["<program>"]) as f:
            interpreter.load_and_run_program(f.read())
    else:
        ReplSession().run()
