import dataclasses
import itertools
import sys
import typing
from functools import cache

from lark import Lark, Token, Tree
from lark.exceptions import UnexpectedInput, VisitError
from lark.visitors import Interpreter as LarkInterpreter
from numpy import int32
from sortedcontainers import SortedKeyList

from ._analysis import ExecutionLocation, ProgramAnalysis, ProgramAnalysisVisitor
from ._expressions import ExpressionTransformer
from .exceptions import (
    BasicBadProgramError,
    BasicMistakeError,
    BasicSyntaxError,
    InternalInterpreterError,
)
from .values import (
    BasicValue,
    is_basic_value,
    is_numeric_basic_value,
    is_string_basic_value,
)

__all__ = ["Interpreter"]


@cache
def _load_parser() -> Lark:
    return Lark.open_from_package(
        __package__,
        "grammar.lark",
        propagate_positions=True,
        parser="lalr",
        cache=True,
        start=["program", "promptline", "expression"],
    )


@dataclasses.dataclass
class _NumberedLine:
    number: int
    statements: list[Tree]
    source: str


class _SortedNumberedLineList(SortedKeyList[_NumberedLine]):
    def __init__(self, iterable=None):
        super().__init__(iterable=iterable, key=lambda nl: nl.number)


@dataclasses.dataclass
class _InterpreterState:
    # Source code which was last parsed.
    source: str = ""

    # Mapping from variable name to current value.
    variables: dict[str, BasicValue] = dataclasses.field(default_factory=dict)

    # Lines which make up the current program.
    lines: _SortedNumberedLineList = dataclasses.field(default_factory=_SortedNumberedLineList)

    # Current execution location or None if we're not executing at the moment.
    execution_location: typing.Optional[ExecutionLocation] = None

    # Execution location for next statement or None if we should stop execution.
    next_execution_location: typing.Optional[ExecutionLocation] = None

    # Analysis of the current program.
    program_analysis: ProgramAnalysis = dataclasses.field(default_factory=ProgramAnalysis)

    def reset(self):
        """Reset state to defaults."""
        self.variables = dict()
        self.lines = _SortedNumberedLineList()
        self.execution_location = None
        self.next_execution_location = None
        self.program_analysis = ProgramAnalysis()


class _ParseTreeInterpreter(LarkInterpreter):
    _state: _InterpreterState
    _expression_transformer: ExpressionTransformer

    def __init__(self):
        super().__init__()
        self._state = _InterpreterState()
        self._expression_transformer = ExpressionTransformer(
            variable_fetcher=lambda v: self._state.variables[v]
        )

    def evaluate_expression(self, expression: str) -> BasicValue:
        """
        Evaluate an expression.

        """
        return self._evaluate_expression_tree(self._parse(expression, start="expression"))

    def _evaluate_expression_tree(self, expression: Tree) -> BasicValue:
        try:
            value = self._expression_transformer.transform(expression)
        except VisitError as e:
            raise e.orig_exc from e
        assert is_basic_value(value)
        return value

    def execute_prompt_line(self, prompt_line: str):
        self.visit(self._parse(prompt_line, start="promptline"))

    def load_program(self, program_source: str):
        self.execute_prompt_line("NEW")
        self.visit(self._parse(program_source, start="program"))

    def _write_output(self, text: str):
        sys.stdout.write(text)

    def _parse(self, input_text: str, **kwargs) -> Tree:
        try:
            tree = _load_parser().parse(input_text, **kwargs)
        except UnexpectedInput as lark_exception:
            raise BasicSyntaxError(str(lark_exception)) from lark_exception
        self._state.source = input_text
        return tree

    def _execute_inline_statements(self, statements: list[Tree]):
        """
        Execute a list of statements. Must only be called if we're evaluating statements from the
        prompt line.

        """
        for statement in statements:
            self.visit(statement)

    def _jump(self, location: ExecutionLocation):
        """
        When RUN-ing a program, jump to a new location.

        """
        self._state.next_execution_location = location

    def _set_variable(self, tree: Tree, variable_name: str, value: BasicValue):
        """
        Set a named variable to a Basic value.

        """
        assert is_basic_value(value)
        if variable_name.endswith("$") and not is_string_basic_value(value):
            raise BasicMistakeError("Cannot assign non-string value to string variable", tree=tree)
        if not variable_name.endswith("$") and not is_numeric_basic_value(value):
            raise BasicMistakeError("Cannot assign non-numeric value to variable", tree=tree)
        if variable_name.endswith("%"):
            # Ensure we only assign numbers to integer variables.
            value = int32(value)
        self._state.variables[variable_name] = value

    # Program definition

    def numbered_line_definition(self, tree: Tree):
        line_number = int(tree.children[0])
        if len(self._state.lines) > 0 and line_number <= self._state.lines[-1].number:
            raise BasicBadProgramError("Line numbers must increase in programs", tree=tree)
        self._add_line_definition(line_number, tree.children[1])

    def unnumbered_line_definition(self, tree: Tree):
        # Unnumbered lines in programs get numbered automagically.
        if len(self._state.lines) == 0:
            line_number = 1
        else:
            line_number = self._state.lines[-1].number + 1
        self._add_line_definition(line_number, tree.children[0])

    def _add_line_definition(self, line_number: int, line_statements: Tree):
        if line_number <= 0:
            raise BasicBadProgramError(
                "Line numbers must be greater than zero", tree=line_statements
            )

        # Extract the source for the statements.
        line_source = (
            self._state.source[line_statements.meta.start_pos : line_statements.meta.end_pos + 1]
            if len(line_statements.children) > 0
            else ""
        ).rstrip()

        # We store the individual statement trees as the line content. We replace any existing
        # lines with the same line number.
        new_line = _NumberedLine(
            number=line_number, statements=line_statements.children, source=line_source
        )
        insert_index = self._state.lines.bisect_key_left(new_line.number)
        if (
            len(self._state.lines) > insert_index
            and self._state.lines[insert_index].number == line_number
        ):
            del self._state.lines[insert_index]

        # Don't add blank lines. This lets one "delete" lines by adding black ones in.
        if line_source != "":
            self._state.lines.add(new_line)

    # Prompt line statements

    def prompt_line_statements(self, tree: Tree):
        # Execute statements within line.
        self._execute_inline_statements(tree.children)

    def numbered_line_update(self, tree: Tree):
        line_number = int(tree.children[0])
        self._add_line_definition(line_number, tree.children[1])

    def new_statement(self, tree: Tree):
        self._state.reset()

    def list_statement(self, tree: Tree):
        if len(self._state.lines) == 0:
            return
        max_line_no = self._state.lines[-1].number
        num_len = len(f"{max_line_no}")
        for line in self._state.lines:
            self._write_output(f"{str(line.number).rjust(num_len)} {line.source}\n")

    def renumber_statement(self, tree: Tree):
        # TODO: if GOTO and friends make an appearance, we'll need some smarts here.
        for new_number, line in zip(itertools.count(10, 10), self._state.lines):
            line.number = new_number

    def run_statement(self, tree: Tree):
        # Nothing to do if there is no program.
        if len(self._state.lines) == 0:
            return

        # Start at first line
        self._state.execution_location = ExecutionLocation(line_index=0, statement_index=0)

        # Analyse program loops, etc.
        analysis_visitor = ProgramAnalysisVisitor()
        analysis_visitor.analyse_lines(line.statements for line in self._state.lines)
        self._state.program_analysis = analysis_visitor.analysis

        # Keep going until we run out of program or the END statement sets the next location to
        # None.
        while self._state.execution_location is not None:
            try:
                current_line_statements = self._state.lines[
                    self._state.execution_location.line_index
                ].statements
            except IndexError:
                # We fell off the bottom of the program, exit the execution loop.
                break

            try:
                current_statement = current_line_statements[
                    self._state.execution_location.statement_index
                ]
            except IndexError:
                # We fell off the line, move to next line and loop.
                self._state.execution_location = ExecutionLocation(
                    line_index=self._state.execution_location.line_index + 1, statement_index=0
                )
                continue

            # Prepare the expected next location to be the following statement.
            self._state.next_execution_location = ExecutionLocation(
                line_index=self._state.execution_location.line_index,
                statement_index=self._state.execution_location.statement_index + 1,
            )

            # Execute the current statement.
            self.visit(current_statement)

            # Move to next location which may have been altered by the statement we just executed.
            self._state.execution_location = self._state.next_execution_location

    # IF... THEN... ELSE

    def inline_if_statement(self, tree: Tree):
        if_header, then_block = tree.children[:2]
        try:
            else_block = tree.children[3]
        except IndexError:
            else_block = None

        if self._test_if_condition(if_header):
            self._execute_inline_statements(then_block.children)
        elif else_block is not None:
            self._execute_inline_statements(else_block.children)

    def if_statement(self, tree: Tree):
        # Skip to ELSE or ENDIF if the condition is not met.
        if not self._test_if_condition(tree):
            assert self._state.execution_location is not None
            self._jump(
                self._state.program_analysis.if_jump_targets[self._state.execution_location]
            )

    def else_statement(self, tree: Tree):
        # Skip this statement. If we're executing it, we fell through from the THEN body.
        assert self._state.execution_location is not None
        self._jump(self._state.program_analysis.else_jump_targets[self._state.execution_location])

    def _test_if_condition(self, if_header: Tree) -> bool:
        condition_expr = if_header.children[1]
        condition_value = self._evaluate_expression_tree(condition_expr)
        if not is_numeric_basic_value(condition_value):
            raise BasicMistakeError("IF conditions must be numeric", tree=condition_expr)
        return condition_value != 0

    # FOR... NEXT loops

    def inline_for_statement(self, tree: Tree):
        for_statement = tree.children[0]
        body_statements = tree.children[1:-1]
        next_statement = tree.children[-1]

        self._begin_for(for_statement)
        while True:
            self._execute_inline_statements(body_statements)
            if not self._process_next(next_statement, for_statement):
                break

    def for_statement(self, tree: Tree):
        self._begin_for(tree)

    def next_statement(self, tree: Tree):
        assert self._state.execution_location is not None
        loop_stmt, jump_loc = self._state.program_analysis.loop_definitions_and_bodies[
            self._state.execution_location
        ]
        if self._process_next(tree, loop_stmt):
            self._jump(jump_loc)

    def _unpack_for_statement(self, for_statement: Tree):
        var_name = for_statement.children[1]
        from_expr = for_statement.children[3]
        to_expr = for_statement.children[5]
        try:
            step_expr = for_statement.children[7]
        except IndexError:
            step_expr = None
        return var_name, from_expr, to_expr, step_expr

    def _begin_for(self, for_statement: Tree):
        # TODO: do we care about repeated variable names?
        var_name, from_expr, _, _ = self._unpack_for_statement(for_statement)
        from_value = self._evaluate_expression_tree(from_expr)
        if not is_numeric_basic_value(from_value):
            raise BasicMistakeError("FOR start value must be numeric", tree=for_statement)
        self._set_variable(for_statement, var_name, from_value)

    def _process_next(self, next_statement: Tree, for_statement: Tree) -> bool:
        index_var_name, _, to_expr, step_expr = self._unpack_for_statement(for_statement)
        if len(next_statement.children) > 1:
            if next_statement.children[1] != index_var_name:
                raise BasicMistakeError(
                    f"Unexpected NEXT variable: {next_statement.children[1]}", tree=next_statement
                )

        to_value = self._evaluate_expression_tree(to_expr)
        if not is_numeric_basic_value(to_value):
            raise BasicMistakeError("FOR to value must be numeric", tree=to_expr)

        if step_expr is not None:
            step = self._evaluate_expression_tree(step_expr)
            if not is_numeric_basic_value(step):
                raise BasicMistakeError("FOR step value must be numeric", tree=step_expr)
        else:
            step = int32(1)

        index_value = self._state.variables[index_var_name]
        assert is_numeric_basic_value(index_value)
        next_index = index_value + step

        # Do we loop?
        if step >= 0 and next_index <= to_value:
            self._set_variable(next_statement, index_var_name, next_index)
            return True
        elif step < 0 and next_index >= to_value:
            self._set_variable(next_statement, index_var_name, next_index)
            return True
        return False

    # Other statements

    def print_statement(self, tree: Tree):
        item_index = 1
        separator = " "
        while item_index < len(tree.children):
            item_expression = tree.children[item_index]
            item_index += 1
            if (
                item_index < len(tree.children)
                and isinstance(tree.children[item_index], Token)
                and tree.children[item_index].type == "PRINT_ITEM_SEPARATOR"
            ):
                separator = tree.children[item_index]
                item_index += 1
            else:
                separator = " "

            value = self._evaluate_expression_tree(item_expression)
            self._write_output(f"{value}")
            match separator:
                case "'":
                    self._write_output("\n")
                case ";" | " ":
                    pass  # no-op separator
                case ",":
                    # FIXME: we don't properly support the column alignment separator "," yet.
                    self._write_output(" ")
                case _:  # pragma: no cover
                    raise InternalInterpreterError(
                        f"Unknown print item separator: {separator!r}", tree=tree
                    )

        if separator != ";":
            self._write_output("\n")

    def let_statement(self, tree: Tree):
        variable_name = tree.children[-3]
        value_expression = tree.children[-1]
        value = self._evaluate_expression_tree(value_expression)
        self._set_variable(tree, variable_name, value)

    def end_statement(self, tree: Tree):
        self._state.next_execution_location = None


class Interpreter:
    _parse_tree_interpreter: _ParseTreeInterpreter

    def __init__(self):
        self._parse_tree_interpreter = _ParseTreeInterpreter()

    def load_program(self, program: str):
        self._parse_tree_interpreter.load_program(program)

    def load_and_run_program(self, program: str):
        self.load_program(program)
        self.execute("RUN")

    def evaluate(self, expression: str) -> typing.Optional[BasicValue]:
        return self._parse_tree_interpreter.evaluate_expression(expression)

    def execute(self, prompt_line: str):
        self._parse_tree_interpreter.execute_prompt_line(prompt_line)
