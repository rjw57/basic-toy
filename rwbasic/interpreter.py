import dataclasses
import enum
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


def _zero_value_for_variable(variable_name: str) -> BasicValue:
    if variable_name.endswith("%"):
        return int32(0)
    elif variable_name.endswith("$"):
        return ""
    else:
        return 0.0


class _ExitReason(enum.Enum):
    END_REACHED = enum.auto()
    PROC_OR_FN_EXIT = enum.auto()


@dataclasses.dataclass
class _ExecutionContext:
    # Current execution location.
    current_location: ExecutionLocation

    # Next execution location or None if we should stop executing.
    next_location: typing.Optional[ExecutionLocation] = None

    # Why we exited this context
    exit_reason: _ExitReason = _ExitReason.END_REACHED

    # Value we exited this context with. (Used with function calls.)
    exit_value: typing.Optional[BasicValue] = None


class _ParseTreeInterpreter(LarkInterpreter):
    # Source code which was last parsed.
    _source: str

    # Mapping from variable name to current value.
    _variables: dict[str, BasicValue]

    # Local variable stack. If a variable name exists in the last entry of the stack, variable
    # reads and writes should come from here.
    _local_variable_stack: list[dict[str, BasicValue]]

    # Lines which make up the current program.
    _lines: _SortedNumberedLineList

    # Analysis of the current program.
    _program_analysis: typing.Optional[ProgramAnalysis]

    # Transformer used to evaluate expressions.
    _expression_transformer: ExpressionTransformer

    # Execution context stack.
    _exec_context: list[_ExecutionContext]

    def __init__(self):
        self._reset()
        self._expression_transformer = ExpressionTransformer(
            variable_fetcher=self._get_variable, function_caller=self._call_function
        )

    def evaluate_expression(self, expression: str) -> BasicValue:
        return self._evaluate_expression_tree(self._parse(expression, start="expression"))

    def execute_prompt_line(self, prompt_line: str):
        self._ensure_program_analysis()
        self.visit(self._parse(prompt_line, start="promptline"))

    def load_program(self, program_source: str):
        self.execute_prompt_line("NEW")
        self.visit(self._parse(program_source, start="program"))

    def _reset(self):
        """Reset state to defaults."""
        self._variables = dict()
        self._local_variable_stack = []
        self._lines = _SortedNumberedLineList()
        self._program_analysis = None
        self._exec_context = []

    def _ensure_program_analysis(self):
        if self._program_analysis is not None:
            return
        analysis_visitor = ProgramAnalysisVisitor()
        analysis_visitor.analyse_lines(line.statements for line in self._lines)
        self._program_analysis = analysis_visitor.analysis

    def _call_function(
        self, tree: Tree, func_name: str, arguments: list[BasicValue]
    ) -> BasicValue:
        if func_name.startswith("FN"):
            try:
                assert self._program_analysis is not None
                fn_stmt, entry_loc = self._program_analysis.proc_or_fn_entry_points[func_name]
            except KeyError:
                raise BasicMistakeError(f"No such function: {func_name}", tree=tree)
            fn_arg_names = fn_stmt.children[1:]
            if len(fn_arg_names) != len(arguments):
                raise BasicMistakeError(
                    f"Incorrect number of arguments for {func_name}", tree=tree
                )

            self._local_variable_stack.append(dict())
            for n, v in zip(fn_arg_names, arguments):
                self._local_variable_stack[-1][n] = _zero_value_for_variable(n)
                self._set_variable(tree, n, v)

            return self._execute(entry_loc).exit_value
        raise BasicMistakeError(f"No such function or array: {func_name}", tree=tree)

    def _get_variable(self, tree: Tree, var_name: str) -> BasicValue:
        if len(self._local_variable_stack) > 0:
            try:
                return self._local_variable_stack[-1][var_name]
            except KeyError:
                pass
        try:
            return self._variables[var_name]
        except KeyError as e:
            raise BasicMistakeError(f"Unknown variable: {var_name}", tree=tree) from e

    def _evaluate_expression_tree(self, expression: Tree) -> BasicValue:
        try:
            value = self._expression_transformer.transform(expression)
        except VisitError as e:
            raise e.orig_exc from e
        assert is_basic_value(value)
        return value

    def _write_output(self, text: str):
        sys.stdout.write(text)

    def _parse(self, input_text: str, **kwargs) -> Tree:
        try:
            tree = _load_parser().parse(input_text, **kwargs)
        except UnexpectedInput as lark_exception:
            raise BasicSyntaxError(str(lark_exception)) from lark_exception
        self._source = input_text
        return tree

    def _execute_inline_statements(self, statements: list[Tree]):
        """
        Execute a list of statements. Must only be called if we're evaluating statements from the
        prompt line.

        """
        self._ensure_program_analysis()
        for statement in statements:
            self.visit(statement)

    def _jump(self, location: ExecutionLocation):
        """
        When RUN-ing a program, jump to a new location.

        """
        self._exec_context[-1].next_location = location

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

        if len(self._local_variable_stack) > 0 and variable_name in self._local_variable_stack[-1]:
            self._local_variable_stack[-1][variable_name] = value
        else:
            self._variables[variable_name] = value

    def _execute(self, starting_from: ExecutionLocation):
        self._exec_context.append(_ExecutionContext(current_location=starting_from))
        context = self._exec_context[-1]

        self._ensure_program_analysis()

        # Keep going until we run out of program or the END statement sets the next location to
        # None.
        while True:
            try:
                current_line_statements = self._lines[
                    self._execution_location.line_index
                ].statements
            except IndexError:
                # We fell off the bottom of the program, exit the execution loop.
                break

            try:
                current_statement = current_line_statements[
                    self._execution_location.statement_index
                ]
            except IndexError:
                # We fell off the line, move to next line and loop.
                context.current_location = ExecutionLocation(
                    line_index=context.current_location.line_index + 1, statement_index=0
                )
                continue

            # Prepare the expected next location to be the following statement.
            context.next_location = ExecutionLocation(
                line_index=context.current_location.line_index,
                statement_index=context.current_location.statement_index + 1,
            )

            # Execute the current statement.
            self.visit(current_statement)

            # Move to next location which may have been altered by the statement we just executed.
            if context.next_location is None:
                break
            context.current_location = context.next_location

        return self._exec_context.pop()

    # Program definition

    def numbered_line_definition(self, tree: Tree):
        line_number = int(tree.children[0])
        if len(self._lines) > 0 and line_number <= self._lines[-1].number:
            raise BasicBadProgramError("Line numbers must increase in programs", tree=tree)
        self._add_line_definition(line_number, tree.children[1])

    def unnumbered_line_definition(self, tree: Tree):
        # Unnumbered lines in programs get numbered automagically.
        if len(self._lines) == 0:
            line_number = 1
        else:
            line_number = self._lines[-1].number + 1
        self._add_line_definition(line_number, tree.children[0])

    def _add_line_definition(self, line_number: int, line_statements: Tree):
        if line_number <= 0:
            raise BasicBadProgramError(
                "Line numbers must be greater than zero", tree=line_statements
            )

        # Extract the source for the statements.
        line_source = (
            self._source[line_statements.meta.start_pos : line_statements.meta.end_pos + 1]
            if len(line_statements.children) > 0
            else ""
        ).rstrip()

        # We store the individual statement trees as the line content. We replace any existing
        # lines with the same line number.
        new_line = _NumberedLine(
            number=line_number, statements=line_statements.children, source=line_source
        )
        insert_index = self._lines.bisect_key_left(new_line.number)
        if len(self._lines) > insert_index and self._lines[insert_index].number == line_number:
            del self._lines[insert_index]

        # Don't add blank lines. This lets one "delete" lines by adding black ones in.
        if line_source != "":
            self._lines.add(new_line)

        # Our program analysis is now out-of-date
        self._program_analysis = None

    # Prompt line statements

    def prompt_line_statements(self, tree: Tree):
        # Execute statements within line.
        self._execute_inline_statements(tree.children)

    def numbered_line_update(self, tree: Tree):
        line_number = int(tree.children[0])
        self._add_line_definition(line_number, tree.children[1])

    def new_statement(self, tree: Tree):
        self._reset()

    def list_statement(self, tree: Tree):
        if len(self._lines) == 0:
            return
        max_line_no = self._lines[-1].number
        num_len = len(f"{max_line_no}")
        for line in self._lines:
            self._write_output(f"{str(line.number).rjust(num_len)} {line.source}\n")

    def renumber_statement(self, tree: Tree):
        # TODO: if GOTO and friends make an appearance, we'll need some smarts here.
        for new_number, line in zip(itertools.count(10, 10), self._lines):
            line.number = new_number

    @property
    def _execution_location(self) -> ExecutionLocation:
        return self._exec_context[-1].current_location

    def run_statement(self, tree: Tree):
        # Nothing to do if there is no program.
        if len(self._lines) == 0:
            return

        # Start at first line
        self._execute(ExecutionLocation(line_index=0, statement_index=0))

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
            assert self._program_analysis is not None
            self._jump(self._program_analysis.if_jump_targets[self._execution_location])

    def else_statement(self, tree: Tree):
        # Skip this statement. If we're executing it, we fell through from the THEN body.
        assert self._program_analysis is not None
        self._jump(self._program_analysis.else_jump_targets[self._execution_location])

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
        assert self._program_analysis is not None
        loop_stmt, jump_loc = self._program_analysis.loop_definitions_and_bodies[
            self._execution_location
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

        index_value = self._variables[index_var_name]
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

    # CASE... ENDCASE

    def case_statement(self, tree: Tree):
        case_expr = tree.children[1]
        case_value = self._evaluate_expression_tree(case_expr)

        assert self._program_analysis is not None
        case_whens = self._program_analysis.case_whens[self._execution_location]
        for when_statement, jump_loc in case_whens:
            for when_expr in when_statement.children[1:]:
                when_value = self._evaluate_expression_tree(when_expr)
                if case_value == when_value:
                    self._jump(jump_loc)
                    return

        # No WHENs matched, look for an OTHERWISE.
        assert self._program_analysis is not None
        case_otherwise_jump_loc = self._program_analysis.case_otherwises.get(
            self._execution_location
        )
        if case_otherwise_jump_loc is not None:
            self._jump(case_otherwise_jump_loc)
            return

        # No match and no OTHERWISE, jump out of the CASE.
        assert self._program_analysis is not None
        self._jump(self._program_analysis.case_exit_points[self._execution_location])

    def when_statement(self, tree: Tree):
        self._exit_case(tree)

    def otherwise_statement(self, tree: Tree):
        self._exit_case(tree)

    def _exit_case(self, tree: Tree):
        # If we hit a WHEN or OTHERWISE statement when executing, we must have fallen through from
        # the case above.
        assert self._program_analysis is not None
        self._jump(self._program_analysis.case_exit_points[self._execution_location])

    # REPEAT... UNTIL loops

    def inline_repeat_statement(self, tree: Tree):
        body_statements = tree.children[1:-1]
        until_statement = tree.children[-1]

        while True:
            self._execute_inline_statements(body_statements)
            if not self._process_until(until_statement):
                break

    def until_statement(self, tree: Tree):
        assert self._program_analysis is not None
        _, jump_loc = self._program_analysis.loop_definitions_and_bodies[self._execution_location]
        if self._process_until(tree):
            self._jump(jump_loc)

    def _process_until(self, until_statement: Tree):
        "Returns True if the loop should continue."
        condition_expr = until_statement.children[1]
        condition_value = self._evaluate_expression_tree(condition_expr)
        if not is_numeric_basic_value(condition_value):
            raise BasicMistakeError("UNTIL conditions must be numeric", tree=condition_expr)
        return condition_value == 0

    # WHILE... ENDWHILE loops

    def while_statement(self, tree: Tree):
        assert self._program_analysis is not None
        exit_loc = self._program_analysis.while_exit_locations[self._execution_location]
        if not self._process_while_statement(tree):
            self._jump(exit_loc)

    def endwhile_statement(self, tree: Tree):
        assert self._program_analysis is not None
        while_loc = self._program_analysis.endwhile_while_locations[self._execution_location]
        self._jump(while_loc)

    def inline_while_statement(self, tree: Tree):
        while_statement = tree.children[0]
        body_statements = tree.children[1:-1]

        while self._process_while_statement(while_statement):
            self._execute_inline_statements(body_statements)

    def _process_while_statement(self, while_statement: Tree) -> bool:
        "Return True if the while body should be entered."
        condition_expr = while_statement.children[1]
        condition_value = self._evaluate_expression_tree(condition_expr)
        if not is_numeric_basic_value(condition_value):
            raise BasicMistakeError("WHILE conditions must be numeric", tree=condition_expr)
        return condition_value != 0

    # DEFPROC... ENDPROC

    def defproc_statement(self, tree: Tree):
        raise BasicMistakeError("Ran into DEFPROC")

    def endproc_statement(self, tree: Tree):
        self._local_variable_stack.pop()
        context = self._exec_context[-1]
        context.next_location = None
        context.exit_reason = _ExitReason.PROC_OR_FN_EXIT

    # DEFFN... =...

    def deffn_statement(self, tree: Tree):
        raise BasicMistakeError("Ran into DEFFN")

    def endfn_statement(self, tree: Tree):
        context = self._exec_context[-1]
        context.exit_value = self._evaluate_expression_tree(tree.children[0])
        self._local_variable_stack.pop()
        context.next_location = None
        context.exit_reason = _ExitReason.PROC_OR_FN_EXIT

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
                and tree.children[item_index].type in {"COMMA", "SEMICOLON", "APOSTROPHE"}
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
        self._exec_context[-1].next_location = None

    def proc_call_statement(self, tree: Tree):
        proc_name = tree.children[0]

        try:
            assert self._program_analysis is not None
            proc_stmt, entry_loc = self._program_analysis.proc_or_fn_entry_points[proc_name]
        except KeyError:
            raise BasicMistakeError(f"No such procedure: {proc_name}")

        proc_arg_names = proc_stmt.children[1:]
        arg_values = [self._evaluate_expression_tree(v) for v in tree.children[1:]]
        if len(proc_arg_names) != len(arg_values):
            raise BasicMistakeError(f"Incorrect number of arguments for {proc_name}", tree=tree)

        self._local_variable_stack.append(dict())
        for n, v in zip(proc_arg_names, arg_values):
            self._local_variable_stack[-1][n] = _zero_value_for_variable(n)
            self._set_variable(tree, n, v)

        self._execute(entry_loc)

    def local_statement(self, tree: Tree):
        assert len(self._local_variable_stack) > 0
        for var_name in tree.children[1:]:
            self._local_variable_stack[-1][var_name] = _zero_value_for_variable(var_name)


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
