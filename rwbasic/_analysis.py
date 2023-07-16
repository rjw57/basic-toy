import dataclasses
import enum
import typing

from lark import Tree, Visitor
from lark.visitors import v_args

from .exceptions import BasicBadProgramError


@dataclasses.dataclass(eq=True, frozen=True)
class ExecutionLocation:
    # Index of statement within execution line.
    statement_index: int

    # Index of line within the program. None indicates we're executing in the prompt line.
    line_index: int


@dataclasses.dataclass
class ProgramAnalysis:
    # Mapping from IF statement locations to locations to jump to if condition *NOT* met.
    if_jump_targets: dict[ExecutionLocation, ExecutionLocation] = dataclasses.field(
        default_factory=dict
    )

    # Mapping from ELSE statement locations to location to jump to if we want to skip it.
    else_jump_targets: dict[ExecutionLocation, ExecutionLocation] = dataclasses.field(
        default_factory=dict
    )

    # Mapping from loop block end statement locations to defining statements and body start
    # locations.
    loop_definitions_and_bodies: dict[
        ExecutionLocation, tuple[Tree, ExecutionLocation]
    ] = dataclasses.field(default_factory=dict)


@v_args(tree=True)
class ProgramAnalysisVisitor(Visitor):
    class _ControlFlowBlockType(enum.Enum):
        IF_THEN = enum.auto()
        FOR_NEXT = enum.auto()

    @dataclasses.dataclass
    class _ControlFlowBlockState:
        # What sort of control flow block is this?
        block_type: "ProgramAnalysisVisitor._ControlFlowBlockType"

        # The statement which initiated the block, (IF, FOR, etc.) and its location
        definition_statement: Tree
        definition_location: ExecutionLocation

        # The location of the start of the block content.
        entry_location: typing.Optional[ExecutionLocation] = None

        # The location to jump to to exit the block.
        exit_location: typing.Optional[ExecutionLocation] = None

        # For if-then blocks only, the location of the start of the ELSE block and the ELSE
        # statement itself.
        else_statement: typing.Optional[Tree] = None
        else_location: typing.Optional[ExecutionLocation] = None
        else_entry_location: typing.Optional[ExecutionLocation] = None

    analysis: ProgramAnalysis

    _control_flow_block_stack: list[_ControlFlowBlockState]
    _current_location: ExecutionLocation

    def __init__(self):
        self._reset()

    def _reset(self):
        self.analysis = ProgramAnalysis()
        self._control_flow_block_stack = []
        self._current_location = ExecutionLocation(statement_index=0, line_index=0)

    def analyse_lines(self, lines: typing.Iterable[typing.Iterable[Tree]]):
        self._reset()
        for line_index, line_statements in enumerate(lines):
            for statement_index, statement in enumerate(line_statements):
                self._current_location = ExecutionLocation(
                    statement_index=statement_index, line_index=line_index
                )
                self.visit(statement)

        if len(self._control_flow_block_stack) > 0:
            raise BasicBadProgramError(
                "Unclosed control flow block",
                tree=self._control_flow_block_stack[-1].definition_statement,
            )

    def _location_following_current(self) -> ExecutionLocation:
        "Helper to return a location pointing to the next statement."
        return ExecutionLocation(
            line_index=self._current_location.line_index,
            statement_index=self._current_location.statement_index + 1,
        )

    def for_statement(self, tree: Tree):
        self._control_flow_block_stack.append(
            ProgramAnalysisVisitor._ControlFlowBlockState(
                block_type=ProgramAnalysisVisitor._ControlFlowBlockType.FOR_NEXT,
                definition_statement=tree,
                definition_location=self._current_location,
                entry_location=self._location_following_current(),
            )
        )

    def next_statement(self, tree: Tree):
        try:
            block_state = self._control_flow_block_stack.pop()
        except IndexError:
            raise BasicBadProgramError("NEXT without matching FOR", tree=tree)

        if block_state.block_type != ProgramAnalysisVisitor._ControlFlowBlockType.FOR_NEXT:
            raise BasicBadProgramError("NEXT within unclosed non-FOR block", tree=tree)

        index_var_name = block_state.definition_statement.children[1]
        if len(tree.children) > 1:
            if tree.children[1] != index_var_name:
                raise BasicBadProgramError(
                    f"Unexpected NEXT variable name: {tree.children[1]}", tree=tree
                )

        assert block_state.entry_location is not None
        self.analysis.loop_definitions_and_bodies[self._current_location] = (
            block_state.definition_statement,
            block_state.entry_location,
        )

    def if_statement(self, tree: Tree):
        self._control_flow_block_stack.append(
            ProgramAnalysisVisitor._ControlFlowBlockState(
                block_type=ProgramAnalysisVisitor._ControlFlowBlockType.IF_THEN,
                definition_statement=tree,
                definition_location=self._current_location,
                entry_location=self._location_following_current(),
            )
        )

    def else_statement(self, tree: Tree):
        if len(self._control_flow_block_stack) == 0:
            raise BasicBadProgramError("ELSE without matching IF", tree=tree)

        block_state = self._control_flow_block_stack[-1]
        if block_state.else_statement is not None:
            raise BasicBadProgramError("Multiple ELSE within single IF", tree=tree)

        if block_state.block_type != ProgramAnalysisVisitor._ControlFlowBlockType.IF_THEN:
            raise BasicBadProgramError("ELSE within unclosed non-IF block", tree=tree)

        # Record the else statement. We jump to the statement following the else location.
        block_state.else_statement = tree
        block_state.else_location = self._current_location
        block_state.else_entry_location = self._location_following_current()

    def endif_statement(self, tree: Tree):
        try:
            block_state = self._control_flow_block_stack.pop()
        except IndexError:
            raise BasicBadProgramError("ENDIF without matching IF", tree=tree)

        if block_state.block_type != ProgramAnalysisVisitor._ControlFlowBlockType.IF_THEN:
            raise BasicBadProgramError("ENDIF within unclosed non-IF block", tree=tree)

        if block_state.else_statement is not None:
            assert block_state.else_location is not None
            assert block_state.else_entry_location is not None
            self.analysis.else_jump_targets[
                block_state.else_location
            ] = self._location_following_current()
            self.analysis.if_jump_targets[
                block_state.definition_location
            ] = block_state.else_entry_location
        else:
            self.analysis.if_jump_targets[
                block_state.definition_location
            ] = self._location_following_current()
