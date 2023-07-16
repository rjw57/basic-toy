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


class _ControlFlowType(enum.Enum):
    IF_THEN = enum.auto()
    FOR_NEXT = enum.auto()
    REPEAT_UNTIL = enum.auto()


@dataclasses.dataclass
class _Block:
    """
    Control flow statements define one or more blocks. Loop statements define one block: the
    loop body. IF... THEN.. statements define the THEN block and, optionally, an ELSE block.
    """

    # The statement which defined this block and its location
    definition_statement: Tree
    definition_location: ExecutionLocation

    # The location of the entry point for this block
    entry_location: ExecutionLocation


@dataclasses.dataclass
class _ControlFlowState:
    # What sort of control flow is this?
    flow_type: _ControlFlowType

    # Blocks making up this control flow statement
    blocks: list[_Block] = dataclasses.field(default_factory=list)


@v_args(tree=True)
class ProgramAnalysisVisitor(Visitor):
    analysis: ProgramAnalysis

    _control_flow_stack: list[_ControlFlowState]
    _current_location: ExecutionLocation

    def __init__(self):
        self._reset()

    def _reset(self):
        self.analysis = ProgramAnalysis()
        self._control_flow_stack = []
        self._current_location = ExecutionLocation(statement_index=0, line_index=0)

    def analyse_lines(self, lines: typing.Iterable[typing.Iterable[Tree]]):
        self._reset()
        for line_index, line_statements in enumerate(lines):
            for statement_index, statement in enumerate(line_statements):
                self._current_location = ExecutionLocation(
                    statement_index=statement_index, line_index=line_index
                )
                self.visit(statement)

        if len(self._control_flow_stack) > 0:
            raise BasicBadProgramError(
                "Unclosed control flow statement",
                tree=self._control_flow_stack[-1].blocks[0].definition_statement,
            )

    def _location_following_current(self) -> ExecutionLocation:
        "Helper to return a location pointing to the next statement."
        return ExecutionLocation(
            line_index=self._current_location.line_index,
            statement_index=self._current_location.statement_index + 1,
        )

    def _push_new_flow(self, initial_statement: Tree, flow_type: _ControlFlowType):
        self._control_flow_stack.append(
            _ControlFlowState(
                flow_type=flow_type,
                blocks=[
                    _Block(
                        definition_statement=initial_statement,
                        definition_location=self._current_location,
                        entry_location=self._location_following_current(),
                    )
                ],
            )
        )

    def _peek_flow(self, end_statement: Tree, flow_type: _ControlFlowType):
        try:
            flow = self._control_flow_stack[-1]
        except IndexError:
            raise BasicBadProgramError(
                "Closing control flow statement outside of control flow", tree=end_statement
            )

        if flow.flow_type != flow_type:
            raise BasicBadProgramError(
                "Incorrect control flow closing statement for open block", tree=end_statement
            )

        return flow

    def _pop_flow(self, end_statement: Tree, flow_type: _ControlFlowType):
        flow = self._peek_flow(end_statement, flow_type)
        self._control_flow_stack.pop()
        return flow

    def for_statement(self, tree: Tree):
        self._push_new_flow(tree, _ControlFlowType.FOR_NEXT)

    def next_statement(self, tree: Tree):
        flow = self._pop_flow(tree, _ControlFlowType.FOR_NEXT)

        assert len(flow.blocks) == 1
        for_block = flow.blocks[0]

        index_var_name = for_block.definition_statement.children[1]
        if len(tree.children) > 1:
            if tree.children[1] != index_var_name:
                raise BasicBadProgramError(
                    f"Unexpected NEXT variable name: {tree.children[1]}", tree=tree
                )

        self.analysis.loop_definitions_and_bodies[self._current_location] = (
            for_block.definition_statement,
            for_block.entry_location,
        )

    def repeat_statement(self, tree: Tree):
        self._push_new_flow(tree, _ControlFlowType.REPEAT_UNTIL)

    def until_statement(self, tree: Tree):
        flow = self._pop_flow(tree, _ControlFlowType.REPEAT_UNTIL)

        assert len(flow.blocks) == 1
        repeat_block = flow.blocks[0]

        self.analysis.loop_definitions_and_bodies[self._current_location] = (
            repeat_block.definition_statement,
            repeat_block.entry_location,
        )

    def if_statement(self, tree: Tree):
        self._push_new_flow(tree, _ControlFlowType.IF_THEN)

    def else_statement(self, tree: Tree):
        flow = self._peek_flow(tree, _ControlFlowType.IF_THEN)

        if len(flow.blocks) > 1:
            raise BasicBadProgramError("Multiple ELSE within single IF", tree=tree)

        flow.blocks.append(
            _Block(
                definition_statement=tree,
                definition_location=self._current_location,
                entry_location=self._location_following_current(),
            )
        )

    def endif_statement(self, tree: Tree):
        flow = self._pop_flow(tree, _ControlFlowType.IF_THEN)

        assert len(flow.blocks) >= 1 and len(flow.blocks) <= 2

        if_block = flow.blocks[0]
        try:
            else_block = flow.blocks[1]
        except IndexError:
            else_block = None

        if else_block is not None:
            self.analysis.else_jump_targets[
                else_block.definition_location
            ] = self._location_following_current()
            self.analysis.if_jump_targets[if_block.definition_location] = else_block.entry_location
        else:
            self.analysis.if_jump_targets[
                if_block.definition_location
            ] = self._location_following_current()
