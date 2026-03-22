"""Adaptive Adversarial Resilience framework — core module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class TaskRequest:
    """A single task submitted to the runtime tier."""
    goal: str
    raw_input: str
    planned_tools: List[str] = field(default_factory=list)
    system_instructions: str = ""
    max_response_length: int = 500


@dataclass
class TraceSegment:
    """One hop in the data-flow trace that AgentArmor inspects."""
    tool: str
    source: str
    data: str
    sink: Optional[str] = None


class ToolDependencyGraph:
    """Directed acyclic graph that describes legal orderings of tools."""

    def __init__(self) -> None:
        self._forward: Dict[str, Set[str]] = {}
        self._all_tools: Set[str] = set()

    def add_tool(self, name: str) -> None:
        self._all_tools.add(name)
        self._forward.setdefault(name, set())

    def add_dependency(self, predecessor: str, successor: str) -> None:
        for t in (predecessor, successor):
            self.add_tool(t)
        self._forward[predecessor].add(successor)

    @property
    def tools(self) -> Set[str]:
        return set(self._all_tools)

    def is_valid_sequence(self, seq: List[str]) -> bool:
        if not seq:
            return True
        position = {tool: idx for idx, tool in enumerate(seq)}
        for pred, succs in self._forward.items():
            if pred not in position:
                continue
            for succ in succs:
                if succ in position and position[pred] >= position[succ]:
                    return False
        return True


def build_default_tdg() -> ToolDependencyGraph:
    """Return a sensible default graph: read -> query -> write."""
    tdg = ToolDependencyGraph()
    tdg.add_dependency("read", "query")
    tdg.add_dependency("query", "write")
    return tdg


class IntentAnalyzer:
    """Base intent-consistency checker (keyword heuristic)."""

    _SUSPICIOUS_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)", re.I),
        re.compile(r"disregard\s+(everything|prior)", re.I),
        re.compile(r"you\s+are\s+now", re.I),
        re.compile(r"forget\s+(everything|your\s+instructions?)", re.I),
        re.compile(r"new\s+instructions?:", re.I),
        re.compile(r"system\s*:\s*override", re.I),
    ]

    def is_intent_consistent(self, goal: str, incoming_text: str) -> bool:
        for pat in self._SUSPICIOUS_PATTERNS:
            if pat.search(incoming_text):
                return False
        goal_tokens = set(goal.lower().split())
        input_tokens = set(incoming_text.lower().split())
        if not goal_tokens:
            return True
        overlap = len(goal_tokens & input_tokens) / len(goal_tokens)
        return overlap >= 0.25


class StructuralFilter:
    """Strip/neutralise content that looks like embedded instructions."""

    _STRIP_PATTERNS: List[re.Pattern[str]] = [
        re.compile(r"<\|.*?\|>", re.S),
        re.compile(r"\{\{.*?\}\}", re.S),
        re.compile(r"(?i)ignore\s+(previous|above|all)\s+\w+"),
        re.compile(r"(?i)disregard\s+\w+"),
        re.compile(r"(?i)system\s*:\s*override"),
    ]

    def sanitize(self, text: str) -> str:
        cleaned = text
        for pat in self._STRIP_PATTERNS:
            cleaned = pat.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned


class ExecutionGuard:
    """Checks that a planned tool sequence is legal."""

    def __init__(self, tdg: ToolDependencyGraph) -> None:
        self._tdg = tdg

    def check(self, planned_tools: List[str]) -> bool:
        unknown = set(planned_tools) - self._tdg.tools
        if unknown:
            return False
        return self._tdg.is_valid_sequence(planned_tools)


class AgentArmor:
    """Lightweight taint tracker that flags unsafe data flows."""

    _SAFE_SOURCES = frozenset({"trusted", "sanitized"})

    def __init__(self) -> None:
        self._segments: List[TraceSegment] = []

    def log_segment(self, segment: TraceSegment) -> None:
        self._segments.append(segment)

    def validate(self) -> bool:
        for seg in self._segments:
            if seg.sink == "high_privilege" and seg.source not in self._SAFE_SOURCES:
                return False
        return True

    def clear(self) -> None:
        self._segments.clear()


class RuntimeTier:
    """Top-level orchestrator that wires analyser -> filter -> guard -> armor."""

    def __init__(self, tdg: ToolDependencyGraph) -> None:
        self.tdg = tdg
        self.intent_analyzer: IntentAnalyzer = IntentAnalyzer()
        self.structural_filter: StructuralFilter = StructuralFilter()
        self.execution_guard: ExecutionGuard = ExecutionGuard(tdg)
        self.agent_armor: AgentArmor = AgentArmor()

    def execute_task(self, request: TaskRequest) -> bool:
        if not self.intent_analyzer.is_intent_consistent(request.goal, request.raw_input):
            return False
        request.raw_input = self.structural_filter.sanitize(request.raw_input)
        if not self.execution_guard.check(request.planned_tools):
            return False
        return True
