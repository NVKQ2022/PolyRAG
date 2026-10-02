"""Data Models and Entities for polyrag."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Represents an input source document."""

    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None


@dataclass
class Chunk:
    """Represents a discrete segmented chunk of a document."""

    source: str
    chunk_id: int | str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_uuid: str | None = None

    @property
    def identifier(self) -> str:
        """Return canonical chunk identifier, e.g. 'rfc1035.txt#42'."""
        return f"{self.source}#{self.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk into a dictionary format."""
        data = {
            "source": self.source,
            "chunk_id": self.chunk_id,
            "text": self.text,
            **self.metadata,
        }
        if self.chunk_uuid:
            data["_id"] = self.chunk_uuid
        return data


@dataclass
class SearchResult:
    """Represents a vector similarity search result."""

    score: float
    distance: float
    document: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.document.get("source", "unknown")

    @property
    def chunk_id(self) -> Any:
        return self.document.get("chunk_id", "")

    @property
    def text(self) -> str:
        return self.document.get("text", "")

    @property
    def identifier(self) -> str:
        return f"{self.source}#{self.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "distance": self.distance,
            "document": self.document,
        }


@dataclass
class RAGResponse:
    """Represents the output of a RAG query pipeline."""

    question: str
    answer: str
    context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    took_ms: int = 0
    confidence: float = 1.0
    reasoning_summary: str = ""
    agent_log: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 1
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "sources": self.sources,
            "took_ms": self.took_ms,
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
            "agent_log": self.agent_log,
            "llm_calls": self.llm_calls,
            "state": self.state,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


@dataclass
class AgentAction:
    """Action selected by an agent in a reasoning step."""

    thought: str
    action: str
    action_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStep:
    """A step in an agent's reasoning trajectory."""

    step_num: int
    thought: str
    action: str
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    chunks_retrieved: int = 0
    took_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
            "chunks_retrieved": self.chunks_retrieved,
            "took_ms": self.took_ms,
        }


@dataclass
class AgentResponse:
    """Final output from an Agent execution."""

    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[AgentStep] = field(default_factory=list)
    reasoning_summary: str = ""
    confidence: float = 1.0
    total_steps: int = 0
    took_ms: int = 0
    llm_calls: int = 0
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "retrieved_evidence": self.retrieved_evidence,
            "trajectory": [s.to_dict() for s in self.trajectory],
            "reasoning_summary": self.reasoning_summary,
            "confidence": self.confidence,
            "total_steps": self.total_steps,
            "took_ms": self.took_ms,
            "llm_calls": self.llm_calls,
            "state": self.state,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


@dataclass
class BaseMessage:
    """Base class for chat messages conforming to modern LangChain message schemas."""

    content: str
    additional_kwargs: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    type: str = "base"

    def __str__(self) -> str:
        return self.content


@dataclass
class HumanMessage(BaseMessage):
    """Message representing user / human input."""

    type: str = "human"


@dataclass
class AIMessage(BaseMessage):
    """Message representing assistant / AI response with optional tool calls."""

    type: str = "ai"
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SystemMessage(BaseMessage):
    """Message representing system / context instructions."""

    type: str = "system"


@dataclass
class ToolMessage(BaseMessage):
    """Message representing output of a tool execution."""

    type: str = "tool"
    tool_call_id: str = ""

