"""Data Models and Entities for polyrag built on LangChain primitives."""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import (
    AIMessage as LCAIMessage,
    BaseMessage as LCBaseMessage,
    HumanMessage as LCHumanMessage,
    SystemMessage as LCSystemMessage,
    ToolMessage as LCToolMessage,
)


class Document(LCDocument):
    """
    LangChain Document entity with PolyRAG backward compatibility.

    Supports both LangChain standard fields (`page_content`, `metadata`, `id`)
    and legacy PolyRAG attributes (`text`, `source`, `doc_id`).
    """

    def __init__(
        self,
        page_content: str = "",
        metadata: dict[str, Any] | None = None,
        text: str | None = None,
        source: str | None = None,
        doc_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        content = text if text is not None else page_content
        meta = dict(metadata or {})
        if source is not None and "source" not in meta:
            meta["source"] = source
        if doc_id is not None and "_id" not in meta:
            meta["_id"] = doc_id
        super().__init__(page_content=content, metadata=meta, **kwargs)

    @property
    def text(self) -> str:
        """Alias for page_content."""
        return self.page_content

    @property
    def source(self) -> str:
        """Source identifier from metadata."""
        return str(self.metadata.get("source", "document"))

    @property
    def doc_id(self) -> str | None:
        """Document ID from metadata or LangChain id."""
        return self.metadata.get("_id") or getattr(self, "id", None)

    def to_dict(self) -> dict[str, Any]:
        """Convert document into dictionary format."""
        return {
            "page_content": self.page_content,
            "text": self.page_content,
            "source": self.source,
            "metadata": self.metadata,
            "id": self.id,
        }


class BaseMessage(LCBaseMessage):
    """Base chat message returning content string on str()."""

    def __str__(self) -> str:
        return str(self.content)


class HumanMessage(LCHumanMessage):
    """Human/User message returning content string on str()."""

    def __str__(self) -> str:
        return str(self.content)


class AIMessage(LCAIMessage):
    """AI/Assistant message returning content string on str()."""

    def __str__(self) -> str:
        return str(self.content)


class SystemMessage(LCSystemMessage):
    """System message returning content string on str()."""

    def __str__(self) -> str:
        return str(self.content)


class ToolMessage(LCToolMessage):
    """Tool message returning content string on str()."""

    def __str__(self) -> str:
        return str(self.content)


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

    @property
    def page_content(self) -> str:
        """LangChain compatibility alias for text."""
        return self.text

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk into a dictionary format."""
        data = {
            "source": self.source,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_content": self.text,
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
    document: dict[str, Any] | Document = field(default_factory=dict)

    @property
    def source(self) -> str:
        if isinstance(self.document, Document):
            return self.document.source
        return self.document.get("source", "unknown")

    @property
    def chunk_id(self) -> Any:
        if isinstance(self.document, Document):
            return self.document.metadata.get("chunk_id", "")
        return self.document.get("chunk_id", "")

    @property
    def text(self) -> str:
        if isinstance(self.document, Document):
            return self.document.page_content
        return self.document.get("text", self.document.get("page_content", ""))

    @property
    def page_content(self) -> str:
        return self.text

    @property
    def identifier(self) -> str:
        return f"{self.source}#{self.chunk_id}"

    def to_dict(self) -> dict[str, Any]:
        doc_dict = self.document.to_dict() if isinstance(self.document, Document) else self.document
        return {
            "score": self.score,
            "distance": self.distance,
            "document": doc_dict,
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


__all__ = [
    "Document",
    "Chunk",
    "SearchResult",
    "RAGResponse",
    "AgentAction",
    "AgentStep",
    "AgentResponse",
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ToolMessage",
]
