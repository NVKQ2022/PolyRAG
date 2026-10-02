"""Backward-compatibility aliases for PolyRAG service interfaces."""

from polyrag.app import PolyRAG as RAGService
from polyrag.pipelines.agentic import AgenticRAG as AgenticRAGService

__all__ = ["RAGService", "AgenticRAGService"]
