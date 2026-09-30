"""
polyrag - A modular, extensible, and production-ready Retrieval-Augmented Generation library.
"""

from polyrag.chunkers.fixed_size import FixedSizeChunker
from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.container import Container
from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import (
    AgentAction,
    AgentResponse,
    AgentStep,
    Chunk,
    Document,
    RAGResponse,
    SearchResult,
)
from polyrag.embeddings.openai import OpenAIEmbedding
from polyrag.embeddings.sentence_transformers import SentenceTransformerEmbedding
from polyrag.exceptions import (
    ConfigurationError,
    IngestionError,
    LLMGenerationError,
    RAGException,
    RetrievalError,
)
from polyrag.llms.openai import OpenAILLM
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent
from polyrag.service import AgenticRAGService, RAGService
from polyrag.vector_stores.chroma import ChromaVectorStore
from polyrag.vector_stores.memory import InMemoryVectorStore

__version__ = "0.1.0"

__all__ = [
    # Dependency Injection
    "Container",
    # Top-level service facades
    "RAGService",
    "AgenticRAGService",
    # Pipelines
    "NaiveRAG",
    "AgenticRAG",
    "ReActAgent",
    # Chunkers
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    # Embeddings
    "BaseEmbeddingModel",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    # Vector Stores
    "BaseVectorStore",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    # LLM
    "BaseLLMClient",
    "OpenAILLM",
    # Core Data Models
    "Document",
    "Chunk",
    "SearchResult",
    "RAGResponse",
    "AgentAction",
    "AgentStep",
    "AgentResponse",
    # Exceptions
    "RAGException",
    "ConfigurationError",
    "IngestionError",
    "RetrievalError",
    "LLMGenerationError",
]
