"""
polyrag - A modular, extensible, and production-ready Retrieval-Augmented Generation library.
"""

from polyrag.chunkers import resolve_chunker
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
from polyrag.embeddings import resolve_embedding_model
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
from polyrag.pipelines.advanced import AdvancedRAG
from polyrag.pipelines.agentic import AgenticRAG
from polyrag.pipelines.base import BaseRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent, ReActRAG
from polyrag.adapters.langchain import (
    LangChainDocumentConverter,
    LangChainEmbeddingAdapter,
)
from polyrag.app import PolyRAG
from polyrag.service import AgenticRAGService, RAGService
from polyrag.vector_stores.chroma import ChromaVectorStore
from polyrag.vector_stores.memory import InMemoryVectorStore
from polyrag.vector_stores.milvus import MilvusVectorStore

__version__ = "0.1.2"

__all__ = [
    # Application Context & Setup Factory
    "PolyRAG",
    # Dependency Injection
    "Container",
    # Service Facades
    "RAGService",
    "AgenticRAGService",
    # RAG Architecture Hierarchy
    "BaseRAG",
    "NaiveRAG",
    "AdvancedRAG",
    "AgenticRAG",
    "ReActAgent",
    "ReActRAG",
    # Chunkers
    "BaseChunker",
    "FixedSizeChunker",
    "RecursiveCharacterChunker",
    "resolve_chunker",
    # Embeddings
    "BaseEmbeddingModel",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "resolve_embedding_model",
    # Vector Stores
    "BaseVectorStore",
    "ChromaVectorStore",
    "InMemoryVectorStore",
    "MilvusVectorStore",
    # Adapters & Bridges
    "LangChainDocumentConverter",
    "LangChainEmbeddingAdapter",
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
