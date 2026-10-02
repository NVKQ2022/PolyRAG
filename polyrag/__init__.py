"""
polyrag - A modular, extensible, and production-ready Retrieval-Augmented Generation library.
"""

from polyrag.chunkers import resolve_chunker
from polyrag.chunkers.fixed_size import FixedSizeChunker
from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.container import Container
from polyrag.core.interfaces import (
    BaseChatModel,
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
    Embeddings,
    TextSplitter,
    VectorStore,
)
from polyrag.core.models import (
    AgentAction,
    AgentResponse,
    AgentStep,
    AIMessage,
    BaseMessage,
    Chunk,
    Document,
    HumanMessage,
    RAGResponse,
    SearchResult,
    SystemMessage,
    ToolMessage,
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
from polyrag.llms import (
    ChatOpenAI,
    OpenAIChatModel,
    OpenAILLM,
    resolve_chat_model,
    resolve_llm_client,
)
from polyrag.pipelines.advanced import AdvancedRAG
from polyrag.pipelines.agentic import AgentTool, AgenticRAG
from polyrag.pipelines.base import BaseRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent, ReActRAG
from polyrag.adapters.langchain import (
    LangChainChatModelAdapter,
    LangChainDocumentConverter,
    LangChainEmbeddingAdapter,
)
from polyrag.app import PolyRAG
from polyrag.service import AgenticRAGService, RAGService
from polyrag.vector_stores import (
    ChromaVectorStore,
    InMemoryVectorStore,
    MilvusLite,
    MilvusLiteVectorStore,
    MilvusVectorStore,
    resolve_vector_store,
)

__version__ = "0.1.5"

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
    "AgentTool",
    "ReActAgent",
    "ReActRAG",
    # LangChain Standard Primitives
    "VectorStore",
    "Embeddings",
    "BaseChatModel",
    "TextSplitter",
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
    "MilvusLiteVectorStore",
    "MilvusLite",
    "resolve_vector_store",
    # Adapters & Bridges
    "LangChainDocumentConverter",
    "LangChainEmbeddingAdapter",
    "LangChainChatModelAdapter",
    # LLM & Modern ChatModel
    "BaseLLMClient",
    "OpenAILLM",
    "ChatOpenAI",
    "OpenAIChatModel",
    "resolve_chat_model",
    "resolve_llm_client",
    # Core Data Models
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
    # Exceptions
    "RAGException",
    "ConfigurationError",
    "IngestionError",
    "RetrievalError",
    "LLMGenerationError",
]
