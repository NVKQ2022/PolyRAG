"""
polyrag - A modular, multi-paradigm RAG orchestration library built natively on LangChain.
"""

from polyrag.app import PolyRAG, RAGService
from polyrag.chunkers import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    resolve_chunker,
)
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
from polyrag.embeddings import (
    LangChainEmbeddingAdapter,
    resolve_embedding_model,
)
from polyrag.exceptions import (
    ConfigurationError,
    IngestionError,
    LLMGenerationError,
    RAGException,
    RetrievalError,
)
from polyrag.llms import (
    ChatOpenAI,
    LangChainChatModelAdapter,
    OpenAIChatModel,
    OpenAILLM,
    resolve_chat_model,
    resolve_llm_client,
)
from polyrag.pipelines.advanced import AdvancedRAG
from polyrag.pipelines.agentic import AgentTool, AgenticRAG, AgenticRAGService
from polyrag.pipelines.base import BaseRAG
from polyrag.pipelines.naive import NaiveRAG
from polyrag.pipelines.react import ReActAgent, ReActRAG
from polyrag.vector_stores import (
    InMemoryVectorStore,
    resolve_vector_store,
)

__version__ = "0.2.0"

__all__ = [
    # Application Context & Setup Factory
    "PolyRAG",
    "RAGService",
    # Dependency Injection
    "Container",
    # RAG Architecture Hierarchy
    "BaseRAG",
    "NaiveRAG",
    "AdvancedRAG",
    "AgenticRAG",
    "AgenticRAGService",
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
    "LangChainEmbeddingAdapter",
    "resolve_embedding_model",
    # Vector Stores
    "BaseVectorStore",
    "InMemoryVectorStore",
    "resolve_vector_store",
    # LLM & ChatModel
    "BaseLLMClient",
    "ChatOpenAI",
    "OpenAILLM",
    "OpenAIChatModel",
    "LangChainChatModelAdapter",
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
