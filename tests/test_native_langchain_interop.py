"""Comprehensive tests verifying pure native LangChain component interoperability with PolyRAG."""

from typing import Any
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage as LCAIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.vectorstores.in_memory import InMemoryVectorStore as LCInMemoryVectorStore
from langchain_text_splitters import CharacterTextSplitter
import pytest

from polyrag import (
    AdvancedRAG,
    AgenticRAG,
    NaiveRAG,
    PolyRAG,
    ReActAgent,
)
from polyrag.core.models import Document


class NativeFakeChatModel(BaseChatModel):
    """Native LangChain BaseChatModel for testing direct integration without custom wrappers."""

    response_text: str = "Native LangChain Answer"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = LCAIMessage(content=self.response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "native-fake-chat-model"


def test_native_langchain_components_in_polyrag():
    # 1. Native LangChain FakeEmbeddings
    lc_embeddings = FakeEmbeddings(size=64)

    # 2. Native LangChain InMemoryVectorStore
    lc_vectorstore = LCInMemoryVectorStore(embedding=lc_embeddings)

    # 3. Native LangChain BaseChatModel
    lc_chat_model = NativeFakeChatModel(response_text="Synthesized from LangChain")

    # 4. Native LangChain CharacterTextSplitter
    lc_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator=" ")

    # Build PolyRAG directly using 100% native LangChain components
    rag = PolyRAG(
        embedding_model=lc_embeddings,
        vector_store=lc_vectorstore,
        llm_client=lc_chat_model,
        chunker=lc_splitter,
    )

    # Ingest text using native LangChain splitter and embeddings into native LangChain vectorstore
    chunks = rag.ingest_text(
        "LangChain and PolyRAG together provide modular and flexible retrieval-augmented generation.",
        source="integration.txt",
    )
    assert len(chunks) >= 1

    # Query Naive RAG pipeline powered by native LangChain components
    response = rag.query("What do LangChain and PolyRAG provide?")
    assert response.answer == "Synthesized from LangChain"
    assert len(response.sources) >= 1


def test_native_langchain_in_hierarchical_pipelines():
    lc_embeddings = FakeEmbeddings(size=32)
    lc_vectorstore = LCInMemoryVectorStore(embedding=lc_embeddings)
    lc_chat_model = NativeFakeChatModel(response_text="Advanced synthesis output")

    # Seed native vector store with a document
    doc = Document(page_content="Distributed algorithms ensure consensus.", metadata={"source": "consensus.md"})
    lc_vectorstore.add_documents([doc])

    # 1. NaiveRAG
    naive = NaiveRAG(
        embedding_model=lc_embeddings,
        vector_store=lc_vectorstore,
        llm_client=lc_chat_model,
    )
    res_naive = naive.execute("Tell me about consensus")
    assert res_naive.answer == "Advanced synthesis output"

    # 2. AdvancedRAG
    advanced = AdvancedRAG(
        embedding_model=lc_embeddings,
        vector_store=lc_vectorstore,
        llm_client=lc_chat_model,
    )
    res_adv = advanced.execute("Tell me about consensus")
    assert res_adv.answer == "Advanced synthesis output"
