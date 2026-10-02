"""Unit tests for modern AgenticRAG with custom tools, state schema, default tools, and LangChain tool bridge."""

from typing import Any, TypedDict
import pytest

from polyrag import (
    AgentTool,
    AgenticRAG,
    BaseLLMClient,
    Container,
    InMemoryVectorStore,
    PolyRAG,
    RAGResponse,
)
from polyrag.core.interfaces import BaseEmbeddingModel


class MockEmbedding(BaseEmbeddingModel):
    @property
    def dim(self) -> int:
        return 4

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class MockToolCallingLLM(BaseLLMClient):
    """Deterministic LLM simulating tool-calling behavior."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "mock-tool-llm"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return "Direct complete response [rfc1035#0]."

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "Mock chat response"

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.call_count += 1
        # If prompt contains available tools instructions:
        if "Available Tools:" in prompt:
            if self.call_count == 1:
                # Step 1: Call custom calculator tool or retrieve_documents
                if "calculate_discount" in prompt:
                    return {
                        "thought": "I will calculate the discount.",
                        "tool": "calculate_discount",
                        "args": {"price": 100.0, "percentage": 20.0},
                    }
                return {
                    "thought": "I need to search technical docs.",
                    "tool": "retrieve_documents",
                    "args": {"query": "DNS architecture"},
                }
            # Step 2: Final answer
            return {
                "thought": "I have collected all necessary facts.",
                "tool": "final_answer",
                "args": {
                    "answer": "Calculated price is $80.00 [pricing#0].",
                    "confidence": 0.98,
                },
            }

        # Legacy decide retrieval
        if "retrieval planner" in prompt:
            return {
                "retrieval_needed": True,
                "query": "DNS RFC 1035",
                "reason": "Technical query",
            }
        # Legacy reflection
        if "reflection and answering" in prompt:
            return {
                "enough": True,
                "confidence": 0.95,
                "reason": "Sufficient context found.",
                "answer": "Grounded answer from reflection [rfc1035#0].",
            }
        return {}


class SimulatedLangChainTool:
    """Simulates a LangChain @tool or BaseTool."""

    def __init__(self, name: str = "lc_calculator"):
        self.name = name
        self.description = "Simulated LangChain tool."

    def invoke(self, args: dict[str, Any]) -> str:
        return f"LC tool result for: {args}"


# =============================================================================
# Unit Tests
# =============================================================================

def test_agentic_rag_default_tools():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    llm = MockToolCallingLLM()

    agentic = AgenticRAG(
        embedding_model=emb,
        vector_store=vs,
        llm_client=llm,
    )

    # Check default tools are registered
    tool_names = [t.name for t in agentic.tools]
    assert "retrieve_documents" in tool_names
    assert "list_sources" in tool_names
    assert len(agentic.custom_tools) == 0


def test_agentic_rag_with_custom_callable_tool():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    llm = MockToolCallingLLM()

    def calculate_discount(price: float, percentage: float) -> dict[str, Any]:
        """Calculate discount and return ticket status."""
        discounted = price * (1.0 - percentage / 100.0)
        return {"discounted_price": discounted, "status": "calculated"}

    agentic = AgenticRAG(
        embedding_model=emb,
        vector_store=vs,
        llm_client=llm,
        tools=[calculate_discount],
    )

    assert len(agentic.custom_tools) == 1
    assert agentic.custom_tools[0].name == "calculate_discount"

    # Query with custom state
    response = agentic.query(
        "Apply 20% discount to $100 order",
        state={"order_id": "ORD-123"},
    )

    assert isinstance(response, RAGResponse)
    assert "Calculated price is $80.00" in response.answer
    assert response.state.get("order_id") == "ORD-123"
    assert response.state.get("discounted_price") == 80.0
    assert response.state.get("status") == "calculated"
    assert len(response.agent_log) >= 2


def test_agentic_rag_with_langchain_duck_typed_tool():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    llm = MockToolCallingLLM()

    lc_tool = SimulatedLangChainTool(name="lc_calculator")
    agentic = AgenticRAG(
        embedding_model=emb,
        vector_store=vs,
        llm_client=llm,
        tools=[lc_tool],
    )

    tool_names = [t.name for t in agentic.tools]
    assert "lc_calculator" in tool_names
    assert "retrieve_documents" in tool_names


def test_agentic_rag_state_schema():
    class CustomTicketState(TypedDict):
        ticket_id: str
        notes: list[str]
        resolved: bool

    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    llm = MockToolCallingLLM()

    agentic = AgenticRAG(
        embedding_model=emb,
        vector_store=vs,
        llm_client=llm,
        state_schema=CustomTicketState,
    )

    assert agentic.state_schema is CustomTicketState

    # Query with ticket state
    response = agentic.query(
        "Resolve ticket 555",
        state={"ticket_id": "TCK-555", "notes": ["Initial report"], "resolved": True},
    )

    assert response.state.get("ticket_id") == "TCK-555"
    assert response.state.get("resolved") is True
    assert "notes" in response.state


def test_agentic_rag_as_langchain_tool():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    llm = MockToolCallingLLM()

    agentic = AgenticRAG(
        embedding_model=emb,
        vector_store=vs,
        llm_client=llm,
    )

    lc_tool = agentic.as_langchain_tool(name="tech_doc_search", description="Search internal docs.")
    assert lc_tool.name == "tech_doc_search"
    assert "Search internal docs." in lc_tool.description

    # Test invoking the exported tool
    res = lc_tool.invoke({"query": "DNS architecture"})
    assert isinstance(res, str)
    assert len(res) > 0


def test_polyrag_create_agentic_rag_with_tools_and_state():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    llm = MockToolCallingLLM()

    rag = PolyRAG(
        embedding_model=emb,
        vector_store=vs,
        llm_client=llm,
    )

    def custom_helper(x: int) -> int:
        """Helper tool."""
        return x * 2

    class MyState(TypedDict):
        session_id: str

    agentic = rag.create_agentic_rag(
        tools=[custom_helper],
        state_schema=MyState,
        top_k=4,
    )

    assert isinstance(agentic, AgenticRAG)
    assert len(agentic.custom_tools) == 1
    assert agentic.state_schema is MyState
    assert agentic.top_k == 4


def test_container_build_agentic_rag_with_tools_and_state():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    llm = MockToolCallingLLM()

    container = Container.create(
        embedding_model=emb,
        vector_store=vs,
        llm_client=llm,
    )

    def dummy_tool() -> str:
        return "dummy"

    agentic = container.build_agentic_rag(
        tools=[dummy_tool],
        state_schema=dict,
    )

    assert isinstance(agentic, AgenticRAG)
    assert len(agentic.custom_tools) == 1
    assert agentic.state_schema is dict
