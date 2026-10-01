"""Unit tests for modern LangChain ChatModel integration, message models, adapters, and parameter injection."""

from typing import Any
import pytest

from polyrag import (
    AIMessage,
    AdvancedRAG,
    AgenticRAG,
    BaseMessage,
    ChatOpenAI,
    ConfigurationError,
    Container,
    HumanMessage,
    LangChainChatModelAdapter,
    NaiveRAG,
    OpenAIChatModel,
    OpenAILLM,
    PolyRAG,
    ReActAgent,
    SystemMessage,
    ToolMessage,
    resolve_chat_model,
    resolve_llm_client,
)
from polyrag.core.interfaces import (
    BaseEmbeddingModel,
    BaseLLMClient,
)
from polyrag.core.models import Document
from polyrag.vector_stores.memory import InMemoryVectorStore


class MockEmbedding(BaseEmbeddingModel):
    @property
    def dim(self) -> int:
        return 4

    def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts: list[str], batch_size: int = 128) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class MockLLM(BaseLLMClient):
    def __init__(self, tag: str = "mock-a"):
        self.tag = tag

    @property
    def model_name(self) -> str:
        return f"mock-llm-{self.tag}"

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return f"Answer from {self.tag}"

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return {"response": f"JSON from {self.tag}"}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return f"Chat reply from {self.tag}"


@pytest.fixture(autouse=True)
def mock_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "mock-api-key")


class SimulatedLangChainMessage:
    def __init__(self, content: str, tool_calls: list[dict[str, Any]] | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def __str__(self) -> str:
        return self.content


class SimulatedLangChainChatModel:
    """Simulates a modern LangChain BaseChatModel (e.g. ChatOpenAI, ChatAnthropic)."""

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.invoked_inputs: list[Any] = []
        self.tools_bound: list[Any] = []

    def invoke(self, input: Any, **kwargs: Any) -> SimulatedLangChainMessage:
        self.invoked_inputs.append(input)
        if isinstance(input, list):
            last = input[-1]
            if isinstance(last, tuple):
                content = last[1]
            else:
                content = getattr(last, "content", str(last))
            return SimulatedLangChainMessage(content=f"LC response to: {content}")
        return SimulatedLangChainMessage(content=f"LC response to: {input}")

    def stream(self, input: Any, **kwargs: Any):
        full = self.invoke(input, **kwargs)
        for token in full.content.split():
            yield SimulatedLangChainMessage(content=token + " ")

    def bind_tools(self, tools: list[Any], **kwargs: Any):
        bound = SimulatedLangChainChatModel(model_name=self.model_name)
        bound.tools_bound = tools
        return bound


# =============================================================================
# Message Models Tests
# =============================================================================

def test_message_models():
    human = HumanMessage(content="Hello AI")
    assert human.type == "human"
    assert human.content == "Hello AI"
    assert str(human) == "Hello AI"

    ai = AIMessage(content="Hello Human", tool_calls=[{"id": "call_1", "name": "search", "args": {}}])
    assert ai.type == "ai"
    assert ai.content == "Hello Human"
    assert len(ai.tool_calls) == 1
    assert str(ai) == "Hello Human"

    sys_msg = SystemMessage(content="System instruction")
    assert sys_msg.type == "system"
    assert str(sys_msg) == "System instruction"

    tool_msg = ToolMessage(content="Tool output", tool_call_id="call_1")
    assert tool_msg.type == "tool"
    assert tool_msg.tool_call_id == "call_1"
    assert str(tool_msg) == "Tool output"


def test_base_llm_client_runnable_interface():
    mock = MockLLM(tag="runnable")
    
    # Test invoke
    ai_msg = mock.invoke("Test prompt")
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.content == "Answer from runnable"

    # Test invoke with message list
    messages = [HumanMessage(content="Hey")]
    ai_msg2 = mock.invoke(messages)
    assert isinstance(ai_msg2, AIMessage)
    assert ai_msg2.content == "Chat reply from runnable"

    # Test stream
    stream_chunks = list(mock.stream("Test prompt"))
    assert len(stream_chunks) > 0
    assert "".join(str(chunk) for chunk in stream_chunks) == "Answer from runnable"

    # Test bind_tools
    bound = mock.bind_tools([{"name": "test"}])
    assert bound is not None


# =============================================================================
# LangChain Adapter Tests
# =============================================================================

def test_langchain_chat_model_adapter():
    lc_model = SimulatedLangChainChatModel(model_name="mock-claude")
    adapter = LangChainChatModelAdapter(lc_model)

    assert adapter.model_name == "mock-claude"

    # Test complete
    resp_text = adapter.complete("Generate ideas")
    assert "LC response to: Generate ideas" in resp_text

    # Test complete_json
    resp_json = adapter.complete_json("Generate json: {'status': 'ok'}")
    assert isinstance(resp_json, dict)

    # Test chat
    chat_resp = adapter.chat([{"role": "user", "content": "Hello"}])
    assert "LC response to: Hello" in chat_resp

    # Test invoke returning AIMessage
    res_msg = adapter.invoke("Direct invoke")
    assert isinstance(res_msg, AIMessage)
    assert "LC response to: Direct invoke" in res_msg.content

    # Test stream
    chunks = list(adapter.stream("Stream this"))
    assert len(chunks) > 0
    assert "LC" in "".join(str(c) for c in chunks)

    # Test bind_tools
    bound_adapter = adapter.bind_tools([{"name": "query_db"}])
    assert isinstance(bound_adapter, LangChainChatModelAdapter)
    assert len(bound_adapter.chat_model.tools_bound) == 1


# =============================================================================
# Resolution Tests
# =============================================================================

def test_resolve_llm_client():
    # 1. None defaults to OpenAILLM
    default_llm = resolve_llm_client(None)
    assert isinstance(default_llm, OpenAILLM)

    # 2. Existing BaseLLMClient
    mock = MockLLM()
    assert resolve_llm_client(mock) is mock

    # 3. String alias / model name
    str_llm = resolve_llm_client("gpt-4o-mini")
    assert isinstance(str_llm, OpenAILLM)
    assert str_llm.model_name == "gpt-4o-mini"

    # 4. Duck-typed LangChain ChatModel
    lc_model = SimulatedLangChainChatModel()
    adapted = resolve_llm_client(lc_model)
    assert isinstance(adapted, LangChainChatModelAdapter)
    assert adapted.chat_model is lc_model

    # 5. Invalid type
    with pytest.raises(TypeError):
        resolve_llm_client(12345)

    # 6. resolve_chat_model alias
    assert resolve_chat_model(mock) is mock


# =============================================================================
# Pipeline Constructor Parameter Tests
# =============================================================================

def test_pipeline_constructors_with_chat_model():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    chat_model = SimulatedLangChainChatModel(model_name="chat-test")

    # NaiveRAG
    naive = NaiveRAG(embedding_model=emb, vector_store=vs, chat_model=chat_model)
    assert isinstance(naive.llm_client, LangChainChatModelAdapter)
    assert naive.chat_model is naive.llm_client

    # AdvancedRAG with llm alias
    adv = AdvancedRAG(embedding_model=emb, vector_store=vs, llm=chat_model)
    assert isinstance(adv.llm_client, LangChainChatModelAdapter)
    assert adv.chat_model is adv.llm_client

    # AgenticRAG
    agentic = AgenticRAG(embedding_model=emb, vector_store=vs, chat_model=chat_model)
    assert isinstance(agentic.llm_client, LangChainChatModelAdapter)
    assert agentic.chat_model is agentic.llm_client

    # ReActAgent
    react = ReActAgent(embedding_model=emb, vector_store=vs, chat_model=chat_model)
    assert isinstance(react.llm_client, LangChainChatModelAdapter)
    assert react.chat_model is react.llm_client


# =============================================================================
# Container Builder Tests
# =============================================================================

def test_container_chat_model_overrides():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    default_llm = MockLLM(tag="default")
    override_lc = SimulatedLangChainChatModel(model_name="override-lc")

    container = Container.create(
        embedding_model=emb,
        vector_store=vs,
        chat_model=default_llm,
    )
    assert container.llm_client is default_llm

    # Default build uses default_llm
    naive = container.build_naive_rag()
    assert naive.llm_client is default_llm

    # Override with chat_model
    naive_ov = container.build_naive_rag(chat_model=override_lc)
    assert isinstance(naive_ov.llm_client, LangChainChatModelAdapter)
    assert naive_ov.chat_model is naive_ov.llm_client

    # Override advanced with llm
    adv_ov = container.build_advanced_rag(llm=override_lc)
    assert isinstance(adv_ov.llm_client, LangChainChatModelAdapter)

    # Override agentic
    agentic_ov = container.build_agentic_rag(chat_model=override_lc)
    assert isinstance(agentic_ov.llm_client, LangChainChatModelAdapter)

    # Override react
    react_ov = container.build_react_agent(chat_model=override_lc)
    assert isinstance(react_ov.llm_client, LangChainChatModelAdapter)


# =============================================================================
# PolyRAG Application Facade Tests
# =============================================================================

def test_polyrag_app_chat_model_parameter():
    emb = MockEmbedding()
    vs = InMemoryVectorStore()
    custom_llm = MockLLM(tag="poly-custom")

    # Initializing PolyRAG with chat_model
    rag = PolyRAG(
        embedding_model=emb,
        vector_store=vs,
        chat_model=custom_llm,
    )
    assert rag.llm_client is custom_llm
    assert rag.chat_model is custom_llm

    # Initializing PolyRAG with llm string
    rag_str = PolyRAG(
        embedding_model=emb,
        vector_store=vs,
        llm="gpt-4o-mini",
    )
    assert isinstance(rag_str.llm_client, OpenAILLM)
    assert rag_str.chat_model.model_name == "gpt-4o-mini"

    # Initializing PolyRAG with simulated LangChain chat model
    lc_model = SimulatedLangChainChatModel(model_name="lc-poly")
    rag_lc = PolyRAG(
        embedding_model=emb,
        vector_store=vs,
        chat_model=lc_model,
    )
    assert isinstance(rag_lc.llm_client, LangChainChatModelAdapter)
    assert rag_lc.chat_model is rag_lc.llm_client

    # Ingest document and query
    rag_lc.ingest("PolyRAG modern chat model integration")
    response = rag_lc.query("What is PolyRAG?")
    assert "LC response to:" in response.answer

    # Per-factory overrides
    override_llm = MockLLM(tag="override")
    naive_sub = rag_lc.create_naive_rag(chat_model=override_llm)
    assert naive_sub.llm_client is override_llm

    adv_sub = rag_lc.create_advanced_rag(llm=override_llm)
    assert adv_sub.llm_client is override_llm

    agent_sub = rag_lc.create_agentic_rag(chat_model=override_llm)
    assert agent_sub.llm_client is override_llm

    react_sub = rag_lc.create_react_agent(chat_model=override_llm)
    assert react_sub.llm_client is override_llm


def test_chat_openai_alias():
    assert ChatOpenAI is OpenAILLM
    assert OpenAIChatModel is OpenAILLM
