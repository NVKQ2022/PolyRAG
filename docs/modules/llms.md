# Module: `polyrag.llms` 🤖

The `polyrag.llms` module provides Language Model and ChatModel integration for text completion, structured JSON synthesis, streaming, and tool-calling agent loops.

PolyRAG is built natively on LangChain's [`BaseChatModel`](https://python.langchain.com/docs/concepts/chat_models/) interface. You can pass **any ChatModel from the LangChain ecosystem** (OpenAI, Anthropic, Gemini, Ollama, Groq, Mistral, Bedrock) directly into PolyRAG pipelines with zero custom wrappers.

---

## 1. Universal LangChain ChatModel Integration

Any chat model subclassing LangChain's `BaseChatModel` plugs directly into PolyRAG pipelines:

```python
# OpenAI Chat Models
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Anthropic Claude
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Google Gemini
# from langchain_google_genai import ChatGoogleGenerativeAI
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

# Local Ollama
# from langchain_community.chat_models import ChatOllama
# llm = ChatOllama(model="llama3:8b")

from polyrag import PolyRAG
rag = PolyRAG(chat_model=llm)
```

---

## 2. Automatic Resolution via `resolve_llm_client`

The `polyrag.llms.resolve_llm_client` (and alias `resolve_chat_model`) handles strings, instances, and fallbacks:

```python
from polyrag.llms import resolve_chat_model, resolve_llm_client

# 1. Defaults to ChatOpenAI(model="gpt-4o-mini") when None
model = resolve_llm_client(None)

# 2. String model identifier
model = resolve_llm_client("gpt-4o")
model = resolve_llm_client("gpt-4o-mini")

# 3. Direct LangChain BaseChatModel instance (passed through unchanged)
model = resolve_llm_client(custom_chat_model)

# 4. Duck-typed object implementing .invoke()
class MyMockLLM:
    def invoke(self, prompt: str):
        return "Synthetic response"

model = resolve_llm_client(MyMockLLM())
```

---

## 3. Core Capabilities

All resolved chat models and LLM clients in PolyRAG support:

### A. Raw Text Completion: `complete(prompt, **kwargs) -> str`
```python
answer = model.complete("Summarize the benefits of TCP flow control.")
print(answer)
```

### B. Structured JSON Output: `complete_json(prompt, **kwargs) -> dict`
Guarantees clean dictionary extraction by stripping Markdown fences (````json ... ````) and using robust JSON fallback parsing:
```python
prompt = """Analyze this query: 'What are the RFC specs for DNS?'
Return JSON in this format:
{"needs_retrieval": true, "query": "RFC DNS"}"""

result = model.complete_json(prompt)
print(result["needs_retrieval"])  # True
```

### C. Conversational Chat: `chat(messages, **kwargs) -> str`
Multi-turn conversational completion:
```python
messages = [
    {"role": "system", "content": "You are a network engineer."},
    {"role": "user", "content": "What port does DNS use?"},
]
reply = model.chat(messages)
print(reply)
```

### D. Streaming: `stream(prompt, **kwargs)`
Yields token chunks or `AIMessage` chunks sequentially:
```python
for chunk in model.stream("Explain QUIC transport protocol"):
    print(chunk, end="", flush=True)
```

### E. Tool Binding: `bind_tools(tools)`
Binds tool schemas for tool-calling agent pipelines (`AgenticRAG`, `ReActAgent`).
