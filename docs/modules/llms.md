# Module: `polyrag.llms` 🤖

The `polyrag.llms` module provides Language Model adapters for text completion, structured JSON extraction, and conversational chat.

---

## 1. `OpenAILLM`

The default LLM client in PolyRAG. Interacts with OpenAI models, Azure OpenAI deployments, and any OpenAI-compatible inference server (vLLM, Ollama, LocalAI, LM Studio).

### Installation
```bash
pip install "polyrag[openai]"
```

### Basic Initialization
```python
from polyrag.llms import OpenAILLM

# Automatically reads OPENAI_API_KEY from environment
llm = OpenAILLM(model_name="gpt-4o-mini")
```

---

## 2. Core Methods

### `complete(prompt, **kwargs) -> str`
Generates a raw string completion:
```python
answer = llm.complete(
    "Explain reciprocal rank fusion in two sentences.",
    temperature=0.2,
)
print(answer)
```

### `complete_json(prompt, **kwargs) -> dict[str, Any]`
Guarantees structured JSON output. Strips Markdown fences (````json ... ````) and uses regex extraction to parse valid JSON dictionaries reliably:
```python
prompt = """Analyze this query: 'What are the RFC specs for DNS?'
Return JSON in this format:
{"needs_retrieval": true, "keywords": ["RFC", "DNS"]}"""

result = llm.complete_json(prompt)
print(result["keywords"])  # ['RFC', 'DNS']
```

### `chat(messages, **kwargs) -> str`
Multi-turn conversational completion:
```python
messages = [
    {"role": "system", "content": "You are a network engineer."},
    {"role": "user", "content": "What port does DNS use?"},
]
reply = llm.chat(messages, temperature=0.0)
print(reply)
```

---

## 3. Connecting to Alternative Providers

### Self-Hosted vLLM or Ollama
```python
llm = OpenAILLM(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)
```

### Azure OpenAI
Configure your Azure endpoint and API version:
```python
from openai import AzureOpenAI
from polyrag.llms import OpenAILLM

azure_client = AzureOpenAI(
    azure_endpoint="https://your-resource.openai.azure.com",
    api_key="AZURE_API_KEY",
    api_version="2024-02-15-preview",
)

llm = OpenAILLM(
    model_name="gpt-4o-mini",
    client=azure_client,
)
```

---

## 4. Custom LLM Client

Implement `BaseLLMClient` to connect alternative providers like Anthropic Claude or Google Gemini:

```python
import anthropic
import json
from polyrag.core.interfaces import BaseLLMClient

class AnthropicLLM(BaseLLMClient):
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: str | None = None):
        self._model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, prompt: str, **kwargs) -> str:
        res = self.client.messages.create(
            model=self._model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=[{"role": "user", "content": prompt}],
        )
        return res.content[0].text

    def complete_json(self, prompt: str, **kwargs) -> dict:
        text = self.complete(prompt, **kwargs)
        return json.loads(text.strip("```json").strip("```"))

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        res = self.client.messages.create(
            model=self._model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            messages=messages,
        )
        return res.content[0].text
```
