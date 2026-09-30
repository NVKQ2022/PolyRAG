"""OpenAI LLM client adapter implementing BaseLLMClient."""

import json
import re
from typing import Any

from polyrag.core.interfaces import BaseLLMClient
from polyrag.exceptions import LLMGenerationError


class OpenAILLM(BaseLLMClient):
    """Adapter for OpenAI, Azure OpenAI, and compatible endpoints."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self._model_name = model_name

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, prompt: str, **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", 0.1)

        # Support Azure OpenAI responses endpoint if available
        if hasattr(self.client, "responses") and hasattr(self.client.responses, "create"):
            try:
                response = self.client.responses.create(
                    model=self._model_name,
                    input=prompt,
                    **{k: v for k, v in kwargs.items() if k not in ("temperature",)},
                )
                return getattr(response, "output_text", str(response)).strip()
            except Exception as e:
                raise LLMGenerationError(f"Azure response creation failed: {e}") from e

        # Support standard ChatCompletions endpoint
        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            try:
                response = self.client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    **{k: v for k, v in kwargs.items() if k not in ("temperature",)},
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                raise LLMGenerationError(f"Chat completion failed: {e}") from e

        raise LLMGenerationError("Client interface not recognized as OpenAI compatible.")

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        raw_text = self.complete(prompt, **kwargs)

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        candidate = match.group(1) if match else raw_text
        candidate = candidate.strip()

        first_brace = candidate.find("{")
        last_brace = candidate.rfind("}")
        if first_brace != -1 and last_brace != -1:
            candidate = candidate[first_brace : last_brace + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return {}

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        temperature = kwargs.get("temperature", 0.1)

        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k not in ("temperature",)},
            )
            return (response.choices[0].message.content or "").strip()

        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        return self.complete(prompt, **kwargs)
