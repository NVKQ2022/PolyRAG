"""LLMs and ChatModels Package for PolyRAG built on langchain_core.language_models."""

from typing import Any
from langchain_core.language_models import BaseChatModel

from polyrag.core.interfaces import BaseLLMClient


class DuckTypedChatModelWrapper(BaseLLMClient):
    """Wraps any duck-typed LangChain chat model to satisfy BaseLLMClient."""

    def __init__(self, chat_model: Any, model_name: str = "custom-chat-model") -> None:
        self.chat_model = chat_model
        self._model_name = (
            getattr(chat_model, "model_name", None)
            or getattr(chat_model, "model", None)
            or getattr(chat_model, "model_id", None)
            or model_name
        )

    @property
    def model_name(self) -> str:
        return str(self._model_name)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        res = self.chat_model.invoke(prompt, **kwargs)
        return str(getattr(res, "content", res)).strip()

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        if hasattr(self.chat_model, "complete_json") and callable(getattr(self.chat_model, "complete_json")):
            return self.chat_model.complete_json(prompt, **kwargs)
        return super().complete_json(prompt, **kwargs)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        tuple_msgs = [(m.get("role", "user"), m.get("content", "")) for m in messages]
        res = self.chat_model.invoke(tuple_msgs, **kwargs)
        return str(getattr(res, "content", res)).strip()

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        from polyrag.core.models import AIMessage

        res = self.chat_model.invoke(input, **kwargs)
        if hasattr(res, "content"):
            return AIMessage(content=str(res.content), tool_calls=getattr(res, "tool_calls", []))
        return AIMessage(content=str(res))

    def stream(self, input: Any, **kwargs: Any) -> Any:
        if hasattr(self.chat_model, "stream") and callable(getattr(self.chat_model, "stream")):
            yield from self.chat_model.stream(input, **kwargs)
        else:
            yield self.invoke(input, **kwargs)

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        if hasattr(self.chat_model, "bind_tools"):
            return DuckTypedChatModelWrapper(
                self.chat_model.bind_tools(tools, **kwargs),
                model_name=self._model_name,
            )
        return self


try:
    from langchain_openai import ChatOpenAI
except ImportError:
    class ChatOpenAI(BaseChatModel):
        """Standard ChatOpenAI conforming to LangChain BaseChatModel."""

        model_name: str = "gpt-4o-mini"

        def __init__(
            self,
            model: str | None = None,
            model_name: str | None = None,
            **kwargs: Any,
        ) -> None:
            resolved = model or model_name or "gpt-4o-mini"
            super().__init__(model_name=resolved, **kwargs)
            self.model_name = resolved

        def _generate(
            self,
            messages: Any,
            stop: Any = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            last = messages[-1] if messages else ""
            content = getattr(last, "content", str(last))
            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=f"ChatOpenAI response to: {content}"))
                ]
            )

        @property
        def _llm_type(self) -> str:
            return "chat-openai"


# Backward compatibility aliases
OpenAILLM = ChatOpenAI
OpenAIChatModel = ChatOpenAI
LangChainChatModelAdapter = DuckTypedChatModelWrapper


def resolve_llm_client(
    llm: BaseLLMClient | BaseChatModel | str | Any | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Resolve an LLM or ChatModel strategy into a concrete LangChain BaseChatModel or BaseLLMClient.

    Supported input types:
    1. Any native LangChain BaseChatModel or PolyRAG BaseLLMClient: returned directly.
    2. Any duck-typed Runnable implementing 'invoke': wrapped in DuckTypedChatModelWrapper.
    3. None or String model identifier:
       - Instantiates ChatOpenAI (or configured BaseChatModel).

    Args:
        llm: LLM client, model name string, or LangChain ChatModel.
        model_name: Optional fallback model name.
        **kwargs: Additional arguments forwarded to constructor.

    Returns:
        BaseChatModel or BaseLLMClient instance.
    """
    if llm is None:
        target_model = model_name or "gpt-4o-mini"
        return ChatOpenAI(model=target_model, **kwargs)

    if isinstance(llm, (BaseLLMClient, BaseChatModel)):
        return llm

    if isinstance(llm, str):
        cleaned = llm.strip()
        target_model = cleaned if cleaned.lower() != "openai" else (model_name or "gpt-4o-mini")
        return ChatOpenAI(model=target_model, **kwargs)

    if hasattr(llm, "invoke"):
        return DuckTypedChatModelWrapper(llm, model_name=model_name or "custom-model")

    raise TypeError(
        f"Expected BaseChatModel, BaseLLMClient instance, or string model name, got {type(llm).__name__}"
    )


resolve_chat_model = resolve_llm_client

__all__ = [
    "BaseLLMClient",
    "BaseChatModel",
    "ChatOpenAI",
    "OpenAILLM",
    "OpenAIChatModel",
    "DuckTypedChatModelWrapper",
    "LangChainChatModelAdapter",
    "resolve_llm_client",
    "resolve_chat_model",
]
