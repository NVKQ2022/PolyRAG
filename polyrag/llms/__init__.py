"""LLMs and ChatModels Package for polyrag."""

from typing import Any

from polyrag.core.interfaces import BaseLLMClient
from polyrag.llms.openai import ChatOpenAI, OpenAIChatModel, OpenAILLM


def resolve_llm_client(
    llm: BaseLLMClient | str | Any | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Resolve an LLM or ChatModel strategy into a concrete BaseLLMClient instance.

    Supported input types:
    1. BaseLLMClient instance: returned directly as-is.
    2. None: defaults to OpenAILLM(model_name=model_name or "gpt-4o-mini", **kwargs).
    3. String model identifier:
       - "openai": instantiates OpenAILLM(model_name=model_name or "gpt-4o-mini", **kwargs)
       - "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", etc.:
         instantiates OpenAILLM(model_name=llm, **kwargs)
       - Any other model name:
         instantiates OpenAILLM(model_name=llm, **kwargs)
    4. LangChain BaseChatModel duck-typed instance (implements 'invoke'):
       wrapped into a LangChainChatModelAdapter.

    Args:
        llm: LLM client, model name string, or LangChain ChatModel.
        model_name: Optional fallback model name.
        **kwargs: Additional arguments forwarded to constructor.

    Returns:
        BaseLLMClient instance.
    """
    if llm is None:
        target_model = model_name or "gpt-4o-mini"
        return OpenAILLM(model_name=target_model, **kwargs)

    if isinstance(llm, BaseLLMClient):
        return llm

    if isinstance(llm, str):
        cleaned = llm.strip()
        target_model = cleaned if cleaned.lower() != "openai" else (model_name or "gpt-4o-mini")
        return OpenAILLM(model_name=target_model, **kwargs)

    # Check duck-typing for modern LangChain ChatModel (has invoke)
    if hasattr(llm, "invoke"):
        from polyrag.adapters.langchain import LangChainChatModelAdapter
        return LangChainChatModelAdapter(chat_model=llm, model_name=model_name, **kwargs)

    raise TypeError(
        f"Expected BaseLLMClient instance, string model name, or LangChain ChatModel, got {type(llm).__name__}"
    )


resolve_chat_model = resolve_llm_client

__all__ = [
    "OpenAILLM",
    "ChatOpenAI",
    "OpenAIChatModel",
    "resolve_llm_client",
    "resolve_chat_model",
]
