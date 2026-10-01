"""LangChain ecosystem bridge adapters for PolyRAG."""

from collections.abc import Iterable
from typing import Any

from polyrag.core.interfaces import BaseEmbeddingModel, BaseLLMClient
from polyrag.core.models import AIMessage, Document as PolyDocument


class LangChainChatModelAdapter(BaseLLMClient):
    """
    Adapter bridging LangChain's modern BaseChatModel (ChatOpenAI, ChatAnthropic,
    ChatOllama, ChatGoogleGenerativeAI, etc.) to PolyRAG's BaseLLMClient interface.

    Enables developers to pass native LangChain chat models directly into PolyRAG,
    maintaining full compatibility with invoke(), complete(), complete_json(), and chat().
    """

    def __init__(self, chat_model: Any, model_name: str | None = None) -> None:
        if not hasattr(chat_model, "invoke"):
            raise TypeError(
                "Provided object does not conform to LangChain ChatModel interface "
                "(missing 'invoke' method)."
            )
        self.chat_model = chat_model
        self._model_name = (
            model_name
            or getattr(chat_model, "model_name", None)
            or getattr(chat_model, "model", None)
            or getattr(chat_model, "model_id", None)
            or chat_model.__class__.__name__
        )

    @property
    def model_name(self) -> str:
        return str(self._model_name)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Execute text completion via chat_model.invoke(prompt)."""
        response = self.chat_model.invoke(prompt, **kwargs)
        if hasattr(response, "content"):
            return str(response.content).strip()
        return str(response).strip()

    def complete_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Execute and parse JSON output from model."""
        import json
        import re

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
        """Execute conversational chat via chat_model.invoke(messages)."""
        tuple_messages = []
        for m in messages:
            role = m.get("role", "user")
            if role in ("assistant", "ai"):
                r = "ai"
            elif role in ("human", "user"):
                r = "human"
            elif role == "system":
                r = "system"
            else:
                r = str(role)
            tuple_messages.append((r, m.get("content", "")))

        response = self.chat_model.invoke(tuple_messages, **kwargs)
        if hasattr(response, "content"):
            return str(response.content).strip()
        return str(response).strip()

    def invoke(self, input: Any, **kwargs: Any) -> AIMessage:
        """Direct delegation to underlying LangChain model's invoke()."""
        response = self.chat_model.invoke(input, **kwargs)
        if isinstance(response, AIMessage):
            return response
        if hasattr(response, "content"):
            tool_calls = getattr(response, "tool_calls", [])
            return AIMessage(content=str(response.content), tool_calls=tool_calls)
        return AIMessage(content=str(response))

    def stream(self, input: Any, **kwargs: Any) -> Any:
        """Stream chunks from underlying LangChain model."""
        if hasattr(self.chat_model, "stream"):
            for chunk in self.chat_model.stream(input, **kwargs):
                if isinstance(chunk, AIMessage):
                    yield chunk
                elif hasattr(chunk, "content"):
                    yield AIMessage(
                        content=str(chunk.content),
                        tool_calls=getattr(chunk, "tool_calls", []),
                    )
                else:
                    yield AIMessage(content=str(chunk))
            return
        yield from super().stream(input, **kwargs)

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> Any:
        """Delegate tool binding to underlying LangChain model."""
        if hasattr(self.chat_model, "bind_tools"):
            return LangChainChatModelAdapter(
                self.chat_model.bind_tools(tools, **kwargs),
                model_name=self._model_name,
            )
        return self



class LangChainEmbeddingAdapter(BaseEmbeddingModel):
    """
    Adapter bridging LangChain Embeddings objects to PolyRAG's BaseEmbeddingModel interface.

    Allows any LangChain-compatible embedding model (e.g., langchain_openai.OpenAIEmbeddings,
    langchain_community.embeddings.HuggingFaceEmbeddings, OllamaEmbeddings, etc.) to be used
    directly in PolyRAG pipelines, vector stores, and ingestion routines without tight coupling.
    """

    def __init__(self, lc_embeddings: Any, dim: int | None = None) -> None:
        if not (hasattr(lc_embeddings, "embed_documents") and hasattr(lc_embeddings, "embed_query")):
            raise TypeError(
                "Provided object does not conform to LangChain Embeddings interface "
                "(missing 'embed_documents' or 'embed_query' methods)."
            )
        self.lc_embeddings = lc_embeddings
        self._dim = dim

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim

        # Check common attributes like dimension, dim, or probe with a sample string
        if hasattr(self.lc_embeddings, "dimension") and isinstance(self.lc_embeddings.dimension, int):
            self._dim = self.lc_embeddings.dimension
            return self._dim
        if hasattr(self.lc_embeddings, "dim") and isinstance(self.lc_embeddings.dim, int):
            self._dim = self.lc_embeddings.dim
            return self._dim

        # Probe embedding dimension dynamically using sample query
        sample = self.lc_embeddings.embed_query("dimension_probe")
        self._dim = len(sample)
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        return self.lc_embeddings.embed_query(text)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 128,
    ) -> list[list[float]]:
        if not texts:
            return []
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if len(texts) <= batch_size:
            return self.lc_embeddings.embed_documents(texts)

        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            results.extend(self.lc_embeddings.embed_documents(chunk))
        return results


class LangChainDocumentConverter:
    """
    Utility to bridge LangChain Document objects and PolyRAG Document models.

    Uses pure duck-typing so users do not need langchain installed unless
    actually calling LangChain loaders.
    """

    @staticmethod
    def to_polyrag_document(lc_doc: Any, default_source: str = "langchain_doc") -> PolyDocument:
        """
        Convert a LangChain Document (or duck-typed object) to a PolyRAG Document.

        Args:
            lc_doc: An object with 'page_content' and 'metadata' attributes or a dict.
            default_source: Fallback source name if not present in metadata.

        Returns:
            PolyRAG Document instance.
        """
        if hasattr(lc_doc, "page_content"):
            text = str(lc_doc.page_content)
            metadata = dict(getattr(lc_doc, "metadata", {}))
            doc_id = getattr(lc_doc, "id", None) or metadata.get("_id")
            source = metadata.get("source", default_source)
        elif hasattr(lc_doc, "text"):
            text = str(lc_doc.text)
            metadata = dict(getattr(lc_doc, "metadata", {}))
            doc_id = getattr(lc_doc, "doc_id", None) or metadata.get("_id")
            source = getattr(lc_doc, "source", default_source)
        elif isinstance(lc_doc, dict):
            text = str(lc_doc.get("text") or lc_doc.get("page_content") or "")
            metadata = {k: v for k, v in lc_doc.items() if k not in ("text", "page_content")}
            doc_id = lc_doc.get("_id") or lc_doc.get("id")
            source = lc_doc.get("source", default_source)
        else:
            text = str(lc_doc)
            metadata = {}
            doc_id = None
            source = default_source

        return PolyDocument(
            source=source,
            text=text,
            metadata=metadata,
            doc_id=doc_id,
        )

    @classmethod
    def to_polyrag_documents(
        cls,
        documents: Iterable[Any],
        default_source: str = "langchain_doc",
    ) -> list[PolyDocument]:
        """Convert an iterable or generator of documents to a list of PolyRAG Documents."""
        return [cls.to_polyrag_document(d, default_source=default_source) for d in documents]
