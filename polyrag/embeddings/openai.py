"""OpenAI Embedding implementation built on langchain_core.embeddings."""

from typing import Any
from langchain_core.embeddings import Embeddings


class OpenAIEmbedding(Embeddings):
    """
    OpenAI embedding adapter conforming to LangChain's Embeddings interface.
    """

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self.model_name = model_name

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
            )

        self._dim = self.MODEL_DIMENSIONS.get(model_name, 1536)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """LangChain standard method: embed multiple documents."""
        if not texts:
            return []
        batch_size = 128
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in batch_embeddings)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """LangChain standard method: embed a single query text."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_text(self, text: str) -> list[float]:
        """PolyRAG backward compatibility alias."""
        return self.embed_query(text)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 128,
    ) -> list[list[float]]:
        """PolyRAG backward compatibility alias."""
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
            )
            batch_embeddings = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in batch_embeddings)
        return embeddings


__all__ = ["OpenAIEmbedding"]
