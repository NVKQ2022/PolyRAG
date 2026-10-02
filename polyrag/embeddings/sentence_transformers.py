"""Local Sentence-Transformers embedding implementation built on langchain_core.embeddings."""

from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbedding(Embeddings):
    """
    Sentence Transformers / HuggingFace embedding adapter conforming to
    LangChain's Embeddings interface.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        if hasattr(self.model, "get_embedding_dimension"):
            dimension = self.model.get_embedding_dimension()
        else:
            dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding dimension for model: {model_name}")

        self._dim = dimension

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """LangChain standard method: embed multiple documents."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """LangChain standard method: embed a single query text."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()

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
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


__all__ = ["SentenceTransformerEmbedding"]
