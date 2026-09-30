"""Local Sentence-Transformers embedding implementation."""

from polyrag.core.interfaces import BaseEmbeddingModel


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Local Sentence Transformers / HuggingFace embedding adapter."""

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

        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError(f"Could not determine embedding dimension for model: {model_name}")

        self._dim = dimension

    @property
    def dim(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 128,
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
