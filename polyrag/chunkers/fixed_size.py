"""Fixed size overlapping character chunker built on langchain_text_splitters."""

from typing import Any
from polyrag.core.interfaces import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Split text into fixed-size overlapping character chunks."""

    def __init__(
        self,
        chunk_size: int = 550,
        chunk_overlap: int = 35,
        drop_empty: bool = True,
        **kwargs: Any,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.drop_empty = drop_empty

    def chunk(self, text: str) -> list[str]:
        if self.drop_empty and not text.strip():
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []

        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            if not self.drop_empty or chunk.strip():
                chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break

        return chunks

    def split_text(self, text: str) -> list[str]:
        return self.chunk(text)


__all__ = ["FixedSizeChunker"]
