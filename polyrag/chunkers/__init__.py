"""Chunkers Package for polyrag."""

from typing import Any

from polyrag.chunkers.fixed_size import FixedSizeChunker
from polyrag.chunkers.recursive import RecursiveCharacterChunker
from polyrag.core.interfaces import BaseChunker


def resolve_chunker(
    chunker: BaseChunker | str | None = None,
    chunk_size: int = 550,
    chunk_overlap: int = 35,
    **kwargs: Any,
) -> BaseChunker:
    """
    Resolve or construct a chunking strategy from an instance or string identifier.

    Args:
        chunker: An existing BaseChunker instance or string name ('recursive', 'fixed_size').
            If None, defaults to RecursiveCharacterChunker.
        chunk_size: Chunk size in characters if instantiating from string or default.
        chunk_overlap: Overlap in characters between adjacent chunks.
        **kwargs: Extra parameters forwarded to the chunker constructor.

    Returns:
        Concrete BaseChunker instance.
    """
    if chunker is None:
        return RecursiveCharacterChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    if isinstance(chunker, BaseChunker):
        return chunker
    if isinstance(chunker, str):
        normalized = chunker.lower().strip().replace("-", "_")
        if normalized in ("fixed", "fixed_size", "fixedsize"):
            return FixedSizeChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )
        if normalized in ("recursive", "recursive_character", "recursivecharacter"):
            return RecursiveCharacterChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )
        raise ValueError(
            f"Unknown chunker strategy: {chunker!r}. "
            "Supported string values: 'recursive', 'fixed_size'."
        )
    raise TypeError(
        f"Expected BaseChunker instance or string strategy, got {type(chunker).__name__}"
    )


__all__ = ["FixedSizeChunker", "RecursiveCharacterChunker", "resolve_chunker"]
