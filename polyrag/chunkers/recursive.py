"""Recursive character chunker that splits hierarchically along natural text boundaries."""

from polyrag.core.interfaces import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """Recursively splits text using a list of natural separators."""

    def __init__(
        self,
        chunk_size: int = 550,
        chunk_overlap: int = 35,
        separators: list[str] | None = None,
        drop_empty: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.drop_empty = drop_empty

    def chunk(self, text: str) -> list[str]:
        if self.drop_empty and not text.strip():
            return []
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []

        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: list[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    other_chunks = self._split_text(s, new_separators)
                    good_splits.extend(other_chunks)
                else:
                    good_splits.append(s)

        # Merge good_splits respecting chunk_size and overlap
        current_chunk: list[str] = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_length + piece_len > self.chunk_size:
                if current_chunk:
                    joined = separator.join(current_chunk)
                    if not self.drop_empty or joined.strip():
                        final_chunks.append(joined)
                    # Keep overlap
                    overlap_chunk: list[str] = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        if overlap_len + len(p) <= self.chunk_overlap:
                            overlap_chunk.insert(0, p)
                            overlap_len += len(p)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_len
            current_chunk.append(piece)
            current_length += len(piece) + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            joined = separator.join(current_chunk)
            if not self.drop_empty or joined.strip():
                final_chunks.append(joined)

        return final_chunks
