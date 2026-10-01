# Module: `polyrag.chunkers` ✂️

The `polyrag.chunkers` module provides text-splitting strategies that divide large documents into smaller, semantically coherent passages for vector embedding and retrieval.

---

## 1. Available Chunkers

| Chunker | Splitting Strategy | Primary Use Case |
| :--- | :--- | :--- |
| **`RecursiveCharacterChunker`** *(Recommended)* | Splits hierarchically along natural punctuation (`\n\n` $\rightarrow$ `\n` $\rightarrow$ `. ` $\rightarrow$ `" "` $\rightarrow$ `""`) | Documentation, Markdown, PDF text, articles, Knowledge Base. |
| **`FixedSizeChunker`** | Slices by strict character length with sliding overlap | Raw byte logs, token-fixed window constraints, unformatted streams. |

---

## 2. `RecursiveCharacterChunker`

Preserves paragraphs, sentences, and words whole by treating `chunk_size` as a budget limit rather than a rigid slicing ruler.

### Parameters
```python
from polyrag.chunkers import RecursiveCharacterChunker

chunker = RecursiveCharacterChunker(
    chunk_size=550,        # Maximum characters per chunk
    chunk_overlap=35,      # Character overlap between adjacent chunks
    separators=None,       # Defaults to ["\n\n", "\n", ". ", " ", ""]
    drop_empty=True,       # Strip whitespace-only chunks
)
```

### How It Works
1. Attempts to split the document using the highest-priority separator (`\n\n` paragraph break).
2. If any piece exceeds `chunk_size`, it recursively splits that piece using the next separator (`\n` line break, then `. ` sentence boundary, then `" "` word space).
3. Merges consecutive pieces together until reaching `chunk_size` while retaining `chunk_overlap`.

### Example
```python
text = """Article ID: KB-001
Title: Authentication Failure

Symptoms:
- User receives HTTP 401
- Session terminates abruptly

Solutions:
1. Renew expired Bearer token
2. Re-authenticate user via SSO"""

chunks = chunker.chunk(text)
for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)
```

---

## 3. `FixedSizeChunker`

Slices strings strictly by index using a sliding window: `text[i : i + chunk_size]`.

### Parameters
```python
from polyrag.chunkers import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=550,
    chunk_overlap=35,
    drop_empty=True,
)
```

> [!WARNING]
> `FixedSizeChunker` does not inspect punctuation. Words that cross chunk boundaries may be sliced in half (e.g. `"inter"` and `"national"`), which can degrade embedding similarity scores.

---

## 4. Custom Chunker Implementation

To create a custom chunking strategy (e.g., token-based, Markdown-header-based, or AST-based), inherit from `BaseChunker`:

```python
from polyrag.core.interfaces import BaseChunker

class SemanticParagraphChunker(BaseChunker):
    """Splits only on explicit double newlines."""

    def __init__(self, max_paragraphs: int = 3):
        self.max_paragraphs = max_paragraphs

    def chunk(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        for i in range(0, len(paragraphs), self.max_paragraphs):
            chunks.append("\n\n".join(paragraphs[i:i + self.max_paragraphs]))
        return chunks
```
