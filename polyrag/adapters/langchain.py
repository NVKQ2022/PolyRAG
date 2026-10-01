"""LangChain ecosystem bridge adapters for PolyRAG."""

from collections.abc import Iterable
from typing import Any

from polyrag.core.models import Document as PolyDocument


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
