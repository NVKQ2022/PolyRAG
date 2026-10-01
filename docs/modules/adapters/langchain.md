# LangChain Document Loader Bridge (`polyrag.adapters.langchain`)

PolyRAG includes a zero-overhead, duck-typed bridge that allows developers to ingest documents directly from any of **LangChain's 150+ battle-tested Document Loaders** (e.g. `PyPDFLoader`, `Docx2txtLoader`, `CSVLoader`, `WebBaseLoader`, `ConfluenceLoader`, `S3FileLoader`).

---

## 1. Zero Dependency Overhead

> [!NOTE]
> PolyRAG **does not require `langchain` as a core dependency**. The bridge uses standard Python duck-typing on attributes (`page_content`, `metadata`, `id`), so you only install specific LangChain loaders if and when your application needs them.

---

## 2. Ingestion Methods

PolyRAG provides three intuitive methods on both [`PolyRAG`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/app.py) and all pipeline subclasses derived from [`BaseRAG`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/pipelines/base.py):

| Method | Accepts | Description |
| :--- | :--- | :--- |
| `rag.ingest_langchain_loader(loader, metadata=None)` | Any LangChain Loader instance | Automatically calls `loader.lazy_load()` (or `loader.load()`) and streams chunks directly into the vector store. |
| `rag.ingest_documents(documents, metadata=None)` | Iterable / Generator of Docs | Ingests any list or generator of LangChain `Document`s, PolyRAG `Document`s, or dicts. |
| `rag.ingest_langchain_documents(docs, metadata=None)` | List of LangChain `Document`s | Convenience alias for `ingest_documents`. |

---

## 3. Practical Examples

### A. Ingesting Complex PDFs (`PyPDFLoader`)
```bash
pip install pypdf langchain-community
```

```python
from langchain_community.document_loaders import PyPDFLoader
from polyrag.app import PolyRAG

rag = PolyRAG.from_env()

# 1. Initialize any LangChain loader
loader = PyPDFLoader("financial_report_2024.pdf")

# 2. Ingest streamingly into PolyRAG (memory-safe via lazy_load)
rag.ingest_langchain_loader(loader, metadata={"department": "Finance"})

# 3. Query using PolyRAG's native RAG pipelines
response = rag.query("What was the annual recurring revenue?")
print(response.answer)
```

---

### B. Ingesting Web Pages (`WebBaseLoader`)
```bash
pip install beautifulsoup4 langchain-community
```

```python
from langchain_community.document_loaders import WebBaseLoader
from polyrag.app import PolyRAG

rag = PolyRAG.from_env()

loader = WebBaseLoader("https://docs.python.org/3/whatsnew/3.12.html")
rag.ingest_langchain_loader(loader, metadata={"topic": "Python 3.12"})

response = rag.query("What are the main performance improvements in Python 3.12?")
print(response.answer)
```

---

### C. Ingesting CSV or Excel Spreadsheets (`CSVLoader`)
```python
from langchain_community.document_loaders import CSVLoader
from polyrag.app import PolyRAG

rag = PolyRAG.from_env()

loader = CSVLoader(file_path="customer_support_tickets.csv", encoding="utf-8")
rag.ingest_langchain_loader(loader, metadata={"type": "Support Tickets"})

response = rag.query("What are the top 3 reported authentication bugs?")
print(response.answer)
```

---

## 4. Standalone Converter: `LangChainDocumentConverter`

If you want to convert LangChain documents into native PolyRAG [`Document`](file:///home/quan/projects/pythonPackage/PolyRAG/polyrag/core/models.py#L8) objects explicitly:

```python
from polyrag.adapters.langchain import LangChainDocumentConverter

# Single document conversion
poly_doc = LangChainDocumentConverter.to_polyrag_document(lc_doc)
print(poly_doc.text)
print(poly_doc.source)
print(poly_doc.metadata)

# Batch conversion from generator or list
poly_docs = LangChainDocumentConverter.to_polyrag_documents(loader.lazy_load())
```
