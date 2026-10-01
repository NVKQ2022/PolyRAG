"""Advanced RAG Pipeline with Pre-Retrieval and Post-Retrieval Optimizations."""

import time
from typing import Any

from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import RAGResponse
from polyrag.pipelines.base import BaseRAG


class AdvancedRAG(BaseRAG):
    """
    Advanced RAG pipeline featuring:
    1. Pre-Retrieval: Multi-query expansion and keyword enrichment.
    2. Retrieval: Multi-query vector search.
    3. Post-Retrieval: Reciprocal Rank Fusion (RRF) and similarity re-ranking.
    4. Generation: Context-optimized synthesis with source citations.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        llm_client: BaseLLMClient,
        chunker: BaseChunker | str | None = None,
        top_k: int = 5,
        num_expanded_queries: int = 3,
        min_relevance_score: float = 0.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm_client=llm_client,
            chunker=chunker,
        )
        self.top_k = top_k
        self.num_expanded_queries = num_expanded_queries
        self.min_relevance_score = min_relevance_score
        self.verbose = verbose

    def expand_query(self, question: str) -> list[str]:
        """Generate multiple perspective queries to improve vector search recall."""
        if self.llm_client is None or self.num_expanded_queries <= 1:
            return [question]

        prompt = f"""You are an expert AI search query optimizer.
Given the user query: "{question}"

Generate {self.num_expanded_queries} diverse search queries that capture different aspects, technical synonyms, and perspectives.
Return strictly one query per line without numbering, bullets, or extra commentary."""

        try:
            output = self.llm_client.complete(prompt)
            lines = [line.strip().lstrip("0123456789.-*• ") for line in output.split("\n") if line.strip()]
            queries = [q for q in lines if q]
            if question not in queries:
                queries.insert(0, question)
            return queries[: self.num_expanded_queries + 1]
        except Exception:
            return [question]

    def reciprocal_rank_fusion(
        self,
        search_runs: list[list[dict[str, Any]]],
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Merge and rank candidate chunks across multiple query runs
        using Reciprocal Rank Fusion (RRF).
        """
        rrf_scores: dict[tuple[str, Any], float] = {}
        doc_map: dict[tuple[str, Any], dict[str, Any]] = {}

        for run in search_runs:
            for rank, item in enumerate(run, start=1):
                doc = item.get("document", {})
                source = doc.get("source", "unknown")
                chunk_id = doc.get("chunk_id", "")
                key = (source, chunk_id)

                if key not in doc_map:
                    doc_map[key] = item

                # RRF Formula: sum(1 / (k + rank))
                rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (rrf_k + rank))

        # Sort descending by fused score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        fused_results: list[dict[str, Any]] = []
        for key in sorted_keys:
            base_item = doc_map[key].copy()
            base_item["rrf_score"] = rrf_scores[key]
            fused_results.append(base_item)

        return fused_results

    def execute(
        self,
        question: str,
        top_k: int | None = None,
        expand_queries: bool = True,
        re_rank: bool = True,
        **kwargs: Any,
    ) -> RAGResponse:
        """Execute the Advanced RAG pipeline."""
        start_time = time.time()
        k = top_k or self.top_k
        llm_calls = 0

        # Step 1: Pre-Retrieval Query Expansion
        t0 = time.time()
        if expand_queries and self.llm_client:
            queries = self.expand_query(question)
            llm_calls += 1
        else:
            queries = [question]
        expansion_ms = int((time.time() - t0) * 1000)

        # Step 2: Multi-Query Retrieval
        t0 = time.time()
        search_runs: list[list[dict[str, Any]]] = []
        for q in queries:
            search_runs.append(self.retrieve(q, top_k=k))
        retrieval_ms = int((time.time() - t0) * 1000)

        # Step 3: Post-Retrieval Fusion & Re-Ranking
        if len(search_runs) > 1 and re_rank:
            fused_results = self.reciprocal_rank_fusion(search_runs)
        else:
            # Flatten & deduplicate
            seen = set()
            fused_results = []
            for run in search_runs:
                for item in run:
                    doc = item.get("document", {})
                    key = (doc.get("source"), doc.get("chunk_id"))
                    if key not in seen:
                        seen.add(key)
                        fused_results.append(item)

        # Filter by minimum relevance score if configured
        if self.min_relevance_score > 0.0:
            fused_results = [r for r in fused_results if r.get("score", 0.0) >= self.min_relevance_score]

        top_results = fused_results[:k]
        context = self.format_context(top_results)

        # Step 4: Generation
        if self.llm_client is None:
            took_ms = int((time.time() - start_time) * 1000)
            return RAGResponse(
                question=question,
                answer="",
                context=context,
                sources=[r.get("document", {}) for r in top_results],
                took_ms=took_ms,
                confidence=1.0 if top_results else 0.0,
                reasoning_summary=f"Advanced RAG retrieved {len(top_results)} chunks across {len(queries)} queries.",
                llm_calls=llm_calls,
            )

        prompt = (
            "You are an expert technical assistant. Answer the user question based on the provided context.\n"
            "Cite sources in the format [source#chunk_id]. "
            "If the context does not contain sufficient facts, clearly explain what is missing.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer with citations:"
        )

        answer = self.llm_client.complete(prompt)
        llm_calls += 1
        took_ms = int((time.time() - start_time) * 1000)

        sources = [r.get("document", {}) for r in top_results]
        summary = (
            f"Expanded into {len(queries)} queries ({expansion_ms}ms). "
            f"Fused {len(fused_results)} chunks down to top-{len(top_results)} ({retrieval_ms}ms)."
        )

        return RAGResponse(
            question=question,
            answer=answer,
            context=context,
            sources=sources,
            took_ms=took_ms,
            confidence=0.90 if top_results else 0.2,
            reasoning_summary=summary,
            llm_calls=llm_calls,
        )


__all__ = ["AdvancedRAG"]
