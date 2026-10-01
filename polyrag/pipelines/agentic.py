"""Agentic RAG Pipeline with Query Rewriting, Multi-Round Retrieval, and Fused Reflection."""

import time
from typing import Any

from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import RAGResponse
from polyrag.exceptions import RetrievalError
from polyrag.pipelines.base import BaseRAG


class AgenticRAG(BaseRAG):
    """
    Agentic RAG pipeline featuring:
    1. Retrieval Planning & Semantic Query Rewriting
    2. Iterative Multi-Round Vector Retrieval & Deduplication
    3. Fused Reflection + Grounded Synthesis with Source Citations
    4. Fallback Direct Answering for non-technical or conversational queries.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel | str | Any | None = None,
        vector_store: BaseVectorStore | None = None,
        llm_client: BaseLLMClient | str | Any | None = None,
        chunker: BaseChunker | str | None = None,
        top_k: int = 3,
        max_rounds: int = 2,
        verbose: bool = False,
        embedding: BaseEmbeddingModel | str | Any | None = None,
        chat_model: BaseLLMClient | str | Any | None = None,
        llm: BaseLLMClient | str | Any | None = None,
    ) -> None:
        super().__init__(
            embedding_model=embedding if embedding is not None else embedding_model,
            vector_store=vector_store,
            llm_client=llm_client,
            chunker=chunker,
            chat_model=chat_model,
            llm=llm,
        )
        self.top_k = top_k
        self.max_rounds = max_rounds
        self.verbose = verbose

    def _log_action(self, action_name: str, details: dict[str, Any]) -> None:
        """Pretty print an observable agent action when verbose is enabled."""
        if not self.verbose:
            return
        print(f"\n>> [ACTION: {action_name.upper()}]")
        for k, v in details.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                print(f"   • {k}: {len(v)} item(s)")
            elif isinstance(v, str) and "\n" in v:
                first_line = v.strip().split("\n")[0]
                print(f"   • {k}: {first_line}...")
            else:
                print(f"   • {k}: {v}")

    def decide_retrieval(self, question: str) -> dict[str, Any]:
        """
        Analyze the question to decide if vector retrieval is needed,
        and rewrite the query to optimize keyword density.
        """
        prompt = f"""You are the retrieval planner of an Agentic RAG system for technical documents and specifications.

Question:
{question}

Decide whether retrieving from local technical documents is needed.

Return ONLY valid JSON:
{{
    "retrieval_needed": true,
    "query": "concise semantic search query with technical keywords",
    "reason": "brief explanation"
}}

Rules:
- retrieval_needed = false for greetings or conversational queries.
- retrieval_needed = true for specific domain, protocol, architecture, or technical questions.
- query should rewrite and expand technical keywords for optimal vector search.
"""
        result = self.llm_client.complete_json(prompt)
        return {
            "retrieval_needed": result.get("retrieval_needed", True),
            "query": (result.get("query") or question).strip(),
            "reason": result.get("reason", "Retrieval is used by default."),
        }

    def direct_answer(self, question: str) -> str:
        """Answer conversational or out-of-domain questions directly without retrieval."""
        prompt = f"Answer the following user query politely and concisely:\n{question}"
        return self.llm_client.complete(prompt)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve top-k chunks from the vector store."""
        k = top_k or self.top_k
        query_vector = self.embedding_model.embed_text(query)
        return self.vector_store.search(query_vector=query_vector, top_k=k)

    def format_context(self, search_results: list[dict[str, Any]]) -> str:
        """Format accumulated search results into a clean context string."""
        blocks: list[str] = []
        for result in search_results:
            document = result.get("document", {})
            source = document.get("source", "unknown")
            chunk_id = document.get("chunk_id", "")
            text = document.get("text", "")
            blocks.append(f"Source: {source}#{chunk_id}\n{text}")
        return "\n\n---\n\n".join(blocks)

    def reflect_and_evaluate(
        self,
        question: str,
        context: str,
        round_number: int,
        max_rounds: int,
    ) -> dict[str, Any]:
        """
        Evaluates evidence sufficiency AND synthesizes the final answer in a single call
        if evidence is sufficient (saving an extra LLM call).
        """
        is_last_round = round_number >= max_rounds

        prompt = f"""You are the reflection and answering engine of an Agentic RAG system.

Original question:
{question}

Retrieval round:
{round_number} of {max_rounds}

Accumulated Evidence:
{context}

Task:
1. Evaluate whether the evidence contains sufficient facts to answer the question completely.
2. If sufficient (or if this is the final round {max_rounds}), set "enough": true and provide the complete "answer" citing sources as [source#chunk_id] (e.g. [document.txt#14]).
3. If NOT sufficient and rounds remain, set "enough": false, identify "missing_gaps", and formulate a concise "next_query" with dense technical keywords.

Return ONLY valid JSON:
{{
    "enough": true,
    "confidence": 0.95,
    "reason": "brief reflection explanation",
    "missing_gaps": [],
    "next_query": "",
    "answer": "Grounded answer with citations [source#chunk_id] if enough=true, otherwise empty"
}}
"""
        result = self.llm_client.complete_json(prompt)
        return {
            "enough": result.get("enough", is_last_round),
            "confidence": float(result.get("confidence", 0.85 if is_last_round else 0.5)),
            "reason": result.get("reason", "Evaluated evidence."),
            "missing_gaps": result.get("missing_gaps", []),
            "next_query": (result.get("next_query") or "").strip(),
            "answer": (result.get("answer") or "").strip(),
        }

    def execute(
        self,
        question: str,
        top_k: int | None = None,
        max_rounds: int | None = None,
    ) -> RAGResponse:
        """Execute the multi-round Agentic RAG workflow."""
        start_time = time.time()
        k = top_k or self.top_k
        rounds_limit = max_rounds or self.max_rounds
        agent_log: list[dict[str, Any]] = []

        if self.verbose:
            print("=" * 65)
            print("                 AGENTIC RAG PIPELINE")
            print("=" * 65)
            print(f"Question: {question}")

        # Step 1: Planning and Query Routing
        t0 = time.time()
        decision = self.decide_retrieval(question)
        planning_ms = int((time.time() - t0) * 1000)

        planning_log = {
            "action": "PLANNING",
            "retrieval_needed": decision["retrieval_needed"],
            "query": decision["query"],
            "reason": decision["reason"],
            "took_ms": planning_ms,
        }
        agent_log.append(planning_log)
        self._log_action("PLANNING", planning_log)

        # Non-retrieval conversational branch
        if not decision["retrieval_needed"]:
            t0 = time.time()
            answer = self.direct_answer(question)
            gen_ms = int((time.time() - t0) * 1000)

            direct_log = {
                "action": "DIRECT_ANSWER",
                "reason": decision["reason"],
                "took_ms": gen_ms,
            }
            agent_log.append(direct_log)
            self._log_action("DIRECT_ANSWER", direct_log)

            return RAGResponse(
                question=question,
                answer=answer,
                context="",
                sources=[],
                took_ms=int((time.time() - start_time) * 1000),
                confidence=1.0,
                reasoning_summary=f"Direct answer generated without retrieval. Reason: {decision['reason']}",
                agent_log=agent_log,
                llm_calls=2,
            )

        # Step 2: Iterative Multi-Round Retrieval Loop
        current_query = decision["query"]
        all_results: list[dict[str, Any]] = []
        seen_chunks: set[tuple[str, Any]] = set()
        final_answer = ""
        final_confidence = 0.5
        accumulated_context = ""

        for round_num in range(1, rounds_limit + 1):
            if self.verbose:
                print(f"\n--- [ROUND {round_num} OF {rounds_limit}] ---")

            t0 = time.time()
            results = self.retrieve(current_query, top_k=k)
            retrieve_ms = int((time.time() - t0) * 1000)

            retrieval_log = {
                "action": "VECTOR_SEARCH",
                "round": round_num,
                "query": current_query,
                "retrieved_count": len(results),
                "took_ms": retrieve_ms,
            }
            agent_log.append(retrieval_log)
            self._log_action("VECTOR_SEARCH", retrieval_log)

            # Deduplication
            new_chunks_count = 0
            new_chunk_ids = []
            for r in results:
                doc = r.get("document", {})
                source = doc.get("source", "unknown")
                chunk_id = doc.get("chunk_id", "")
                chunk_key = (source, chunk_id)

                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    all_results.append(r)
                    new_chunks_count += 1
                    new_chunk_ids.append(f"{source}#{chunk_id}")

            dedup_log = {
                "action": "DEDUPLICATION",
                "round": round_num,
                "new_chunks_added": new_chunks_count,
                "total_cumulative_chunks": len(all_results),
                "new_chunk_ids": new_chunk_ids,
            }
            agent_log.append(dedup_log)
            self._log_action("DEDUPLICATION", dedup_log)

            accumulated_context = self.format_context(all_results)

            # Fused Reflection and Evaluation
            t0 = time.time()
            eval_result = self.reflect_and_evaluate(
                question=question,
                context=accumulated_context,
                round_number=round_num,
                max_rounds=rounds_limit,
            )
            eval_ms = int((time.time() - t0) * 1000)
            final_confidence = eval_result["confidence"]

            reflect_log = {
                "action": "REFLECTION_AND_EVALUATION",
                "round": round_num,
                "is_sufficient": eval_result["enough"],
                "confidence": final_confidence,
                "reason": eval_result["reason"],
                "missing_gaps": eval_result.get("missing_gaps", []),
                "next_query": eval_result.get("next_query"),
                "took_ms": eval_ms,
            }
            agent_log.append(reflect_log)
            self._log_action("REFLECTION_AND_EVALUATION", reflect_log)

            if eval_result["enough"] and eval_result.get("answer"):
                final_answer = eval_result["answer"]
                if self.verbose:
                    print(">> [DECISION] Evidence sufficient! Answer synthesized directly in reflection.")
                break

            next_query = eval_result.get("next_query")
            if not next_query or not next_query.strip() or next_query.strip().lower() == current_query.strip().lower():
                if self.verbose:
                    print(">> [DECISION] No further unique query. Concluding.")
                final_answer = eval_result.get("answer", "")
                break

            refine_log = {
                "action": "QUERY_REFINEMENT",
                "round": round_num,
                "previous_query": current_query,
                "refined_next_query": next_query,
            }
            agent_log.append(refine_log)
            self._log_action("QUERY_REFINEMENT", refine_log)

            current_query = next_query

        # Fallback synthesis if answer was not generated
        if not final_answer:
            prompt = (
                f"Answer the question based only on the evidence:\n"
                f"Context:\n{accumulated_context}\n\n"
                f"Question: {question}\n"
                f"Answer with citations [source#chunk_id]:"
            )
            final_answer = self.llm_client.complete(prompt)

        total_took_ms = int((time.time() - start_time) * 1000)
        rounds_executed = len([l for l in agent_log if l.get("action") == "VECTOR_SEARCH"])
        llm_calls_made = len(
            [l for l in agent_log if l.get("action") in ("PLANNING", "REFLECTION_AND_EVALUATION", "DIRECT_ANSWER")]
        )

        sources = [r.get("document", {}) for r in all_results]
        summary = (
            f"Executed {rounds_executed} retrieval round(s) with {llm_calls_made} LLM call(s). "
            f"Gathered {len(all_results)} unique chunks across {len(seen_chunks)} sources. "
            f"Evidence confidence: {final_confidence:.2f}."
        )

        return RAGResponse(
            question=question,
            answer=final_answer,
            context=accumulated_context,
            sources=sources,
            took_ms=total_took_ms,
            confidence=round(final_confidence, 2),
            reasoning_summary=summary,
            agent_log=agent_log,
            llm_calls=llm_calls_made,
        )

    def query(self, question: str, **kwargs: Any) -> RAGResponse:
        """Alias for execute."""
        return self.execute(question, **kwargs)
