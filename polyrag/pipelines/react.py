"""ReAct (Reasoning + Acting) Agent Pipeline."""

import json
import re
import time
from typing import Any

from polyrag.core.interfaces import (
    BaseChunker,
    BaseEmbeddingModel,
    BaseLLMClient,
    BaseVectorStore,
)
from polyrag.core.models import AgentAction, AgentResponse, AgentStep
from polyrag.pipelines.base import BaseRAG


class ReActAgent(BaseRAG):
    """
    ReAct Agent implementing a structured Thought-Action-Observation loop
    using tool invocation to solve single-hop and multi-hop queries.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        chunker: BaseChunker | None = None,
        max_steps: int = 4,
        default_top_k: int = 5,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm_client=llm_client,
            chunker=chunker,
        )
        self.max_steps = max_steps
        self.default_top_k = default_top_k
        self.verbose = verbose

    def _build_system_prompt(self) -> str:
        return f"""You are an expert technical ReAct (Reasoning + Acting) Agent.
Your job is to answer questions accurately by retrieving evidence from the indexed document knowledge base.

You have access to the following tools:
1. `search`: Search the vector knowledge base for relevant passages.
   Input: {{"query": "<keywords/phrases>", "top_k": <integer, default {self.default_top_k}>}}
2. `list_docs`: Inspect sample document titles or collection overview.
   Input: {{}}
3. `final_answer`: Emit the final grounded answer once sufficient evidence is gathered.
   Input: {{"answer": "<Grounded answer with citations [source#chunk_id]>", "confidence": <float 0.0-1.0>}}

You must operate in a strict JSON Thought -> Action loop.
At each step, respond with EXACTLY ONE JSON object:
{{
    "thought": "<Reasoning about current evidence and what information is still missing>",
    "action": "<search | list_docs | final_answer>",
    "action_input": {{ ... }}
}}

Rules:
- Never assume facts without citing evidence from retrieved chunks.
- If you have gathered sufficient evidence to answer the question, immediately emit 'final_answer'.
- Always return strictly valid JSON.
"""

    def _parse_action(self, llm_output: str, fallback_query: str) -> AgentAction:
        """Safely extract and parse JSON action from LLM output."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
        clean = match.group(1) if match else llm_output
        clean = clean.strip()

        first_brace = clean.find("{")
        last_brace = clean.rfind("}")
        if first_brace != -1 and last_brace != -1:
            clean = clean[first_brace : last_brace + 1]

        try:
            data = json.loads(clean)
            return AgentAction(
                thought=str(data.get("thought", "Evaluating next step...")),
                action=str(data.get("action", "search")),
                action_input=dict(data.get("action_input", {})),
            )
        except Exception:
            return AgentAction(
                thought="Searching knowledge base for relevant context.",
                action="search",
                action_input={"query": fallback_query, "top_k": self.default_top_k},
            )

    def _execute_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search vector store and return results."""
        query_vec = self.embedding_model.embed_text(query)
        return self.vector_store.search(query_vector=query_vec, top_k=top_k)

    def execute(
        self,
        question: str,
        top_k: int | None = None,
        max_steps: int | None = None,
    ) -> AgentResponse:
        """Execute the ReAct workflow for a question."""
        start_time = time.time()
        k = top_k or self.default_top_k
        limit = max_steps or self.max_steps

        trajectory: list[AgentStep] = []
        cumulative_chunks: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()

        system_prompt = self._build_system_prompt()
        conversation_history = f"User Question: {question}\n\nBegin your reasoning loop."

        final_answer = ""
        confidence = 0.85

        for step_idx in range(1, limit + 1):
            prompt = (
                f"{system_prompt}\n\n"
                f"{conversation_history}\n\n"
                f"Current Step: {step_idx} of {limit}\n"
                "Provide your next Thought and Action in strictly valid JSON format."
            )

            t0 = time.time()
            llm_output = self.llm_client.complete(prompt)
            action = self._parse_action(llm_output, question)

            if action.action == "final_answer":
                final_answer = str(action.action_input.get("answer", ""))
                confidence = float(action.action_input.get("confidence", 0.95))
                step_ms = int((time.time() - t0) * 1000)
                trajectory.append(
                    AgentStep(
                        step_num=step_idx,
                        thought=action.thought,
                        action="final_answer",
                        action_input=action.action_input,
                        observation="Final answer formulated.",
                        chunks_retrieved=0,
                        took_ms=step_ms,
                    )
                )
                if self.verbose:
                    print(f">> [ReAct Step {step_idx}] Final Answer Emitted.")
                break

            # Execute tool
            obs = ""
            new_chunks_count = 0
            if action.action in ("search", "search_rfc"):
                query_str = str(action.action_input.get("query", question))
                step_k = int(action.action_input.get("top_k", k))
                retrieved = self._execute_search(query_str, top_k=step_k)

                obs_lines = []
                for res in retrieved:
                    doc = res.get("document", {})
                    source = doc.get("source", "unknown")
                    cid = doc.get("chunk_id", "")
                    ident = f"{source}#{cid}"
                    text = doc.get("text", "")

                    if ident not in seen_chunk_ids:
                        seen_chunk_ids.add(ident)
                        cumulative_chunks.append(doc)
                        new_chunks_count += 1
                    obs_lines.append(f"[{ident}]: {text[:220]}...")

                obs = f"Retrieved {len(retrieved)} chunks:\n" + "\n".join(obs_lines)

            elif action.action in ("list_docs", "list_available_docs"):
                count = self.vector_store.count()
                sample = self.vector_store.peek(limit=3)
                obs = f"Vector database contains {count} indexed chunks. Sample: {sample}"

            else:
                obs = f"Unknown tool '{action.action}'. Please use 'search', 'list_docs', or 'final_answer'."

            step_ms = int((time.time() - t0) * 1000)
            trajectory.append(
                AgentStep(
                    step_num=step_idx,
                    thought=action.thought,
                    action=action.action,
                    action_input=action.action_input,
                    observation=obs[:500],
                    chunks_retrieved=new_chunks_count,
                    took_ms=step_ms,
                )
            )

            if self.verbose:
                print(f">> [ReAct Step {step_idx}] Action: {action.action} | New chunks: {new_chunks_count}")

            conversation_history += (
                f"\n\nThought {step_idx}: {action.thought}\n"
                f"Action {step_idx}: {action.action}({action.action_input})\n"
                f"Observation {step_idx}: {obs}"
            )

        # Fallback if no final answer reached
        if not final_answer:
            evidence_text = "\n\n".join(
                f"[{c.get('source', 'unknown')}#{c.get('chunk_id', '')}]\n{c.get('text', '')}"
                for c in cumulative_chunks
            )
            fallback_prompt = (
                f"Based on the following collected evidence:\n{evidence_text}\n\n"
                f"Answer the question: {question}\n"
                "Cite sources as [source#chunk_id]."
            )
            final_answer = self.llm_client.complete(fallback_prompt)
            confidence = 0.70

        total_took_ms = int((time.time() - start_time) * 1000)
        sources = [
            {
                "source": c.get("source", "unknown"),
                "chunk_id": c.get("chunk_id", ""),
            }
            for c in cumulative_chunks
        ]
        summary = (
            f"Executed {len(trajectory)} reasoning steps. "
            f"Retrieved {len(cumulative_chunks)} unique chunks across {len(seen_chunk_ids)} unique references."
        )

        return AgentResponse(
            question=question,
            answer=final_answer,
            sources=sources,
            retrieved_evidence=cumulative_chunks,
            trajectory=trajectory,
            reasoning_summary=summary,
            confidence=confidence,
            total_steps=len(trajectory),
            took_ms=total_took_ms,
            llm_calls=len(trajectory) + (1 if not final_answer else 0),
        )

    def query(self, question: str, **kwargs: Any) -> AgentResponse:
        """Alias for execute."""
        return self.execute(question, **kwargs)


# Alias for consistent RAG hierarchy naming
ReActRAG = ReActAgent

__all__ = ["ReActAgent", "ReActRAG"]
