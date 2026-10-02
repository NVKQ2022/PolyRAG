"""Agentic RAG Pipeline with Query Rewriting, Multi-Round Retrieval, Custom Tools, and State Schema."""

from collections.abc import Callable
import inspect
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
from polyrag.core.models import BaseMessage, HumanMessage, RAGResponse
from polyrag.exceptions import RetrievalError
from polyrag.pipelines.base import BaseRAG


class AgentTool:
    """Unified wrapper around LangChain tools, functions, or callables."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Any,
        args_schema: Any = None,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema

    def invoke(self, input_args: Any) -> Any:
        """Invoke underlying tool whether it's a LangChain tool or callable."""
        # 1. LangChain tool with invoke method
        if hasattr(self.func, "invoke") and callable(getattr(self.func, "invoke")):
            try:
                return self.func.invoke(input_args)
            except Exception:
                pass

        # 2. Dictionary input arguments
        if isinstance(input_args, dict):
            try:
                return self.func(**input_args)
            except TypeError:
                if len(input_args) == 1:
                    return self.func(next(iter(input_args.values())))
                return self.func(input_args)

        # 3. Scalar input
        return self.func(input_args)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            return self.invoke(kwargs)
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                return self.invoke(args[0])
            return self.func(*args)
        return self.func()

    def __repr__(self) -> str:
        return f"AgentTool(name='{self.name}', description='{self.description}')"


def normalize_tool(tool: Any) -> AgentTool:
    """Normalize a tool into an AgentTool instance."""
    if isinstance(tool, AgentTool):
        return tool

    # LangChain BaseTool or @tool decorated object
    if hasattr(tool, "name") and (hasattr(tool, "description") or hasattr(tool, "func")):
        name = str(tool.name)
        description = str(getattr(tool, "description", "") or "")
        args_schema = getattr(tool, "args_schema", None)
        return AgentTool(name=name, description=description, func=tool, args_schema=args_schema)

    # Standard callable function
    if callable(tool):
        name = getattr(tool, "__name__", "custom_tool")
        description = getattr(tool, "__doc__", "") or f"Custom tool {name}"
        return AgentTool(name=name, description=description.strip(), func=tool)

    raise TypeError(f"Expected callable or LangChain tool, got {type(tool).__name__}")


class AgenticRAG(BaseRAG):
    """
    Modern Agentic RAG pipeline featuring:
    1. Built-in default retrieval & collection inspection tools
    2. Support for custom external tools (LangChain @tool, BaseTool, or callables)
    3. State Schema support (TypedDict, Pydantic, or dictionary)
    4. Retrieval Planning, Semantic Query Rewriting, and Multi-Round Verification
    5. Fallback Direct Answering for non-technical or conversational queries.
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
        # Modern Agent parameters
        tools: list[Any] | None = None,
        state_schema: type | dict | None = None,
        system_prompt: str | None = None,
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
        self.state_schema = state_schema
        self.system_prompt = system_prompt

        # Normalize and register tools
        self.custom_tools: list[AgentTool] = [normalize_tool(t) for t in (tools or [])]
        self.default_tools: list[AgentTool] = [
            self._create_retrieve_tool(),
            self._create_list_sources_tool(),
        ]
        self.tools: list[AgentTool] = self.default_tools + self.custom_tools

        # Compile internal agent (LangGraph if available, or native PolyRAG engine)
        self.agent: Any = self._init_agent()

    def _create_retrieve_tool(self) -> AgentTool:
        """Create the default vector retrieval tool."""

        def retrieve_documents(query: str, top_k: int | None = None) -> str:
            """Search the indexed vector store for relevant chunks and text evidence using semantic retrieval."""
            k = top_k or self.top_k
            results = self.retrieve(query, top_k=k)
            if not results:
                return "No matching documents found in vector store."
            return self.format_context(results)

        return AgentTool(
            name="retrieve_documents",
            description="Search the indexed vector store for relevant chunks and text evidence using semantic retrieval.",
            func=retrieve_documents,
        )

    def _create_list_sources_tool(self) -> AgentTool:
        """Create the default source inspection tool."""

        def list_sources() -> str:
            """List document filenames or source identifiers currently indexed in the vector store."""
            sources = self.list_sources()
            if not sources:
                return "No document sources currently indexed."
            return "Indexed sources:\n" + "\n".join(f"- {s}" for s in sources)

        return AgentTool(
            name="list_sources",
            description="List document filenames or source identifiers currently indexed in the vector store.",
            func=list_sources,
        )

    def _init_agent(self) -> Any:
        """
        Compile the internal agent if LangGraph is available.
        Otherwise falls back to the native PolyRAG tool-calling engine.
        """
        try:
            from langgraph.prebuilt import create_react_agent

            raw_model = getattr(self.chat_model, "chat_model", self.chat_model)
            # Adapt tools for LangChain if needed
            return create_react_agent(
                model=raw_model,
                tools=[t.func if hasattr(t.func, "name") else t for t in self.tools],
                state_schema=self.state_schema,
                prompt=self.system_prompt or "You are a grounded RAG agent. Retrieve facts before answering.",
            )
        except Exception:
            return None

    def as_langchain_tool(
        self,
        name: str = "agentic_rag_search",
        description: str | None = None,
    ) -> Any:
        """Export this AgenticRAG instance as a LangChain-compatible tool."""
        desc = description or (
            "Search technical documentation and specifications with multi-round retrieval "
            "and grounded reflection. Returns verified answer and citations."
        )

        def rag_search_tool(query: str) -> str:
            res = self.query(query)
            return res.answer

        rag_search_tool.__name__ = name
        rag_search_tool.__doc__ = desc

        try:
            from langchain_core.tools import tool

            return tool(rag_search_tool)
        except Exception:
            return AgentTool(name=name, description=desc, func=rag_search_tool)

    def list_sources(self) -> list[str]:
        """List distinct document sources currently indexed in the vector store."""
        if hasattr(self.vector_store, "get_all_documents"):
            docs = self.vector_store.get_all_documents()
            return sorted(list({d.get("source", "unknown") for d in docs if isinstance(d, dict)}))
        if hasattr(self.vector_store, "documents"):
            docs = self.vector_store.documents
            return sorted(list({d.get("source", "unknown") for d in docs if isinstance(d, dict)}))
        return ["Indexed document collection"]

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

Tasks:
1. Determine if the evidence is sufficient to thoroughly answer the question.
2. If sufficient, generate the complete, grounded answer with source citations [source#chunk_id].
3. If insufficient and more rounds remain, specify missing information gaps and a refined query for the next round.
4. If this is the final round, synthesize the best possible answer using the available evidence.

Return ONLY valid JSON:
{{
    "enough": true,
    "confidence": 0.95,
    "reason": "Clear explanation of why evidence is or is not sufficient",
    "missing_gaps": ["gap 1 if not enough"],
    "next_query": "Refined semantic query for next round if not enough",
    "answer": "Grounded answer with citations [source#chunk_id] if enough or last round"
}}

Rules:
- If round == {max_rounds}, set enough = true and provide the final answer.
- Answer MUST use citations in the format [source#chunk_id].
- Never hallucinate facts outside the provided context.
"""
        result = self.llm_client.complete_json(prompt)

        return {
            "enough": result.get("enough", is_last_round),
            "confidence": float(result.get("confidence", 0.5)),
            "reason": result.get("reason", "Evaluation complete."),
            "missing_gaps": result.get("missing_gaps", []),
            "next_query": result.get("next_query"),
            "answer": result.get("answer"),
        }

    def _execute_with_custom_tools(
        self,
        question: str,
        current_state: dict[str, Any],
        k: int,
        rounds_limit: int,
        start_time: float,
    ) -> RAGResponse:
        """Execute agent reasoning loop when custom tools are provided."""
        tool_map = {t.name: t for t in self.tools}
        agent_log: list[dict[str, Any]] = current_state.setdefault("agent_log", [])
        all_results: list[dict[str, Any]] = current_state.setdefault("retrieved_evidence", [])
        seen_chunks: set[tuple[str, Any]] = set()

        tools_desc = "\n".join(f"- `{t.name}`: {t.description}" for t in self.tools)
        history_trace = f"User Question: {question}\n\nReasoning history:\n"

        final_answer = ""
        final_confidence = 0.85
        accumulated_context = ""

        system_instruction = self.system_prompt or (
            "You are an expert AI agent with access to specialized tools.\n"
            "Answer questions accurately by selecting tools, inspecting evidence, and formulating final answers."
        )

        for step_idx in range(1, rounds_limit + 2):
            prompt = f"""{system_instruction}

Available Tools:
{tools_desc}
- `final_answer`: Emit the final grounded answer once all needed evidence is collected. (Input: {{"answer": "<Grounded answer with citations [source#chunk_id]>", "confidence": <float 0.0-1.0>}})

{history_trace}

Step {step_idx} of {rounds_limit + 1}:
Provide your next Thought and Tool in strictly valid JSON format:
{{
    "thought": "<reasoning>",
    "tool": "<tool_name>",
    "args": {{ ... }}
}}
"""
            t0 = time.time()
            decision = self.llm_client.complete_json(prompt)
            thought = str(decision.get("thought", "Analyzing question..."))
            tool_name = str(decision.get("tool", "retrieve_documents"))
            args = dict(decision.get("args", {}))

            if tool_name == "final_answer":
                final_answer = str(args.get("answer", ""))
                final_confidence = float(args.get("confidence", 0.95))
                step_log = {
                    "action": "FINAL_ANSWER",
                    "step": step_idx,
                    "thought": thought,
                    "answer": final_answer,
                    "confidence": final_confidence,
                    "took_ms": int((time.time() - t0) * 1000),
                }
                agent_log.append(step_log)
                self._log_action("FINAL_ANSWER", step_log)
                break

            # Execute tool
            obs = ""
            if tool_name in tool_map:
                try:
                    tool_inst = tool_map[tool_name]
                    if tool_name == "retrieve_documents":
                        q_str = args.get("query", question)
                        results = self.retrieve(q_str, top_k=k)
                        for r in results:
                            doc = r.get("document", {})
                            chunk_key = (doc.get("source", "unknown"), doc.get("chunk_id", ""))
                            if chunk_key not in seen_chunks:
                                seen_chunks.add(chunk_key)
                                all_results.append(r)
                                current_state.setdefault("citations", []).append(
                                    {"source": chunk_key[0], "chunk_id": chunk_key[1]}
                                )
                        obs = self.format_context(results)
                        accumulated_context = self.format_context(all_results)
                    else:
                        obs_res = tool_inst.invoke(args)
                        obs = str(obs_res)
                        # If custom tool returned a dictionary, update custom state fields
                        if isinstance(obs_res, dict):
                            for k_res, v_res in obs_res.items():
                                if k_res not in ("messages", "agent_log"):
                                    current_state[k_res] = v_res
                except Exception as e:
                    obs = f"Error executing tool '{tool_name}': {e}"
            else:
                obs = f"Unknown tool '{tool_name}'."

            step_log = {
                "action": "TOOL_EXECUTION",
                "step": step_idx,
                "thought": thought,
                "tool": tool_name,
                "args": args,
                "observation": obs[:200] + "..." if len(obs) > 200 else obs,
                "took_ms": int((time.time() - t0) * 1000),
            }
            agent_log.append(step_log)
            self._log_action(f"TOOL:{tool_name}", step_log)

            history_trace += f"\nThought: {thought}\nAction: {tool_name}({args})\nObservation: {obs}\n"

        if not final_answer:
            prompt = f"Answer the question based on accumulated evidence:\n{accumulated_context or history_trace}\nQuestion: {question}\nAnswer:"
            final_answer = self.llm_client.complete(prompt)

        current_state["answer"] = final_answer
        total_took_ms = int((time.time() - start_time) * 1000)
        sources = [r.get("document", {}) for r in all_results]

        return RAGResponse(
            question=question,
            answer=final_answer,
            context=accumulated_context,
            sources=sources,
            took_ms=total_took_ms,
            confidence=round(final_confidence, 2),
            reasoning_summary=f"Executed agent loop with {len(agent_log)} action(s).",
            agent_log=agent_log,
            llm_calls=len(agent_log) + 1,
            state=current_state,
        )

    def execute(
        self,
        question: str,
        top_k: int | None = None,
        max_rounds: int | None = None,
        state: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """
        Execute the Agentic RAG workflow for a given question.
        Manages state conforming to state_schema, executes tools, and returns RAGResponse with state.
        """
        start_time = time.time()
        k = top_k or self.top_k
        rounds_limit = max_rounds or self.max_rounds

        # Initialize state conforming to state_schema or default dictionary
        current_state: dict[str, Any] = dict(state or {})
        if "messages" not in current_state:
            current_state["messages"] = [HumanMessage(content=question)]
        if "citations" not in current_state:
            current_state["citations"] = []
        if "retrieved_evidence" not in current_state:
            current_state["retrieved_evidence"] = []
        if "agent_log" not in current_state:
            current_state["agent_log"] = []

        # If custom tools were provided, run the custom tools agent execution loop
        if len(self.custom_tools) > 0:
            return self._execute_with_custom_tools(
                question=question,
                current_state=current_state,
                k=k,
                rounds_limit=rounds_limit,
                start_time=start_time,
            )

        # Standard Multi-Round Retrieval & Reflection flow
        agent_log: list[dict[str, Any]] = current_state["agent_log"]

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

            current_state["answer"] = answer
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
                state=current_state,
            )

        # Step 2: Iterative Multi-Round Retrieval Loop
        current_query = decision["query"]
        all_results: list[dict[str, Any]] = current_state["retrieved_evidence"]
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
                    current_state["citations"].append({"source": source, "chunk_id": chunk_id})

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

        current_state["answer"] = final_answer
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
            state=current_state,
        )

    def query(
        self,
        question: str,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RAGResponse:
        """Alias for execute with optional state parameter."""
        return self.execute(question, state=state, **kwargs)
