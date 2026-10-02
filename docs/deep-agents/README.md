# Deep Agents in RAG: Cost, Complexity, and Workflow Analysis 🧠

This guide provides an engineering and architectural analysis of applying **Deep Agents** (and the modern LangChain Deep Agents harness) to Retrieval-Augmented Generation (RAG). It evaluates whether Deep Agents represent **overkill** for standard RAG, examines their **cost profile**, details their **architectural complexity**, and outlines the **step-by-step workflow** required to build and operate them.

---

## 1. Executive Summary & Verdict: Is Deep Agent an Overkill for RAG?

> [!IMPORTANT]
> **Verdict**: For **90% of production RAG use cases, Deep Agents are complete overkill.**
> 
> However, for the remaining **10% of open-ended, multi-document synthesis and long-horizon research tasks**, they solve fundamental context window and reasoning failure modes that traditional RAG cannot overcome.

### Why It Is Overkill for 90% of RAG Tasks
Most RAG applications solve direct question-answering over indexed documentation:
* *"What is the cancellation policy?"*
* *"How do I configure the Redis port in setting.json?"*
* *"Summarize section 4 of the uploaded quarterly report."*

For these queries:
1. **Latency Penalty**: Users expect answers in **0.8s – 3s**. Deep Agents take **30s – 5+ minutes**.
2. **Cost Explosion**: Deep Agents consume **20x – 100x more tokens** per query than standard RAG pipelines.
3. **Operational Overhead**: Requires distributed state persistence, job queues (Celery/Temporal), sandboxed execution, and subagent concurrency management.
4. **Diminishing Returns**: If the answer is contained within 1 to 3 distinct text chunks, running hierarchical task decomposition, subagent delegation, and workspace scratchpads yields zero accuracy benefit.

### When Deep Agents Are Justified (The 10% Sweet Spot)
Deep Agents become essential when the query demands **long-horizon research, cross-document comparison, and multi-faceted synthesis**:
* *"Compare the risk factors and capital expenditure trajectories across all 25 indexed 10-K filings for Fortune 500 tech companies and synthesize a 10-page audit report."*
* *"Perform an exhaustive codebase security review against OWASP Top 10 and produce a prioritized remediation matrix with code diffs."*
* Tasks where human analysts would spend 2–6 hours searching, reading, taking notes, cross-referencing, and synthesizing.

---

## 2. Standard Agent vs. Deep Agent

In the modern AI ecosystem (specifically LangChain's Deep Agents harness and LangGraph runtime), agents are divided into two distinct architectural classes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Standard Agent (Shallow / Single-Loop)               │
│                                                                        │
│   User Query ──► [ LLM (Prompt + Full History) ] ◄──► [ Tools ]        │
│                         │                                              │
│                         ▼ (Appends all raw tool results to history)     │
│                   Final Answer (2-5 steps)                             │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                   Deep Agent (Hierarchical / Multi-Harness)             │
│                                                                        │
│   User Query ──► [ Lead Orchestrator ] ──► [ Task Planning DAG ]       │
│                         │                                              │
│         ┌───────────────┼───────────────┐                              │
│         ▼               ▼               ▼                              │
│   [ Subagent A ]  [ Subagent B ]  [ Subagent C ]                       │
│    (Topic Alpha)   (Topic Beta)    (Fact Check)                        │
│         │               │               │                              │
│         └───────────────┼───────────────┘                              │
│                         ▼                                              │
│             [ Workspace / Filesystem ]  (Notes, Memos, Outlines)       │
│                         │                                              │
│                         ▼                                              │
│             [ Reflection & Auditor ]                                   │
│                         │                                              │
│                         ▼                                              │
│             Long-Form Research Report (20-100+ steps)                  │
└────────────────────────────────────────────────────────────────────────┘
```

### The 4 Pillars of Deep Agents
1. **Explicit Planning Tools**: The agent creates and maintains a structured task backlog (`pending`, `in_progress`, `completed`). It dynamically adapts its plan as new findings emerge or information gaps are detected.
2. **Workspace / Persistent Filesystem**: Offloads raw data, intermediate summaries, and notes to a virtual or disk-backed filesystem. The LLM's active prompt stays clean, avoiding "lost in the middle" degradation.
3. **Hierarchical Subagent Delegation**: The lead orchestrator spawns specialized, isolated worker subagents. Each subagent solves a specific sub-problem, extracts key insights, and returns only a distilled memo, keeping the main context pristine.
4. **Context Engineering & Compaction**: Employs Recursive Language Models (RLMs), map-reduce passes over document batches, prompt caching, and thread summarization to manage massive token volumes efficiently.

---

## 3. Cost Analysis

The cost difference between standard RAG and Deep Agent RAG is orders of magnitude apart.

### Token Consumption Breakdown per Query

| Metric | Naive RAG | Advanced RAG (RRF) | Agentic RAG (PolyRAG) | Deep Agent RAG |
| :--- | :--- | :--- | :--- | :--- |
| **Number of LLM Calls** | 1 call | 2 – 3 calls | 2 – 5 calls | **20 – 100+ calls** |
| **Input Tokens** | ~1,500 tokens | ~4,000 tokens | ~6,000 tokens | **100,000 – 500,000+ tokens** |
| **Output Tokens** | ~300 tokens | ~400 tokens | ~600 tokens | **5,000 – 25,000 tokens** |
| **Est. Cost (GPT-4o)** | **$0.007** | **$0.015** | **$0.025** | **$0.40 – $2.50+** |
| **Est. Cost (GPT-4o-mini)**| **$0.0003** | **$0.0008** | **$0.0015** | **$0.02 – $0.15** |
| **Median Execution Time**| **0.8 seconds** | **1.8 seconds** | **3.5 seconds** | **45 – 300 seconds** |

### Why Deep Agents Consume So Many Tokens
1. **Planning & Replanning Multipliers**: Formulating a research plan, checking task completion, and adjusting the DAG after every round requires multiple reasoning passes.
2. **Context Duplication Across Subagents**: Every spawned subagent requires its own system prompt, instruction context, and tool definitions.
3. **Exhaustive Retrieval Scraping**: A Deep Agent may retrieve and inspect 50 to 150 document chunks across 10 search iterations, compared to 3–5 chunks in standard RAG.
4. **Intermediate Map-Reduce Passes**: Summarizing large document chunks into research notes before writing the final draft.
5. **Auditor & Grounding Pass**: A dedicated verifier compares the generated draft against all cited passages to eliminate hallucinations.

### Hidden Infrastructure Costs
* **Concurrency & Rate Limits**: Running 4–8 subagents concurrently can instantly exhaust provider TPM (Tokens Per Minute) and RPM (Requests Per Minute) limits without robust queuing.
* **Persistent Storage & Memory**: Requires durable databases (Redis, Postgres, or disk stores) to checkpoint execution state and scratchpad files across long-running tasks.
* **Job Queue Infrastructure**: Cannot run synchronously inside an HTTP request; requires Celery, Temporal, or BullMQ with WebSocket status streaming.

---

## 4. Architectural Complexity Analysis

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      System Complexity Spectrum                         │
│                                                                         │
│  Naive RAG        Advanced RAG       Agentic RAG       Deep Agent RAG   │
│  [Low]            [Low-Medium]       [Medium]          [High-Extreme]   │
│  • Single script  • Multi-query      • Loop controller • State machine  │
│  • Vector store   • RRF merge        • Router JSON     • Subagents      │
│  • 1 LLM call     • Re-ranker        • Reflection      • Filesystem     │
│                                                        • Async queues   │
│                                                        • Checkpoints    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Dimensions of Complexity

#### 1. State Machine & Checkpointing
* **Standard RAG**: Stateless. A query goes in, chunks are retrieved, an answer is emitted.
* **Deep Agent**: Stateful. Requires a durable state graph (e.g., LangGraph or custom state machine) that records the current task index, scratchpad files, active subagent conversation threads, and memory checkpoints. If a container crashes on step 37 of 50, the agent must resume from its last checkpoint without restarting from scratch.

#### 2. Cascading Failure Modes
In a multi-agent system, errors compound exponentially:
* **Drift / Topic Hallucination**: A subagent misunderstands an ambiguous sub-question, retrieves irrelevant documents, and writes a flawed memo. The lead orchestrator assumes the memo is factual, polluting the final report.
* **Infinite Replanning Loop**: If the indexed knowledge base does not contain the answer, the agent may perpetually decompose the query into new subtasks, hitting maximum step limits or blowing through token budgets.
* **Subagent Output Formatting Failures**: A subagent returning non-JSON or broken formatting can crash the lead agent's parsing pipeline without defensive validation layers.

#### 3. Asynchronous Orchestration & Timeouts
* Standard REST API endpoints time out after 30 to 60 seconds (Cloudflare, Nginx, AWS ALB).
* Deep Agent execution takes minutes, forcing the architecture to switch to **asynchronous task execution**:
  - Request submission returns `job_id`.
  - Background worker picks up the job.
  - Client polls via SSE (Server-Sent Events) or WebSockets to track live progress, tool invocations, and subagent outputs.

---

## 5. Step-by-Step Workflow (Steps of Work)

Below is the complete end-to-end lifecycle of a Deep Agent RAG execution:

```mermaid
flowchart TD
    Start(["User Complex Research Query"]) --> Step1["Step 1: Query Analysis & Goal Decomposition"]
    Step1 --> Step2["Step 2: Structured Plan & Task DAG Generation"]
    
    subgraph ExecutionLoop ["Deep Execution & Exploration Loop"]
        Step2 --> Step3["Step 3: Select Next Milestone & Dispatch Subagents"]
        Step3 --> Step4A["Subagent A: Targeted Vector Retrieval"]
        Step3 --> Step4B["Subagent B: Technical Spec Scraping"]
        Step4A --> Step5["Step 4: Offload Extracted Findings to Workspace Filesystem"]
        Step4B --> Step5
        Step5 --> Step6["Step 5: Progress Evaluation & Dynamic Replanning"]
        Step6 -- "Gaps Found / Subtasks Remaining" --> Step3
    end
    
    Step6 -- "All Subtasks Satisfied" --> Step7["Step 6: Map-Reduce Synthesis from Workspace Notes"]
    Step7 --> Step8["Step 7: Critique & Grounding Auditor Verification"]
    Step8 -- "Unverified Claims Detected" --> Step3
    Step8 -- "Passed Grounding Audit" --> Step9["Step 8: Final Cited Report & Artifact Delivery"]
    Step9 --> End(["Completed Report"])
```

### Detailed Breakdown of Work Steps

#### Step 1: Query Analysis & Goal Decomposition
* The lead orchestrator analyzes the user's objective, identifying:
  - Primary goal.
  - Multi-part constraints (e.g., date ranges, comparisons, file filters).
  - Required output structure (tables, sections, technical memos).

#### Step 2: Structured Plan & Task DAG Generation
* Formulates an explicit, serialized task backlog:
  ```json
  {
    "tasks": [
      {"id": 1, "description": "Retrieve OAuth 2.0 grant types from RFC 6749", "status": "pending"},
      {"id": 2, "description": "Retrieve SAML 2.0 Web Browser SSO profile specifications", "status": "pending"},
      {"id": 3, "description": "Compare token lifecycle, security vulnerabilities, and latency", "depends_on": [1, 2], "status": "pending"}
    ]
  }
  ```

#### Step 3: Hierarchical Subagent Dispatch
* The orchestrator assigns tasks to isolated worker subagents.
* Worker subagents run in sandboxed contexts with specialized instructions and tools (`search_vector_store`, `read_chunk_range`, `filter_by_metadata`).
* They perform iterative searches, inspect candidate chunks, and filter out noise.

#### Step 4: Workspace Filesystem / Scratchpad State Offloading
* Subagents do **not** return raw chunks to the orchestrator.
* Instead, they write findings to the workspace scratchpad:
  - `workspace/notes/oauth2_summary.md`
  - `workspace/notes/saml2_summary.md`
  - `workspace/citations/evidence_index.json`
* Only a concise status memo (`"Completed task 1: extracted 4 grant types and 3 security constraints"`) is returned to the lead orchestrator.

#### Step 5: Progress Evaluation & Dynamic Replanning
* The orchestrator inspects the updated workspace and checks remaining tasks.
* **Gap Analysis**: If findings for task 2 were insufficient, the orchestrator updates the plan, adds subtask `2b`, and dispatches another search with revised query keywords.

#### Step 6: Map-Reduce Synthesis from Workspace Notes
* Once all DAG tasks are completed, the orchestrator reads the organized workspace notes.
* Uses a map-reduce pattern to aggregate notes across sections, structuring the comprehensive report with section headers, comparison tables, and code snippets.

#### Step 7: Critique & Grounding Auditor Verification
* A dedicated **Auditor Subagent** inspects the draft report against `workspace/citations/evidence_index.json`.
* For every factual claim, it checks if a corresponding source chunk exists.
* Unsubstantiated claims are flagged for deletion or trigger a quick targeted re-retrieval.

#### Step 8: Final Report & Artifact Delivery
* The final, verified document is formatted in Markdown with standardized citations (`[RFC6749#chunk_12]`).
* Artifacts (summary tables, JSON metadata, task execution logs) are emitted alongside the response.

---

## 6. The RAG Complexity & Cost Spectrum: Detailed Contextual Guide

The RAG design space is not binary. It spans a continuous trade-off curve balancing **speed and cost** against **depth and autonomy**:

```
Fast / Cheap                                                  Deep / Heavy
─────────────────────────────────────────────────────────────────────────►
[NaiveRAG]     ──►  [AdvancedRAG]  ──►  [AgenticRAG]  ──►  [DeepResearchRAG]
Single-shot         Multi-query         Dynamic routing    Multi-subagent
Vector search       Rank fusion         Multi-round        Workspace notes
Latency: ~1s        Latency: ~2s        Latency: ~3-5s     Latency: 30s-5m
Cost: $             Cost: $$            Cost: $$$          Cost: $$$$$
Calls: 1            Calls: 2-3          Calls: 3-6         Calls: 20-100+
Tokens: ~1.5k       Tokens: ~4k         Tokens: ~8k        Tokens: 100k-500k+
Mode: Sync HTTP     Mode: Sync HTTP     Mode: Sync HTTP    Mode: Async/Queue
```

---

### Tier 1: `NaiveRAG` — "The High-Speed Factual Retriever"

#### 1. Core Philosophy & Architecture
The foundational **Retrieve-then-Read** pipeline. A user query is embedded directly, top-$k$ nearest neighbor chunks are retrieved from the vector store via cosine/dot-product similarity, and concatenated into a prompt template alongside the user question for a single completion pass.

```
User Query ──► [ Embed Query ] ──► [ Vector Store Top-K ] ──► [ Prompt + Chunks ] ──► [ LLM ] ──► Answer
```

#### 2. Key Metrics & Profile
* **Latency**: **0.5s – 1.2s** (p50: 0.7s, p95: 1.5s).
* **LLM Invocations**: Exactly **1 call**.
* **Token Footprint**: ~1,000 – 1,800 tokens total (depends on chunk size & top-$k$).
* **Cost Profile**:
  * GPT-4o: ~$0.007 per query ($7.00 per 1,000 queries).
  * GPT-4o-mini: ~$0.0003 per query ($0.30 per 1,000 queries).
* **Execution Mode**: Synchronous HTTP request-response. High-throughput (50–200 req/sec with pooled vector DB).

#### 3. Internal Mechanics
* **Query Handling**: Raw string pass-through. No rewriting, expansion, or spell correction.
* **Retrieval Strategy**: Single vector search (ANN index like HNSW or Flat).
* **Context Handling**: Direct concatenation of retrieved chunks into the prompt.
* **Reasoning**: Single forward generation pass. No reflection, iteration, or validation.

#### 4. Ideal Use Cases
* High-volume customer support FAQ lookups (*"What are your business hours?", "How do I initiate a return?"*).
* Static software documentation search with explicit keyword alignment.
* Real-time autocomplete or inline coding suggestion augmentations.

#### 5. Failure Modes & Limitations
* **Semantic Mismatch**: If the user's phrasing differs significantly from the document's vocabulary, retrieval fails completely.
* **Single Point of Failure**: If the top-$k$ search returns irrelevant or incomplete chunks, the LLM hallucinates or admits ignorance.
* **Context Clutter**: Cannot handle queries that require synthesizing facts scattered across 10+ pages.

#### 6. PolyRAG Usage
```python
from polyrag import PolyRAG

rag = PolyRAG()
rag.ingest_text("PolyRAG is a modular, multi-paradigm RAG framework.")
response = rag.query("What is PolyRAG?")
print(response.answer)
```

---

### Tier 2: `AdvancedRAG` — "The Precision Multi-Query Synthesizer"

#### 1. Core Philosophy & Architecture
Addresses the primary weakness of Naive RAG (poor query formulation and vocabulary mismatch) using **pre-retrieval query expansion** and **post-retrieval rank fusion**. The LLM first generates alternative search perspectives or hypothetical document embeddings (HyDE). All queries run in parallel, and results are merged using **Reciprocal Rank Fusion (RRF)** and relevance thresholds before synthesis.

```
User Query ──► [ LLM Query Expansion ] ──► [ Query 1, Query 2, Query 3 ]
                                                    │
                                                    ▼ (Parallel Searches)
                                            [ Candidate Pools ]
                                                    │
                                                    ▼
                                            [ RRF Re-ranking ]
                                                    │
                                                    ▼
                                            [ Filtered Chunks ] ──► [ LLM Synthesis ] ──► Answer
```

#### 2. Key Metrics & Profile
* **Latency**: **1.5s – 2.5s** (p50: 1.8s, p95: 3.0s).
* **LLM Invocations**: **2 calls** (1 for query generation, 1 for final synthesis).
* **Token Footprint**: ~3,000 – 5,000 tokens total.
* **Cost Profile**:
  * GPT-4o: ~$0.015 per query ($15.00 per 1,000 queries).
  * GPT-4o-mini: ~$0.0008 per query ($0.80 per 1,000 queries).
* **Execution Mode**: Synchronous HTTP. Suitable for user-facing search bars with slightly higher latency budgets.

#### 3. Internal Mechanics
* **Query Handling**: Generates $N$ distinct queries exploring synonyms, technical terms, and alternative facets.
* **Retrieval Strategy**: Multi-query parallel retrieval. Merges candidate lists using RRF score: $RRF(d) = \sum \frac{1}{k + rank_i(d)}$.
* **Context Handling**: Deduplicates overlapping chunks, ranks by fused score, and filters out chunks below `min_relevance_score`.
* **Reasoning**: Two-stage reasoning (exploration + synthesis).

#### 4. Ideal Use Cases
* Technical documentation search where users ask vague or conversational questions (*"Why is my connection dropping on port 443?"*).
* Knowledge bases with domain-specific terminology where users don't know the exact internal acronyms.
* Enterprise search portals requiring higher recall without human intervention.

#### 5. Failure Modes & Limitations
* **Cannot do Multi-Hop Reasoning**: If answering query B requires first discovering an intermediate fact A from document 1, parallel query expansion fails because it expands before reading.
* **Always Retrieves**: Even if the user asks a conversational question like *"Hello, how are you?"*, it will wastefully expand and search the vector database.

#### 6. PolyRAG Usage
```python
from polyrag import PolyRAG

rag = PolyRAG()
advanced_pipeline = rag.create_advanced_rag(
    num_expanded_queries=3,
    top_k=5,
    min_relevance_score=0.1,
)
response = advanced_pipeline.query("How to handle SSL handshake errors?")
```

---

### Tier 3: `AgenticRAG` & `ReActAgent` — "The Dynamic Adaptive Reasoner"

#### 1. Core Philosophy & Architecture
Introduces an **autonomous control loop with decision-making capabilities**. Instead of blindly searching, the agent first **decides whether retrieval is even needed** (intent classification & routing). If needed, it rewrites the query, searches, inspects the retrieved evidence, and evaluates whether the information is sufficient. If insufficient, it initiates a **second round of retrieval** with refined queries before performing **fused self-reflection** and generating the grounded response.

Alternatively, `ReActAgent` provides a formal **Thought-Action-Observation loop**, allowing dynamic tool calls (`search`, `list_docs`, `final_answer`) until the agent is confident.

```
User Query ──► [ Intent Router & Retrieval Planner ]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
  [ No Retrieval Needed ]   [ Retrieval Needed ]
         │                       │
         ▼ (Direct Answer)       ▼
       Answer              [ Round 1 Search ] ──► [ Evaluate Evidence ]
                                                        │
                                    ┌───────────────────┴───────────────────┐
                                    ▼                                       ▼
                             [ Sufficient ]                          [ Insufficient ]
                                    │                                       │
                                    ▼                                       ▼
                       [ Fused Reflection & Synthesis ]              [ Round 2 Search (Refined) ]
                                    │                                       │
                                    ▼                                       └──────► [ Fused Synthesis ]
                                  Answer
```

#### 2. Key Metrics & Profile
* **Latency**: **2.5s – 5.5s** (p50: 3.5s, p95: 7.0s).
* **LLM Invocations**: **2 to 5 calls** (Router + Round 1 Eval + Optional Round 2 + Final Synthesis).
* **Token Footprint**: ~5,000 – 12,000 tokens total.
* **Cost Profile**:
  * GPT-4o: ~$0.035 per query ($35.00 per 1,000 queries).
  * GPT-4o-mini: ~$0.002 per query ($2.00 per 1,000 queries).
* **Execution Mode**: Synchronous HTTP or streaming response. Ideal for interactive Copilot interfaces.

#### 3. Internal Mechanics
* **Query Handling**: Semantic query rewriting based on technical intent analysis.
* **Retrieval Strategy**: Multi-round adaptive retrieval with cumulative deduplication across rounds.
* **Context Handling**: Filters and deduplicates chunks from multiple rounds; tracks provenance (`source#chunk_id`).
* **Reasoning**: Cyclic loop with self-correction, gap detection, and grounded citation verification.

#### 4. Ideal Use Cases
* Interactive AI Copilots where users mix casual conversation (*"Thanks!", "Help me with..."*) with complex domain questions.
* Multi-hop technical queries (*"Find the database timeout setting in config.yaml, then explain how that affects the retry policy in client.py"*).
* Scenarios requiring high precision and citation transparency (legal, compliance, technical specifications).

#### 5. Failure Modes & Limitations
* **Step Limit Constraints**: Bound by a maximum step or round limit (e.g., `max_rounds=2` or `max_steps=5`). Cannot tackle open-ended research spanning 50+ documents.
* **Context Clutter in Long Conversations**: In `ReActAgent`, each thought, tool call, and observation stays in the prompt history, degrading attention on complex chains.

#### 6. PolyRAG Usage
```python
from polyrag import PolyRAG

rag = PolyRAG()

# Agentic RAG with multi-round retrieval and reflection
agentic = rag.create_agentic_rag(max_rounds=2, top_k=3, verbose=True)
response = agentic.query("What are the timeout settings and how do they impact failovers?")

# Or ReAct Agent with dynamic tool invocation
react = rag.create_react_agent(max_steps=4, verbose=True)
response = react.query("Compare API rate limits across v1 and v2 specs.")
```

---

### Tier 4: `DeepResearchRAG` — "The Autonomous Long-Horizon Investigator"

#### 1. Core Philosophy & Architecture
Designed for **exhaustive, open-ended research over massive document corpora**. Instead of a single LLM trying to hold everything in its context window, an **Orchestrator** decomposes the objective into an explicit **Task DAG (Directed Acyclic Graph)**. It spawns independent, scoped **Worker Subagents** to explore specific sub-topics in parallel.

Raw chunks and intermediate findings are **offloaded to a persistent workspace filesystem / scratchpad**, preventing context window exhaustion. Once all tasks are satisfied, notes are condensed via **map-reduce**, verified by an **Auditor Subagent** for groundedness, and compiled into a long-form cited report.

```
                                  [ User Complex Research Objective ]
                                                   │
                                                   ▼
                                       [ Lead Orchestrator ]
                                       [ (Task Planning DAG) ]
                                                   │
                   ┌───────────────────────────────┼───────────────────────────────┐
                   ▼                               ▼                               ▼
          [ Subagent 1: Topic A ]         [ Subagent 2: Topic B ]         [ Subagent 3: Topic C ]
          (Isolated Vector Search)        (Isolated Vector Search)        (Isolated Vector Search)
                   │                               │                               │
                   └───────────────────────────────┼───────────────────────────────┘
                                                   ▼
                                  [ Workspace Filesystem Scratchpad ]
                                  • notes/topic_a.md
                                  • notes/topic_b.md
                                  • citations/evidence_registry.json
                                                   │
                                                   ▼
                                  [ Map-Reduce Draft Synthesis ]
                                                   │
                                                   ▼
                                  [ Critique & Grounding Auditor ]
                                  (Checks every claim against chunks)
                                                   │
                                                   ▼
                                  [ Final Exhaustive Research Report ]
```

#### 2. Key Metrics & Profile
* **Latency**: **30 seconds to 5+ minutes** (p50: 90s, p95: 240s).
* **LLM Invocations**: **20 to 100+ calls**.
* **Token Footprint**: **100,000 to 500,000+ tokens** total across orchestrator, subagents, and auditor.
* **Cost Profile**:
  * GPT-4o: ~$0.50 – $2.50+ per query ($500 – $2,500 per 1,000 queries).
  * GPT-4o-mini: ~$0.03 – $0.15 per query ($30 – $150 per 1,000 queries).
* **Execution Mode**: **Asynchronous background task** with WebSocket status streaming, job checkpoints, and durable persistence.

#### 3. Internal Mechanics
* **Query Handling**: High-level goal decomposition into prioritized task dependencies.
* **Retrieval Strategy**: Multi-subagent parallel retrieval; may inspect 50–200 chunks across multiple iterations.
* **Context Handling**: Filesystem-backed state isolation. LLM context stays below 8k tokens while workspace holds hundreds of thousands of words.
* **Reasoning**: Hierarchical multi-actor state machine with dynamic replanning, gap analysis, and auditor verification.

#### 4. Ideal Use Cases
* Comprehensive competitive intelligence or market analysis across hundreds of uploaded PDFs.
* Multi-document regulatory compliance audits (e.g., verifying 50 internal policy files against ISO-27001).
* Generating 10-to-20 page executive briefing dossiers with verified citations.

#### 5. Failure Modes & Limitations
* **Extreme Latency & Cost**: Completely unusable for interactive user-facing chatbots.
* **Over-Exploration / Stalls**: Can get trapped in recursive subtask creation if the knowledge base lacks necessary data.
* **Infrastructure Complexity**: Requires background workers (Celery/Temporal), durable databases (Redis/Postgres), and robust rate-limit management.

#### 6. PolyRAG Usage (Conceptual Roadmap)
```python
from polyrag import PolyRAG

rag = PolyRAG()

# Optional specialized deep research pipeline
deep_research = rag.create_deep_research_rag(
    max_subagents=4,
    max_planning_depth=3,
    workspace_dir="./workspace_scratchpad",
    verbose=True,
)

# Returns a task ID or runs asynchronously
report = deep_research.research(
    "Perform a comprehensive comparative analysis of OAuth2 vs SAML2 security architectures in our indexed RFCs."
)
print(report.markdown_report)
print(report.task_dag_history)
```

---

## 7. Deep Comparative Matrix Across All 4 Tiers

| Dimension | `NaiveRAG` | `AdvancedRAG` | `AgenticRAG` | `DeepResearchRAG` |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Goal** | Fast factual answer | High recall & precision | Multi-hop reasoning & routing | Exhaustive multi-doc synthesis |
| **P95 Latency** | **1.5 seconds** | **3.0 seconds** | **7.0 seconds** | **4 – 5 minutes** |
| **Cost per 1k (GPT-4o)**| **$7.00** | **$15.00** | **$35.00** | **$500 – $2,500** |
| **Cost per 1k (4o-mini)**| **$0.30** | **$0.80** | **$2.00** | **$30 – $150** |
| **LLM Calls** | 1 | 2 | 2 – 5 | 20 – 100+ |
| **Token Budget** | < 2,000 | ~4,000 | ~8,000 | 100,000 – 500,000+ |
| **Retrieval Strategy** | Single-shot vector top-$k$ | Multi-query expansion + RRF | Multi-round adaptive retrieval | Hierarchical subagent searches |
| **Chunks Inspected** | 1 – 5 chunks | 5 – 15 chunks | 5 – 25 chunks | 50 – 200+ chunks |
| **Context Window Pressure**| Negligible | Low | Moderate | High (mitigated by filesystem) |
| **Execution Architecture** | Stateless function | Stateless pipeline | Stateful single loop | Stateful multi-agent graph |
| **Transport Model** | Sync HTTP Request | Sync HTTP Request | Sync HTTP / Streaming | Async Job Queue + WebSocket |
| **Error Recovery** | None (fails silently) | None | Self-correction / Round 2 | Replanning, gap analysis, auditor |
| **Best Fit** | FAQs, doc search bars | Technical search with jargon | Copilots, multi-hop Q&A | Multi-document audit dossiers |
| **Worst Fit** | Multi-hop or complex queries | Real-time < 500ms SLA | High-throughput batch APIs | Synchronous user chat interfaces |

---

## 8. Summary & Architect's Rule of Thumb

> [!TIP]
> **The 10-Second Selection Rule**:
> 1. If the user is waiting on a screen and needs an answer in 1 second ➡️ **`NaiveRAG`**.
> 2. If search keywords often miss internal document phrasing ➡️ **`AdvancedRAG`**.
> 3. If the user asks multi-hop questions, mixes chit-chat, or needs self-correction ➡️ **`AgenticRAG`**.
> 4. If the user asks you to *"investigate, read dozens of files, and write an exhaustive report"* ➡️ **`DeepResearchRAG`**.

