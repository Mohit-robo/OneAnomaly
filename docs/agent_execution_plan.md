# AI Agent Execution Plan — anomaly_app
**Model:** Qwen3.5-2B (text-only, hybrid thinking)
**Scope:** Full agent layer over existing Flask + Node.js + FAISS pipeline

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Browser UI                          │
│         (Existing HTML/JS + new Agent Chat Panel)       │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────────┐
│                Node.js Proxy (existing)                 │
│         + New: /agent route → Agent API                 │
└──────────┬─────────────────────────┬────────────────────┘
           │                         │
┌──────────▼──────────┐   ┌──────────▼──────────────────┐
│  Flask Backend      │   │   Agent Service (Python)     │
│  (existing)         │   │   LangGraph + Qwen3.5-2B    │
│  /preprocess        │   │   + ChromaDB RAG             │
│  /spatial           │   │   + Tool Registry            │
│  /memory_bank       │   │   + Session Manager          │
│  /validate          │   └──────────┬───────────────────┘
│  /inference         │              │
└─────────────────────┘   ┌──────────▼───────────────────┐
                           │   Ollama Server (local)      │
                           │   Qwen3.5-2B                 │
                           │   nomic-embed-text           │
                           └──────────────────────────────┘
```

**Key architectural decisions:**
- Agent service is a **separate Python process** — does not modify Flask backend
- Agent calls Flask endpoints as tools — identical to how the UI does
- LangGraph manages phase state — the model never decides phase transitions
- Qwen3.5-2B hybrid thinking: CoT ON for complex decisions, OFF for simple tool param extraction
- No VLM checkpoints — structural validation replaces visual verification

---

## 2. Component Inventory

### 2.1 New Components to Build

| Component | Type | Purpose |
|-----------|------|---------|
| Agent Service | Python FastAPI process | Hosts LangGraph, tool registry, RAG |
| ChromaDB Instance | Local vector DB | Stores embedded knowledge chunks |
| Session Manager | Python module | Reads/writes session JSON, validates schema |
| Tool Registry | Python module | Wraps Flask endpoints as typed tools |
| RAG Retriever | Python module | Handles query → ChromaDB → context |
| Ingestion Script | One-time Python script | Populates ChromaDB from docs |
| Agent Chat Panel | HTML/JS addition to existing UI | User-facing agent interface |
| Pydantic Schema | Python module | Validates all session config changes |

### 2.2 Existing Components (unchanged)

| Component | Role in Agent Context |
|-----------|----------------------|
| Flask `/preprocess` | Called by `run_preprocessing` tool |
| Flask `/spatial` | Called by `set_spatial_regions` tool |
| Flask `/memory_bank` | Called by `create_memory_bank` tool |
| Flask `/validate` | Called by `run_validation` tool |
| Flask `/inference` | Called by `run_final_inference` tool |
| Node.js proxy | Gets new `/agent` route added |
| FAISS index | Unchanged — agent just triggers its creation |
| Session history JSON | Promoted to primary agent state carrier |

---

## 3. RAG Knowledge Base — Setup Plan

### 3.1 Documents to Ingest

Collect and organize all existing documentation into these categories before ingestion:

| Doc Category | Content | Chunk Strategy |
|-------------|---------|----------------|
| Setup docs | Installation, model paths, dependencies | By section header |
| Phase docs | Phase 1–5 descriptions, parameters | By phase + sub-step |
| Dev plan | This document + handwritten notes transcription | By phase |
| Lessons learned | Known issues, hardware migration notes, false positive patterns | By topic |
| Session history JSONs | Past successful session configs | Whole JSON per chunk |
| API reference | Flask endpoint inputs/outputs | By endpoint |
| Parameter glossary | τ, k, σ, n_max, BS definitions | One chunk per param |

### 3.2 Chunking Rules

- **Unit:** One concept = one chunk. Never split a parameter explanation across chunks.
- **Overlap:** 50-token overlap between adjacent chunks from same document
- **Metadata per chunk** (mandatory):
  ```
  phase: [1|2|3|4|5|"cross"]
  topic: "spatial_tiling" | "memory_bank" | "threshold_tuning" | etc.
  doc_type: "setup" | "plan" | "lesson" | "session" | "api"
  criticality: "high" | "medium" | "reference"
  ```
- Metadata enables **filtered retrieval** — agent queries only Phase 3 docs when in Phase 3

### 3.3 Embedding Model

- **Model:** `nomic-embed-text` via Ollama
- **Runs:** Locally on IPC during both ingestion and runtime
- **Embedding dimension:** 768
- **Why nomic:** Same model for ingestion and retrieval — embedding space consistency is critical. Switching models later requires full re-ingestion.

### 3.4 Vector DB

- **Tool:** ChromaDB (local persistent mode)
- **Collections:** One collection per doc_type — `setup_docs`, `phase_docs`, `session_history`, `api_reference`
- **Storage:** `/data/chromadb/` on IPC disk
- **Persistence:** Survives process restarts — no re-ingestion needed unless docs change

### 3.5 One-Time Ingestion Pipeline (Tasks)

```
Task 1: Collect all markdown/txt/JSON docs into /data/raw_docs/
Task 2: Write chunker script — splits by headers, applies metadata tagging
Task 3: Run chunker → generates /data/chunks/ as JSON objects
Task 4: Review chunk quality manually — ensure no concept is split
Task 5: Run embedding via nomic-embed-text (Ollama batch call)
Task 6: Insert into ChromaDB with metadata
Task 7: Verify retrieval — run 10 test queries, check top-3 returned chunks
Task 8: Adjust chunk boundaries where retrieval misses
```

### 3.6 Runtime Retrieval Strategy

Two retrieval modes used depending on context:

**Mode A — Filtered retrieval (most calls):**
Agent specifies `phase=3` + `topic="memory_bank"` → ChromaDB filters metadata first, then ranks by similarity → returns top 3 chunks

**Mode B — Open retrieval (for ambiguous user queries):**
No metadata filter → pure similarity → returns top 5 chunks across all collections

Context injection format into Qwen3.5-2B prompt:
```
[CONTEXT FROM KNOWLEDGE BASE]
Source: phase_docs / Phase 3 / memory_bank_creation
Content: <chunk text>
---
Source: api_reference / Flask / /memory_bank endpoint
Content: <chunk text>
[END CONTEXT]
```

---

## 4. Qwen3.5-2B Integration Plan

### 4.1 Serving

- **Runtime:** Ollama on IPC (RTX 5060)
- **Model tag:** `qwen3.5:2b` (or GGUF Q4_K_M quantized for VRAM efficiency)
- **Ollama endpoint:** `http://localhost:11434/api/chat`
- **Context window:** 32K tokens — sufficient for session JSON + RAG chunks + tool history
- **Thinking mode toggle:**
  - `thinking: true` → complex decisions (is this config valid? which tool to call next?)
  - `thinking: false` → simple extraction (parse user input for tool params)

### 4.2 System Prompt Structure

The system prompt injected at every call has four sections:

```
Section 1: Role definition
  "You are an orchestration agent for an industrial anomaly detection pipeline.
   You do not generate images or interpret visuals. You manage a structured
   5-phase workflow by calling tools and reading session state."

Section 2: Current session state (injected dynamically)
  Full session JSON for current session

Section 3: Available tools at this graph node (constrained list)
  Only tools valid for current phase — not full registry

Section 4: Retrieved RAG context
  2–4 chunks relevant to current task
```

### 4.3 Thinking Mode Usage Map

| Agent Task | Thinking Mode | Reason |
|-----------|--------------|--------|
| Parsing user intent into tool params | OFF | Fast, deterministic |
| Deciding if session config is valid | ON | Needs multi-step checking |
| Selecting pre-proc mode recommendation | ON | Depends on multiple factors |
| Synthesizing RAG answer for user question | OFF | Retrieval + summarize |
| Detecting BS mismatch across phases | ON | Cross-phase logic |
| Generating results summary for Phase 5 | OFF | Straightforward |
| Deciding if Phase 4 scores are stable | ON | Threshold analysis |

---

## 5. LangGraph State Machine — Design

### 5.1 Graph Nodes (one per phase + utilities)

```
Nodes:
  START
  → node_intake          (parse user message, identify intent)
  → node_phase1          (pre-processing setup)
  → node_phase2          (spatial region configuration)
  → node_phase3          (memory bank creation)
  → node_phase4          (validation + threshold tuning)
  → node_phase5          (final inference + export)
  → node_rag_query       (answer knowledge questions, can be called from any node)
  → node_session_load    (load existing session, jump to correct phase)
  → node_error_handler   (catch tool failures, propose recovery)
  END
```

### 5.2 State Object (passed between all nodes)

```
AgentState {
  current_phase: int              # 1–5
  session_config: dict            # full session JSON (see Section 6)
  conversation_history: list      # all user/agent turns
  last_tool_call: str             # name of last tool called
  last_tool_result: dict          # output of last tool
  rag_context: list               # chunks retrieved for current turn
  pending_confirmation: bool      # waiting for user yes/no
  error_count: int                # consecutive tool failures
  available_tools: list           # constrained to current phase
}
```

### 5.3 Edge Conditions (deterministic — not decided by model)

```
START → node_intake
  always

node_intake → node_session_load
  if: user message contains session name or "load session"

node_intake → node_phase1
  if: no active session OR session.current_phase == 1

node_intake → node_phase{N}
  if: session.current_phase == N (resume from where left off)

node_intake → node_rag_query
  if: user message is a question (no tool action needed)

node_phase1 → node_phase2
  if: session_config.phase1.status == "confirmed"
  guard: Pydantic validation passes for phase1 config

node_phase2 → node_phase3
  if: session_config.phase2.status == "confirmed"
  guard: spatial_bs set, model_path set, preview_confirmed == true

node_phase3 → node_phase4
  if: session_config.phase3.status == "confirmed"
  guard: faiss_index_path exists on disk, bank_size > 0

node_phase4 → node_phase5
  if: user explicitly confirms params (pending_confirmation resolved)
  guard: all threshold params (τ, k, σ, n_max) are set

node_any → node_error_handler
  if: error_count >= 2

node_error_handler → node_any
  after: user acknowledges error and provides correction
```

### 5.4 Tool Constraint Per Node

Each node receives only the tools valid at that phase:

```
node_phase1 tools:    [run_preprocessing, query_knowledge_base, save_session]
node_phase2 tools:    [set_spatial_regions, export_dinov3_trt, query_knowledge_base, save_session]
node_phase3 tools:    [create_memory_bank, query_knowledge_base, save_session]
node_phase4 tools:    [run_validation, update_threshold, query_knowledge_base, save_session]
node_phase5 tools:    [run_final_inference, download_results, save_session]
node_rag_query tools: [query_knowledge_base]
```

This constraint is the primary guard against model hallucinating wrong tool calls.

---

## 6. Session Config JSON Schema

This JSON is the single source of truth. Agent reads it at every node. Pydantic validates it before any phase transition.

```
session_config {

  meta {
    session_name: str
    created_at: datetime
    last_updated: datetime
    current_phase: int          # 1–5
    part_type: str              # e.g. "bottle_cap", "label_zone"
  }

  phase1 {
    status: "pending"|"configured"|"confirmed"
    mode: "threshold"|"template_matching"|"unet"
    params {
      threshold_min: int
      threshold_max: int
      channel: str              # if mode=threshold
      template_path: str        # if mode=template_matching
      unet_model_path: str      # if mode=unet (TRT engine)
      unet_onnx_path: str
    }
    preview_confirmed: bool
  }

  phase2 {
    status: "pending"|"configured"|"confirmed"
    mode: "grid"|"draw"|"none"
    grid_config {
      rows: int
      cols: int
    }
    drawn_regions: list[{label: str, coords: list[int]}]
    num_regions: int            # BS for DINOv3 export
    image_size: int             # 224
    dinov2_onnx_path: str
    dinov2_trt_path: str
    preview_confirmed: bool
  }

  phase3 {
    status: "pending"|"configured"|"confirmed"
    good_samples_path: str
    num_shots: int
    bank_name: str
    faiss_index_path: str
    bank_size: int              # BS × num_shots × feature_dim
    feature_dim: int
    normalization: "l2"|"none"
  }

  phase4 {
    status: "pending"|"tuning"|"confirmed"
    test_images_path: str
    raw_scores_cache_path: str  # cached scores for instant threshold updates
    params {
      tau: float                # decision threshold
      k: int                   # FAISS neighbors
      sigma: float             # heatmap smoothing
      n_max: int               # max anomalies per image
    }
    validation_summary {
      total_images: int
      ok_count: int
      not_ok_count: int
      score_distribution: list[float]
    }
  }

  phase5 {
    status: "pending"|"running"|"complete"
    test_set_path: str
    results_zip_path: str
    csv_report_path: str
    gt_mask_path: str           # optional
    iou_scores: list[float]     # populated if gt_mask provided
    session_snapshot_path: str
  }

}
```

**Pydantic guards at each phase transition:**
- Phase 1 → 2: `phase1.status == confirmed`, unet_model_path exists if mode=unet
- Phase 2 → 3: `phase2.num_regions > 0`, TRT path exists, `phase2.num_regions == BS used in TRT export`
- Phase 3 → 4: FAISS index file exists, `bank_size > 0`
- Phase 4 → 5: All 4 threshold params set, `pending_confirmation == false`

---

## 7. Tool Registry — Detailed Spec

### Tool 1: `run_preprocessing`
```
Maps to: POST /preprocess
Inputs: mode (str), params (dict), sample_image_path (str)
Outputs: mask_path (str), preview_image_path (str), success (bool)
Agent use: Called in Phase 1 after user selects mode
Session update: phase1.params, phase1.status = "configured"
Error handling: If mask_path missing → node_error_handler
```

### Tool 2: `confirm_preprocessing_preview`
```
Maps to: No backend call — agent presents preview_image_path to user
Inputs: preview_image_path (str)
Outputs: user_confirmed (bool)
Agent use: After run_preprocessing, agent asks user to verify mask quality in text
Session update: phase1.preview_confirmed, phase1.status = "confirmed"
Note: Since no VLM, agent describes the mask qualitatively from metadata
       (e.g., "mask covers X pixels, Y% of image area")
```

### Tool 3: `set_spatial_regions`
```
Maps to: POST /spatial
Inputs: mode (str), grid_config (dict) OR drawn_regions (list), image_size (int)
Outputs: region_coords (list), num_regions (int), preview_path (str)
Agent use: Phase 2 — after user specifies mode via chat
Session update: phase2.mode, phase2.num_regions, phase2.grid_config or drawn_regions
```

### Tool 4: `export_dinov3_trt`
```
Maps to: POST /export_model
Inputs: batch_size (int), image_size (int), onnx_output_path (str)
Outputs: trt_path (str), export_time_sec (float), success (bool)
Agent use: Phase 2 — triggered after spatial regions confirmed
Session update: phase2.dinov2_onnx_path, phase2.dinov2_trt_path
Guard: batch_size must == phase2.num_regions (Pydantic validates before call)
```

### Tool 5: `create_memory_bank`
```
Maps to: POST /memory_bank/create
Inputs: good_samples_path (str), num_shots (int), bank_name (str),
        session_config (dict) — includes phase1 + phase2 params
Outputs: faiss_index_path (str), bank_size (int), feature_dim (int)
Agent use: Phase 3 core tool
Session update: phase3.faiss_index_path, phase3.bank_size, phase3.status = "confirmed"
Note: Flask internally applies phase1 pre-proc + phase2 tiling before extraction
```

### Tool 6: `run_validation`
```
Maps to: POST /validate
Inputs: test_images_path (str), session_config (dict), tau (float),
        k (int), sigma (float), n_max (int)
Outputs: raw_scores_cache_path (str), heatmap_paths (list),
         ok_count (int), not_ok_count (int), score_distribution (list)
Agent use: Phase 4 — first validation run
Session update: phase4.raw_scores_cache_path, phase4.validation_summary
```

### Tool 7: `update_threshold`
```
Maps to: POST /validate/update_threshold
Inputs: raw_scores_cache_path (str), tau (float), k (int),
        sigma (float), n_max (int)
Outputs: updated heatmap_paths, ok_count, not_ok_count (recomputed from cache)
Agent use: Phase 4 — every time user adjusts a slider/param via chat
Session update: phase4.params
Note: No re-inference — uses cached raw scores. Must be fast (<1s).
```

### Tool 8: `run_final_inference`
```
Maps to: POST /inference
Inputs: test_set_path (str), session_config (dict), gt_mask_path (str, optional)
Outputs: results_zip_path (str), csv_report_path (str), iou_scores (list, optional)
Agent use: Phase 5 after user confirmation
Session update: phase5 (all fields)
```

### Tool 9: `query_knowledge_base`
```
Maps to: ChromaDB query (internal to agent service — no Flask call)
Inputs: query (str), phase_filter (int, optional), topic_filter (str, optional)
Outputs: list of {content: str, source: str, metadata: dict}
Agent use: Any phase — when user asks a question or agent needs doc context
Session update: none
Thinking mode: OFF — retrieval + summarize is straightforward
```

### Tool 10: `load_session`
```
Maps to: Read session JSON from disk
Inputs: session_name (str)
Outputs: full session_config dict
Agent use: node_session_load — when user wants to resume
Session update: replaces entire AgentState.session_config
Post-action: LangGraph routes to node_phase{session.current_phase}
```

### Tool 11: `save_session`
```
Maps to: Write session JSON to disk
Inputs: session_config (dict), session_name (str)
Outputs: saved_path (str)
Agent use: Called after every phase confirmation + at end of Phase 5
Session update: meta.last_updated
```

### Tool 12: `get_validation_stats`
```
Maps to: Read from phase4.validation_summary in session JSON (no Flask call)
Inputs: none (reads from current AgentState)
Outputs: formatted stats string (score distribution, ok/not-ok ratio)
Agent use: Phase 4 — when agent recommends threshold adjustments
Thinking mode: ON — agent reasons about score distribution to recommend τ
```

---

## 8. Agent Chat Panel — UI Integration

### 8.1 What to Add to Existing UI

The existing UI stays untouched. Add a **collapsible side panel** (right side) with:

```
Components:
  - Chat message window (scrollable)
  - Text input + Send button
  - Session selector dropdown (load existing sessions)
  - Phase indicator bar (shows current phase: 1–5)
  - "Thinking..." spinner (shown when Qwen3.5-2B thinking=true)
  - Tool call log (collapsible — shows what tool was called + result summary)
  - Session JSON viewer (collapsible — shows current state for transparency)
```

### 8.2 New Node.js Route

Add `/agent` POST route to existing Node.js proxy:
```
Input:  { user_message: str, session_name: str, conversation_history: list }
Output: { agent_reply: str, tool_calls: list, session_state: dict, phase: int }
Proxies to: Agent Service FastAPI at localhost:8001/agent
```

### 8.3 Agent Service API (FastAPI)

Single endpoint exposed by the agent service:
```
POST /agent
  Body: { user_message, session_name, conversation_history }
  Returns: { reply, tool_calls_made, updated_session, current_phase, error }
```

Internally this triggers the full LangGraph run and returns when the graph reaches an END or user-input-needed state.

---

## 9. Validation Without VLM — Replacement Strategy

Since Qwen3.5-2B cannot see images, visual checkpoints are replaced with **quantitative text-based checks**:

### Phase 1 (Pre-processing quality)
- Tool returns: mask pixel coverage %, connected component count, foreground/background ratio
- Agent receives these numbers and reasons: "40% coverage with 3 connected components is consistent with expected bottle cap region — proceeding" or flags anomaly
- User sees the text verdict + can view mask image in the main UI panel themselves

### Phase 2 (Spatial tiling verification)
- Tool returns: per-region pixel count, region overlap %, aspect ratio per region
- Agent validates: no region is <5% of image area, no overlap >10%, regions sum to ~95% of image
- Pydantic schema enforces hard limits; agent provides soft reasoning above that

### Phase 4 (Anomaly score stability)
- Agent uses `get_validation_stats` tool to read score distribution
- With thinking=ON: reasons about whether score gap between OK and Not-OK is wide enough
- Recommends τ adjustments: "Your Not-OK scores cluster between 0.7–0.9 and OK scores between 0.1–0.3. Setting τ=0.55 gives maximum separation margin."
- User views heatmaps in main UI and confirms via chat: "looks good" → agent updates session

---

## 10. Development Sequence

### Sprint 1 — Foundation (Week 1–2)
```
Task 1.1: Design and finalize session_config JSON schema (Pydantic models)
Task 1.2: Set up Ollama on IPC — install Qwen3.5-2B + nomic-embed-text
Task 1.3: Set up ChromaDB local persistent instance
Task 1.4: Write document chunker + metadata tagger
Task 1.5: Run one-time ingestion of all docs into ChromaDB
Task 1.6: Test RAG retrieval manually — 10 query coverage test
```

### Sprint 2 — Tool Registry (Week 2–3)
```
Task 2.1: Write Flask endpoint wrappers as typed Python tool functions
Task 2.2: Add missing Flask endpoints if any (update_threshold, get_stats)
Task 2.3: Test each tool independently — mock agent calls, verify session updates
Task 2.4: Add Pydantic validation layer on top of each tool
Task 2.5: Write tool response normalizer — all tools return uniform dict structure
```

### Sprint 3 — Agent Core (Week 3–4)
```
Task 3.1: Build minimal ReAct loop first (no LangGraph) — single phase test
Task 3.2: Verify Qwen3.5-2B tool calling works with Ollama function-calling API
Task 3.3: Test thinking mode toggle — measure latency difference
Task 3.4: Build LangGraph state graph — all nodes, edges, guards
Task 3.5: Implement AgentState schema
Task 3.6: Test phase transitions with mock tool results
Task 3.7: Test error_handler node — simulate tool failure, verify recovery flow
```

### Sprint 4 — Agent Service + UI (Week 4–5)
```
Task 4.1: Wrap LangGraph in FastAPI — single /agent endpoint
Task 4.2: Add /agent route to Node.js proxy
Task 4.3: Build agent chat panel in existing UI (HTML/JS, collapsible)
Task 4.4: Wire chat panel → Node.js /agent → FastAPI → LangGraph → response
Task 4.5: Add phase indicator bar and tool call log to UI
Task 4.6: Test full end-to-end: user types → agent runs Phase 1 tool → session updated
```

### Sprint 5 — Session Management + Validation Logic (Week 5–6)
```
Task 5.1: Build session load/save tools
Task 5.2: Test session resume — load Phase 3 session, agent jumps to node_phase3
Task 5.3: Implement quantitative replacement checks for Phase 1, 2, 4 (no VLM)
Task 5.4: Test threshold recommendation logic (Phase 4, thinking=ON)
Task 5.5: Test full 5-phase run end-to-end with real bottle images
Task 5.6: Stress test: wrong params, missing files, BS mismatch — verify error handling
```

### Sprint 6 — Polish + Optimization (Week 6–7)
```
Task 6.1: Tune RAG chunk quality based on real agent queries
Task 6.2: Optimize Qwen3.5-2B prompt length (trim session JSON for irrelevant phases)
Task 6.3: Add conversation memory compression (summarize old turns, keep recent)
Task 6.4: Benchmark agent latency per phase — target <5s per agent turn (thinking=OFF)
Task 6.5: Add session history browser to UI (list all saved sessions)
Task 6.6: Write agent README — how to run, how to extend tool registry
```

---

## 11. Tool & Library Stack Summary

| Layer | Tool | Version Target | Purpose |
|-------|------|---------------|---------|
| LLM runtime | Ollama | Latest | Serves Qwen3.5-2B locally |
| LLM model | Qwen3.5-2B (Q4_K_M GGUF) | — | Agent reasoning + tool calling |
| Embedding model | nomic-embed-text (Ollama) | — | RAG query + ingestion embedding |
| Vector DB | ChromaDB | 0.5.x | Knowledge base storage + retrieval |
| Agent framework | LangGraph | 0.2.x | Phase state machine |
| Agent API | FastAPI | 0.11x | Expose agent as HTTP service |
| Schema validation | Pydantic v2 | 2.x | Session config + tool input/output |
| LLM client | LangChain-Ollama | Latest | Qwen3.5-2B ↔ LangGraph bridge |
| Document chunker | LangChain text splitters | — | Chunking raw docs for ingestion |
| Existing backend | Flask | Unchanged | All inference tools |
| Existing proxy | Node.js / Express | Unchanged + /agent route | Proxies agent requests |
| Existing UI | HTML/CSS/JS | + agent panel | User-facing interface |

---

## 12. Key Integration Rules

1. **Agent never touches model weights directly** — all inference goes through Flask tools
2. **Session JSON is the only shared state** — agent reads it, tools write to it, UI can display it
3. **LangGraph decides phase transitions** — Qwen3.5-2B never outputs "go to phase 3"
4. **Pydantic validates before every tool call** — invalid config fails before Flask is called
5. **RAG context is always injected** — no agent turn calls the LLM without relevant KB context
6. **Tool list is always constrained** — model never sees tools outside its current phase node
7. **Raw scores are always cached after Phase 4 first run** — threshold updates never re-infer
8. **Every phase confirmation triggers `save_session`** — no state is ever lost between turns
