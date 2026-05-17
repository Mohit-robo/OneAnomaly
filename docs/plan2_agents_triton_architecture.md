# anomaly_app — Infrastructure Plan 2
## DINOv3 + Triton + Agents (Qwen3.5-2B via Ollama)
**Scope:** Full production deployment with agent orchestration layer.
**Models:** DINOv3 (Triton/TRT) + Qwen3.5-2B (Ollama, text-only)
**Principle:** UI + Agent Service on cloud, all GPU compute local, one HTTP call per image across the wire.

> **Note:** Qwen3.5-2B is text-only — no vision input. Visual verification checkpoints
> (Phase 1 mask quality, Phase 2 tiling confirmation, Phase 4 heatmap verdict) are handled
> via quantitative text-based checks from gateway metrics, not image interpretation.
> See Section 7 for the replacement strategy.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [How This Extends Plan 1](#2-how-this-extends-plan-1)
3. [Local Stack — IPC RTX 5060](#3-local-stack--ipc-rtx-5060)
4. [Cloud Stack — Google Cloud](#4-cloud-stack--google-cloud)
5. [Connectivity Layer — Tailscale](#5-connectivity-layer--tailscale)
6. [Qwen3.5-2B via Ollama — Setup](#6-qwen35-2b-via-ollama--setup)
7. [Agent Service — LangGraph Design](#7-agent-service--langgraph-design)
8. [RAG Layer — ChromaDB](#8-rag-layer--chromadb)
9. [Request Flow — Agent-Mediated vs Manual](#9-request-flow--agent-mediated-vs-manual)
10. [Latency Analysis](#10-latency-analysis)
11. [Tech Stack Summary](#11-tech-stack-summary)
12. [Cost Breakdown](#12-cost-breakdown)

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD (CPU only)                     │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐     │
│   │              Cloud Run (UI Service)                  │     │
│   │   UI (HTML/JS/CSS)                                   │     │
│   │   Session API (/session CRUD)                        │     │
│   │   Infer proxy (/infer → local gateway)               │     │
│   │   Results API (/results → Cloud Storage)             │     │
│   └──────────────────────────────────────────────────────┘     │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐     │
│   │           Cloud Run (Agent Service)                  │     │
│   │                                                      │     │
│   │   LangGraph state machine (5-phase graph)            │     │
│   │   Tool Registry (wraps local gateway endpoints)      │     │
│   │   RAG Retriever (queries ChromaDB)                   │     │
│   │   Session Manager (reads/writes session configs)     │     │
│   │                                                      │     │
│   │   Calls across Tailscale:                            │     │
│   │     → Local Gateway (inference tools)                │     │
│   │     → Ollama server (LLM reasoning)                  │     │
│   └──────────────────────────────────────────────────────┘     │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐     │
│   │               ChromaDB (GCP VM or Cloud Run)         │     │
│   │   Knowledge base: phase docs, setup, lessons,        │     │
│   │   session history, parameter glossary                │     │
│   └──────────────────────────────────────────────────────┘     │
│                                                                │
│   Cloud Storage: sessions, results, good sample ZIPs          │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                   Tailscale VPN (WireGuard)
                   Two call types:
                     A) Image inference: image in → result JSON out
                     B) LLM reasoning: prompt in → text out (Ollama)
                           │
┌──────────────────────────▼─────────────────────────────────────┐
│                    LOCAL IPC — RTX 5060                        │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐     │
│   │           Inference Gateway (FastAPI :8080)          │     │
│   │   POST /infer            → full image pipeline       │     │
│   │   POST /build_memory_bank → FAISS index creation     │     │
│   │   POST /export_trt       → TRT engine export         │     │
│   │   POST /sync_session     → session config update     │     │
│   │   GET  /health           → status check              │     │
│   │   GET  /gateway_stats    → quantitative metrics      │     │
│   │         (mask coverage, region areas, score dist.)   │     │
│   └────────────────────┬─────────────────────────────────┘     │
│                        │ gRPC loopback                         │
│   ┌────────────────────▼─────────────────────────────────┐     │
│   │         Triton Inference Server (:8001 gRPC)         │     │
│   │         Backend: TensorRT, dynamic shapes            │     │
│   │                                                      │     │
│   │   dinov3_encoder: [1-8, 3, 224, 224] → [1-8, 768]   │     │
│   │   unet_mask (future): [1-4, 3, H, W] → [1-4, 1, H, W]│   │
│   └──────────────────────────────────────────────────────┘     │
│                                                                │
│   ┌──────────────────────────────────────────────────────┐     │
│   │         Ollama Server (:11434)                       │     │
│   │         Model: Qwen3.5-2B (Q4_K_M GGUF)             │     │
│   │         Context: 32K tokens                          │     │
│   │         Thinking mode: toggled per call type         │     │
│   └──────────────────────────────────────────────────────┘     │
│                                                                │
│   FAISS indexes (local disk)                                   │
│   TRT engine cache (local disk)                                │
│   Session config cache (local disk)                            │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. How This Extends Plan 1

Plan 2 is **Plan 1 with three additions.** Nothing in Plan 1 is removed or modified.

| Component | Plan 1 | Plan 2 Change |
|-----------|--------|---------------|
| Inference Gateway | ✅ Same | + `GET /gateway_stats` endpoint added |
| Triton + TRT | ✅ Same | Unchanged |
| FAISS | ✅ Same | Unchanged |
| Cloud Run (UI) | ✅ Same | + Agent chat panel added to HTML/JS |
| Cloud Storage | ✅ Same | Unchanged |
| Tailscale | ✅ Same | + Ollama added as second local target |
| **Ollama (Qwen3.5-2B)** | ❌ Not present | **NEW: local, called by Agent Service** |
| **Agent Service** | ❌ Not present | **NEW: cloud, LangGraph + tool registry** |
| **ChromaDB** | ❌ Not present | **NEW: cloud, RAG knowledge base** |

**Migration path:** A team running Plan 1 can add the agent layer (Plan 2) without touching the gateway or Triton. The agent calls the same gateway endpoints that the UI already calls.

---

## 3. Local Stack — IPC RTX 5060

### 3.1 Process Map (Plan 2 additions in bold)

| Process | Port | Protocol | Responsibility |
|---------|------|----------|---------------|
| Triton Server | 8001 | gRPC (internal) | Model inference |
| Inference Gateway | 8080 | HTTP | Pipeline orchestration |
| **Ollama Server** | **11434** | **HTTP** | **Qwen3.5-2B LLM serving** |

### 3.2 Updated Memory Budget

| Component | VRAM Usage |
|-----------|-----------|
| TRT DINOv3 (BS=4, Triton) | ~1.8 GB |
| TRT UNet (BS=2, future) | ~0.6 GB |
| Triton framework overhead | ~0.3 GB |
| FAISS GPU index (optional) | ~0.5 GB |
| **Qwen3.5-2B Q4_K_M (Ollama)** | **~1.4 GB** |
| **Total** | **~4.6 GB** |
| **Headroom remaining** | **~7.4 GB** |

Still within RTX 5060 budget. Ollama and Triton share the GPU without conflict — Ollama calls are infrequent (agent turns), Triton calls are fast GPU bursts. They don't overlap in practice.

### 3.3 Ollama Configuration

```
OLLAMA_NUM_GPU=1                  # use RTX 5060
OLLAMA_GPU_MEMORY_FRACTION=0.15   # ~1.4GB ceiling, leaves Triton headroom
OLLAMA_KEEP_ALIVE=30m             # keep model warm for 30 min after last call
OLLAMA_MAX_QUEUE=4                # max concurrent agent reasoning calls
```

### 3.4 Gateway Stats Endpoint (New in Plan 2)

The agent needs quantitative metrics to make text-based decisions (since no VLM). The gateway exposes these after each inference call:

```
GET /gateway_stats?session={name}&last_n={20}

Response: {
  phase1_stats: {
    mask_coverage_pct: float,       # % of image covered by mask
    num_connected_components: int,  # should be 1–3 for clean masks
    fg_bg_ratio: float
  },
  phase2_stats: {
    region_areas_pct: [float],      # % of image per region
    region_overlaps_pct: [float],   # overlap between adjacent regions
    smallest_region_pct: float
  },
  phase4_stats: {
    score_distribution: [float],    # all anomaly scores from last run
    ok_count: int,
    not_ok_count: int,
    score_mean: float,
    score_std: float,
    score_gap: float                # mean(not_ok) - mean(ok) — separation quality
  }
}
```

Agent calls this endpoint (via tool) and reasons over the numbers with `thinking=ON`.

---

## 4. Cloud Stack — Google Cloud

### 4.1 Services Map

| GCP Service | Purpose | Plan 1 | Plan 2 |
|-------------|---------|--------|--------|
| Cloud Run (UI) | UI + session API + infer proxy | ✅ | ✅ (+ agent chat panel) |
| Cloud Run (Agent) | Agent Service — LangGraph + tool calls | ❌ | ✅ NEW |
| Cloud Storage | Sessions, results, good samples | ✅ | ✅ |
| GCP e2-micro VM | Tailscale network hop | ✅ | ✅ |
| **ChromaDB** | **RAG knowledge base** | ❌ | ✅ NEW (on e2-micro VM) |
| Secret Manager | Tailscale key | ✅ | ✅ |

### 4.2 ChromaDB Placement

ChromaDB runs on the **same e2-micro VM** that hosts Tailscale. Avoids an extra instance — the VM is already running 24/7 as the network bridge.

```
GCP e2-micro VM:
  ├── Tailscale daemon          (network bridge to local IPC)
  └── ChromaDB (persistent)    (:8000, only accessible within VPC)
      └── /data/chromadb/
          ├── phase_docs/
          ├── setup_docs/
          ├── session_history/
          └── api_reference/
```

ChromaDB is not exposed to the public internet — only the Agent Service Cloud Run instance can reach it (via VPC internal routing).

### 4.3 Agent Service — Cloud Run

**What runs here:**
- FastAPI app exposing `POST /agent`
- LangGraph state machine (in-process)
- RAG Retriever (calls ChromaDB on e2-micro)
- Tool Registry (calls local gateway via Tailscale)
- Session Manager (reads/writes Cloud Storage)

**What it does NOT contain:**
- No model weights
- No FAISS
- No ChromaDB data (only the client)

**Cloud Run sizing for Agent Service:**
- 2 vCPU, 2GB RAM — LangGraph + Python async is lightweight
- Concurrency: 1 (agent sessions are stateful, don't parallelize)
- Min instances: 0 (scales to zero when not in use)
- Max instances: 2 (one per concurrent session)

---

## 5. Connectivity Layer — Tailscale

### 5.1 Two Local Targets (Plan 2 addition)

```
Cloud Agent Service reaches:
  ├── Local Gateway     → 100.x.x.1:8080   (inference tools)
  └── Local Ollama      → 100.x.x.1:11434  (LLM reasoning)

Cloud UI Service reaches:
  └── Local Gateway     → 100.x.x.1:8080   (direct infer proxy)
```

Both targets are on the same IPC machine — same Tailscale IP, different ports.

### 5.2 Latency Profile for Agent Calls

```
Agent turn (one reasoning step):
  Cloud Agent Service
    → Ollama (Tailscale, 30–60ms round trip)
    → Qwen3.5-2B processes (1–4 seconds, thinking=ON)
    → Agent parses tool call
    → Local Gateway (Tailscale, 30–60ms round trip)
    → Gateway runs full inference (50–120ms)
    → Agent receives result
    → Ollama again (30–60ms) for final response

Total per agent turn: ~2–6 seconds
```

This is acceptable for agent use — users expect agent responses in seconds, not milliseconds. Inference latency (image → result) is dominated by Triton GPU time + network, same as Plan 1.

---

## 6. Qwen3.5-2B via Ollama — Setup

### 6.1 Serving

```
ollama serve                          # starts server on :11434
ollama pull qwen3.5:2b               # downloads model
ollama run qwen3.5:2b --verbose      # verify it loads on GPU
```

Ollama with Qwen3.5-2B uses approximately 1.2–1.5GB VRAM at Q4_K_M quantization. Confirmed within RTX 5060 budget alongside Triton.

### 6.2 Thinking Mode Per Call Type

Qwen3.5-2B supports hybrid thinking — CoT reasoning chain runs internally before output. Toggle via the `/think` flag in the prompt or system message.

| Agent Task | Thinking | Reason | Est. Latency |
|-----------|----------|--------|-------------|
| Parse user intent → tool params | OFF | Deterministic extraction | 0.5–1s |
| Validate session config for consistency | ON | Multi-field cross-check | 2–4s |
| Recommend pre-proc mode from gateway stats | ON | Quantitative reasoning | 2–4s |
| Synthesize RAG answer for user question | OFF | Retrieve + summarize | 1–2s |
| Detect BS mismatch (phase2 vs phase3) | ON | Schema cross-validation | 2–3s |
| Recommend τ threshold from score distribution | ON | Statistical reasoning | 3–5s |
| Generate Phase 5 results summary | OFF | Structured formatting | 0.5–1s |

### 6.3 Context Window Management

Qwen3.5-2B has a 32K context window. Budget per agent turn:

| Content | Tokens |
|---------|--------|
| System prompt (role + rules) | ~300 |
| Current session JSON | ~800 |
| Conversation history (last 5 turns) | ~1500 |
| RAG context (3 chunks) | ~900 |
| Available tools (constrained list) | ~400 |
| **Total per turn** | **~3900 tokens** |

Well within 32K. For long sessions (Phase 5 with large results), compress conversation history by summarizing turns older than 10 exchanges.

---

## 7. Agent Service — LangGraph Design

### 7.1 Graph Structure

```
START
  ↓
node_intake ──────────────────────────────────────────────────┐
  ├── user asks question → node_rag_query → END               │
  ├── session load requested → node_session_load              │
  ├── no active session → node_phase1                         │
  └── active session → node_phase{current_phase}              │
                                                              │
node_session_load → node_phase{N} (resume at correct phase)  │
                                                              │
node_phase1 ─────────────────────────────────────────────────┤
  Tools: run_preprocessing, get_gateway_stats,               │
         confirm_phase, query_knowledge_base, save_session   │
  → (phase1.confirmed) → node_phase2                         │
                                                              │
node_phase2 ─────────────────────────────────────────────────┤
  Tools: set_spatial_regions, export_dinov3_trt,             │
         get_gateway_stats, confirm_phase,                   │
         query_knowledge_base, save_session                  │
  → (phase2.confirmed) → node_phase3                         │
                                                              │
node_phase3 ─────────────────────────────────────────────────┤
  Tools: build_memory_bank, query_knowledge_base,            │
         save_session                                        │
  → (phase3.confirmed) → node_phase4                         │
                                                              │
node_phase4 ─────────────────────────────────────────────────┤
  Tools: run_validation, update_threshold,                   │
         get_gateway_stats, confirm_phase,                   │
         query_knowledge_base, save_session                  │
  → (user confirms) → node_phase5                            │
                                                              │
node_phase5 ─────────────────────────────────────────────────┤
  Tools: run_final_inference, save_session,                  │
         download_results                                    │
  → END                                                       │
                                                              │
node_error_handler ──────────────────────────────────────────┘
  Triggered: error_count >= 2
  Tools: query_knowledge_base (fetch recovery docs)
  → back to current phase after user acknowledges
```

### 7.2 AgentState Schema

```
AgentState {
  current_phase: int (1–5)
  session_config: SessionConfig      # full Pydantic model
  conversation_history: list[dict]   # {role, content} pairs
  last_tool: str
  last_tool_result: dict
  rag_context: list[str]
  pending_confirmation: bool
  error_count: int
  available_tools: list[str]         # constrained per node
}
```

### 7.3 Tool Registry (Agent Service Side)

Each tool is a Python function that makes an HTTP call to the local gateway via Tailscale OR to ChromaDB directly. All tools return a typed dict.

| Tool | Calls | Thinking Mode |
|------|-------|--------------|
| `run_preprocessing` | Gateway POST /infer (phase=1) | — |
| `get_gateway_stats` | Gateway GET /gateway_stats | ON (agent reasons on result) |
| `set_spatial_regions` | Gateway POST /infer (phase=2) | — |
| `export_dinov3_trt` | Gateway POST /export_trt | — |
| `build_memory_bank` | Gateway POST /build_memory_bank | — |
| `run_validation` | Gateway POST /infer (phase=4, batch) | — |
| `update_threshold` | Gateway POST /sync_session + re-infer | — |
| `run_final_inference` | Gateway POST /infer (phase=5, batch) | — |
| `query_knowledge_base` | ChromaDB client (direct) | OFF |
| `load_session` | Cloud Storage client | — |
| `save_session` | Cloud Storage + Gateway /sync_session | — |
| `confirm_phase` | Internal state update only | ON (validate before confirming) |
| `download_results` | Cloud Storage signed URL generation | — |

### 7.4 Quantitative Validation (Replacing VLM Checkpoints)

Since Qwen3.5-2B cannot interpret images, all three former VLM checkpoints are replaced by the `get_gateway_stats` tool + agent reasoning with `thinking=ON`.

**Phase 1 — Mask quality:**
```
Agent calls get_gateway_stats → reads phase1_stats
Qwen3.5-2B (thinking=ON) reasons:
  "mask_coverage_pct=38%, num_components=2, fg_bg_ratio=0.61
   Coverage is within expected range for cap-zone masking (30–50%).
   2 connected components suggests clean separation.
   Proceeding to Phase 2."

OR flags:
  "mask_coverage_pct=89%, num_components=47
   Coverage is too high — mask likely includes background.
   Recommend lowering threshold_min from 120 to 80."
```

**Phase 2 — Spatial tiling:**
```
Agent calls get_gateway_stats → reads phase2_stats
Pydantic guard: smallest_region_pct >= 5% (hard rule)
Agent (thinking=ON):
  "region_areas = [24%, 26%, 23%, 27%] — balanced split.
   region_overlaps = [1.2%, 0.8%, 1.1%] — within 2% tolerance.
   Tiling config is valid."
```

**Phase 4 — Threshold recommendation:**
```
Agent calls get_gateway_stats → reads phase4_stats
Agent (thinking=ON):
  "score_distribution shows bimodal pattern.
   score_gap=0.52 — good separation between OK and Not-OK.
   ok cluster mean=0.18, not_ok cluster mean=0.70.
   Recommend τ=0.44 for maximum margin.
   Current τ=0.50 is conservative — will miss 8% of defects."
```

### 7.5 Agent API Endpoint

The Agent Service exposes one endpoint, called by the UI's agent chat panel:

```
POST /agent
  Body: {
    user_message: string,
    session_name: string,
    conversation_history: list[dict]
  }
  Response: {
    agent_reply: string,
    tools_called: list[{name, params, result_summary}],
    updated_session: dict,
    current_phase: int,
    pending_confirmation: bool,
    error: string | null
  }
```

---

## 8. RAG Layer — ChromaDB

### 8.1 Collections

| Collection | Content | Chunk Size |
|-----------|---------|-----------|
| `phase_docs` | Phase 1–5 step-by-step docs | ~300 tokens |
| `setup_docs` | Installation, model paths, dependencies | ~400 tokens |
| `lessons_learned` | Known bugs, false positive patterns, hw migration notes | ~250 tokens |
| `session_history` | Past successful session configs as JSON | Whole JSON |
| `api_reference` | Gateway endpoint specs | One chunk per endpoint |
| `parameter_glossary` | τ, k, σ, n_max, BS definitions | One chunk per param |

### 8.2 Metadata Per Chunk

```json
{
  "phase": 3,
  "topic": "memory_bank_creation",
  "doc_type": "phase_docs",
  "criticality": "high"
}
```

Metadata enables filtered queries — agent in Phase 3 only retrieves Phase 3 + cross-phase chunks.

### 8.3 Retrieval Mode Per Graph Node

| Node | ChromaDB Filter | k | Mode |
|------|----------------|---|------|
| node_phase1 | phase=1 OR phase="cross" | 3 | Filtered |
| node_phase2 | phase=2 OR phase="cross" | 3 | Filtered |
| node_phase3 | phase=3 | 3 | Filtered |
| node_phase4 | phase=4 | 3 | Filtered |
| node_phase5 | phase=5 | 2 | Filtered |
| node_rag_query | no filter | 5 | Open similarity |

### 8.4 Ingestion Pipeline (One-Time)

```
Step 1: Collect all docs → /data/raw_docs/
Step 2: Chunk by section headers, apply metadata
Step 3: Embed via nomic-embed-text (Ollama on IPC)
Step 4: Insert into ChromaDB collections on GCP e2-micro
Step 5: Test: 15 queries covering all phases → verify top-3 accuracy
Step 6: Fix chunk boundaries where retrieval misses
```

Embedding model (`nomic-embed-text`) runs locally during ingestion. At runtime, query embeddings also computed locally and sent to ChromaDB for search — embedding computation is on IPC, only the query vector transits to cloud ChromaDB.

---

## 9. Request Flow — Agent-Mediated vs Manual

Both modes coexist. User can switch between chat (agent) and direct UI (Plan 1 behavior) at any time. Same session config, same gateway.

### Manual Mode (Plan 1 behavior, still available)

```
Browser UI controls → Cloud Run UI service → Local Gateway → result
```

### Agent Mode (Plan 2 addition)

```
Browser Agent Chat → Cloud Run Agent Service
                          ↓
                    Qwen3.5-2B (Ollama, local, via Tailscale)
                    "Which tool to call?"
                          ↓
                    Tool: run_validation
                          ↓
                    Local Gateway (via Tailscale)
                    Full inference pipeline
                          ↓
                    Tool: get_gateway_stats
                          ↓
                    Local Gateway stats endpoint
                          ↓
                    Qwen3.5-2B (thinking=ON)
                    "Threshold recommendation"
                          ↓
                    Agent updates session config
                          ↓
                    Browser receives reply + updated state
```

### Full Phase 4 Example (Agent-Mediated)

```
User: "Run validation on the test images and tune the threshold"

Agent turn 1:
  RAG retrieval → Phase 4 docs loaded
  Ollama: decide tool → run_validation
  Tool: run_validation (20 images) → raw scores cached

Agent turn 2:
  Tool: get_gateway_stats → score distribution
  Ollama (thinking=ON): analyze bimodal distribution
  Recommends τ=0.44
  Updates session: phase4.params.tau = 0.44
  Tool: save_session

Agent turn 3:
  Tool: update_threshold (τ=0.44, cached scores)
  No re-inference — uses raw score cache from turn 1
  Gets updated ok_count, not_ok_count

Agent reply:
  "Validation complete. 18/20 images passed. 2 flagged as anomalous.
   Score gap between OK (mean 0.18) and Not-OK (mean 0.71) is 0.53 —
   strong separation. Threshold set to τ=0.44 for maximum margin.
   Ready to confirm and proceed to final inference."

User: "Looks good, confirm"
Agent: phase4.status = "confirmed" → transition to node_phase5
```

---

## 10. Latency Analysis

### Inference Path (Same as Plan 1 — Unchanged)

| Hop | Latency |
|----|---------|
| Browser → Cloud UI | 10–30ms |
| Cloud UI → GCP VM → Local Gateway (Tailscale) | 30–70ms |
| Gateway preprocessing | 5–15ms |
| Triton DINOv3 (BS=4) | 4–8ms |
| FAISS search | 1–3ms |
| Heatmap generation | 5–10ms |
| Result back to browser | 30–70ms |
| **Total per image** | **~85–175ms** |

### Agent Turn (New in Plan 2)

| Step | Latency |
|------|---------|
| User message → Agent Service | 10–30ms |
| ChromaDB RAG retrieval | 10–30ms |
| Tailscale hop to Ollama | 30–60ms |
| Qwen3.5-2B (thinking=OFF) | 500ms–1.5s |
| Qwen3.5-2B (thinking=ON) | 2–5s |
| Tool call (gateway) | 85–175ms (same as inference path) |
| Tailscale return hop | 30–60ms |
| **Total agent turn (simple)** | **~1.5–3s** |
| **Total agent turn (with thinking)** | **~3–8s** |

Agent turns feel slow compared to direct UI interaction. This is the known tradeoff — agent turns involve multiple network hops and LLM reasoning. For parameter configuration and session orchestration, 3–8 seconds is acceptable. For rapid threshold tuning (slider interaction), users should use the direct UI, not the agent.

---

## 11. Tech Stack Summary

### Local IPC

| Component | Tool | Purpose |
|-----------|------|---------|
| Inference server | Triton Inference Server | DINOv3 + UNet model serving |
| Backend | TensorRT (dynamic shapes) | GPU inference |
| Gateway framework | FastAPI + Uvicorn | Pipeline orchestration |
| Triton client | tritonclient[grpc] | loopback gRPC calls |
| Preprocessing | OpenCV | Masking, spatial tiling |
| Similarity search | FAISS-GPU | Memory bank queries |
| Heatmap | NumPy + PIL | Score visualization |
| **LLM server** | **Ollama** | **Serves Qwen3.5-2B** |
| **LLM model** | **Qwen3.5-2B Q4_K_M** | **Agent reasoning** |
| Connectivity | Tailscale | Cloud ↔ local bridge |

### Cloud (Google Cloud)

| Component | GCP Service | Notes |
|-----------|------------|-------|
| UI + infer proxy | Cloud Run | Serverless |
| **Agent Service** | **Cloud Run** | **LangGraph + tool registry** |
| Sessions + results | Cloud Storage | Standard tier |
| Network + ChromaDB | GCP e2-micro VM | Free tier, always-on |
| **Knowledge base** | **ChromaDB on e2-micro** | **RAG store** |
| Secrets | Secret Manager | Tailscale key |

### Agent Service Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| LangGraph | 0.2.x | Phase state machine |
| LangChain-Ollama | Latest | Ollama ↔ LangGraph bridge |
| ChromaDB client | 0.5.x | RAG retrieval |
| Pydantic v2 | 2.x | Session + tool schema validation |
| httpx | Latest | Async gateway tool calls |
| FastAPI | 0.111.x | /agent endpoint |

---

## 12. Cost Breakdown

### Monthly Estimates

| Item | Service | Plan 1 Cost | Plan 2 Cost |
|------|---------|------------|------------|
| Cloud Run (UI) | Cloud Run | $0–3 | $0–3 |
| **Cloud Run (Agent Service)** | Cloud Run | — | **$0–3** |
| GCP e2-micro VM (Tailscale + ChromaDB) | Compute | Free | Free |
| Cloud Storage | Storage | $0.50 | $0.50 |
| Triton + FAISS + DINOv3 | Local | Electricity | Electricity |
| **Qwen3.5-2B (Ollama)** | Local | — | **Electricity** |
| Tailscale | — | Free | Free |
| **Total cloud** | | **~$1–5/mo** | **~$1–8/mo** |

### vs GPU Cloud

| Approach | Monthly |
|----------|---------|
| All-cloud GPU (GCP T4, 24/7) | ~$350/month |
| **Plan 1 (Triton local, no agent)** | **~$1–5/month** |
| **Plan 2 (Triton + Agent local)** | **~$1–8/month** |

Agent Service adds virtually nothing to cloud cost — Cloud Run scales to zero, and each agent turn is one serverless invocation measured in seconds. The Ollama LLM compute (RTX 5060, local) costs only electricity.

---

## Key Design Rules (Plan 2)

1. **Plan 1 rules all carry forward** — one HTTP call per image, Triton resident, FAISS local
2. **Agent and UI are independent paths** — both call the same gateway, user can switch between them anytime
3. **Ollama is always local** — LLM reasoning never leaves the IPC; only text prompts and responses cross Tailscale
4. **Agent never bypasses the gateway** — even when the agent is in control, all inference goes through the same FastAPI gateway that the UI uses
5. **ChromaDB is cloud-side** — agent service and ChromaDB are collocated (same VPC), no retrieval latency across Tailscale
6. **LangGraph decides phase transitions, not the model** — Qwen3.5-2B selects tools and reasons; it never outputs a phase number
7. **Quantitative stats replace VLM checkpoints** — gateway_stats endpoint is the agent's only source of image-derived information
8. **Raw score cache eliminates re-inference** — threshold tuning in Phase 4 is a stat re-computation, not a new Triton call
9. **Session JSON is the only shared state** — agent writes it, UI reads it, gateway executes from it
10. **Pydantic validates before every tool call** — wrong session state fails before any Triton or FAISS call is made
