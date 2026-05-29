# anomaly_app — Infrastructure Plan 1
## DINOv3 + Triton (No Agents)
**Scope:** Production deployment without any agent layer.
**Models:** DINOv3 (current) + UNet (future phase)
**Principle:** UI on cloud, all GPU compute local, one HTTP call per image across the wire.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Local Stack — IPC](#2-local-stack--ipc)
3. [Cloud Stack — Amazon Web Services (AWS)](#3-cloud-stack-amazon-web-services-aws)
4. [Connectivity Layer — Tailscale](#4-connectivity-layer-tailscale)
5. [Triton Server — Detailed Setup](#5-triton-server-detailed-setup)
6. [Inference Gateway — Detailed Design](#6-inference-gateway-detailed-design)
7. [Request Flow Per Phase](#7-request-flow-per-phase)
8. [Session Management](#8-session-management)
9. [Future: Adding UNet to the Stack](#9-future-adding-unet-to-the-stack)
10. [Tech Stack Summary](#10-tech-stack-summary)
11. [Cost Breakdown](#11-cost-breakdown)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                      AWS CLOUD (CPU only)                │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │              AWS App Runner / EC2 Instance       │   │
│   │                                                  │   │
│   │   UI (Flask-served HTML/JS/CSS)                  │   │
│   │   Session Store (JSON files / Amazon S3)         │   │
│   │   Results Store (annotated images, CSV)          │   │
│   │                                                  │   │
│   │   Exposes:                                       │   │
│   │     /ui         → browser                        │   │
│   │     /session    → CRUD for session configs       │   │
│   │     /infer      → proxies to local gateway       │   │
│   └──────────────────────────────────────────────────┘   │
└─────────────────────────┬────────────────────────────────┘
                          │
                  Tailscale VPN (WireGuard)
                  One HTTP call per image
                  Payload: image bytes in
                  Payload: result JSON + heatmap out
                          │
┌─────────────────────────▼────────────────────────────────┐
│                LOCAL IPC —                      │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │           Inference Gateway (FastAPI)             │   │
│   │                                                  │   │
│   │   POST /infer   → full pipeline per image        │   │
│   │   POST /build   → memory bank creation           │   │
│   │   POST /export  → TRT model export               │   │
│   │                                                  │   │
│   │   Internally:                                    │   │
│   │     Step 1: Pre-processing (OpenCV/UNet)         │   │
│   │     Step 2: Spatial tiling                       │   │
│   │     Step 3: Triton gRPC call (loopback)          │   │
│   │     Step 4: FAISS search                         │   │
│   │     Step 5: Heatmap generation                   │   │
│   └────────────────────┬─────────────────────────────┘   │
│                        │ gRPC loopback (0ms network)     │
│   ┌────────────────────▼─────────────────────────────┐   │
│   │         Triton Inference Server                  │   │
│   │         Backend: TensorRT                        │   │
│   │                                                  │   │
│   │   Model: dinov3_encoder                          │   │
│   │     Input:  [1-8, 3, 224, 224] float32           │   │
│   │     Output: [1-8, 768] float32                   │   │
│   │     Batching: dynamic (preferred: 1, 4)          │   │
│   │                                                  │   │
│   │   Model: unet_mask (FUTURE)                      │   │
│   │     Input:  [1-4, 3, H, W] float32               │   │
│   │     Output: [1-4, 1, H, W] float32               │   │
│   │     Batching: dynamic (preferred: 1, 2)          │   │
│   └──────────────────────────────────────────────────┘   │
│                                                          │
│   ┌──────────────────────────────────────────────────┐   │
│   │   FAISS Index (local disk)                       │   │
│   │   Session Configs (local disk)                   │   │
│   │   TRT Engine Cache (local disk)                  │   │
│   └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

**Core principle:** The boundary between cloud and local is exactly one HTTP call. Cloud never receives raw feature tensors. Cloud never runs GPU code. Everything GPU-related is encapsulated behind the gateway.

---

## 2. Local Stack — IPC

### 2.1 Process Map

| Process | Port | Protocol | Responsibility |
|---------|------|----------|---------------|
| Triton Server | 8000 (HTTP), 8001 (gRPC) | gRPC (internal) | Model inference only |
| Inference Gateway | 8080 | HTTP | Pipeline orchestration, receives from cloud |
| FAISS (in-process) | — | Library call | Similarity search, memory bank I/O |

### 2.2 Disk Layout

```
/opt/anomaly_app/
├── triton_models/               # Triton model repository
│   ├── dinov3_encoder/
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       └── model.plan       # TRT engine
│   └── unet_mask/               # Added when ready (Phase 1.3)
│       ├── config.pbtxt
│       └── 1/
│           └── model.plan
├── gateway/                     # FastAPI inference gateway
│   ├── main.py
│   ├── preprocessing.py
│   ├── tiling.py
│   ├── faiss_search.py
│   └── heatmap.py
├── faiss_indexes/               # One subdir per saved memory bank
│   └── bottle_cap_v1/
│       ├── index.faiss
│       └── metadata.json
├── sessions/                    # Session config JSONs (synced with cloud)
│   └── session_abc123.json
└── trt_cache/                   # TRT engines keyed by BS + image size
    ├── dinov3_bs1_224.plan
    └── dinov3_bs4_224.plan
```

### 2.3 Memory Budget  — 4GB VRAM assumed)

| Component | VRAM Usage |
|-----------|-----------|
| TRT DINOv3 (BS=4) | ~1.8 GB |
| TRT UNet (BS=2, future) | ~0.6 GB |
| Triton framework overhead | ~0.3 GB |
| FAISS GPU index (optional) | ~0.5 GB |
| **Total** | **~3.2 GB** |
| **Headroom remaining** | **~0.8 GB** |

Headroom is sufficient. Triton holds models resident — no loading between calls.

---

## 3. Cloud Stack — Amazon Web Services (AWS)

### 3.1 Services Used

| AWS Service | Purpose | Tier |
|-------------|---------|------|
| AWS App Runner or ECS Fargate | Hosts the Flask UI + session API | Low-cost managed container hosting |
| Amazon S3 | Stores session JSONs, result ZIPs, CSVs | Standard storage (5 GB Free Tier) |
| AWS Systems Manager Parameter Store | Stores Tailscale auth key | Free (Standard parameters) |
| Amazon ECR | Container image registry for App Runner | 500 MB/month free tier |

### 3.2 App Runner / ECS Fargate Service

**What it does:**
- Serves the full existing HTML/JS/CSS UI
- Exposes `/session` endpoints — CRUD for session config files
- Exposes `/infer` endpoint — receives image from browser, proxies to local gateway via Tailscale, returns result
- Exposes `/results` endpoint — download ZIP / CSV from Amazon S3

**What it does NOT do:**
- No model loading
- No GPU
- No FAISS
- No preprocessing

**Sizing:** 1 vCPU, 2GB RAM is the minimum App Runner configuration, which is extremely lightweight and fast. It handles low-volume scaling perfectly.

### 3.3 Session Config Sync Strategy

Session configs (JSON) live in two places:
- Amazon S3 (source of truth, accessible from browser)
- Local IPC disk (gateway reads at inference time)

Sync mechanism: **on session save**, the cloud server writes to Amazon S3 AND triggers a lightweight webhook to the local gateway (`POST /sync_session`). Gateway pulls the JSON and writes to local disk. No polling, no lag.

### 3.4 Result Storage Flow

```
Local gateway generates:
  ├── heatmap images (PNG)
  └── anomaly scores (in-memory dict)

Cloud UI/API Server receives result JSON (scores + heatmap base64)
  → writes heatmap PNGs to Amazon S3
  → writes CSV report to Amazon S3
  → returns signed download URLs (S3 Pre-signed URLs) to browser
```

---

## 4. Connectivity Layer — Tailscale

### 4.1 Setup

```
Devices on Tailscale network:
  ├── Local IPC                   → tailscale IP: 100.x.x.1
  └── AWS EC2 Jump Node Instance   → tailscale IP: 100.x.x.2
```

Since AWS App Runner or serverless Fargate cannot host a persistent Tailscale sidecar simply, **use a t2.micro or t3.micro EC2 instance (AWS Free Tier, 750 hours/month)** as the cloud-side network node/NAT gateway:

```
Browser → AWS App Runner (UI + session API)
                ↓ (VPC Connector egress)
          EC2 t2.micro / t3.micro Instance (AWS Free Tier)
          [Tailscale installed & routing enabled]
                ↓ Tailscale VPN tunnel
          Local IPC Inference Gateway :8080
```

This adds one ultra-low latency internal hop but keeps the UI hosting serverless/lightweight, and is fully covered under the AWS Free Tier.

### 4.2 Latency Profile

| Hop | Latency |
|----|---------|
| Browser → AWS App Runner | 15–35ms |
| App Runner → EC2 Instance (VPC) | < 1ms (same availability zone/VPC) |
| EC2 VM → Local Gateway (Tailscale) | 20–60ms (ISP dependent) |
| Gateway → Triton (gRPC loopback) | < 1ms |
| Triton inference (DINOv3, BS=4) | 4–8ms |
| FAISS search | 1–3ms |
| **Total per image (round trip)** | **~45–110ms** |

With 20 images in validation (Phase 4):
- Sequential: 20 × 110ms = 2.2 seconds network overhead
- Parallel (async gateway): 8-10 seconds for 20 images with 4 concurrent requests

---

## 5. Triton Server — Detailed Setup

### 5.1 DINOv3 Model Config (`config.pbtxt`)

```
name: "dinov3_encoder"
backend: "tensorrt"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [768]
  }
]

dynamic_batching {
  preferred_batch_size: [1, 4]
  max_queue_delay_microseconds: 5000
}

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }
]
```

### 5.2 Dynamic Shape TRT Engine Export Requirements

When compiling the TRT engine for use with Triton dynamic batching, the engine must be compiled with an **optimization profile**:

- Min shape: `[1, 3, 224, 224]`
- Optimal shape: `[4, 3, 224, 224]` — your most common spatial grid case
- Max shape: `[8, 3, 224, 224]`

This replaces the current fixed-BS TRT engines. One engine handles all batch sizes from 1 to 8. No engine swapping between Phase 2 modes.

### 5.3 Dynamic Batching Behavior

With `max_queue_delay_microseconds: 5000`:
- Triton waits up to 5ms for more requests to accumulate before dispatching
- If 4 spatial patch requests arrive within 5ms → batched as BS=4, one forward pass
- If only 1 arrives → dispatched immediately after 5ms (no stuck requests)
- At → requests naturally batch; no explicit client-side batching needed

### 5.4 Triton Metrics (Prometheus)

Triton exposes metrics at `:8002/metrics`:

| Metric | What to watch |
|--------|--------------|
| `nv_inference_queue_duration_us` | Queue time — indicates if batching is working |
| `nv_inference_compute_infer_duration_us` | Pure GPU time |
| `nv_inference_request_success` | Throughput counter |
| `nv_gpu_memory_used_bytes` | VRAM usage |

Connect to Grafana locally for dashboard visibility.

---

## 6. Inference Gateway — Detailed Design

### 6.1 API Surface

```
POST /infer
  Body: {
    image_b64: string,          # base64-encoded image
    session_name: string,        # which session config to load
    phase: int                   # 1–5 controls which pipeline runs
  }
  Response: {
    ok: bool,
    anomaly_score: float,
    region_scores: [float],      # one per spatial region
    heatmap_b64: string,         # base64 PNG
    inference_ms: float
  }

POST /build_memory_bank
  Body: {
    good_images_b64: [string],   # list of base64 images
    session_name: string
  }
  Response: {
    bank_name: string,
    bank_size: int,
    build_time_sec: float
  }

POST /export_trt
  Body: {
    model: "dinov3" | "unet",
    batch_sizes: [int],          # e.g. [1, 4, 8]
    image_size: int
  }
  Response: {
    engine_path: string,
    export_time_sec: float
  }

POST /sync_session
  Body: { session_name: string, config: dict }
  Response: { saved: bool }

GET /health
  Response: { triton: bool, faiss_loaded: bool, gpu_mem_free_gb: float }
```

### 6.2 Internal Pipeline Per `/infer` Call

```
Receive image_b64 + session_name
        ↓
Load session config from local disk (cached in memory after first load)
        ↓
Decode image bytes
        ↓
─── Phase 1: Pre-processing ───────────────────────────────
Apply mask (thresholding / template / UNet via Triton)
        ↓
─── Phase 2: Spatial Tiling ───────────────────────────────
Split image into N regions per session config
Stack into tensor [N, 3, 224, 224]
        ↓
─── Triton gRPC Call (loopback) ───────────────────────────
Send tensor to dinov3_encoder
Receive feature vectors [N, 768]
        ↓
─── Phase 4: FAISS Search ─────────────────────────────────
L2-normalize vectors
Query FAISS index → top-k distances per region
Compute anomaly scores per region
Apply threshold τ → OK / Not-OK per region
Aggregate to image-level decision
        ↓
─── Heatmap Generation ────────────────────────────────────
Map region scores → spatial heatmap
Apply Gaussian smoothing (sigma from session)
Overlay on original image
Encode as base64 PNG
        ↓
Return result JSON
```

### 6.3 Session Config Caching

Gateway holds an in-memory dict of `{session_name → config}`. On `/sync_session` call, cache is invalidated for that session and reloaded from disk. Avoids disk reads on every inference call.

### 6.4 FAISS Index Management

- On startup: load all FAISS indexes from `/opt/anomaly_app/faiss_indexes/` into memory dict
- On `/build_memory_bank`: build new index, add to memory dict and save to disk
- Index switching per request: keyed by `session_config.phase3.bank_name`
- No index reload between requests — all loaded, switch is a dict lookup

---

## 7. Request Flow Per Phase

### Phase 1 — Pre-processing Setup
```
Browser: user adjusts threshold / uploads template
      ↓
Cloud Run: stores pre-proc params in session config
      ↓ (POST /sync_session)
Local Gateway: updates local session config
      ↓
Browser: requests preview → cloud proxies image to /infer with phase=1
      ↓
Gateway: applies pre-proc, returns masked image (heatmap_b64 = mask preview)
      ↓
Browser: displays mask overlay
```

### Phase 2 — Spatial Region Selection
```
Browser: user selects grid or draws regions
      ↓
Cloud Run: stores region config in session (no inference needed)
      ↓
Browser: requests split preview → /infer with phase=2
      ↓
Gateway: tiles image per config, returns preview showing region boundaries
      ↓
Browser: user confirms
```

### Phase 3 — Memory Bank Creation
```
Browser: user uploads good samples ZIP (up to cloud)
      ↓
Cloud Run: stores ZIP to Cloud Storage, triggers /build_memory_bank
      ↓ (one call, image list in body)
Gateway: applies Phase 1 + 2 pipeline per image
         extracts features via Triton
         builds FAISS index
         saves locally
         returns bank_size, build_time
      ↓
Cloud Run: updates session config with bank metadata
```

### Phase 4 — Validation (≤20 images)
```
Browser: user uploads test images
      ↓
Cloud Run: for each image → POST /infer (can parallelize 4 concurrent)
      ↓ (one call per image)
Gateway: full pipeline → score + heatmap per image
      ↓
Cloud Run: collects all results, renders validation dashboard
Browser: displays heatmaps, threshold slider adjusts τ in session config
         (threshold change = new /infer call with updated session)
```

### Phase 5 — Final Inference (large test set)
```
Browser: uploads test set ZIP → Cloud Storage
      ↓
Cloud Run: iterates images, calls /infer for each
           collects results → CSV + annotated images → Cloud Storage
      ↓
Browser: downloads ZIP from Cloud Storage signed URL
```

---

## 8. Session Management

### Session Config Location Strategy

```
Source of truth:     Cloud Storage bucket /sessions/{session_name}.json
Local cache:         /opt/anomaly_app/sessions/{session_name}.json
Sync trigger:        POST /sync_session (cloud → local on every save)
```

### What the session config controls at the gateway

On every `/infer` call, the gateway reads:
- `phase1.mode` + `phase1.params` → which preprocessing to apply
- `phase2.mode` + `phase2.num_regions` + `phase2.grid_config` → how to tile
- `phase3.bank_name` → which FAISS index to query
- `phase4.params.tau` / `k` / `sigma` / `n_max` → scoring thresholds

Changing any param via the UI → session saved → `/sync_session` → next `/infer` uses updated params. No gateway restart required.

---

## 9. Future: Adding UNet to the Stack

When Phase 1.3 (UNet segmentation) is implemented, the gateway adds it as a Triton model. Zero impact on existing DINOv3 pipeline.

### Addition steps:

**Step 1: Export UNet to TRT**
- Convert trained PyTorch UNet → ONNX with dynamic spatial dims
- Compile ONNX → TRT engine with optimization profile
- Min: `[1, 3, 480, 640]`, Opt: `[2, 3, 480, 640]`, Max: `[4, 3, 480, 640]`

**Step 2: Add to Triton model repository**
```
triton_models/
└── unet_mask/
    ├── config.pbtxt    # backend: tensorrt, dynamic_batching enabled
    └── 1/
        └── model.plan  # TRT engine
```

**Step 3: Gateway preprocessing.py update**
- If `session.phase1.mode == "unet"`:
  - Send image to `unet_mask` Triton model (gRPC loopback)
  - Receive binary mask
  - Apply mask before tiling
- Else: existing threshold / template path (unchanged)

**Step 4: Triton Ensemble (optional optimization)**

Chain UNet → DINOv3 as a Triton ensemble pipeline:
```
triton_models/
└── anomaly_pipeline/       # ensemble model
    └── config.pbtxt
        steps:
          1. unet_mask      → mask output
          2. dinov3_encoder → feature output
```
One gRPC call from gateway → both models run in sequence inside Triton → one response back. Eliminates intermediate Python tensor handling between the two models.

---

## 10. Tech Stack Summary

### Local IPC

| Component | Tool | Version |
|-----------|------|---------|
| Inference server | Triton Inference Server | 23.x (NGC container) |
| Backend | TensorRT | 8.6+ |
| Gateway framework | FastAPI | 0.111.x |
| Gateway server | Uvicorn (async) | Latest |
| Triton client | tritonclient[grpc] | Latest |
| Pre-processing | OpenCV | 4.9.x |
| Similarity search | FAISS-GPU | 1.7.x |
| Heatmap generation | NumPy + PIL | Latest |
| Connectivity | Tailscale | Latest |
| Metrics | Prometheus + Grafana | Latest |

### Cloud (AWS)

| Component | AWS Service | Notes |
|-----------|-------------|-------|
| UI + API | AWS App Runner / ECS Fargate | Serverless container hosting |
| Static assets | Amazon S3 | Served via S3 bucket (5 GB Free Tier) |
| Session + results | Amazon S3 | Standard storage tier |
| Network proxy | EC2 t2.micro / t3.micro | Free Tier (750 hrs/month), runs Tailscale |
| Secrets | Systems Manager Parameter Store | Free standard parameter storage |
| Container registry | Amazon ECR | 500 MB/month free tier |

---

## 11. Cost Breakdown

### Monthly Estimates

| Item | Service | Est. Cost/Month |
|------|---------|----------------|
| App Runner (UI + API) | AWS App Runner | ~$5–10 (depends on request rate and active hours) |
| EC2 VM (Tailscale hop) | EC2 t2.micro / t3.micro | **Free** (AWS 12-month Free Tier) |
| Amazon S3 (sessions + results) | Amazon S3 | $0.02–0.50 (mostly covered by 5GB free tier) |
| Triton + DINOv3 + FAISS | Local IPC | Electricity only |
| Tailscale | Tailscale | Free (personal/starter plan) |
| **Total cloud** | | **~$5–10/month** |

### Comparison

| Approach | Monthly Cost |
|----------|-------------|
| All-cloud GPU (AWS g4dn.xlarge VM, 24/7) | ~$520/month |
| All-cloud GPU (Serverless GPU) | ~$250–450/month |
| **This architecture** | **~$5–10/month** |

---

## Key Design Rules

1. **One HTTP call per image across the wire** — all preprocessing, Triton calls, and FAISS search happen locally before response
2. **Triton stays resident** — models loaded at startup, never unloaded between requests
3. **Dynamic batch profiles** — one TRT engine handles BS=1 through BS=8, no engine swapping
4. **Session config is the control plane** — changing any param via UI triggers sync, no gateway restart
5. **Cloud handles zero GPU work** — Cloud Run is purely UI + routing + storage
6. **UNet slots in without breaking DINOv3** — Triton ensemble path is reserved from day one
7. **FAISS stays local** — feature vectors never transit the network
