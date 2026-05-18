# Triton Distributed Architecture Implementation Plan

This document breaks down the execution strategy to migrate OneAnomaly to a scalable, distributed cloud-edge architecture, based on `plan_triton_architecture.md`.

## Goal
Decouple the frontend/UI from the heavy GPU computing. The UI will live on serverless Google Cloud Run (CPU-only), while the heavy tensor operations (DINOv3, FAISS, heatmaps) will run on a local Edge GPU node. The two systems will securely communicate over a Tailscale VPN.

## Phase 1: Local Triton Backend Configuration
*Goal: Remove PyTorch/TRT coupling from our web server entirely by pushing model execution into an independent optimized C++ runtime (Triton).*
- [ ] Implement TRT exporter to support **Dynamic Batching** shape profiles instead of fixed batch sizes (e.g., Min [1, 3, 224, 224], Opt [4, 3, 224, 224], Max [8, 3, 224, 224]).
- [ ] Construct the Triton model repository directory structure (`/opt/anomaly_app/triton_models/dinov3_encoder/1/`).
- [ ] Create the Triton `config.pbtxt` defining the TensorRT backend, dynamic batching settings, and instance groups.
- [ ] Launch the local Triton NGC Docker container binding ports `8000` (HTTP) and `8001` (gRPC).

## Phase 2: Local Inference Gateway (FastAPI)
*Goal: Abstract Triton gRPC calls, FAISS memory banks, and OpenCV processing behind a clean local HTTP REST API.*
- [ ] Build `gateway/main.py` using FastAPI. 
- [ ] Migrate image Pre-processing (Morphology/Thresholding) logic into the gateway.
- [ ] Migrate Spatial Tiling logic into the gateway.
- [ ] Implement `tritonclient` gRPC loopback to send spatial image patches to the local Triton Server.
- [ ] Migrate FAISS index creation and similarity search into the gateway.
- [ ] Define gateway endpoints:
    - `POST /infer`: Receives base64 image + session ID → returns heatmap base64 + scores.
    - `POST /build_memory_bank`: Receives good images base64 → builds local FAISS index.
    - `POST /export_trt`: Compiles dynamic shape `.engine`.
    - `POST /sync_session`: Webhook for syncing session JSON from the cloud.

## Phase 3: Cloud Run Server Migration (GCP)
*Goal: Strip standard Flask `api_server.py` of all machine learning libraries, leaving only the UI hosting, storage, and networking layers.*
- [ ] Remove `torch`, `tensorrt`, and `faiss` from the frontend Cloud server. 
- [ ] Implement Google Cloud Storage hooks for saving and retrieving pipeline configurations (`session_id.json`).
- [ ] Update `/detect` and `/extract_features` routes in Flask to act purely as proxies. They will wrap uploaded images into Base64 payloads and POST them over HTTP to the local FastAPI Gateway.
- [ ] Update artifact generation: The Cloud proxy receives results and uploads `_overlay.png` results into Cloud Storage instead of the local filesystem.

## Phase 4: Network & Transport Bridge (Tailscale)
*Goal: Establish secure, fast 1-to-1 connectivity between the Serverless Cloud and the Local GPU.*
- [ ] Register Tailscale Auth Key in GCP Secret Manager.
- [ ] Provision a free-tier Google Compute Engine (`e2-micro`) instance to run Tailscale.
- [ ] Assign sub-routes so Cloud Run can push requests through the `e2-micro` VPN jump node straight to the Local FastAPI Gateway.
- [ ] Stress-test edge connectivity, ensuring round-trip image ingestion and Base64 mask return takes `<100ms` total. 

## Phase 5: Verification & Production Launch
- [ ] Conduct E2E parallel inference checks (uploading a ZIP folder of 20 images to Cloud Run should trigger async batched handling at the local Triton wrapper).
- [ ] Confirm Cloud VRAM usage remains strictly unaffected by parallel requests.
- [ ] Audit Grafana/Prometheus metrics directly exposed by Triton on `:8002` to confirm queue delays and inference timing match theoretical maximums. 
