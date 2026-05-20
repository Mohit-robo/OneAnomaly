# Execution Checkpoints (TODO)

## Stage 1: Preprocessing Pipeline

**Local test scripts** → `python/src/`

### ✅ Done
- [x] `test_thresholding.py` — intensity thresholding with interactive OpenCV trackbar preview + batch mode
- [x] `test_bg_subtraction.py` — averaged bg subtraction, interactive + batch mode
- [x] `test_preprocess_pipeline.py` — end-to-end: preprocessing → ONNX → FAISS → heatmap with per-image latency
- [x] UI panel for Threshold Controls (min/max intensity, morphological open/close)
- [x] UI panel for Background Subtraction Controls (diff threshold, morph kernels, fill holes, min component)
- [x] Upload reference blank images for averaged background subtraction
- [x] Live preview of threshold / bg-subtraction masked result
- [x] Image preprocessing applied before feature extraction in `api_server.py`

---

## Stage 2: Spatial Region Configuration

**Reference** → `python/spatial_anomaly_scripts/`

### ✅ Done
- [x] UI toggles for Region Selection Modes (Quadrant Grid / Manual Draw / No Split)
- [x] Mode A: 2×2 Auto Quadrant Split
- [x] Mode B: Interactive canvas drawing for up to N manual bounding boxes + labels
- [x] Mode C: Single-region (no splitting)
- [x] Preview section showing how sample images will be tiled
- [x] DINOv3 engine exporter: ONNX → TensorRT export with UI button
- [x] Multi-page stepper UI with stage locking
- [x] Persistent Right Inspector Panel (live parameter summary throughout session)
- [x] Stage navigation gating: blocks progress if engine not selected before export

---

## Stage 3: Memory Bank Creation

**Reference** → `python/spatial_anomaly_scripts/`

### ✅ Done
- [x] Upload good samples (ZIP or multi-file)
- [x] Feature extraction through TensorRT engine with spatial region support
- [x] `SpatialMemoryBank` integration for region-aware FAISS indexing
- [x] Save / load named memory banks to `memory_banks/`
- [x] **Skip Build & Use Existing** option — bypass extraction if bank already exists

### 🔜 Pending
- [ ] Show per-region feature count in the inspector during extraction
- [ ] Validate memory bank integrity before allowing proceed to detection

---

## Stage 4: Detection & Analysis

### ✅ Done
- [x] Single image upload for inference
- [x] Multi-file / ZIP batch upload
- [x] Preprocessing applied to test images before inference (matches training pipeline)
- [x] FAISS similarity search per region, anomaly score computation
- [x] Session-based output management (`outputs/sessions/<session_id>/`)
- [x] Backend saves: `_source.png`, `_overlay.png`, `_heatmap.png`, `_annotated.png` per image
- [x] **Detailed Analysis** side-by-side view:
    - Left: `_source.png` (clean input reference)
    - Right: `_overlay.png` (heatmap baked over original by backend)
- [x] Centered anomaly score summary card below inspection panels
- [x] Anomaly status badge (NORMAL / ANOMALY) reactive to threshold slider
- [x] Heatmap alpha slider controlling overlay opacity (CSS-based, no canvas)
- [x] Download results as ZIP

### ✅ Done (Phase 4 & 5)
- [x] Region-level score breakdown in the detailed analysis view
- [x] Paginated results when batch size > 1 (Stage 5 grid)
- [x] CSV metadata export alongside ZIP download
- [x] "Save session history" as JSON for reproducibility
- [x] Batch filter (All / Anomaly / Normal)
- [x] Enlarge / lightbox view for side-by-side original + heatmap overlays

---

## Stage 5: UNet + SAM Segmentation (Future)
- [ ] Integrate Meta SAM for clean binary mask generation from point/box prompts
- [ ] UNet training UI (epochs, LR, BS, loss) with MLflow progress tracking
- [ ] UNet export pipeline: UNet → ONNX → TensorRT

---

## Stage 6: Production Deployment (Triton + GCP)

**Reference** → `docs/plan_triton_architecture.md` / `docs/triton_implementation_plan.md`

### ✅ Done
- [x] **Infrastructure Setup Design** (Tailscale & GCP layout planned in docs)
- [x] **Triton Local Server**
  - [x] Reconfigure DINOv3 TRT Export to use dynamic batch profiles `[Min:1, Opt:4, Max:8]`
  - [x] Build `triton_models/dinov3_encoder` repo structure + `config.pbtxt`
  - [x] Deploy high-fidelity dynamic ONNX Runtime model (`dinov3_onnx`) to resolve degenerate TRT features compile bug
- [x] **Local Inference Gateway**
  - [x] Build robust `gateway/main.py`-based API using Uvicorn
  - [x] Refactor PyTorch OpenCV pipelines (Crop, Heatmap) over to gateway
  - [x] Move dynamic FAISS index memory bank creation over to gateway
  - [x] Bind gateway endpoints (`/infer`, `/build_memory_bank`, `/sync_session`)
  - [x] Fix `IsADirectoryError` when serializing standard single-region memory bank to file path (`bank.pkl`)
  - [x] Implement in-memory `MemoryBank` caching with `/sync_session` auto-invalidation to bypass FAISS rebuilding on every inference
- [x] **Cloud API Proxy**
  - [x] Strip out ML imports (PyTorch, TensorRT, FAISS, cv2 constraints) from existing `api_server.py`
  - [x] Refactor Web API calls to Proxy Image Arrays externally through gateway
  - [x] Unwanted scratch scripts removed
- [x] **Phase 5 Frontend Polish**
  - [x] Paged UI filtering and pagination works seamlessly
  - [x] Single-click Unified Downloads logic extracts `stacked_b64` (original + overlay)


### 🔜 Pending Next (Cloud Prep & Security)
- [ ] Set up Tailscale VPN on local IPC network
- [ ] Provision GCP server and set up Tailscale proxy
- [ ] Set up Google Cloud Storage bucket
- [ ] Bind `/session/` interactions to Read/Write JSON configs securely against Google Cloud Storage
- [ ] Guarantee Cloud Run utilizes purely CPU while inference achieves 100ms processing loopback target

---

## Known Limitations / Tech Debt
- [ ] Engine must be manually selected — no auto-detection of the most recent export
- [ ] Memory bank list not auto-refreshed after a new bank is saved (requires page reload or re-click)
