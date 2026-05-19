# Changelog

All notable changes to the **OneAnomaly** / **Anomaly Detection App** project.

---


## [v1.6.0] - 2026-05-18

### 🚀 Production Migration: Triton Distributed Architecture (Stage 6)
- **Triton Local Server Orchestration**: The core DINOv3 inference engine has been securely exported to TensorRT utilizing dynamic batching profiles (`[Min:1, Opt:4, Max:8]`). It runs via NVIDIA Triton (`nvcr.io/nvidia/tritonserver:24.08-py3`). Output dimensions corrected to exact spatial sizes `[196, 768]`.
- **FastAPI Edge Gateway**: A dedicated worker (`gateway/main.py`) acts under Uvicorn routing the loopback connection natively. The gateway now handles FAISS memory bank generation, raw OpenCV tiling arrays, and score extraction asynchronously to reduce computational payload over the outer edge limits.
- **Cloud Run Proxy (API Server Refactoring)**: Stripped the bulky ML logic (`PyTorch`, `TensorRT`, `FAISS`) directly away from the `api_server.py`. The backend codebase is now exceptionally clean, routing all inference (`/detect_anomalies`), bank creation (`/build_memory_bank`), and session config updates over purely HTTP REST payloads containing `base64` strings encoded sequentially.
- **Removed Unwanted Scratch Scripts**: Extraneous ML testing logic (`feature_extractor.py`, `dinov3_faiss.py`, `test_dinov3_pytorch.py`, `simple_inference_onnx.py`, `simple_inference_onnx_spatial.py`) removed fully optimizing for deployment readiness.
- **Phase 5 Feature Complete: Stacked Visualization**: When initiating a JSON / CSV / Image batch export on the frontend, the UI triggers a side-by-side array composition handled completely server-side via `np.hstack([image_bgr, overlay])` ensuring the user has a single cohesive `.zip` folder.

## [v1.5.0] - 2026-05-17

### 🚀 Phase 5: Confirmation & Final Inference (Batch)
- **Batch Processing Workspace**: Implemented a comprehensive Step 5 dashboard for high-volume inference, summary statistics, and final exports.
- **Side-by-Side Batch Grid**: Added a paginated (10 per page) results grid that renders both the clean source image and the anomaly heatmap side-by-side within each card.
- **Interactive Filtering**: Added All / Anomaly / Normal tab filters to instantly sort batch results.
- **Lightbox Inspection**: Clicking any result card now launches a fluid, full-screen lightbox displaying the high-resolution Matplotlib `torch_res_*.png` composite (Escape key supported).
- **Consolidated Export Pipeline**: The "Download Results" button now sequentially yields a triple-payload:
  1. **ZIP Archive**: Filtered to contain only the `torch_res_*.png` side-by-side composite images.
  2. **CSV Report**: Filename, Anomaly Score, and Status mapping based on the live UI threshold.
  3. **Session JSON**: Comprehensive state dump of all Pre-processing, Spatial, Memory Bank, and Detection parameters for 100% reproducibility.
- **Memory Bank Override Fix**: Resolved an issue where the memory bank dropdown remained empty by aligning the API payload key (`memory_banks`).



## [v1.4.0] - 2026-05-16

### 🔧 Critical Fix: Anomaly Heatmap Display (Black Canvas → img tag)
- **Root Cause**: The right-side "Anomaly Detection Result" panel used an HTML `<canvas>` for rendering. Multiple race conditions caused it to render as solid black:
  - `sourceImg.complete` check failed silently because `src` was set and `complete` reset atomically
  - `naturalWidth === 0` early-return path triggered due to image not fully materialized before draw
  - Potential canvas CORS taint when frontend and API differ in port (cross-origin without `crossOrigin="anonymous"` on `<img>`)
- **Fix**: Removed canvas entirely. Replaced with a plain `<img id="result-overlay-img">` that loads `_overlay.png` directly from the backend session directory. Heatmap alpha slider now controls CSS `opacity` on the element — no canvas, no race conditions, guaranteed render.
- **Dead Code Removal**: Removed the unused `imgPath` variable and the entire `data.results.forEach` tile-building loop from the detect handler. Detection now immediately triggers `showDetailedAnalysis` for the first result.
- **`renderBlendedCanvas` removed**: Entire function deleted. `showDetailedAnalysis` is now synchronous with a single direct `img.src` assignment.

### 🔧 Backend: Correct Heatmap Artifact Export
- `anomaly_detector.py`: `_heatmap.png` now saved as a **JET colormap image** (via `cv2.applyColorMap`) instead of a raw grayscale float array. Makes it usable for visual inspection and future frontend blending.

---

## [v1.3.0] - 2026-05-15

### 🚀 Expert-Level Detection Workspace
- **Removed Results Grid**: Eliminated the redundant "Detection Results" tile grid that was duplicating info already shown in the Detailed Analysis view.
- **Auto-Activate First Result**: After detection, `showDetailedAnalysis` is called immediately for the primary result — no user click required.
- **Centered Summary Card**: Anomaly summary card (score + NORMAL/ANOMALY verdict) is centered below the side-by-side inspection panels.
- **Reactive Threshold Badge**: Status badge (NORMAL/ANOMALY) and card border color update live as the threshold slider moves.
- **Memory Bank Skip**: Added "Skip Build & Use Existing" button in Stage 3 for users who have already built a bank and want to proceed directly to detection.

### 🐛 Bug Fixes
- **Export Modal Close**: Fixed the export completion popup not dismissing after closing — close button event listener was not re-attached after SSE stream completion.
- **Engine Validation**: Stage 3 now blocks proceeding to detection if no engine has been selected in Stage 2. Shows a warning toast.

---

## [v1.2.0] - 2026-05-15

### 🚀 Stage 2: Spatial Region Architecture
- **Multi-Page Stepper UI**: Sequential stage-locked workflow (Preprocessing → Spatial → Memory Bank → Detect).
- **Stage Locking**: Future stages locked until current stage config is saved.
- **Persistent Inspector Panel**: Right-hand live auditor showing all current session parameters.
- **Spatial Modes**: Implemented Quadrant Grid (2×2), Manual Region Drawing (canvas overlay), and No Split.
- **Export Button Redesign**: Blue primary, Black hover state with increased hit area.
- **Dropdown Dark Mode**: Engine select menu forced to `background: black` for theme consistency.

---

## [v1.1.0] - 2026-05-14

### 🚀 Stage 1: Preprocessing UI Overhaul
- **Premium Dark Mode UI**: Roboflow-inspired design with Inter/Geist typography, glassmorphic elements, border-glow interactions.
- **Live Interactive Previews**: Real-time preprocessing preview grid via API.
- **Preprocessing Modes**: Binary Thresholding and Averaged Background Subtraction with full parameter controls.
- **Session State**: Preprocessing config stored in Flask global state, persisted across requests within a session.
- **Directory Refactor**: `python/tests/` → `python/src/` for cleaner module organization.

---

## [v1.0.0] - 2026-01-05

### 🚀 Initial Release
- **Full Web Interface**: Replaced CLI workflow with responsive Web UI.
- **Live Terminal**: Real-time Python stdout streaming to browser via log polling.
- **Anomaly Detection Core**:
  - DINOv3 ViT-S/16 feature extraction (PyTorch + TensorRT)
  - FAISS IndexFlatIP for Cosine Similarity search
  - PatchCore-style memory bank with coreset subsampling
  - JET colormap heatmap with Gaussian smoothing
- **Bug Fixes**:
  - Button stability (`type="button"` to prevent form submission)
  - Matplotlib thread safety (`Agg` backend)
  - Recursive ZIP / nested folder support for uploads
  - DINOv3 local weights fallback before downloading
