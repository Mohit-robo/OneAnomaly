# Anomaly Detection Web App — Project Reference

## Overview

A production-ready **web application** for detecting surface anomalies in industrial parts using Meta's **DINOv3 Vision Transformer** + **FAISS** for memory-bank similarity search. The system features a multi-stage stepper UI, live parameter inspection, hardware-accelerated **TensorRT** inference, and an interactive heatmap analysis workspace.

---

## Technology Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vanilla HTML5 / CSS3 / JavaScript |
| **Backend (ML API)** | Python / Flask / Flask-CORS |
| **Feature Extraction** | DINOv3 ViT-S/16 (PyTorch → TensorRT) |
| **Similarity Search** | FAISS (GPU-accelerated) |
| **Image Processing** | OpenCV, scipy, matplotlib |
| **Engine Files** | TensorRT `.engine` (ONNX intermediate) |

> **Note**: The app is served entirely by the Flask server on port **5000**. There is no separate Node.js proxy — `server.js` is not used in production.

---

## Architecture

```
anomaly_app/
├── public/                  # Frontend (served by Flask as static)
│   ├── index.html           # Main single-page application
│   ├── style.css            # Premium dark-mode design system
│   └── app.js               # All frontend logic (stepper, API calls, viz)
│
├── python/                  # Python ML backend
│   ├── api_server.py        # Flask REST API (all endpoints)
│   ├── feature_extractor.py # DINOv3 TRT / PyTorch feature extraction
│   ├── memory_bank.py       # MemoryBank + SpatialMemoryBank (FAISS)
│   ├── anomaly_detector.py  # Detection, heatmap generation, file saving
│   ├── dinov3/              # Meta DINOv3 repository (git submodule)
│   ├── src/                 # Preprocessing test scripts
│   │   ├── test_thresholding.py
│   │   ├── test_bg_subtraction.py
│   │   └── test_preprocess_pipeline.py
│   └── requirements.txt
│
├── models/
│   ├── dinov3_vits16_pretrain_lvd1689m.pth   # PyTorch weights
│   └── engine_files/        # TensorRT compiled engines (.engine, .onnx)
│       ├── dinov3_manual_int30-255_bs3.engine
│       └── ...
│
├── memory_banks/            # Saved FAISS index banks (.pkl)
├── uploads/                 # Temporarily stored uploads
│   └── temp_test/           # Test images before inference
├── outputs/
│   └── sessions/            # Per-run results
│       └── <session_id>/
│           ├── <stem>_source.png     # Clean input image
│           ├── <stem>_overlay.png    # Heatmap baked over source
│           ├── <stem>_heatmap.png    # Raw JET colormap heatmap
│           └── <stem>_annotated.png  # Source with score annotation
└── docs/
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check, model loaded status |
| GET | `/api/logs` | Stream backend logs to frontend terminal |
| POST | `/api/configure_preprocess` | Save preprocessing config (threshold/bg-sub) |
| POST | `/api/configure_spatial` | Save spatial region config + selected engine |
| POST | `/api/extract_features` | Run feature extraction on good images → build memory bank |
| POST | `/api/save_memory_bank` | Persist current FAISS index to disk |
| POST | `/api/load_memory_bank` | Load a named memory bank from disk |
| GET | `/api/list_memory_banks` | List all `.pkl` files in `memory_banks/` |
| GET | `/api/list_engines` | List all `.engine` files in `models/engine_files/` |
| POST | `/api/detect_anomalies` | Run inference; returns scores + session_id |
| GET | `/api/get_image/<session>/<file>` | Serve a result image from a session directory |
| GET | `/api/download_results/<session>` | Download session output as ZIP |
| POST | `/api/export_engine` | Stream TensorRT export (SSE) |
| POST | `/api/preview_preprocess` | Generate preprocessing preview grid |
| GET | `/api/spatial_config` | Return current spatial configuration |

---

## UI Workflow (4-Stage Stepper)

### Stage 1 — Preprocessing
- Choose **Intensity Thresholding** or **Averaged Background Subtraction**
- Tune morphological kernel sizes, threshold bounds
- Live preview grid shows masked result on sample images
- Save config → unlocks Stage 2

### Stage 2 — Spatial Region Definition
- Choose **Manual Draw** (interactive canvas), **2×2 Quadrant**, or **No Split**
- Select a TensorRT engine (batch size must match region count)
- **Export** button compiles ONNX → TRT engine with SSE progress stream
- Save regions → unlocks Stage 3

### Stage 3 — Memory Bank
- Upload good sample images (ZIP or multi-file)
- **Extract Features**: runs preprocessing → spatial tiling → DINOv3 → FAISS index
- **Skip & Use Existing**: load a previously built bank directly
- Save bank with a named identifier → unlocks Stage 4

### Stage 4 — Detection
- Upload single test image or batch ZIP
- Preprocessing config applied automatically (same as Stage 1)
- Performs FAISS similarity search → anomaly score per region
- **Detailed Analysis view**:
  - Left: `_source.png` (clean input reference)
  - Right: `_overlay.png` (JET heatmap baked over original — pre-computed by backend)
  - Heatmap alpha slider adjusts CSS `opacity` of overlay image (no canvas)
  - Centered anomaly summary card (score + NORMAL/ANOMALY badge)
  - Badge and card color are reactive to the threshold slider

---

## Key Design Decisions

### Why no canvas for the heatmap?
The original implementation used an HTML `<canvas>` to blend the source image and heatmap at runtime. This caused persistent black-box rendering due to race conditions (image `complete` state unreliable in setTimeout), potential canvas CORS taint, and `naturalWidth === 0` early-exit paths. The fix was to remove the canvas entirely and use a plain `<img>` pointing at the pre-baked `_overlay.png` from the backend.

### Why SpatialMemoryBank?
Standard FAISS indexes store all patch features flat. `SpatialMemoryBank` partitions them by region so that anomaly scores can be computed per spatial zone (e.g., top-left vs. bottom-right), enabling localized defect reporting.

### Why TensorRT batch size matters?
The spatial pipeline extracts features from `N` regions per image. The TensorRT engine batch size must match `N` exactly so all regions are processed in a single forward pass. Mismatched batch sizes cause shape errors or silent incorrect outputs.

### Preprocessing consistency
The same preprocessing config (mode, thresholds, morphological params) saved in Stage 1 is re-applied to test images in Stage 4. This ensures the decision boundary in feature space matches the training distribution (good images were also preprocessed).

---

## Global State (api_server.py)

```python
feature_extractor   # DINOv3 feature extractor (TRT or PyTorch)
memory_bank         # Loaded FAISS bank (SpatialMemoryBank)
anomaly_detector    # AnomalyDetector instance wrapping above
last_initialized_engine  # Tracks which engine is loaded (avoids redundant init)
preprocess_config   # Dict: mode, thresholds, morphology params
spatial_config      # Dict: spatial_mode, engine, regions, grid_ratios
averaged_bg_reference  # numpy array — loaded from disk on startup
```

---

## Session Output Files

For each detected image `<stem>.png` in session `<session_id>`:

| File | Description |
|---|---|
| `<stem>_source.png` | Original image after preprocessing, saved as clean reference |
| `<stem>_overlay.png` | Heatmap (JET colormap) blended over source at `alpha=0.5` |
| `<stem>_heatmap.png` | Raw JET colormap heatmap (pure, for potential future blending) |
| `<stem>_annotated.png` | Source with anomaly score text annotation |
| `torch_res_<stem>.png` | Matplotlib side-by-side plot (original + overlay) |
