# Phase 0 — Environment Setup & Server Launch

## 1. Intent

Validate that all dependencies, model weights, and TensorRT engines are in place, then bring the Flask API + frontend online. This is the pre-flight check before any pipeline work begins.

---

## 2. Steps to Perform

### Step 0.1 — Verify Python Environment

```bash
cd anomaly_app/python
pip install -r requirements.txt   # First-time or after dependency changes
python -c "import torch, faiss, tensorrt, cv2, flask; print('OK')"
```

**Expected:** prints `OK` with no import errors.

### Step 0.2 — Verify Model Weights

```bash
ls -lh ../models/dinov3_vits16_pretrain_lvd1689m.pth
```

**Expected:** file exists, size ~85–90 MB.  
**If missing:** see `docs/SETUP_DINOV3.md` → Section 1.

### Step 0.3 — Verify TensorRT Engine Files

```bash
ls ../models/engine_files/*.engine
```

**Expected:** at least one `.engine` file present.  
**If missing:** you will need to export one in Phase 2. The pipeline can still proceed using the PyTorch fallback but will be slower.

### Step 0.4 — Start the Flask Server

```bash
cd anomaly_app/python
python api_server.py
```

**Expected terminal output:**
```
============================================================
OneAnomaly API Server
============================================================
  POST /api/configure_preprocess
  POST /api/configure_spatial
  POST /api/extract_features
  ...
  GET  /api/get_image/<session_id>/<filename>
============================================================
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Step 0.5 — Open the UI

Open a browser and navigate to: **http://localhost:5000**

---

## 3. Verification Criteria

| Check | Pass Condition |
|---|---|
| Browser loads app | Page title "ONEANOMALY" visible, dark background |
| Stage 1 panel visible | "Preprocessing" stage is expanded and not locked |
| Stages 2–4 locked | Grey locked icon on Stage 2, 3, 4 badges |
| Right inspector | Shows "Session" section with parameter placeholders |
| Health check | `curl http://localhost:5000/health` returns `{"status":"healthy","models_loaded":false}` |

---

## 4. UI Interactions

| UI Element | Location | Action |
|---|---|---|
| Stage 1 header | Top of main content | Click to expand if collapsed |
| Live Terminal | Bottom of right inspector panel | Click to verify no startup errors |
| Console (browser DevTools) | F12 → Console | Check for `Using API_BASE: http://localhost:5000` log |

---

## 5. Error Reference

| Error | Likely Cause | Fix |
|---|---|---|
| `404 Not Found` on `localhost:5000` | Flask server not running | Run `python api_server.py` |
| `ImportError: No module named 'tensorrt'` | TRT not installed system-wide | Install TRT or remove TRT import if using PyTorch fallback |
| `ImportError: No module named 'faiss'` | faiss-gpu not installed | `pip install faiss-gpu` |
| Browser shows blank white page | Flask cannot find `../public/index.html` | Ensure you run from `python/` directory: `cd python && python api_server.py` |
| `RuntimeError: CUDA error: no kernel image` | GPU / CUDA version mismatch | Rebuild FAISS for correct CUDA version |

---

## 6. Phase Outputs

- Flask server running on `http://0.0.0.0:5000`
- Frontend accessible in browser
- Global state initialized: `feature_extractor=None`, `memory_bank=None`, `preprocess_config={"mode":"thresholding"}`
- Averaged background reference loaded from disk if `uploads/averaged_bg_reference.npy` exists

---

## 7. Ready for Phase 1 When

- [ ] Browser shows the app without errors
- [ ] Stage 1 (Preprocessing) is the only unlocked stage
- [ ] `curl http://localhost:5000/health` returns HTTP 200
- [ ] No red errors in browser DevTools console on page load
