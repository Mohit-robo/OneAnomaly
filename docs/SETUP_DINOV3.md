# DINOv3 Setup & TensorRT Engine Guide

## Overview

This project uses **Meta DINOv3 ViT-S/16** as the backbone feature extractor. Inference runs via **TensorRT** for maximum GPU throughput on Jetson / NVIDIA workstations.

---

## 1. Model Weights

### Current Status
The weights file is already present in the repo at:
```
models/dinov3_vits16_pretrain_lvd1689m.pth
```

### Re-Downloading (if needed)
DINOv3 weights require Meta access approval:
1. Visit: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
2. Submit the access form
3. Download `dinov3_vits16_pretrain_lvd1689m.pth` from the email link
4. Place at: `anomaly_app/models/dinov3_vits16_pretrain_lvd1689m.pth`

```bash
# Once you have the link:
wget -O models/dinov3_vits16_pretrain_lvd1689m.pth "YOUR_LINK_FROM_EMAIL"
```

---

## 2. DINOv3 Repository

The DINOv3 codebase (needed for model definition) is included as a submodule at `python/dinov3/`.

```bash
# If cloning fresh:
git submodule update --init --recursive
```

If the submodule is already present, no action needed.

---

## 3. TensorRT Engine Files

### Current Engine Files (in `models/engine_files/`)

| Engine | Batch Size | Notes |
|---|---|---|
| `dinov3_manual_int30-255_bs3.engine` | 3 | 3-region manual, threshold 30-255 ✅ Default |
| `dinov3_manual_int30-255_bs2.engine` | 2 | 2-region manual |
| `dinov3_manual_int30-220_bs3.engine` | 3 | 3-region, tighter threshold |
| `dinov3_manual_int30-220_bs1.engine` | 1 | Single-region |
| `dinov3_manual_avgD85_bs4.engine` | 4 | Average-diff bg-sub preprocess |
| `dinov3_manual_int18-255_bs1.engine` | 1 | Broad threshold |

**Batch size must match the number of spatial regions configured in Stage 2.**

### Exporting a New Engine (via UI)
1. In Stage 2, configure your spatial regions
2. Select batch size matching region count
3. Click **Export** — the UI streams TRT compilation logs in real time
4. Engine is saved to `models/engine_files/` automatically

### Exporting a New Engine (CLI)
```bash
cd python
python -c "
from feature_extractor import FeatureExtractor
fe = FeatureExtractor(backend='pytorch')
fe.export_to_onnx('my_model.onnx', batch_size=3)
"
# Then compile with trtexec:
trtexec --onnx=my_model.onnx \
        --saveEngine=models/engine_files/my_engine_bs3.engine \
        --fp16
```

---

## 4. Python Requirements

```bash
cd python
pip install -r requirements.txt
```

Key dependencies:
```
torch>=2.0
torchvision
faiss-gpu
opencv-python
scipy
matplotlib
flask
flask-cors
numpy
Pillow
tensorrt  # Must match system TRT version
```

> **FAISS-GPU Note**: `faiss-gpu` requires CUDA. For CPU-only testing use `faiss-cpu`.

---

## 5. Running the Server

```bash
cd python
python api_server.py
```
In another terminal run 

```bash
cd public
python3 -m http.server 8000
```

Server starts on **http://0.0.0.0:8000**. The frontend is served from `../public/`.

---

## 6. Feature Extractor Backends

`feature_extractor.py` supports two backends:

| Backend | When Used |
|---|---|
| `tensorrt` | Production — requires `.engine` file in `engine_files/` |
| `pytorch` | Fallback — uses `dinov3_vits16_pretrain_lvd1689m.pth` |

Backend is selected based on `spatial_config['engine']`. If no engine is selected or the file is missing, it falls back to PyTorch automatically.

---

## 7. Troubleshooting

### Engine batch size mismatch
**Symptom**: Silent wrong outputs or shape errors during detection  
**Fix**: Ensure the engine batch size matches the number of configured spatial regions exactly

### `CUDA out of memory`
**Fix**: Reduce image resolution, or use a smaller batch size engine

### `dinov3` repo not found
**Fix**: `git submodule update --init --recursive` from the project root

### `libcuda.so not found` / TRT init failure
**Fix**: Ensure TensorRT is installed system-wide and `LD_LIBRARY_PATH` includes TRT libs:
```bash
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### Memory bank incompatible
**Symptom**: FAISS dimension mismatch error when loading a bank  
**Fix**: The bank was built with a different engine (different feature dimension). Rebuild the bank with the current engine.
