# Stage 1 — Local Testing Scripts

Testing scripts for **Phase 1.1 (Thresholding)** and **Phase 1.2 (Background Subtraction)** preprocessing.  
Run these from the `python/` directory.

---

## File Overview

| Script | Phase | Purpose |
|--------|-------|---------|
| `test_thresholding.py` | 1.1 | Intensity thresholding → binary BG mask. Interactive + batch. |
| `test_bg_subtraction.py` | 1.2 | Template / averaged / MOG2 / KNN BG subtraction. Interactive + batch. |
| `test_preprocess_pipeline.py` | 1.1 + 1.2 | End-to-end: preprocessing → ONNX feature extraction → FAISS → heatmap. |

---

## Quick Start

```bash
# Run from python/ directory
cd /home/silicon/projects/anomaly_app/python
```

---

## Phase 1.1 — Thresholding

### Interactive (live trackbar preview)
```bash
python tests/test_thresholding.py \
    --interactive \
    --image /path/to/single_image.jpg
```
- Opens OpenCV window with sliders for **Channel**, **Thresh Min**, **Thresh Max**, **Morph Open**, **Morph Close**
- Press **S** to save current result, **Q** to quit
- Prints final params on exit — copy them into the batch command below

### Batch (process a folder)
```bash
python tests/test_thresholding.py \
    --batch \
    --input_dir dataset/good \
    --output_dir outputs/thresh_test \
    --channel gray \
    --thresh_min 30 --thresh_max 220 \
    --morph_open 3 --morph_close 5 \
    --save_masks
```

### Channel options
| Value | Meaning |
|-------|---------|
| `gray` | Grayscale intensity |
| `red` / `green` / `blue` | Individual BGR channels |
| `saturation` | HSV saturation |
| `value` | HSV value (brightness) |

### Output layout (per image)
```
┌─────────────┬─────────────┐
│  Original   │   Channel   │
├─────────────┼─────────────┤
│    Mask     │ Masked Result│
└─────────────┴─────────────┘
     [Param strip]
```

---

## Phase 1.2 — Averaged Background Subtraction

### Interactive (live trackbar preview)
```bash
python tests/test_bg_subtraction.py \
    --ref_dir ../dataset/metal_plate/bg \
    --input_dir ../dataset/metal_plate/test/scratches/ \
    --image ../dataset/metal_plate/test/scratches/001.png \
    --interactive
```
- Opens OpenCV window with sliders for **Diff Threshold**, **Morph Open**, **Morph Close**, **Fill Holes**, and **Min Comp %**
- Press **S** to save current result, **Q** to quit

### Batch (process a folder)
Builds a mean reference from the empty background images in `--ref_dir`, then applies subtraction to all images in `--input_dir`.
```bash
python tests/test_bg_subtraction.py \
    --ref_dir ../dataset/metal_plate/bg \
    --input_dir ../dataset/metal_plate/test/scratches/ \
    --diff_threshold 85 --morph_open 5 --morph_close 7 \
    --min_component_ratio 0.1 \
    --output_dir outputs/bg_sub_test --save_masks
```

---

## End-to-End Pipeline Test

Chains Stage 1 preprocessing → ONNX feature extraction → FAISS search → heatmap.

```bash
# Thresholding preprocess
python tests/test_preprocess_pipeline.py \
    --preprocess thresh \
    --onnx_path ../models/dinov3_vits16.onnx \
    --good_dir dataset/good \
    --test_dir dataset/test \
    --output_dir outputs/pipeline_test \
    --channel gray --thresh_min 30 --thresh_max 220 \
    --k 1 --sigma 4 --threshold 0.10

# Averaged BG subtraction preprocess
python tests/test_preprocess_pipeline.py \
    --preprocess average \
    --onnx_path ../models/dinov3_vits16.onnx \
    --good_dir dataset/good \
    --test_dir dataset/test \
    --output_dir outputs/pipeline_test \
    --diff_threshold 25 \
    --k 1 --sigma 4 --threshold 0.10

# Save the memory bank for reuse
python tests/test_preprocess_pipeline.py \
    --preprocess average \
    --onnx_path ../models/dinov3_vits16.onnx \
    --good_dir dataset/good \
    --test_dir dataset/test \
    --output_dir outputs/pipeline_test \
    --save_bank outputs/stage1_bank.npy
```

### Pipeline output (per test image)
```
┌────────────┬─────────────────┬───────────────┬──────────────┐
│  Original  │  Preprocessed   │ Anomaly Heatmap│   Overlay    │
│            │  (mask applied) │   (jet cmap)   │ score=0.1234 │
└────────────┴─────────────────┴───────────────┴──────────────┘
```

### Console output includes per-image score, OK/ANOMALY status, and latency:
```
  [  1/ 20] img_001.jpg                     score=0.0321  [OK]       lat=47.3ms
  [  2/ 20] img_002.jpg                     score=0.1892  [ANOMALY]  lat=45.1ms
```

---

## Key Parameters Reference

| Param | Script | Default | Meaning |
|-------|--------|---------|---------|
| `--channel` | thresh | `gray` | Channel to threshold |
| `--thresh_min` | thresh | `30` | Lower bound intensity |
| `--thresh_max` | thresh | `220` | Upper bound intensity |
| `--diff_threshold` | bg_sub / pipeline | `25.0` | Max pixel diff to call background |
| `--morph_open` | all | `3` | Morphological open kernel (denoise) |
| `--morph_close` | all | `5`/`7` | Morphological close kernel (fill holes) |
| `--k` | pipeline | `1` | FAISS k-NN neighbors |
| `--sigma` | pipeline | `4` | Heatmap Gaussian smoothing |
| `--threshold` | pipeline | `0.10` | Anomaly score τ cutoff |
| `--no_mask` | pipeline | off | Disable feature masking |

---

## Recommended Workflow

1. **Tune Phase 1.1** interactively on a good representative image:
   ```bash
   python tests/test_thresholding.py --interactive --image dataset/good/img_001.jpg
   ```
2. **Validate batch coverage** — aim for `coverage ≥ 70%` on all good images.
3. **Compare Phase 1.2** — run `average` mode and compare coverage to thresholding.
4. **Run end-to-end pipeline** with the best preprocessing params.
5. Propagate confirmed params into the main `api_server.py` pre-processing step.
