# Anomaly App — Development Plan
  
**Stack:** DINOv3 + FAISS | Python/Flask backend | Node.js proxy | HTML/JS frontend

---

## Current State (from Repo)

The existing app implements a baseline DINOv3 anomaly detection pipeline:
- Memory bank creation from "good" ZIP uploads
- FAISS-based cosine similarity scoring
- Anomaly heatmap overlay visualization
- Batch detection with downloadable reports
- Dark-mode UI with live terminal and progress bar

The notes define **5 major phases** of new capability to build on top of this foundation.

---

## Phase 1 — Pre-processing Setup (Complete)

**Goal:** Let users define how background is removed / regions of interest are masked before features are extracted. This directly controls what DINOv3 sees.

### 1.1 Thresholding (Quick Option)
- Add a UI panel with threshold controls (min/max intensity, channel selection)
- Apply thresholding to generate a binary BG mask
- Show live preview of masked result

### 1.2 Template Matching / Advanced BG Subtraction
- Let user upload a reference "blank" template image, or if multiple uploaded create a average of all images as a reference image
- Run OpenCV template matching or MOG2/KNN BG subtraction
- Output foreground mask for downstream use
- Show live preview of masked result

### 1.3 UNet-based Segmentation (Primary Option, to be developed after the entire application is built) 
This is the production path. Full sub-pipeline:

1. **Provide BG images** — user uploads a set of background-only images via UI
2. **SAM + Point/ Box Prompt for mask creation**
   - Integrate Meta SAM (Segment Anything Model)
   - User clicks point(s)/ Draws box(es) on the image in UI to define the ROI
   - SAM generates a clean binary mask
3. **Train a UNet** on the (image, mask) pairs [refer to](https://github.com/Mohit-robo/edge_optimisation/blob/main/train_advanced.py)
   - Set training params on UI: epochs, LR, batch size, loss type or default values
   - Show training progress + loss curve on dashboard.
   - Use MLflow for tracking experiments
   - Use DVC for Data Version Control

4. **Check Results** — overlay predicted masks on validation images in the dashboard
5. **Export pipeline:**
   - UNet → ONNX
   - ONNX → TensorRT (for IPC/ Jetson deployment)
   - Store exported model path in session config
   - Load exported model in memory for inference during later phases and save it for future use


## Phase 2 — Spatial Region Selection (Complete)

**Goal:** Divide the image into sub-regions so each region gets its own memory bank/ or feature at 'n' index in memory bank and anomaly score, enabling spatially-aware detection.


### 2.1 Region Selection Modes (UI toggle)

**Mode A — 2×2 Grid (automatic)**
- Divide background subtracted image from Phase 1 into 4 equal quadrants
- Preview split in UI before confirming

**Mode B — Draw Regions (manual)**
- Canvas overlay where user draws up to 5 bounding polygons/rectangles
- Label each region (e.g., "cap thread zone", "label area", "body")
- These become partial spatial regions for independent feature extraction

**Mode C — No Splitting**
- Run DINOv3 on the whole image as a single patch
- Useful for small or simple parts

### 2.2 Preview & Verification
- After user defines regions, show a preview of how a sample image will be tiled
- User confirms before proceeding.
- Display region count, patch resolution, and expected feature vector size

### 2.3 DINOv3 Export for Spatial Batch Size

- **Condition** User mentions to export model for spatial batch size. Else ask for previosuly saved model
- If user selects previously saved model, skip this step
- Determine batch size (BS) = number of spatial regions
- Export DINOv3 with this fixed BS:
  - DINOv3 → ONNX (image size = 224×224 per patch)
  - ONNX → TensorRT
- Store: `model_bs`, `image_size`, `trt_path` in session config
- Save: User names the model and save it for future use
---

## Phase 3 — Memory Bank Creation

**Goal:** Extract DINOv3 features from verified "good" samples using the pre-processing and spatial config from Phases 1 & 2, and build the FAISS index. Use same model converted to TensorRT from Phase 2.

### Step-by-step flow:

1. **Pre-processing from Phase 1** is applied first (whichever mode user selected and confirmed)
2. **Spatial region split from Phase 2** is applied (or skipped if Mode C)

   Branch point:
   - **Spatial regions selected (Yes):** use params from Phase 2 (BS=n, region coords)
   - **No spatial regions:** run on 1 single image pass, BS=1

3. **Upload Good Samples**
   - User selects number of shots (K good images)
   - Upload via ZIP or folder picker

4. **Feature Extraction**
   - Load TRT DINOv3 model (BS=n from Phase 2)
   - Run inference on each image × each spatial region
   - Memory bank size = `BS × num_images × feature_dim`

5. **Build FAISS Index**
   - L2-normalize all feature vectors
   - Index with FAISS (flat or IVF depending on bank size)
   - Save index + metadata (region coords, normalization params) to disk

6. **Name & Save Memory Bank**
   - User names the bank (e.g., `bottle_cap_v2`)
   - All config (pre-proc params, spatial params, model paths) saved alongside index

---

## Phase 4 — Validate Results

**Goal:** Test the memory bank on a small set of images (≤20) with adjustable thresholds to tune detection sensitivity before final inference.

### Upload & Inference
1. User uploads images or ZIP (restricted to 20 images for validation speed)
2. Apply Phase 1 pre-processing (loaded from saved session)
3. Apply Phase 2 spatial tiling (loaded from saved session)
4. Model already loaded in memory — no reload needed. If selecting already present memory bank load the model from phase 2. Dropdown to select memory bank and model from phase 2.
5. Run FAISS similarity search per region → compute anomaly scores
6. Display Latency for inference

### UI Controls (interactive tuning)

| Parameter | UI Element | Effect |
|-----------|-----------|--------|
| **Threshold (τ)** | Slider | Binary OK/Not-OK cutoff on anomaly score |
| **k** (nearest neighbors) | Slider / input | Number of FAISS neighbors for score aggregation |
| **Sigma (σ)** | Slider | Gaussian smoothing on raw score map before heatmap |
| **v_max** | Slider | Fixed maximum for heatmap scaling |

### Visualization
- Anomaly heatmap overlay (red/blue, same as current)
- **Dynamic annotation:** user can adjust threshold slider and region annotations update live (no re-inference needed — recompute from cached raw scores)
- Side-by-side: original vs. heatmap + overlay

---

## Phase 5 — Confirmation & Final Inference

**Goal:** Lock in all parameters, run on a large test set, and export results.

### 5.1 Parameter Summary Screen
- Display all selected params consolidated from all phases:
  - Phase 1: pre-proc mode + params
  - Phase 2: region config, model BS, image size
  - Phase 3: memory bank name, bank size, k
  - Phase 4: τ (threshold), σ, n_max
- User can override any value manually at this step before final run

### 5.2 Upload Test Set & Run
- User uploads a larger test set (ZIP, no 20-image limit)
- Confirm → trigger full inference pipeline

**Branch on results:**
- **Good images** → flagged as conforming (score < τ)
- **Not OK images** → flagged as anomalous; user can optionally upload `gt_mask` per image for IoU-based accuracy measurement

### 5.3 Results Display
- Show a subset of Not-OK detections in UI (paginated, configurable)
- For each: anomaly score, heatmap overlay, region-level breakdown
- If `gt_mask` provided: compute pixel-level IoU, show per-image accuracy

### 5.4 Export & Session Management
- **Download all results:** ZIP with annotated images + CSV (filename, score, region scores, OK/Not-OK label)
- **Save session history:** full session config snapshot (all params + model paths + FAISS index reference) saved as JSON — enables reproducibility and reloading

---

## Implementation Sequence (Recommended Order)

| Priority | Phase | Why |
|----------|-------|-----|
| 1 | Phase 1.1 and Phase 1.2 | Easy Start |
| 2 | Phase 2 — Spatial Regions (Grid + Preview) | Unlocks spatially-aware memory banks; high ROI |
| 3 | Phase 3 — Memory Bank with Spatial Tiling | Core pipeline improvement over current flat extraction |
| 4 | Phase 4 — Slider-based Threshold Tuning | Immediate UX value; builds on existing heatmap display |
| 5 | Phase 5 — Session Save + Download All | Productionizes the workflow |
| 6 | Phase 1.3 — UNet + SAM Segmentation | Most complex; build after spatial + tuning are stable |

---

## Key Engineering Notes

- **TRT batch size must be fixed at export time.** If user switches between Grid (BS=4) and Single (BS=1), different TRT engines are needed — cache both or force re-export.
- **Raw score caching is critical** for Phase 4 UX. Store per-image, per-region raw FAISS distances so threshold/sigma/n_max changes are instant (no re-inference).
- **SAM integration** (Phase 1.3) can be run as a separate Flask endpoint — heavy model, load on demand.
- **Session JSON** (Phase 5.4) should be designed early as a unified config schema — it will be referenced across all phases.
- Consider migrating from Vanilla JS to a lightweight reactive framework (Alpine.js or HTMX) to handle the multi-phase UI state without a full React rewrite.



## Future Scope

- **Phase 2** - Add model export image size selection.
- **Phase 3** - Create seperate memory bank for each spatial region. Select the size of memory bank for each region.
