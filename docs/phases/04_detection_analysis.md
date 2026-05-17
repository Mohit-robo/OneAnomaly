# Phase 4 — Detection & Analysis

## 1. Intent

Perform inference on suspect images to identify any surface defects. This phase compares test image features against the memory bank "normal" baseline. The results are presented in a high-fidelity workspace allowing for expert-level inspection of heatmaps and anomaly scores.

---

## 2. Steps to Perform

### Step 4.1 — Upload Test Images

**UI:** Stage 4 card → **"Upload Test Image"** file input
Supports: single PNG/JPG image, multiple files, or a ZIP folder of test samples.

The image(s) are uploaded to `uploads/temp_test/` before processing begins.

---

### Step 4.2 — Run Detection

**UI:** Click **"Run Detection"** button.

This triggers `POST /api/detect_anomalies` which executes the full inference pipeline:
1.  **Preprocessing**: Applies the exactly same `preprocess_config` saved in Phase 1 to the test image.
2.  **Spatial Tiling**: Splits the processed image into the regions defined in Phase 2.
3.  **Inference**:
    -   Extracts features from each region using the selected TensorRT engine.
    -   Performs FAISS similarity search against the loaded memory bank.
    -   Calculates the nearest-neighbor distance (anomaly score) for each patch.
4.  **Heatmap Generation**:
    -   Aggregates patch scores into a high-resolution heatmap.
    -   Applies Gaussian smoothing and JET colormapping.
    -   Bends the heatmap over the original image at α=0.5 to create `_overlay.png`.
5.  **Artifact Persistence**: Saves all results to `outputs/sessions/detect_<timestamp>/`.

---

### Step 4.3 — Inspect Results (Detailed Analysis)

After detection, the "Detailed Analysis" workspace automatically expands below Stage 4.

**The workspace contains two primary columns:**

| Left Column: Input Reference | Right Column: Anomaly Detection Result |
| :--- | :--- |
| Shows the preprocessed image (`_source.png`). | Shows the heatmap blended with the original image (`_overlay.png`). |

**Tuning the inspection:**
-   **Heatmap Alpha Slider**: Move the slider in Stage 4 to adjust the transparency of the red/blue overlay on the right panel.
-   **Anomaly Threshold Slider**: Move this slider to set the decision boundary.
    -   If `Max Score > Threshold`, the status badge turns **RED (ANOMALY)**.
    -   If `Max Score < Threshold`, the status badge turns **GREEN (NORMAL)**.
    -   The summary card below the inspection panels updates its color and text in real-time.

---

## 3. Verification Criteria

| Check | Pass Condition |
| --- | --- |
| Detection completes | Live Terminal shows "Detection complete" |
| Detailed analysis visible | Side-by-side images appear below Step 4 |
| Right image is layered | Heatmap overlay (red/blue) visible on top of original image |
| Alpha slider works | Moving slider changes transparency of the heatmap in real-time |
| Threshold slider works | Moving slider toggles badge between NORMAL and ANOMALY based on score |
| Artifacts present | `outputs/sessions/detect_<timestamp>/` contains `_source`, `_overlay`, `_heatmap`, and `_annotated` images |

---

## 4. UI Element Map

| What you want to do | UI Element | Location |
| :--- | :--- | :--- |
| Upload test image | "Upload Test Image" file input | Stage 4 card |
| Start inference | "Run Detection" button | Stage 4 card |
| Adjust overlay intensity | "Heatmap Alpha" slider | Stage 4 card (inspector section) |
| Change sensitivity | "Anomaly Threshold" slider | Stage 4 card (inspector section) |
| Download full batch results | "Download Results (ZIP)" button | Bottom of Stage 4 card (visible after detection) |
| View specific file name | Filename text | Header of Detailed Analysis card |

---

## 5. Error Reference

| Error / Symptom | Likely Cause | Fix |
| :--- | :--- | :--- |
| "No memory bank loaded" error | Step 3 was not completed or bank wasn't loaded | Go back to Stage 3, extract features or load a bank |
| Right detection panel is black | Race condition in legacy code or image not found | Fixed in v1.4.0 (uses `_overlay.png` directly). Refresh page and rerun if it persists |
| Heatmap doesn't match object | Misalignment in spatial region configuration | Check Stage 2 to ensure regions cover the object correctly |
| Batch detection only shows one result | App currently auto-activates the *first* result only | Check `outputs/sessions/` for the other result files |
| Anomaly score is extremely high for normal images | Preprocessing config mismatch | Ensure Stage 1 thresholds are same as used during bank build |
| Memory error on large ZIP upload | Server file limit or memory exhaustion | Upload in smaller batches or increase server memory |

**Log to check**: Flask terminal for score calculation logs (`Max score: 0.XXXX`).
**Network check**: POST `/api/detect_anomalies` → returns `results` array + `session_id`.

---

## 6. Phase Outputs

- **JSON Data**: Scores per region and max global score.
- **Session Artifacts** (`outputs/sessions/detect_<timestamp>/`):
  - `_source.png`: Reference preprocessed source image.
  - `_overlay.png`: Pre-baked heatmap overlay (JET) at 0.5 alpha.
  - `_heatmap.png`: Pure heatmap colormap.
  - `_annotated.png`: Image with score text burned in.
  - `torch_res_*.png`: Matplotlib debug side-by-side plot.
- **Interactive UI view**: Live reactive status board.
- **ZIP Export**: Consolidated archive of all artifacts.

---

## 7. Pipeline Finalized When

- [ ] All 4 stages show completed status.
- [ ] Multiple test images have been validated with the threshold slider.
- [ ] Final results ZIP has been downloaded and verified.
- [ ] Optimal threshold (τ) has been determined for the specific product line.
