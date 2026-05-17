# Phase 1 — Preprocessing Configuration

## 1. Intent

Define how raw camera images are cleaned before feature extraction. The goal is to isolate the object-of-interest from background noise (conveyor belts, fixtures, shadows) so that DINOv3 only sees the relevant surface. This config is applied **identically** at both training time (Phase 3) and inference time (Phase 4) — consistency here is critical.

Two modes are available:

| Mode | When to Use |
|---|---|
| **Intensity Thresholding** | Static, bright background. Object has distinctly different intensity from background. |
| **Averaged Background Subtraction** | Dynamic or textured backgrounds. You have access to reference "empty scene" images. |

---

## 2. Steps to Perform

### Step 1.1 — Select Mode

**UI:** Stage 1 card → Radio buttons at the top  
Choose either **"Intensity Thresholding"** or **"Background Subtraction"**.

The corresponding parameter panel slides in below the mode selector.

---

### Mode A: Intensity Thresholding

**Parameters to configure:**

| Slider | What it Controls | Recommended Starting Point |
|---|---|---|
| Min Intensity | Lower bound — pixels below this value are masked out | 30 |
| Max Intensity | Upper bound — pixels above are masked out | 255 |
| Morph Open | Removes small noise blobs (erosion then dilation) | 3–5 |
| Morph Close | Fills small holes in the mask (dilation then erosion) | 5–7 |

**Step 1.2A — Upload a sample image for preview**  
UI: Click **"Upload Preview Image"** → select a representative good image.  
The system sends it to `POST /api/preview_mask` and renders a 2-panel grid:
- Left: original image
- Right: masked result (white = kept, black = removed)

**Step 1.2B — Tune sliders**  
Adjust Min/Max and morphological kernels while watching the live preview update. Target: the object fills the white region with minimal noise and no missing chunks.

---

### Mode B: Averaged Background Subtraction

**Step 1.2B-i — Upload background reference images**  
UI: Click **"Upload Background References"** → select 5–20 images of the **empty scene** (no object, same lighting/angle). As a ZIP or multi-file.

The backend:
1. Loads all reference images
2. Averages them pixel-wise into a mean background
3. Saves the result to `uploads/averaged_bg_reference.npy`
4. This reference persists across server restarts

**Parameters to configure:**

| Parameter | What it Controls | Recommended Start |
|---|---|---|
| Diff Threshold | Pixel difference threshold to call foreground | 85 |
| Morph Open | Noise removal kernel | 5 |
| Morph Close | Hole filling kernel | 7 |
| Fill Holes | Fill enclosed regions in mask | ON |
| Min Component Ratio | Discard blobs smaller than X% of frame | 0.10 |

**Step 1.2B-ii — Preview with a sample image**  
Upload a sample object image. The preview shows how well the object is isolated from the background.

---

### Step 1.3 — Save Configuration

**UI:** Click **"Save Configuration & Continue"** at the bottom of Stage 1.

This calls `POST /api/configure_preprocess` with the current mode and parameters. The server stores them in the `preprocess_config` global.

**Expected response:** Toast notification "Preprocessing config saved. Stage 2 unlocked."

---

## 3. Verification Criteria

| Check | Pass Condition |
|---|---|
| Preview renders | Two-panel grid appears after uploading sample image |
| Object correctly isolated | White mask covers full object, minimal background leakage |
| No over-masking | Object itself is not partially blacked out |
| Save toast appears | Green toast: "Preprocessing config saved" |
| Stage 2 unlocks | Stage 2 badge changes from locked (grey) to active |
| Inspector updated | Right panel shows current Mode + parameter values |

**Quantitative check:** Visually confirm the masked image retains >85% of the object surface and <5% background pixels appear white.

---

## 4. UI Element Map

| What you want to do | UI Element | Location |
|---|---|---|
| Switch mode | Radio buttons: "Intensity Thresholding" / "Background Subtraction" | Top of Stage 1 card |
| Upload preview image | "Upload Preview Image" file input | Below mode selector |
| Adjust threshold bounds | "Min Intensity" / "Max Intensity" sliders | Thresholding panel |
| Adjust morphology | "Morph Open" / "Morph Close" sliders | Both panels |
| Upload BG references | "Upload Background References" file input | BG Subtraction panel |
| Refresh preview | Happens automatically on slider change | Preview grid area |
| Save and continue | "Save Configuration & Continue" button | Bottom of Stage 1 card |

---

## 5. Error Reference

| Error / Symptom | Likely Cause | Fix |
|---|---|---|
| Preview stays blank after upload | Image format not supported | Use PNG or JPG; avoid TIFF |
| Preview shows all-black mask | Min threshold too high, masking everything | Lower Min Intensity to 18–30 |
| Preview shows all-white mask | Thresholds too permissive | Raise Min or lower Max |
| BG reference upload fails | ZIP too large or corrupt | Try uploading as individual files |
| "No config provided" error in logs | Stage 1 save button form data missing | Reload page and retry |
| Stage 2 does not unlock after save | Server returned error on `/api/configure_preprocess` | Check Flask terminal for stack trace |
| Averaged reference produces poor mask | Lighting changed between BG capture and test | Recapture BG references under same lighting |

**Log to check:** Flask terminal → look for `Preprocessing config saved:` followed by the config dict.  
**Browser check:** F12 → Network → POST `/api/configure_preprocess` → Response Body should be `{"success": true}`.

---

## 6. Phase Outputs

- `preprocess_config` dict stored in Flask global state:
  ```python
  {
    "mode": "thresholding",           # or "average"
    "min_intensity": 30,
    "max_intensity": 255,
    "morph_open": 3,
    "morph_close": 5,
    "diff_threshold": 85,             # BG subtraction only
    "fill_holes": True,               # BG subtraction only
    "min_component_ratio": 0.10       # BG subtraction only
  }
  ```
- (If BG mode) `uploads/averaged_bg_reference.npy` saved to disk
- Stage 2 unlocked in the UI

---

## 7. Ready for Phase 2 When

- [ ] Live preview mask shows clean object isolation
- [ ] Green "saved" toast appeared after clicking Save
- [ ] Stage 2 badge is no longer locked (grey → colored)
- [ ] Right inspector panel shows the selected mode and parameter values
- [ ] Flask terminal shows no errors from `/api/configure_preprocess`
