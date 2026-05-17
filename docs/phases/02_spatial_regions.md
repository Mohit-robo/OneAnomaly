# Phase 2 — Spatial Region Definition & Engine Export

## 1. Intent

Define **where** on the image to look for anomalies by splitting it into spatial inspection zones. The DINOv3 engine will process each zone as a separate tensor in a single batched forward pass. This enables localized defect reporting (e.g., "top-left region is anomalous, bottom-right is normal").

Additionally this phase handles **TensorRT engine selection or export** — the engine batch size must exactly match the number of configured regions.

Three spatial modes are available:

| Mode | Description | Batch Size |
|---|---|---|
| **Manual Draw** | User draws N bounding boxes directly on a sample image | N boxes drawn |
| **Quadrant Grid** | Automatic 2×2 equal split of the full image | 4 |
| **No Split** | Full image processed as one region | 1 |

---

## 2. Steps to Perform

### Step 2.1 — Upload a Sample Reference Image

**UI:** Stage 2 card → **"Upload Reference Image"** file input  
This image is used as the canvas background for drawing region boxes. Use a representative object image (same resolution as your dataset).

---

### Step 2.2 — Select a Spatial Mode

**UI:** Radio buttons — "Manual Draw" / "Quadrant Grid" / "No Split"

#### Mode A: Manual Draw

1. After uploading the reference image, a canvas overlay appears on the image.
2. **Click and drag** to draw a bounding box around your first inspection zone.
3. Box appears with a label (e.g., "Region 1").
4. Repeat for additional regions (up to the engine's max batch size).
5. To remove a box: click the **×** on the box label or the **"Clear Regions"** button.

> **Rule:** The number of boxes drawn = the batch size your TRT engine must be compiled for.

#### Mode B: Quadrant Grid

No action needed — the system automatically divides the image into 4 equal quadrants. Batch size = 4.

#### Mode C: No Split

No action needed. The full image is one region. Batch size = 1.

---

### Step 2.3 — Select a TensorRT Engine

**UI:** Stage 2 card → **"Select Engine"** dropdown

The dropdown lists all `.engine` files found in `models/engine_files/`. Engine filenames encode their batch size (e.g., `dinov3_manual_int30-255_bs3.engine` → batch size 3).

**Select the engine whose `_bs<N>` matches your region count.**

> **Example:** 3 manual regions → select an engine with `_bs3`.

Available engines (as of current repo):

| Engine File | Batch Size | Preprocessing Mode |
|---|---|---|
| `dinov3_manual_int30-255_bs3.engine` | 3 | Thresholding 30–255 |
| `dinov3_manual_int30-255_bs2.engine` | 2 | Thresholding 30–255 |
| `dinov3_manual_int30-220_bs3.engine` | 3 | Thresholding 30–220 |
| `dinov3_manual_int30-220_bs1.engine` | 1 | Thresholding 30–220 |
| `dinov3_manual_avgD85_bs4.engine` | 4 | Avg BG diff=85 |
| `dinov3_manual_int18-255_bs1.engine` | 1 | Broad threshold |

---

### Step 2.4 — (Optional) Export a New TensorRT Engine

If no existing engine matches your batch size, export one:

**UI:** Click **"Export"** button in Stage 2.

This triggers `POST /api/export_trt` which:
1. Loads the DINOv3 PyTorch model
2. Exports to ONNX (batch size = region count)
3. Compiles ONNX → TRT engine using `trtexec`
4. Saves to `models/engine_files/`

The export streams progress via **Server-Sent Events (SSE)** — a live log window opens in the modal overlay. Export time is typically **5–20 minutes** depending on GPU.

**When complete:** The modal shows "Export Complete" + a **Close** button. Click Close. The new engine appears in the dropdown. Select it.

> ⚠️ Do not close the browser tab during export — the SSE connection will be lost.

---

### Step 2.5 — Save Spatial Configuration

**UI:** Click **"Save Regions & Continue"** button at the bottom of Stage 2.

This calls `POST /api/configure_spatial` with:
- Spatial mode
- Region coordinates (for manual mode)
- Selected engine filename
- Grid ratio (if quadrant mode)

**Expected:** Toast "Spatial config saved. Stage 3 unlocked."

---

## 3. Verification Criteria

| Check | Pass Condition |
|---|---|
| Reference image visible as canvas background | Image renders inside the draw area |
| Drawn boxes visible on image | Dashed colored boxes with region labels |
| Engine dropdown populated | At least one `.engine` file listed |
| Batch size matches region count | Engine `_bsN` number = number of drawn boxes |
| Save toast appears | "Spatial config saved" green toast |
| Stage 3 unlocked | Stage 3 badge no longer grey |
| Inspector updated | Right panel shows mode, engine name, region count |

---

## 4. UI Element Map

| What you want to do | UI Element | Location |
|---|---|---|
| Upload reference image | "Upload Reference Image" file input | Top of Stage 2 card |
| Switch spatial mode | Radio buttons | Below file input |
| Draw a region box | Click + drag on the image canvas | Canvas overlay on image |
| Remove all boxes | "Clear Regions" button | Below canvas |
| Select engine | "Select Engine" dropdown | Lower section of Stage 2 |
| Export new engine | "Export" button | Next to engine dropdown |
| Watch export progress | SSE log modal | Auto-opens on Export click |
| Close export modal | "Close" button in modal | Appears when export finishes |
| Save and continue | "Save Regions & Continue" button | Bottom of Stage 2 card |

---

## 5. Error Reference

| Error / Symptom | Likely Cause | Fix |
|---|---|---|
| Engine dropdown empty | No `.engine` files in `models/engine_files/` | Export an engine or place one manually |
| Export hangs > 30 min | TRT compilation OOM or CUDA error | Check Flask terminal for `trtexec` stderr |
| Export modal does not close | Close button event not reattached | Refresh page; engine file was likely still saved — check `ls models/engine_files/` |
| "Batch size mismatch" at inference time | Wrong engine selected for region count | Re-select engine in Stage 2 and re-save |
| Canvas does not appear for drawing | Reference image upload failed | Re-upload; confirm image is PNG/JPG |
| Stage 3 not unlocking after save | API error on `/api/configure_spatial` | F12 → Network → check response; Flask terminal for stack trace |
| Quadrant mode with `_bs3` engine | Quadrant always produces 4 regions | Use a `_bs4` engine with quadrant mode |

**Log to check:** Flask terminal → `Spatial config saved:` + dict  
**CLI verify:** `ls -lh models/engine_files/` to confirm new engine file is present.

---

## 6. Phase Outputs

- `spatial_config` global updated in Flask:
  ```python
  {
    "spatial_mode": "manual",          # or "quadrant", "none"
    "engine": "dinov3_manual_int30-255_bs3.engine",
    "regions": [[x, y, w, h], ...],   # manual mode only
    "grid_ratios": {"x": 50, "y": 50} # quadrant mode
  }
  ```
- (If exported) New `.engine` + `.onnx` files in `models/engine_files/`
- Stage 3 unlocked in the UI

---

## 7. Ready for Phase 3 When

- [ ] Regions are defined (boxes visible or mode selected)
- [ ] An engine is selected and its `_bsN` matches the region count
- [ ] "Spatial config saved" toast appeared
- [ ] Stage 3 is unlocked (not grey)
- [ ] Right inspector shows engine name and region info
- [ ] Flask terminal shows no errors from `/api/configure_spatial`
