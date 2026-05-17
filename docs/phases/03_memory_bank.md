# Phase 3 — Memory Bank Creation

## 1. Intent

Build the **reference feature database** — a FAISS index populated with DINOv3 patch features extracted from known-good (defect-free) images. This is the "normal" baseline the system compares every test image against in Phase 4.

Two paths are available:

| Path | When to Use |
|---|---|
| **Build New** | First run for a product type, or after changing engine/preprocessing |
| **Skip & Use Existing** | Bank already built and saved; want to go straight to detection |

> **Critical:** The bank must be built using the **same engine and preprocessing config** that will be used during detection. A mismatch in feature dimensions or input distribution will cause poor anomaly scores.

---

## 2. Steps to Perform

---

### Path A: Build a New Memory Bank

#### Step 3.1 — Prepare Good Sample Images

Collect **defect-free** images of the product under consistent lighting, angle, and framing. Minimum: 20–50 images. More is better (100–500 is typical for industrial use).

Images should represent the **full range of normal variation** (e.g., slight rotations, minor color variance from different batches) but contain **no visible defects**.

#### Step 3.2 — Upload Good Samples

**UI:** Stage 3 card → **"Upload Good Samples"** file input  
Accept: ZIP file or multi-file selection (PNG, JPG, BMP).

The files are stored temporarily in `uploads/` before extraction begins.

#### Step 3.3 — Start Feature Extraction

**UI:** Click **"Extract Features"** button.

This calls `POST /api/extract_features` which:
1. Applies `preprocess_config` (masks/thresholds) to each image
2. Splits each image into regions defined by `spatial_config`
3. Feeds each region crop through the DINOv3 TRT engine
4. Collects patch-level feature vectors
5. Builds a FAISS `IndexFlatIP` (cosine similarity) from all features
6. Stores the index in the `SpatialMemoryBank` in memory

The Live Terminal in the right inspector panel streams progress:
```
Processing image 1/120: img_001.jpg
Processing image 2/120: img_002.jpg
...
Feature extraction complete. Total features: 45600
FAISS index built successfully.
```

**Expected duration:** ~2–5 seconds per image on TRT; longer on PyTorch fallback.

#### Step 3.4 — Save the Memory Bank

**UI:** Enter a name in the **"Bank Name"** text field (e.g., `fabric_v1`, `metal_cap_line_A`)  
Then click **"Save Memory Bank"**.

This calls `POST /api/save_memory_bank` which serializes the FAISS index to `memory_banks/<bank_name>.pkl`.

**Expected:** Toast "Memory bank saved: fabric_v1"

---

### Path B: Skip & Use Existing Bank

**UI:** Click **"Skip Build & Use Existing"** button in Stage 3.

A dropdown appears listing all `.pkl` files in `memory_banks/`.

1. Select the desired bank from the dropdown.
2. Click **"Load Bank"**.

This calls `POST /api/load_memory_bank` which deserializes the `.pkl` into the `memory_bank` global.

**Warning:** If the bank was built with a different engine (different feature dimension), loading will fail with a FAISS dimension mismatch. See Error Reference below.

---

## 3. Verification Criteria

| Check | Pass Condition |
|---|---|
| Upload accepted | File input shows filename; no error toast |
| Extraction starts | Live Terminal shows "Processing image 1/N" |
| Extraction completes | Terminal shows "FAISS index built successfully" |
| Save succeeds | Green toast "Memory bank saved: <name>" |
| `.pkl` file exists on disk | `ls memory_banks/*.pkl` shows the new file |
| Stage 4 unlocks | Stage 4 badge changes from locked to active |
| Load succeeds (Path B) | Green toast "Memory bank loaded: <name>" |

**Quantitative check:** Feature count in the terminal should be approximately:  
`N_images × N_regions × patches_per_region`  
e.g., 100 images × 3 regions × 152 patches = 45,600 features

---

## 4. UI Element Map

| What you want to do | UI Element | Location |
|---|---|---|
| Upload good samples | "Upload Good Samples" file input | Top of Stage 3 card |
| Start feature extraction | "Extract Features" button | Stage 3 card |
| Monitor progress | Live Terminal | Right inspector panel, bottom |
| Name the bank | "Bank Name" text field | Stage 3 card, below extract button |
| Save the bank | "Save Memory Bank" button | Stage 3 card |
| Load an existing bank | "Skip Build & Use Existing" button | Stage 3 card |
| Select from saved banks | Dropdown (appears after Skip button click) | Stage 3 card |
| Load selected bank | "Load Bank" button | Next to dropdown |

---

## 5. Error Reference

| Error / Symptom | Likely Cause | Fix |
|---|---|---|
| "No memory bank loaded" on detection | Bank not saved or loaded after extraction | Repeat Step 3.4 or Step Path B |
| FAISS dimension mismatch on load | Bank built with different engine | Rebuild bank with current engine |
| Feature extraction crashes immediately | Engine not initialized | Check Stage 2 — engine must be selected and saved first |
| Extraction very slow (>30s/image) | Falling back to PyTorch (no TRT) | Verify engine file exists and `spatial_config['engine']` is set |
| "No images found" error | ZIP structure not recognized | Ensure images are at root of ZIP or in one subfolder level |
| Memory bank file not in dropdown | File not in `memory_banks/` or wrong suffix | Confirm file is `.pkl` and in correct directory |
| Out of GPU memory during extraction | Too many images in one batch | Split upload into two batches; build bank incrementally |

**Log to check:** Flask terminal for `Processing image X/N` progress and any Python exceptions.  
**CLI verify:** `ls -lh memory_banks/` to confirm `.pkl` size is non-zero.

---

## 6. Phase Outputs

- `memory_bank` (SpatialMemoryBank) loaded into Flask global state with FAISS index ready for search
- `memory_banks/<bank_name>.pkl` saved to disk (Path A)
- Stage 4 unlocked in the UI

**Bank file contents (inside .pkl):**
```
- FAISS IndexFlatIP (cosine similarity, GPU)
- feature_dim: int (matches engine output dimension)
- num_regions: int
- is_fitted: True
```

---

## 7. Ready for Phase 4 When

- [ ] Live terminal shows "FAISS index built successfully" (Path A) or "Memory bank loaded" (Path B)
- [ ] Green toast appeared for save or load
- [ ] `memory_banks/<name>.pkl` exists and is non-zero in size
- [ ] Stage 4 badge is unlocked
- [ ] Right inspector shows "Memory Bank: <name>" in the session summary
- [ ] Flask terminal shows `memory_bank.is_fitted = True`
