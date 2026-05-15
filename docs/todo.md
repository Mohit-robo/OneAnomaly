# Execution Checkpoints (TODO)

## Stage 1: Phase 1.1 and Phase 1.2 (Easy Start)

**Local test scripts** → `python/tests/`  (see `python/tests/README.md`)

### Testing (Done)
- [x] `test_thresholding.py` — intensity thresholding with interactive OpenCV trackbar preview + batch mode.
- [x] `test_bg_subtraction.py` — averaged bg subtraction, interactive + batch mode.
- [x] `test_preprocess_pipeline.py` — end-to-end: preprocessing → ONNX → FAISS → heatmap with per-image latency.

### Integration into App (Next)
- [X] Add UI panel for Threshold Controls (min/max intensity, channel selection, morph open/close).
- [X] Add UI panel for Background Subtraction Controls (diff threshold, morph open/close, fill holes, min component).
- [X] Add support to upload a reference "blank" folder/images for averaged BG subtraction.
- [X] Implement live preview of threshold/bg-subtraction masked result.
- [X] Implement image preprocessing to generate binary BG mask before feature extraction in api_server.py/feature_extractor.py.

## Stage 2: Phase 2 — Spatial Regions

**Reference** - python/spatial_anomaly_scripts

- [x] Add UI toggles for Region Selection Modes (Mode A: 2x2 Grid, Mode B: Draw Regions, Mode C: No Splitting).
- [x] Implement Mode A: 2x2 Grid logic on background subtracted image.
- [x] Implement Mode B: Canvas overlay for drawing up to 5 bounding polygons/rectangles + labeling.
- [x] Implement Mode C: Single patch processing.
- [x] Build preview section showing how sample images will be tiled.
- [x] Implement DINOv3 exporter: Export ONNX to TensorRT with "Export" UI button.
- [x] **New**: Multi-page (Step-by-Step) UI architecture with stage locking.
- [x] **New**: Persistent Right Inspector Panel for "on-the-go" parameter tracking throughout the session.

## Stage 3: Phase 3 — Memory Bank Creation

**Reference** - python/spatial_anomaly_scripts

- [ ] Integrate Phase 1 pre-processing and Phase 2 spatial split into feature extraction flow.
- [ ] Retain existing "Upload Good Samples" flow but enhance it to pass images through new pipeline.
- [ ] Update Feature Extraction to load TRT DINOv3 model (BS=n) and extract spatial region features.
- [ ] Refactor FAISS index generation to handle memory bank size = `BS × num_images × feature_dim`.
- [ ] Implement naming and saving of the Memory Bank with comprehensive session config (pre-proc, spatial params, index).

## Stage 4: Phase 4 — Validate Results
- [ ] Build UI to upload a small test set (≤20 images) for validation with memory bank/model selection.
- [ ] Implement pipeline combining pre-processing + spatial tiling from saved session during inference.
- [ ] Perform FAISS similarity search per region, compute anomaly scores, and display inference latency.
- [ ] Add interactive UI controls: Threshold (τ), k (nearest neighbors), Sigma (σ), and v_max.
- [ ] Implement dynamic visualization logic to update overlays and heatmaps live via cached raw scores.

## Stage 5: Phase 5 — Confirmation & Final Inference
- [ ] Create Parameter Summary Screen displaying all consolidated config params before final execution.
- [ ] Support large batch upload for full inference.
- [ ] Setup result flag logic: Good images (score < τ) vs Not OK images. Add optional `gt_mask` upload.
- [ ] Build detailed Results Display (paginated, region-level score breakdown, pixel-level IoU if `gt_mask` present).
- [ ] Implement "Download all results" feature (ZIP of images + CSV metadata).
- [ ] Implement "Save session history" exporting full state to JSON for reproducibility.

## Stage 6: Phase 1.3 — UNet + SAM Segmentation
- [ ] Integrate Meta SAM to generate clean binary mask using point/box prompts over BG images.
- [ ] Provide UI for configuring UNet training parameters (epochs, LR, BS, loss).
- [ ] Implement background UNet training with progress tracking (MLflow) and dataset tracking (DVC).
- [ ] Overlay predicted masks on validation images in the dashboard.
- [ ] Process UNet export pipeline: UNet → ONNX → TensorRT, saving it to session config for future inference.
