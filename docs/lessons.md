# Lessons Learned — OneAnomaly Project

This document captures engineering decisions, bugs encountered, root cause analyses, and solutions applied throughout the development of the OneAnomaly anomaly detection platform. Intended for use in a RAG knowledge base to accelerate future debugging and onboarding.

---

## 1 — Architecture: Flask serves both frontend and API

**Issue:**  
The original plan included a Node.js Express proxy (`server.js`) sitting between the frontend and the Python Flask API. This added complexity with two servers to keep running, CORS configurations on both sides, and an extra network hop.

**Solution used:**  
Dropped `server.js` entirely. Flask serves the static frontend directly (from `../public/`) and handles all API routes on the same port 5000. This eliminates CORS entirely for same-origin calls, reduces the startup process to one command, and removes a layer of indirection from every API request.

---

## 2 — TensorRT Batch Size Must Match Region Count

**Issue:**  
The spatial pipeline feeds `N` image crops (one per region) as a batch into the DINOv3 engine. If the compiled engine expects batch size 3 but 2 regions are configured, inference either crashes or silently returns incorrect feature tensors.

**Solution used:**  
Enforce engine selection in Stage 2 UI. The engine filename encodes its batch size (e.g., `_bs3.engine`). Stage 3 is blocked (via toast + navigation guard) if no engine is selected. Documentation was updated to keep a table of available engines with their batch sizes.

---

## 3 — Black Canvas: HTML Canvas Race Condition

**Issue:**  
The "Anomaly Detection Result" right panel used an HTML `<canvas>` to blend the source image and heatmap. It consistently rendered as solid black. Three overlapping bugs caused this:
1. `sourceImg.complete` check immediately after setting `src` is unreliable — the browser may report `complete = true` with `naturalWidth = 0` during the transition frame
2. A `setTimeout(150ms)` was used for "layout stabilization" but 150ms is not guaranteed to be enough if the server is under load
3. If any element in `updateDynamicVisualization` threw (e.g., `.querySelector('.badge-icon')` on a null badge), the async function rejected silently before ever calling `renderBlendedCanvas`

**Solution used:**  
Removed the canvas entirely. The backend already generates `_overlay.png` (heatmap baked over source at α=0.5). The right panel now uses a plain `<img id="result-overlay-img">` with `src` set directly to the `_overlay.png` API endpoint. The alpha slider controls `img.style.opacity` — pure CSS, no rendering engine, no race conditions.

---

## 4 — Dead Code: `imgPath` Variable Built But Never Used

**Issue:**  
In the detection response handler, a `data.results.forEach` loop built result tiles and constructed:
```js
const imgPath = `${API_BASE}/api/get_image/${data.session_id}/${stem}_overlay.png`;
```
This variable was never assigned to any DOM element. The loop also created tiles that were appended to `resultsGrid` which was immediately hidden (`display: none`). The tile click handler called `showDetailedAnalysis`, which actually worked — but `showDetailedAnalysis` was also called unconditionally for `idx === 0` at the end of the loop. Result: duplicate calls, redundant DOM work, and confused state.

**Solution used:**  
Deleted the entire `forEach` loop. The detection handler now directly calls `showDetailedAnalysis(data.results[0], data.session_id)` after hiding `resultsArea`. Single source of truth, no duplicate calls.

---

## 5 — Canvas CORS Taint Potential

**Issue:**  
HTML canvas has a security rule: if you draw a cross-origin image onto a canvas (even with CORS headers on the server), the canvas becomes "tainted" unless the `<img>` element has `crossOrigin="anonymous"` set BEFORE its `src` is assigned. The `<img id="detailed-original">` in the HTML did not have this attribute. On systems where the frontend is served on a different port than the Flask API (e.g., port 8000 vs 5000), this would silently taint the canvas and cause `drawImage` to produce a black rectangle.

**Solution used:**  
Moot after solution #3 (canvas removed). But for future reference: always set `img.crossOrigin = "anonymous"` before setting `img.src` if that image will ever be drawn to a canvas. And serve frontend + API from the same origin to avoid the problem entirely.

---

## 6 — `sourceImg.complete` Is Unreliable After `src` Change

**Issue:**  
JavaScript's `img.complete` property has a subtle behavior: after you change `img.src`, `complete` briefly becomes `false`. But if the new image is already cached, the browser can set `complete = true` synchronously before the next microtask — without firing `onload`. The pattern `if (!img.complete) { await new Promise(r => { img.onload = r; }) }` fails when the image is cached: `complete` is `true`, the wait is skipped, but the image's layout dimensions are not yet committed to the DOM.

**Solution used:**  
Switch to `img.decode()` for reliable await-able loading, OR eliminate the need entirely by not relying on `img` dimensions for canvas sizing. In our case, removing the canvas removed this entire class of problem.

---

## 7 — Preprocessing Config Must Be Applied Consistently at Both Train and Test Time

**Issue:**  
Early versions applied background subtraction during feature extraction (Stage 3 / memory bank build) but forgot to re-apply it to test images in Stage 4. The result: the memory bank was built from masked/cropped feature embeddings, but test images were processed as raw — causing a distribution mismatch and artificially high anomaly scores even on good test images.

**Solution used:**  
In `api_server.py` detect endpoint, the same `preprocess_config` global is applied to test images (lines ~677-701) before they are passed to the anomaly detector. This matches exactly what was applied to good images during extraction.

---

## 8 — Export Modal Does Not Close After SSE Stream Ends

**Issue:**  
The TRT export progress is streamed via Server-Sent Events (SSE). When the stream ends (export complete), the modal overlay showed a "Close" button. Clicking it did not close the modal because the event listener was attached only once on page load — but the close button was inside the modal with `display: none`, so the listener fired against a detached or hidden element in some render states.

**Solution used:**  
Moved the close button event listener to be re-attached each time the modal is shown (inside the SSE `onmessage` handler on stream close). This ensures the listener is always live when the button becomes visible.

---

## 9 — Engine Not Selected Validation

**Issue:**  
Users could click through Stage 2 and begin feature extraction in Stage 3 without having selected or exported a TensorRT engine. The backend would then attempt to initialize `FeatureExtractor` with `model_path = None`, falling back to PyTorch — but without warning the user, who expected TRT-accelerated extraction.

**Solution used:**  
Added a validation check in the Stage 3 "Extract Features" handler in `app.js`. If `spatial_config.engine` is not set (from the Stage 2 inspector), a warning toast is shown and `fetch` is never called. The proceed button is also visually dimmed until an engine is confirmed.

---

## 10 — Memory Bank: Skip Build Option

**Issue:**  
Every session required running full feature extraction even when the user had already built and saved a memory bank for the same product. This was time-consuming (minutes for large datasets) and blocked rapid iterative testing of different thresholds.

**Solution used:**  
Added a **"Skip Build & Use Existing"** button in Stage 3. This calls `/api/load_memory_bank` with a selected bank name rather than `/api/extract_features`. If the load succeeds, the stage is marked complete and the user proceeds directly to detection.

---

## 11 — FAISS Index Dimension Mismatch After Engine Change

**Issue:**  
A memory bank built with one engine (e.g., `bs3.engine` with feature dim 384×3=1152 stored regionally) cannot be used with a different engine that produces different feature dimensions. The FAISS index was built for a fixed dimension at creation time; passing differently-shaped query vectors causes a runtime error or silent garbage results.

**Solution used:**  
The bank's feature dimension is saved in its metadata. At load time, the API checks that the loaded bank's dimension matches the current engine's output dimension. If mismatched, an error is returned asking the user to rebuild the bank. Documented in `SETUP_DINOV3.md`.

---

## 12 — Backend Heatmap Saved as Grayscale Float

**Issue:**  
`_heatmap.png` was originally saved as:
```python
cv2.imwrite(str(heatmap_path), (heatmap * 255).astype(np.uint8))
```
This writes a single-channel grayscale uint8 image. When the frontend tried to use it as an overlay (even before the canvas was removed), blending a grayscale image over a color source image produced a washed-out grey tint instead of the expected red/blue JET thermal visualization.

**Solution used:**  
Changed to apply `cv2.COLORMAP_JET` before saving:
```python
heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite(str(heatmap_path), heatmap_colored)
```
The `_overlay.png` (baked blend) was already correct because `_create_overlay()` applied the colormap internally. The `_heatmap.png` fix makes it usable independently.

---

## 13 — Matplotlib Thread Safety: Use `Agg` Backend

**Issue:**  
`matplotlib.pyplot` uses a GUI backend (Tkinter) by default. When called from a Flask request thread, it raises `RuntimeError: main thread is not in main loop` because Tkinter requires the main thread.

**Solution used:**  
Set the non-interactive backend before any pyplot import:
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```
This is done at the top of `anomaly_detector.py` and must come before any `plt` usage.

---

## 14 — DINOv3 Submodule Initialization

**Issue:**  
The `python/dinov3/` directory (Meta's DINOv3 codebase) is a git submodule. On a clean clone, the directory exists but is empty, causing `ImportError` when `feature_extractor.py` tries to import from it.

**Solution used:**  
```bash
git submodule update --init --recursive
```
Must be run after cloning. Documented in `SETUP_DINOV3.md`.

---

## 15 — Session Artifacts Are Cleared on Each Detection Run

**Issue:**  
The session directory (`outputs/sessions/<session_id>/`) is wiped at the start of each detection run. This means previous run results are lost if the user runs detection again with the same session ID.

**Solution used:**  
Each detection run generates a unique session ID: `detect_<timestamp>`. The clearing logic only wipes the current session's folder, so previous sessions are unaffected — they accumulate in `outputs/sessions/`. This is intentional: it prevents disk bloat within a single session but preserves history across sessions. Future work: add a "clear history" button to the UI.

---

## 16 — `showDetailedAnalysis` Was `async` But Called in `setTimeout`

**Issue:**  
`showDetailedAnalysis` was declared `async` and called inside `setTimeout(() => { showDetailedAnalysis(...) }, 150)`. The `setTimeout` callback does not propagate promise rejections — any `await` failure inside the async function was silently swallowed, leaving the UI in a partially initialized state with no error feedback.

**Solution used:**  
Made `showDetailedAnalysis` synchronous (it no longer needs to `await` image loading since we use direct `img.src` assignment). Direct call without `setTimeout` — no delay needed since there is no canvas to initialize.

---

## 17 — General: img vs canvas for result display

**Lesson (not a bug):**  
HTML `<canvas>` should only be used when you need to perform pixel-level operations that cannot be achieved with CSS (e.g., per-pixel blend modes, drawing shapes, reading pixel data). For simple image display with opacity control, a plain `<img>` + CSS `opacity` is always preferable:
- No CORS taint risk
- No race conditions with `complete`/`naturalWidth`
- Works even when images come from a different origin
- The browser's native image renderer is more optimized than canvas `drawImage` for display purposes

---

## 18 — Multiple File Downloads from a Single Button Click

**Issue:**  
In Phase 5, the "Download Results" button needs to trigger the download of three distinct files (ZIP from server, CSV generated client-side, JSON generated client-side). Attempting to trigger all three sequentially without yielding caused browsers (particularly Chrome) to block the second and third downloads as a "Pop-up/Anti-Spam" protection measure.

**Solution used:**  
Introduced artificial asynchronous delays between the download triggers:
```javascript
window.open(url1, '_blank');
await new Promise(r => setTimeout(r, 600)); // Yield to browser UI thread
// Trigger URL 2...
await new Promise(r => setTimeout(r, 400));
// Trigger URL 3...
```
This spacing guarantees the browser processes each as a distinct user-initiated pseudo-action, bypassing the simultaneous-download blocker without requiring users to tweak site permission settings.
