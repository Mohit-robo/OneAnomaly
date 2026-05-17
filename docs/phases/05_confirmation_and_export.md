# Phase 5: Confirmation & Final Inference (Batch)

## 1. What the session is intended for
This is the final phase of the OneAnomaly pipeline. It provides a read-only "Single Source of Truth" summary of all configurations built in Stages 1–4. Users use this stage to run high-volume **batch inference** on test datasets, review paginated results, dial in the final threshold, and export the comprehensive session package (images + metadata + pipeline state) for external auditing or CI/CD deployment.

## 2. Steps to perform / Selections
1. **Review Parameters:** Look at the left panel to ensure Preprocessing, Region Splitting, and Engine selection are correct.
2. **Override Memory Bank (Optional):** If you want to test against a previously saved memory bank instead of the one just built, select it from the dropdown.
3. **Upload Test Images:** Drag and drop a folder, ZIP file, or multiple images into the uploader.
4. **Adjust Threshold:** Move the threshold slider. Notice how the statistics (Total, Anomaly, Normal) and the result grid automatically update.
5. **Filter & Inspect Results:** Use the `All | Anomaly | Normal` tabs to filter the grid. Click any image pair to open the enlarged side-by-side Lightbox for detailed pixel inspection.
6. **Export:** Click the "Download Results" button to generate the final artifact package.

## 3. Verification Criteria
- The "Configuration Summary" panel must display valid data (no missing/undefined fields).
- Uploading test images triggers processing, resulting in the Grid View populating with exactly 10 images per page.
- The dual-image cards (Original | Anomaly Map) match the aspect ratio of the inputs.
- Changing the threshold slider instantly updates the OK/Not-OK badges and re-sorts the filter tabs without requiring a network request.

## 4. What to hit on UI to perform a certain task
- **To test images:** Drag onto the "Upload Test Images" drop zone.
- **To see only anomalies:** Click the `Anomaly` filter tab above the grid.
- **To inspect a false positive:** Click the image card in the grid to open the full-screen Lightbox. Press `Esc` to close.
- **To change pages:** Click the `Previous` / `Next` buttons at the bottom of the grid.
- **To get outputs:** Click the `⬇ Download Results` button in the Export section.

## 5. What to verify or refer to in case of error
- **Empty Memory Bank Dropdown:** Ensure `outputs/banks/` contains `.pkl` files and `api_server.py:list_memory_banks()` is returning a valid JSON string `{ "memory_banks": [...] }`.
- **Results Not Appearing:** Check the browser DevTools console. The response size might be too large if trying to push thousands of images over the JSON socket at once. Look for `net::ERR_CONNECTION_RESET` or Flask timeouts.
- **"ZIP Download Failed":** Ensure the backend has permissions to write to `outputs/sessions/<id>/` and that the `zipfile` module executed correctly.

## 6. What the phase outputs
When "Download Results" is clicked, the system sequences 3 distinct downloads:
1. **`[session_id]_overlays.zip`**: A ZIP archive containing `torch_res_*.png` side-by-side (Original + Heatmap overlay) composite images.
2. **`[session_id]_results.csv`**: A spreadsheet mapping `filename` -> `anomaly_score` -> `status` (NORMAL/ANOMALY), based on the currently selected UI threshold.
3. **`[session_id]_session_config.json`**: An exhaustive JSON dump of every single pipeline parameter (intensity mins/maxs, morph ops, region layouts, engine name, bank name) required to perfectly reproduce the session.

## 7. Verifications for the next step
As Phase 5 is the final phase of the workflow, verification means checking the downloaded artifacts:
1. Open the CSV and confirm the counts match the UI summary.
2. Open the ZIP and confirm the images are valid side-by-side `.png` files.
3. Open the JSON file to ensure the configuration isn't empty.
Once validated, the model pipeline is officially ready for edge deployment.
