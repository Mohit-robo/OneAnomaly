# Changelog

All notable changes to the **Anomaly Detection App** project.

## [v1.1.0] - 2026-05-14

### 🚀 Major UI/UX Overhaul (Stage 1.2: Preprocessing)
-   **Roboflow-Inspired Aesthetics**: Migrated UI to a highly-polished, premium dark mode featuring custom typography (Inter/Geist), glassmorphic elements, and border glow interactions.
-   **Stage 1: Preprocessing Mode**: Focused the interface explicitly on image preprocessing (Background Subtraction and Binary Thresholding).
-   **Live Interactive Previews**: Added real-time layout grid visualization driven by instantaneous API endpoints, removing the clunky `morphological` components from background subtraction routines for cleaner results.
-   **Workflow-Oriented Layout**: Widened the main central canvas by dropping the static right-sided metadata inspector, placing emphasis on an immersive workspace.
-   **Contextual Tooltips**: Overhauled frontend elements to feature custom non-clipping hover tooltips for concise parameter guidance.

### 🐛 Backend Refinements
-   **Session Memory State**: Shifted intermediate pipeline configuration storage to Server RAM instead of immediate JSON disk saving, building state dynamically across tools.
-   **Directory Restructuring**: Refactored `python/tests` directory to `python/src` for standardized module organization.

## [v1.0.0] - 2026-01-05

### 🚀 Major Features
-   **Full Web Interface**: Replaced CLI-only workflow with a modern, responsive Web UI.
-   **Live System Feedback**: Integrated a real-time terminal log viewer and progress bar into the frontend, bridging the gap between the Python backend and the user.
-   **Optimized Anomaly Detection**:
    -   Aligned preprocessing with the reference script (256x256 resize, L2 Normalization).
    -   Switched to **Cosine Similarity** (FAISS IndexFlatIP) for more accurate defect scoring.
    -   Implemented Reference-Standard heatmap generation (Cubic Upscaling -> Gaussian Smoothing -> Min-Max Norm).

### 🐛 Bug Fixes & Improvements
-   **Button Stability**: Fixed issue where the "Run Detection" button would vanish by enforcing `type="button"` to prevent accidental form submissions.
-   **Error Reporting**: exposed detailed Python stack traces in the Web UI, replacing generic error messages.
-   **Recursive Processing**: Added support for nested ZIP files and folder structures during image upload.
-   **Robustness**: Added checks for corrupt images and thread-safety fixes for Matplotlib (switched to `Agg` backend).
-   **Dependency Handling**: Improved DINOv3 model loading to check for local weights first, simplifying setup.

### 📜 Original Ask vs. Delivered
*   **Original Ask**: A script to run anomaly detection using DINOv3.
*   **Delivered**: A complete **End-to-End Application**.
    -   Instead of just updated scripts, we built a **Client-Server Architecture**.
    -   Added a robust **Memory Bank Management** system.
    -   Added visualization tools (Heatmap overlays, side-by-side plots) directly in the browser.
    -   Ensured **Pixel-Perfect Alignment** with the user's "Gold Standard" reference script (`dinov3_faiss.py`).
