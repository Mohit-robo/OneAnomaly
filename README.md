# DINOv3 Anomaly Detection Web Platform

A premium, industrial-grade **Web Application** for detecting surfacing defects using Meta's DINOv3 Vision Transformer. This tool replaces complex command-line workflows with a modern, dark-themed GUI that provides real-time feedback, interactive visualizations, and simplified asset management.

## 🌟 Web App Features

### 🖥️ Modern User Interface
-   **Roboflow-Inspired Workflow**: A premium, linear stage-by-stage workflow guiding the user from preprocessing to deployment.
-   **Sleek Dark Mode**: Designed for low-eye-strain usage in industrial environments with glassmorphic elements and modern typography.
-   **Live Interactive Previews**: Instantly visualize thresholding and background subtraction results within inline grid canvases.
-   **Contextual Assistance**: Beautiful, non-clipping hover tooltips attached to all complex parameters.

### ⚙️ Stage 1: Pipeline Preprocessing
-   **Averaged Background Subtraction**: Upload empty reference backgrounds to subtract static noise from industrial camera feeds.
-   **Binary Thresholding**: High-fidelity parameter extraction with min/max bounds and granular morphological denoising (Open/Close kernels).
-   **Session Memory**: Configurations are safely retained across server states during active workflow sessions.

### 🧠 Intelligent Memory Management
-   **One-Click Training**: Simply upload "Good" images (ZIP or folders) and click "Extract". The app handles the complexities of DINOv3 feature extraction and FAISS indexing.
-   **Memory Bank Manager**: Save, load, and switch between different defect datasets (e.g., "Bottle Caps", "Fabric", "Metal Sheets") instantly.
-   **Efficiency**: Uses 100% of extracted features (no subsampling) for maximum accuracy, optimized with L2 normalization.

### 🔍 Advanced Detection & Visualization
-   **Visual Heatmaps**: Instantly see *where* the defect is with high-resolution red/blue heatmaps overlays.
-   **Side-by-Side View**: Compare original images vs. anomaly overlays.
-   **Scoring System**: Automatic "Anomaly Score" calculation based on Cosine Similarity.
-   **Batch Processing**: Upload a ZIP of 100+ test images and get a downloadable report of all defects.

---

## 🚀 Quick Start Guide

### Prerequisites
-   **NVIDIA GPU** (Required for DINOv3 acceleration)
-   **Python 3.8+**
-   **Node.js 14+**

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Mohit-robo/anomaly_app.git
cd anomaly_app

# Install Python Dependencies (Backend)
cd python
pip install -r requirements.txt

# Install Node.js Dependencies (Frontend)
cd ..
npm install
```

### 2. Model Setup
Download the `dinov3_vits16` weights:
1.  Create a `models/` folder in the root.
2.  Download `dinov3_vits16_pretrain_lvd1689m.pth`.
3.  Place it in `models/`.

### 3. Launching the App
You need two terminals running:

**Terminal 1 (Python Brain):**
```bash
cd python
python api_server.py
```

**Terminal 2 (Web Server):**
```bash
# Main project folder
node server.js
```

Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 📖 Web Workflow

The platform uses a **sequential stepper architecture**. Complete each stage to unlock the next.

### Stage 1: Pre-processing Configuration
1.  **Select Mode**: Choose between *Intensity Thresholding* or *Averaged Background Subtraction*.
2.  **Tune Sliders**: Adjust parameters like thresholds and morphological kernels. Watch the **Live Preview** update instantly.
3.  **Confirm**: Click "Save Configuration & Continue" to lock in basic pipeline settings and unlock Spatial Region selection.

### Stage 2: Spatial Region Partitioning
1.  **Define Areas**: Choose *Automatic Quadrant Split* for grid analysis or *Manual Selection* to draw specific ROIs.
2.  **Export**: Click **Export** to compile the DINOv3 pipeline into an optimized TensorRT engine for your specific batch size.

### Stage 3: Memory Bank Creation
1.  **Upload**: Drag & drop a ZIP file containing *only good samples*.
2.  **Extract**: Click "Extract Features". The system processes the good images through your custom Pre-processing and Spatial filters.
3.  **Save**: Name and save your Memory Bank (e.g., `fabric_v1`).

### Stage 4: Inference & Detection
1.  **Load**: Select your saved Memory Bank.
2.  **Test**: Upload suspect images or a batch folder.
3.  **Analyze**: 
    -   Review **Anomaly Scores** for both the whole image and specific regions.
    -   Inspect **Heatmap Overlays** in the result gallery.
    -   Download a full ZIP report including CSV metadata.


---

## 🛠️ Tech Stack
-   **Frontend**: HTML5, CSS3 (Custom Dark Theme), Vanilla JS.
-   **Backend Proxy**: Node.js / Express / Multer.
-   **ML Engine**: Python / Flask / PyTorch / DINOv3 / FAISS.
