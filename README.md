# DINOv3 Anomaly Detection Web Platform

A premium, industrial-grade **Web Application** for detecting surfacing defects using Meta's DINOv3 Vision Transformer. This tool replaces complex command-line workflows with a modern, dark-themed GUI that provides real-time feedback, interactive visualizations, and simplified asset management.

## 🌟 Web App Features

### 🖥️ Modern User Interface
-   **Sleek Dark Mode**: Designed for low-eye-strain usage in industrial environments.
-   **Reactive Design**: Responsive UI that works seamlessly across devices.
-   **Live Terminal**: Integrated black-box terminal showing real-time Python backend logs directly in the browser. 
-   **Progress Tracking**: Precision progress bar for long-running extraction and detection tasks.

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

### Phase 1: Teaching the AI (Memory Bank)
1.  Go to the **"Memory Bank Setup"** card.
2.  Select **"Create New Memory Bank"**.
3.  **Upload**: Drag & drop a ZIP file containing *only good samples*.
4.  **Extract**: Click the button. Watch the **Live Terminal** and **Progress Bar** as the system processes thousands of patches.
5.  **Save**: Name your bank (e.g., `fabric_v1`) and save it.

### Phase 2: Detecting Defects
1.  Load your saved Memory Bank.
2.  Scroll to **"Detect Anomalies"**.
3.  **Upload**: Select a suspect image or a batch of images.
4.  **Run**: Click "Run Detection".
5.  **Analyze**: 
    -   See the **Anomaly Score** (0.0 to 1.0).
    -   Inspect the **Heatmap Overlay** to pinpoint the defect location.
    -   Download high-res result images.

---

## 🛠️ Tech Stack
-   **Frontend**: HTML5, CSS3 (Custom Dark Theme), Vanilla JS.
-   **Backend Proxy**: Node.js / Express / Multer.
-   **ML Engine**: Python / Flask / PyTorch / DINOv3 / FAISS.
