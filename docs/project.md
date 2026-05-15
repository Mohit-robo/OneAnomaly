# Anomaly Detection Web App - Implementation Plan

## Overview

Building a production-ready anomaly detection web application that uses DINOv3 for feature extraction and FAISS for similarity search to detect anomalies in industrial parts. The system features a **multi-page stepper UI** (Stage-by-Stage) for structured configuration, a **Live Parameter Inspector** for real-time tracking, and hardware-accelerated TensorRT inference.


## User Review Required

> [!IMPORTANT]
> **Model Selection**: Using DINOv3 ViT-S/16 distilled model. The model will be loaded via PyTorch Hub from the cloned DINOv3 repository initially, then converted to TensorRT for production.

> [!IMPORTANT]
> **Technology Stack**: 
> - Backend: Node.js with Express
> - Python Bridge: Flask API for ML operations
> - Frontend: Vanilla HTML/CSS/JavaScript
> - ML: PyTorch → TensorRT conversion pipeline
> - Storage: Pickle files for feature banks

> [!WARNING]
> **FAISS-GPU Requirement**: This requires NVIDIA GPU with CUDA support. The application will not work on CPU-only systems for production use.

## Proposed Changes

### Phase 1: Python ML Backend (Complete)

#### [NEW] [requirements.txt](file:///d:/Mohit/anomaly_app/python/requirements.txt)
Python dependencies including torch, torchvision, faiss-gpu, opencv-python, numpy, pillow, flask, and flask-cors.

#### [NEW] [test_dinov3_pytorch.py](file:///d:/Mohit/anomaly_app/python/test_dinov3_pytorch.py)
Standalone PyTorch testing script that:
- Clones DINOv3 repository and loads ViT-S/16 distilled model via torch.hub
- Processes a folder of images
- Extracts patch-wise features (using intermediate layer outputs)
- Saves feature visualizations for manual verification
- Outputs feature dimensions and processing time

#### [NEW] [feature_extractor.py](file:///d:/Mohit/anomaly_app/python/feature_extractor.py)
Core feature extraction module that:
- Loads DINOv3 ViT-S/16 distilled model (PyTorch or TensorRT)
- Implements patch-wise feature extraction
- Handles batch processing
- **Supports recursive image search in nested directories (flat or nested structure)**

#### [NEW] [memory_bank.py](file:///d:/Mohit/anomaly_app/python/memory_bank.py)
PatchCore-inspired memory bank implementation:
- Stores good image features using coreset subsampling
- Implements FAISS index for fast similarity search
- Supports save/load functionality via pickle
- Provides methods for adding features and querying

#### [NEW] [anomaly_detector.py](file:///d:/Mohit/anomaly_app/python/anomaly_detector.py)
Anomaly detection and visualization:
- Performs FAISS similarity search on test images
- Generates anomaly heatmaps
- Overlays heatmaps on original images
- Returns anomaly scores and visualizations

#### [NEW] [api_server.py](file:///d:/Mohit/anomaly_app/python/api_server.py)
Flask API server exposing endpoints:
- POST `/extract_features` - Extract features from good images (**Handling Zip files with flat or nested structure**)
- POST `/save_memory_bank` - Save feature bank to disk
- POST `/load_memory_bank` - Load existing feature bank
- POST `/detect_anomalies` - Process test images (**Supports Zip, multiple files, or single image**)
- GET `/health` - Health check endpoint

---

### Phase 2: TensorRT Conversion (In Progress)

#### [NEW] [convert_to_tensorrt.py](file:///d:/Mohit/anomaly_app/python/convert_to_tensorrt.py)
Script to convert PyTorch model to TensorRT:
- Exports DINOv3 ViT-S/16 to ONNX format
- Converts ONNX to TensorRT engine
- Validates output consistency
- Benchmarks performance

#### [NEW] [test_dinov3_tensorrt.py](file:///d:/Mohit/anomaly_app/python/test_dinov3_tensorrt.py)
TensorRT testing script similar to PyTorch test but using TensorRT engine.

---

### Phase 3: Node.js Web Application

#### [NEW] [package.json](file:///d:/Mohit/anomaly_app/package.json)
Node.js dependencies: express, multer (file uploads), archiver/unzipper, axios, uuid.

#### [NEW] [server.js](file:///d:/Mohit/anomaly_app/server.js)
Express server that:
- Handles file uploads (zip, folders, single images)
- Manages session IDs
- Proxies requests to Python API
- Serves static frontend files
- Manages upload/output directories

#### [MODIFY] [public/index.html](file:///home/silicon/projects/anomaly_app/public/index.html)
Main UI with modern multi-page architecture:
- **Stepper Navigation**: Sequential stages (Pre-processing, Spatial, Memory Bank, Detect).
- **Stage Locking**: Future stages are locked until preceding ones are configured.
- **Section 1 (Pre-processing)**: Threshold or Background subtraction with live preview.
- **Section 2 (Spatial Regions)**: Grid splits or Manual box drawing for localized detection.
- **Persistent Inspector**: Right-hand panel summarizing all session parameters (Thresholds, Channels, Modes, Ratios).
- **Export System**: Direct TensorRT conversion with specialized blue-to-black interactive button.

#### [NEW] [public/style.css](file:///d:/Mohit/anomaly_app/public/style.css)
Clean black & white styling:
- Minimalist design with proper spacing
- Button styles with hover effects
- Responsive layout
- Image gallery grid
- Progress indicators

#### [NEW] [public/app.js](file:///d:/Mohit/anomaly_app/public/app.js)
Frontend JavaScript:
- File upload handling
- API communication
- Progress tracking
- Results rendering
- Download functionality
- Error handling

---

### Phase 4: Project Structure

#### [NEW] [.gitignore](file:///d:/Mohit/anomaly_app/.gitignore)
Ignore node_modules, Python cache, uploads, outputs, model weights, etc.

#### [NEW] [README.md](file:///d:/Mohit/anomaly_app/README.md)
Comprehensive documentation:
- Project overview
- Installation instructions
- Usage guide
- API documentation
- Troubleshooting

Directory structure:
```
anomaly_app/
├── python/                    # Python ML backend
│   ├── api_server.py
│   ├── feature_extractor.py
│   ├── memory_bank.py
│   ├── anomaly_detector.py
│   ├── test_dinov3_pytorch.py
│   ├── test_dinov3_tensorrt.py
│   ├── convert_to_tensorrt.py
│   └── requirements.txt
├── public/                    # Frontend files
│   ├── index.html
│   ├── style.css
│   └── app.js
├── uploads/                   # User uploaded files
│   ├── good/
│   └── test/
├── outputs/                   # Processing results
│   └── sessions/
├── memory_banks/              # Saved feature banks
├── models/                    # Model weights (TensorRT)
├── server.js                  # Node.js server
├── package.json
├── .gitignore
└── README.md
```

## User Guide & File Structure

### Zip File Structure
The application supports both **flat** and **nested** directory structures within uploaded ZIP files. The system recursively searches for images.

**Option A: Flat Structure (Recommended)**
```
my_images.zip
├── image1.jpg
├── image2.png
└── image3.jpg
```

**Option B: Nested Structure**
```
my_dataset.zip
├── train/
│   ├── good/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── bad/  (ignored during good image upload if you only select the good folder/zip)
```
*Note: When uploading "Good Images" for training, the system will process ALL images found in the zip. Ensure the zip contains ONLY good images.*

### Workflow
1. **Setup**: Ensure DINOv3 weights are downloaded to `models/` directory.
2. **Training (Memory Bank)**:
   - Select "Create New Memory Bank".
   - Upload a ZIP file containing ONLY normal/good images.
   - Click "Extract Features".
   - Save the memory bank with a unique name.
3. **Detection**:
   - Load your saved memory bank.
   - Upload test images (Single image, multiple files, or ZIP).
   - Click "Run Detection".
   - View results or download full report.

## Verification Plan

### Automated Tests

#### 1. PyTorch Model Testing
**Command**: 
```bash
cd d:/Mohit/anomaly_app/python
python test_dinov3_pytorch.py --input_folder <path_to_test_images> --output_folder <path_to_output>
```
**Expected Output**:
- Feature maps saved as images
- Console output showing feature dimensions (should be 384 for ViT-S/16)
- Processing time per image
- No errors during model loading or inference

**Manual Verification**: You will review the output feature visualizations to ensure they look reasonable.

#### 2. Memory Bank Functionality
**Command**:
```bash
cd d:/Mohit/anomaly_app/python
python -c "from memory_bank import MemoryBank; import numpy as np; mb = MemoryBank(); mb.add_features(np.random.rand(100, 384)); mb.save('test_bank.pkl'); mb2 = MemoryBank(); mb2.load('test_bank.pkl'); print('Success' if mb2.feature_count() == 100 else 'Failed')"
```
**Expected Output**: "Success"

#### 3. Python API Server
**Command**:
```bash
cd d:/Mohit/anomaly_app/python
python api_server.py
```
Then in another terminal:
```bash
curl http://localhost:5000/health
```
**Expected Output**: JSON response with status "healthy"

#### 4. Node.js Server
**Command**:
```bash
cd d:/Mohit/anomaly_app
npm install
npm start
```
**Expected Output**: Server running on http://localhost:3000

### Manual Verification

#### 1. End-to-End Workflow Test
**Steps**:
1. Start Python API server: `cd python && python api_server.py`
2. Start Node.js server: `npm start`
3. Open browser to http://localhost:3000
4. Upload a zip file with 10-20 good images
5. Click "Extract Features" button
6. Enter a filename (e.g., "test_run_1") and click "Save Memory Bank"
7. Verify file appears in memory_banks/ directory
8. Upload a test image with potential anomaly
9. Click "Run Detection"
10. Verify anomaly heatmap is displayed
11. Click "Download Results"

**Expected Results**:
- All uploads complete without errors
- Progress indicators work correctly
- Anomaly heatmap overlays on original image
- Download produces a zip file with results

#### 2. Feature Extraction Verification
**You will manually verify**:
- Feature extraction output images show meaningful patterns
- Processing time is reasonable (< 1 second per image on GPU)
- Memory bank file size is appropriate

#### 3. TensorRT Conversion (Future Phase)
**Steps**:
1. Run conversion script: `python convert_to_tensorrt.py`
2. Run TensorRT test: `python test_dinov3_tensorrt.py --input_folder <path>`
3. Compare outputs with PyTorch version
4. Benchmark performance improvement

**You will manually verify**:
- TensorRT outputs match PyTorch outputs (within tolerance)
- Performance improvement is significant (3-5x faster)

## Implementation Order

1. **Phase 1a**: Create Python testing scripts (PyTorch only)
   - You verify PyTorch model works correctly
2. **Phase 1b**: Implement core ML modules (feature extractor, memory bank, anomaly detector)
3. **Phase 1c**: Create Flask API server
4. **Phase 3**: Build Node.js web application
5. **Phase 2**: TensorRT conversion (after PyTorch verification)

## Notes

- Starting with PyTorch for initial development and testing
- TensorRT conversion will be done after verifying PyTorch pipeline works
- PatchCore memory bank approach uses coreset subsampling to reduce memory footprint
- FAISS will use GPU for fast similarity search
- Session management ensures multiple users can use the app simultaneously
