import sys
import os
import base64
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2

# Import local functionality
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / 'python'))

from preprocessing import apply_preprocessing
from tiling import crop_regions, build_global_score_map
from triton_extractor import TritonFeatureExtractor
from anomaly_detector import AnomalyDetector
from memory_bank import MemoryBank, SpatialMemoryBank

# Triton client
try:
    import tritonclient.grpc as grpcclient
except ImportError:
    print("Warning: tritonclient not installed")

app = FastAPI(title="OneAnomaly Inference Gateway")

class SessionConfig(BaseModel):
    session_name: str
    config: dict

class InferRequest(BaseModel):
    image_b64: str
    session_name: str
    phase: int = 4

# In-memory session cache
session_cache: Dict[str, dict] = {}
# Triton client connection
triton_client = None

@app.on_event("startup")
async def startup_event():
    global triton_client
    try:
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001")
        print("Connected to local Triton gRPC at localhost:8001")
    except Exception as e:
        print(f"Failed to connect to Triton: {e}")

@app.get("/health")
def health_check():
    triton_ready = False
    if triton_client:
        try:
            triton_ready = triton_client.is_server_ready()
        except:
            pass
    return {
        "status": "ok", 
        "triton_connected": triton_client is not None,
        "triton_ready": triton_ready
    }

@app.post("/sync_session")
def sync_session(payload: SessionConfig):
    session_cache[payload.session_name] = payload.config
    print(f"Synced session config: {payload.session_name}")
    return {"saved": True, "session_name": payload.session_name}

def b64_to_img(b64_str: str) -> np.ndarray:
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_data = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def img_to_b64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

@app.post("/infer")
async def infer(payload: InferRequest):
    start_time = time.time()
    
    # 1. Get session config
    config = session_cache.get(payload.session_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Session {payload.session_name} not found in gateway cache. Sync first.")
        
    # 2. Decode image
    try:
        image_bgr = b64_to_img(payload.image_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 image")
        
    # 3. Phase 1: Preprocessing
    mask, masked_img = apply_preprocessing(image_bgr, config.get("phase1", {}))
    
    if payload.phase == 1:
        # Just return preview of preprocessing
        return {
            "ok": True, 
            "heatmap_b64": img_to_b64(masked_img),
            "inference_ms": (time.time() - start_time) * 1000
        }
        
    # 4. Phase 2: Spatial Tiling
    regions = config.get("phase2", {}).get("regions", [])
    crops = crop_regions(masked_img, regions)
    
    if payload.phase == 2:
        # Return preview of spatial regions (draw boxes on image)
        preview_img = masked_img.copy()
        for r in regions:
            x, y, w, h = r['x'], r['y'], r['w'], r['h']
            cv2.rectangle(preview_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return {
            "ok": True,
            "heatmap_b64": img_to_b64(preview_img),
            "inference_ms": (time.time() - start_time) * 1000
        }
        
    # 5. Phase 4: Anomaly Detection
    session_dir = Path("memory_banks") / payload.session_name
    if not session_dir.exists():
        raise HTTPException(status_code=400, detail="Memory bank not found on gateway for this session. Sync/Build first.")
        
    try:
        if len(regions) > 1:
            mb = SpatialMemoryBank.load(str(session_dir))
        else:
            mb = MemoryBank.load(str(session_dir))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load memory bank: {e}")
        
    # We use AnomalyDetector from original logic but plug in the new Triton extractor!
    triton_extractor = TritonFeatureExtractor(url="localhost:8001", model_name="dinov3_encoder")
    detector = AnomalyDetector(memory_bank=mb, feature_extractor=triton_extractor, k_neighbors=3, gaussian_sigma=2.0)
    
    score, heatmap, overlay = detector.detect_anomalies(image_bgr, return_heatmap=True, alpha=0.5, regions=regions)
    
    # Convert overlay and stacked to base64
    overlay_b64 = img_to_b64(overlay)
    try:
        stacked = np.hstack([image_bgr, overlay])
        stacked_b64 = img_to_b64(stacked)
    except:
        stacked_b64 = overlay_b64
    
    return {
        "ok": True,
        "inference_ms": (time.time() - start_time) * 1000,
        "score": float(score),
        "heatmap_b64": overlay_b64,
        "stacked_b64": stacked_b64
    }

class BuildMemoryBankRequest(BaseModel):
    session_name: str
    images_b64: List[str]

@app.post("/build_memory_bank")
async def build_memory_bank(payload: BuildMemoryBankRequest):
    start_time = time.time()
    
    config = session_cache.get(payload.session_name)
    if not config:
        raise HTTPException(status_code=400, detail="Sync session first")
        
    regions = config.get("phase2", {}).get("regions", [])
    
    # 1. Initialize Memory Bank
    if len(regions) > 1:
        mb = SpatialMemoryBank(regions_coords=[[r['x'], r['y'], r['w'], r['h']] for r in regions], feature_dim=768)
    else:
        mb = MemoryBank(feature_dim=768)
        
    triton_extractor = TritonFeatureExtractor(url="localhost:8001", model_name="dinov3_encoder")
    
    # 2. Extract features
    all_crops = []
    for b64 in payload.images_b64:
        try:
            image_bgr = b64_to_img(b64)
            _, masked_img = apply_preprocessing(image_bgr, config.get("phase1", {}))
            if len(regions) > 1:
                crops = crop_regions(masked_img, regions)
                all_crops.extend(crops)
            else:
                all_crops.append(masked_img)
        except Exception:
            continue
            
    if not all_crops:
        raise HTTPException(status_code=400, detail="No valid crops extracted")
        
    features, _ = triton_extractor.extract_batch(all_crops)
    
    if len(regions) > 1:
        num_images = len(all_crops) // len(regions)
        num_regions = len(regions)
        patch_dim = features.shape[-1]
        features_reshaped = features.reshape(num_images, num_regions, -1, patch_dim)
        mb.add_features(features_reshaped)
    else:
        mb.add_features(features)
        
    # 3. Save Bank
    session_dir = Path("memory_banks") / payload.session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    mb.save(str(session_dir))
    
    return {
        "ok": True,
        "n_features_saved": mb.features.shape[0] if mb.features is not None else 0,
        "ms": (time.time() - start_time) * 1000
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
