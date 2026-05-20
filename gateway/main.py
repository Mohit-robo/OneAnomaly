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

triton_extractor = TritonFeatureExtractor(url="localhost:8001", model_name="dinov3_onnx")

class SessionConfig(BaseModel):
    session_name: str
    config: dict

class InferRequest(BaseModel):
    image_b64: str
    session_name: str
    phase: int = 4

# In-memory session cache
session_cache: Dict[str, dict] = {}
# In-memory memory bank cache to prevent reloading on every inference request
memory_bank_cache: Dict[str, Any] = {}
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
    # Invalidate cached memory bank when session is re-synced to prevent stale configuration
    if payload.session_name in memory_bank_cache:
        del memory_bank_cache[payload.session_name]
        print(f"Invalidated cached memory bank for session: {payload.session_name}")
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

    if payload.phase == 2:
        # Return preview of spatial regions (draw boxes on image)
        preview_img = masked_img.copy()
        for r in regions:
            x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])
            cv2.rectangle(preview_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return {
            "ok": True,
            "heatmap_b64": img_to_b64(preview_img),
            "inference_ms": (time.time() - start_time) * 1000
        }
        
    # 5. Phase 4: Anomaly Detection
    session_dir = BASE_DIR / "memory_banks" / payload.session_name
    if not session_dir.exists():
        raise HTTPException(status_code=400, detail="Memory bank not found on gateway for this session. Sync/Build first.")
        
    mb = memory_bank_cache.get(payload.session_name)
    if mb is None:
        try:
            if len(regions) > 1:
                mb = SpatialMemoryBank.load(str(session_dir))
            else:
                mb = MemoryBank(feature_dim=768)
                mb.load(str(session_dir / "bank.pkl"))
            memory_bank_cache[payload.session_name] = mb
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load memory bank: {e}")
    else:
        # Debug print showing we reused the cached instance
        print(f"Using cached memory bank for session '{payload.session_name}'")
        
    # Convert once to RGB for the feature extractor (DINO is RGB-trained)
    masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    
    detector = AnomalyDetector(memory_bank=mb, feature_extractor=triton_extractor, k_neighbors=3, gaussian_sigma=2.0)
    
    score, heatmap, overlay_rgb = detector.detect_anomalies(masked_img_rgb, return_heatmap=True, alpha=0.5, regions=regions)
    
    # Convert overlay back to BGR for UI display and hstacking
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    overlay_b64 = img_to_b64(overlay_bgr)
    
    try:
        stacked = np.hstack([image_bgr, overlay_bgr])
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
        
    errors = []
    # crops_per_region[region_idx] = list of crop images from each training image
    crops_per_region = [[] for _ in range(len(regions) if len(regions) > 1 else 1)]
    
    for i, b64 in enumerate(payload.images_b64):
        try:
            image_bgr = b64_to_img(b64)
            if image_bgr is None:
                errors.append(f"img[{i}]: cv2.imdecode returned None (bad base64)")
                continue
            _, masked_img = apply_preprocessing(image_bgr, config.get("phase1", {}))
            masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
            if len(regions) > 1:
                crops = crop_regions(masked_img_rgb, regions)
                print(f"[build_memory_bank] img[{i}]: {len(crops)} crops from {len(regions)} regions")
                for r_idx, crop in enumerate(crops):
                    crops_per_region[r_idx].append(crop)
            else:
                crops_per_region[0].append(masked_img_rgb)
        except Exception as e:
            import traceback
            err_msg = f"img[{i}]: {type(e).__name__}: {e}"
            errors.append(err_msg)
            print(f"[build_memory_bank] ERROR {err_msg}")
            traceback.print_exc()
            continue

    total_crops = sum(len(c) for c in crops_per_region)
    print(f"[build_memory_bank] crops per region: {[len(c) for c in crops_per_region]}, errors: {len(errors)}")
    if errors:
        print(f"[build_memory_bank] error details: {errors}")

    if total_crops == 0:
        detail = f"No valid crops extracted. Errors: {'; '.join(errors[:3])}"
        raise HTTPException(status_code=400, detail=detail)

    # Extract features and add to respective banks
    if len(regions) > 1:
        for region_idx, region_crops in enumerate(crops_per_region):
            if not region_crops:
                print(f"[build_memory_bank] WARNING: No crops for region {region_idx}")
                continue
            print(f"[build_memory_bank] Extracting features for region {region_idx} ({len(region_crops)} crops)...")
            features, _ = triton_extractor.extract_batch(region_crops)
            print(f"[build_memory_bank] Region {region_idx} features shape: {features.shape}")
            mb.banks[region_idx].add_features(features)
    else:
        features, _ = triton_extractor.extract_batch(crops_per_region[0])
        mb.add_features(features)
        
        
    # 3. Save Bank
    session_dir = BASE_DIR / "memory_banks" / payload.session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    if len(regions) > 1:
        mb.save(str(session_dir))
    else:
        mb.save(str(session_dir / "bank.pkl"))
        
    # Cache the newly built memory bank immediately
    memory_bank_cache[payload.session_name] = mb
    
    return {
        "ok": True,
        "n_features_saved": mb.feature_count(),
        "ms": (time.time() - start_time) * 1000
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
