"""
Flask API Server
Provides REST API endpoints for the anomaly detection web application.
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional
import json

from flask import Flask, request, jsonify, send_file, stream_with_context
from flask_cors import CORS
import numpy as np
import cv2

from feature_extractor import FeatureExtractor
from memory_bank import MemoryBank, SpatialMemoryBank
from anomaly_detector import AnomalyDetector

# Import from local test scripts
import sys
sys.path.append(str(Path(__file__).parent))
from src.test_thresholding import apply_threshold_mask
from src.test_thresholding import build_result_grid as build_thresh_grid
from src.test_bg_subtraction import apply_mask as apply_bg_mask
from src.test_bg_subtraction import build_result_grid as build_bg_grid
from src.test_bg_subtraction import build_averaged_reference, load_images_from_dir


app = Flask(__name__)
CORS(app)

# Global state
feature_extractor: Optional[FeatureExtractor] = None
memory_bank: Optional[MemoryBank] = None
anomaly_detector: Optional[AnomalyDetector] = None
last_initialized_engine: Optional[str] = None

# Global Preprocessing State
preprocess_config = {"mode": "thresholding"}
averaged_bg_reference: Optional[np.ndarray] = None

# Global Spatial State
spatial_config = {
    "spatial_mode": "manual",
    "engine": "dinov3_manual_int30-255_bs3.engine",
    "regions": [],
    "grid_ratios": {"x": 50, "y": 50}
}

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
MEMORY_BANK_DIR = BASE_DIR / 'memory_banks'
MODELS_DIR = BASE_DIR / 'models'
ENGINE_DIR = MODELS_DIR / 'engine_files'
BG_REF_PATH = UPLOAD_DIR / 'averaged_bg_reference.npy'

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MEMORY_BANK_DIR.mkdir(exist_ok=True)
ENGINE_DIR.mkdir(exist_ok=True, parents=True)

def load_bg_ref():
    if BG_REF_PATH.exists():
        try:
            return np.load(str(BG_REF_PATH))
        except Exception as e:
            print(f"Error loading bg reference: {e}")
    return None

def save_bg_ref(ref):
    if ref is not None:
        try:
            np.save(str(BG_REF_PATH), ref)
        except Exception as e:
            print(f"Error saving bg reference: {e}")

# Initial Load
averaged_bg_reference = load_bg_ref()


def initialize_models(force_reinit=False):
    """Initialize ML models on first request or if engine changed."""
    global feature_extractor, memory_bank, anomaly_detector, last_initialized_engine
    
    current_engine = spatial_config.get('engine')
    
    if feature_extractor is None or current_engine != last_initialized_engine or force_reinit:
        print(f"Initializing feature extractor (Engine: {current_engine})...")
        
        # Determine engine path
        model_path = None
        if current_engine:
            model_path = str(ENGINE_DIR / current_engine)
            print(f"Using selected engine: {model_path}")
            
        feature_extractor = FeatureExtractor(
            model_name='dinov3_vits16',
            device='cuda',
            backend='tensorrt',
            model_path=model_path
        )
        
        last_initialized_engine = current_engine
        
        if memory_bank is None:
            print("Initializing Spatial Memory Bank...")
            # Start with default 1 region, it will be recreated/adjusted during extraction based on phase 2 context
            memory_bank = SpatialMemoryBank(
                num_regions=1,
                feature_dim=feature_extractor.get_feature_dimension(),
                use_gpu=True
            )
        
        print("Models initialized successfully!")


# Log Capture
import sys
import io
import time
from queue import Queue

class LogCapture:
    def __init__(self):
        self.terminal = sys.stdout
        self.log_queue = []
        self.start_time = time.time()
        
    def write(self, message):
        self.terminal.write(message)
        
        # Filter out polling logs to prevent spam in the web terminal
        if 'GET /api/logs' in message or 'GET /health' in message:
            return
            
        if message.strip():  # Only log non-empty messages
            # Store with timestamp
            self.log_queue.append({
                'time': time.strftime('%H:%M:%S'),
                'message': message
            })
            # Keep only last 1000 logs
            if len(self.log_queue) > 1000:
                self.log_queue.pop(0)

    def flush(self):
        self.terminal.flush()
        
    def get_logs(self, after_index=0):
        return self.log_queue[after_index:], len(self.log_queue)

    def clear(self):
        self.log_queue = []

# Redirect stdout and stderr
log_capture = LogCapture()
sys.stdout = log_capture
sys.stderr = log_capture


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get logs since a specific index."""
    try:
        after_index = int(request.args.get('after', 0))
        logs, current_index = log_capture.get_logs(after_index)
        return jsonify({
            'logs': logs,
            'next_index': current_index
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': feature_extractor is not None
    })


@app.route('/api/configure_preprocess', methods=['POST'])
def configure_preprocess():
    """Save preprocessing config and optionally build averaged background reference."""
    global preprocess_config, averaged_bg_reference
    try:
        data = request.form.get('config')
        if not data:
            return jsonify({'error': 'No config provided'}), 400
            
        config = json.loads(data)
        preprocess_config = config
        
        if config.get('mode') == 'average':
            files = request.files.getlist('ref_files')
            if files:
                bg_dir = UPLOAD_DIR / 'temp_bg'
                bg_dir.mkdir(exist_ok=True)
                # clear previous
                for item in bg_dir.iterdir():
                    if item.is_file(): item.unlink()
                    else: shutil.rmtree(item)
                
                # save items
                for file in files:
                    if file.filename.endswith('.zip'):
                        zip_path = bg_dir / 'upload.zip'
                        file.save(zip_path)
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(bg_dir)
                        zip_path.unlink()
                    elif file.filename:
                        file.save(bg_dir / file.filename)
                
                try:
                    images = load_images_from_dir(bg_dir)
                    averaged_bg_reference = build_averaged_reference(images)
                    save_bg_ref(averaged_bg_reference)
                except Exception as e:
                    return jsonify({'error': f"Failed to build reference: {str(e)}"}), 400
                    
        return jsonify({'success': True, 'config': preprocess_config})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/preview_mask', methods=['POST'])
def preview_mask():
    """Apply the current masking configuration to a test image and return the preview grid."""
    global averaged_bg_reference
    try:
        test_file = request.files.get('test_image')
        if not test_file:
            return jsonify({'error': 'No test image provided'}), 400
            
        config_data = request.form.get('config')
        return_result = request.form.get('return_result') == 'true'
        config = json.loads(config_data) if config_data else preprocess_config
        
        # Save temp test image
        test_dir = UPLOAD_DIR / 'temp_preview'
        test_dir.mkdir(exist_ok=True)
        test_path = test_dir / test_file.filename
        test_file.save(test_path)
        
        image_bgr = cv2.imread(str(test_path))
        if image_bgr is None:
            return jsonify({'error': 'Invalid test image format'}), 400
            
        # If average mode, we might need to build the reference locally if not saved
        local_reference = averaged_bg_reference
        if config.get('mode') == 'average':
            files = request.files.getlist('ref_files')
            if files:
                bg_dir = UPLOAD_DIR / 'temp_preview_bg'
                bg_dir.mkdir(exist_ok=True)
                for item in bg_dir.iterdir():
                    if item.is_file(): item.unlink()
                    else: shutil.rmtree(item)
                    
                for file in files:
                    if file.filename.endswith('.zip'):
                        zip_path = bg_dir / 'upload.zip'
                        file.save(zip_path)
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(bg_dir)
                        zip_path.unlink()
                    elif file.filename:
                        file.save(bg_dir / file.filename)
                try:
                    images = load_images_from_dir(bg_dir)
                    averaged_bg_reference = build_averaged_reference(images)
                    save_bg_ref(averaged_bg_reference)
                    local_reference = averaged_bg_reference
                except Exception as e:
                    print(f"Failed to build local reference in preview: {e}")
                    pass
            
            if local_reference is None:
                return jsonify({'error': 'No averaged background reference available.'}), 400
                
        # Apply mask based on config
        if config.get('mode') == 'thresholding':
            mask, masked_img, channel_img = apply_threshold_mask(
                image_bgr,
                channel=config.get('channel', 'gray'),
                thresh_min=config.get('thresh_min', 30),
                thresh_max=config.get('thresh_max', 220),
                morph_open=config.get('morph_open', 3),
                morph_close=config.get('morph_close', 5)
            )
            grid = build_thresh_grid(image_bgr, mask, masked_img, channel_img, config)
        else: # average
            mask, masked_img = apply_bg_mask(
                image_bgr,
                local_reference,
                diff_threshold=config.get('diff_threshold', 85),
                morph_open=config.get('morph_open', 5),
                morph_close=config.get('morph_close', 7),
                fill_holes=config.get('fill_holes', True),
                min_component_ratio=config.get('min_component_ratio', 0.1)
            )
            grid = build_bg_grid(image_bgr, local_reference, mask, masked_img, config, label="Preview")
            
        # Save output grid or single result
        if return_result:
            out_path = test_dir / f"masked_{test_file.filename}.jpg"
            cv2.imwrite(str(out_path), masked_img)
        else:
            out_path = test_dir / f"grid_{test_file.filename}.png"
            cv2.imwrite(str(out_path), grid)
        
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def get_spatial_regions(img_shape, config):
    """Get regional bounding boxes based on spatial config."""
    h_img, w_img = img_shape[:2]
    mode = config.get('spatial_mode', 'whole')
    
    if mode == 'whole':
        return [[0, 0, w_img, h_img]]
        
    elif mode == 'manual':
        regions = config.get('regions', [])
        if not regions:
            return [[0, 0, w_img, h_img]]
        
        coords = []
        for r in regions:
            x, y, w, h = int(r['x']), int(r['y']), int(r['w']), int(r['h'])
            # Ensure bounds
            x_start = max(min(x, w_img - 1), 0)
            y_start = max(min(y, h_img - 1), 0)
            x_end = max(min(x + w, w_img), x_start + 1)
            y_end = max(min(y + h, h_img), y_start + 1)
            coords.append([x_start, y_start, x_end - x_start, y_end - y_start])
        return coords if coords else [[0, 0, w_img, h_img]]
        
    elif mode == 'grid':
        ratios = config.get('grid_ratios', {'x': 50, 'y': 50})
        split_x = int(w_img * ratios['x'] / 100.0)
        split_y = int(h_img * ratios['y'] / 100.0)
        
        return [
            [0, 0, split_x, split_y],
            [split_x, 0, w_img - split_x, split_y],
            [0, split_y, split_x, h_img - split_y],
            [split_x, split_y, w_img - split_x, h_img - split_y]
        ]
        
    return [[0, 0, w_img, h_img]]

def crop_regions(image_bgr, config):
    """Crop image into regions based on spatial config."""
    coords = get_spatial_regions(image_bgr.shape, config)
    crops = []
    for x, y, w, h in coords:
        crops.append(image_bgr[y:y+h, x:x+w])
    return crops


@app.route('/api/extract_features', methods=['POST'])
def extract_features():
    def generate():
        yield json.dumps({'title': 'Extracting...', 'message': 'Initializing extraction...', 'progress': 0}) + '\n'
        
        try:
            initialize_models()
            
            # Create temporary directory for this request
            temp_dir = UPLOAD_DIR / 'good'
            temp_dir.mkdir(exist_ok=True)
            
            # Clear previous temp files
            for item in temp_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            
            # Handle file uploads
            if 'zip_file' in request.files:
                # Handle zip file
                zip_file = request.files['zip_file']
                zip_path = temp_dir / 'upload.zip'
                zip_file.save(zip_path)
                
                # Extract zip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                zip_path.unlink()
            
            elif 'files' in request.files:
                # Handle multiple files
                files = request.files.getlist('files')
                for file in files:
                    if file.filename:
                        save_path = temp_dir / file.filename
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        file.save(save_path)
            
            else:
                yield json.dumps({'error': 'No files provided'}) + '\n'
                return
            
            # Identify valid image files
            image_files = []
            for item in temp_dir.rglob('*'):
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.append(item)
                    
            # Subsampling if num_shots forms a limit
            num_shots_str = request.form.get('num_shots')
            if num_shots_str and num_shots_str.strip():
                num_shots = int(num_shots_str)
                if 0 < num_shots < len(image_files):
                    import random
                    image_files = random.sample(image_files, num_shots)

            yield json.dumps({'title': 'Preprocessing', 'message': f'Processing {len(image_files)} images...', 'progress': 10}) + '\n'

            # Ensure spatial memory bank is exactly matching logic
            all_crops = []
            import cv2
            
            total_images = len(image_files)
            for idx, item in enumerate(image_files):
                image_bgr = cv2.imread(str(item))
                if image_bgr is not None:
                    # 1. Preprocessing mask
                    if preprocess_config.get('mode') == 'thresholding':
                        _, masked_img, _ = apply_threshold_mask(
                            image_bgr,
                            channel=preprocess_config.get('channel', 'gray'),
                            thresh_min=preprocess_config.get('thresh_min', 30),
                            thresh_max=preprocess_config.get('thresh_max', 220),
                            morph_open=preprocess_config.get('morph_open', 3),
                            morph_close=preprocess_config.get('morph_close', 5)
                        )
                        image_bgr = masked_img
                    elif preprocess_config.get('mode') == 'average' and averaged_bg_reference is not None:
                        _, masked_img = apply_bg_mask(
                            image_bgr,
                            averaged_bg_reference,
                            diff_threshold=preprocess_config.get('diff_threshold', 85),
                            morph_open=preprocess_config.get('morph_open', 5),
                            morph_close=preprocess_config.get('morph_close', 7),
                            fill_holes=preprocess_config.get('fill_holes', True),
                            min_component_ratio=preprocess_config.get('min_component_ratio', 0.1)
                        )
                        image_bgr = masked_img
                    
                    # 2. Spatial crop
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    crops = crop_regions(image_rgb, spatial_config)
                    all_crops.extend(crops)
                
                if (idx + 1) % 5 == 0 or idx == total_images - 1:
                    progress = 10 + int(70 * (idx + 1) / total_images)
                    yield json.dumps({'title': 'Preprocessing', 'message': f'Image {idx+1}/{total_images}', 'progress': progress}) + '\n'
                    
            global memory_bank
            num_regions = len(crops) if all_crops else 1
            
            # Reset memory bank to correctly reflect the regions
            memory_bank = SpatialMemoryBank(
                num_regions=num_regions,
                feature_dim=feature_extractor.get_feature_dimension(),
                use_gpu=True
            )
            if all_crops:
                 memory_bank.regions_coords = get_spatial_regions(cv2.imread(str(UPLOAD_DIR / image_files[0])).shape, spatial_config)

            if all_crops:
                yield json.dumps({'title': 'Extraction', 'message': 'Extracting and indexing features...', 'progress': 85}) + '\n'
                features, _ = feature_extractor.extract_batch(all_crops)
                memory_bank.add_features_batch(features, apply_coreset=True)
                num_features = features.shape[0]
            else:
                num_features = 0

            yield json.dumps({
                'success': True,
                'num_images': len(image_files),
                'num_features': num_features,
                'num_features_after_coreset': memory_bank.feature_count(),
                'num_regions': memory_bank.num_regions,
                'feature_dim': feature_extractor.get_feature_dimension(),
                'progress': 100
            }) + '\n'

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield json.dumps({'error': str(e)}) + '\n'

    return app.response_class(stream_with_context(generate()), mimetype='application/json')


@app.route('/api/save_memory_bank', methods=['POST'])
def save_memory_bank():
    """
    Save current memory bank to disk.
    
    Expected JSON:
        - filename: Name for the memory bank file
    """
    try:
        if memory_bank is None or not memory_bank.is_fitted:
            return jsonify({'error': 'No memory bank to save'}), 400
        
        data = request.json
        filename = data.get('filename', 'memory_bank')
        
        # Ensure .pkl extension
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        
        filepath = MEMORY_BANK_DIR / filename
        memory_bank.save(str(filepath))
        
        return jsonify({
            'success': True,
            'num_features': memory_bank.feature_count()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_memory_bank', methods=['POST'])
def load_memory_bank():
    """
    Load memory bank from disk.
    
    Expected JSON:
        - filename: Name of the memory bank file
    """
    try:
        initialize_models()
        
        data = request.json
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        filepath = MEMORY_BANK_DIR / filename
        
        if not filepath.exists():
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        memory_bank.load(str(filepath))
        
        return jsonify({
            'success': True,
            'num_features': memory_bank.feature_count(),
            'feature_dim': memory_bank.feature_dim
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list_memory_banks', methods=['GET'])
def list_memory_banks():
    """List all saved memory banks."""
    try:
        banks = [f.name for f in MEMORY_BANK_DIR.glob('*.pkl')]
        return jsonify({
            'memory_banks': banks
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect_anomalies', methods=['POST'])
def detect_anomalies():
    """
    Detect anomalies in uploaded test images.
    
    Expected form data:
        - files: Multiple image files OR
        - zip_file: Single zip file containing images OR
        - single_image: Single image file
        - session_id: Unique session ID for this detection run
    """
    try:
        if memory_bank is None or not memory_bank.is_fitted:
            return jsonify({'error': 'No memory bank loaded'}), 400
        
        initialize_models()
        
        # Create anomaly detector
        global anomaly_detector
        anomaly_detector = AnomalyDetector(
            memory_bank=memory_bank,
            feature_extractor=feature_extractor
        )
        
        # Get session ID
        session_id = request.form.get('session_id', 'default_session')
        
        # Create session directory
        session_dir = OUTPUT_DIR / 'sessions' / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear previous session files
        for item in session_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Create temp directory for test images
        temp_dir = UPLOAD_DIR / 'temp_test'
        temp_dir.mkdir(exist_ok=True)
        
        # Clear previous temp files
        for item in temp_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Handle file uploads
        is_single_image = False
        
        if 'single_image' in request.files:
            # Single image
            is_single_image = True
            image_file = request.files['single_image']
            image_path = temp_dir / image_file.filename
            image_file.save(image_path)
        
        elif 'zip_file' in request.files:
            # Zip file
            zip_file = request.files['zip_file']
            zip_path = temp_dir / 'upload.zip'
            zip_file.save(zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            zip_path.unlink()
        
        elif 'files' in request.files:
            # Multiple files
            files = request.files.getlist('files')
            for file in files:
                if file.filename:
                    save_path = temp_dir / file.filename
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    file.save(save_path)
        
        else:
            return jsonify({'error': 'No files provided'}), 400
        
        # Preprocess images before detection
        for item in temp_dir.rglob('*'):
            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_bgr = cv2.imread(str(item))
                if image_bgr is not None:
                    if preprocess_config.get('mode') == 'thresholding':
                        _, masked_img, _ = apply_threshold_mask(
                            image_bgr,
                            channel=preprocess_config.get('channel', 'gray'),
                            thresh_min=preprocess_config.get('thresh_min', 30),
                            thresh_max=preprocess_config.get('thresh_max', 220),
                            morph_open=preprocess_config.get('morph_open', 3),
                            morph_close=preprocess_config.get('morph_close', 5)
                        )
                        cv2.imwrite(str(item), masked_img)
                    elif preprocess_config.get('mode') == 'average' and averaged_bg_reference is not None:
                        _, masked_img = apply_bg_mask(
                            image_bgr,
                            averaged_bg_reference,
                            diff_threshold=preprocess_config.get('diff_threshold', 85),
                            morph_open=preprocess_config.get('morph_open', 5),
                            morph_close=preprocess_config.get('morph_close', 7),
                            fill_holes=preprocess_config.get('fill_holes', True),
                            min_component_ratio=preprocess_config.get('min_component_ratio', 0.1)
                        )
                        cv2.imwrite(str(item), masked_img)

        anomaly_threshold = float(request.form.get('threshold', 0.5))
        heatmap_alpha = float(request.form.get('alpha', 0.5))

        # Process images
        results_dict = anomaly_detector.detect_from_folder(
            str(temp_dir),
            str(session_dir),
            alpha=heatmap_alpha
        )
        
        # Convert results to list for frontend
        results_list = []
        max_score = 0
        for fname, info in results_dict.items():
            score = info.get('anomaly_score')
            if score is None: score = 0
            
            if score > max_score:
                max_score = score
            results_list.append({
                'filename': fname,
                'anomaly_score': score,
                'is_anomaly': score > anomaly_threshold,
                'processed': info.get('processed', False)
            })

        # Prepare response
        response_data = {
            'success': True,
            'session_id': session_id,
            'num_images': len(results_list),
            'max_score': max_score,
            'results': results_list,
            'output_dir': str(session_dir)
        }
        
        # If single image, include image data
        if is_single_image and results_list:
            image_name = results_list[0]['filename']
            overlay_path = session_dir / f"{Path(image_name).stem}_overlay.png"
            
            if overlay_path.exists():
                response_data['overlay_image'] = str(overlay_path)
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_results/<session_id>', methods=['GET'])
def download_results(session_id):
    """Download results as a zip file."""
    try:
        session_dir = OUTPUT_DIR / 'sessions' / session_id
        
        if not session_dir.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        # Create zip file — overlay images only (heatmap baked over source)
        zip_path = OUTPUT_DIR / f"{session_id}_results.zip"
        
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            overlay_files = sorted(session_dir.glob('torch_res_*.png'))
            if not overlay_files:
                # Fallback: include everything if no overlays found
                overlay_files = [f for f in session_dir.rglob('*') if f.is_file()]
            for file in overlay_files:
                zipf.write(file, file.name)
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"{session_id}_overlays.zip"
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_image/<session_id>/<filename>', methods=['GET'])
def get_image(session_id, filename):
    """Get a specific image from session results."""
    try:
        image_path = OUTPUT_DIR / 'sessions' / session_id / filename
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path, mimetype='image/png')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/save_session', methods=['POST'])
def save_session():
    """Save the full pipeline session config as a JSON file for reproducibility."""
    try:
        payload = request.json
        if not payload:
            return jsonify({'error': 'No session data provided'}), 400

        sessions_dir = OUTPUT_DIR / 'sessions'
        sessions_dir.mkdir(parents=True, exist_ok=True)

        session_id = payload.get('session_id') or ('session_' + str(int(time.time() * 1000)))
        filename = f"{session_id}_config.json"
        save_path = sessions_dir / filename

        with open(save_path, 'w') as f:
            json.dump(payload, f, indent=4)

        print(f"Session config saved: {save_path}")
        return jsonify({'success': True, 'filename': filename, 'path': str(save_path)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list_engines', methods=['GET'])
def list_engines():
    """List all available TensorRT engine files."""
    engines = [f.name for f in ENGINE_DIR.glob('*.engine')]
    return jsonify({'engines': engines})


@app.route('/api/configure_spatial', methods=['POST'])
def configure_spatial():
    """Receive and store the active spatial partitioning configuration."""
    global spatial_config
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No config provided'}), 400
        
        if not data.get('engine'):
            return jsonify({'error': 'Memory Bank creation requires a TensorRT engine. Please select or export one in Stage 2.'}), 400
            
        spatial_config.update(data)
        
        # Save to persistent JSON
        config_path = BASE_DIR / 'pipeline_config.json'
        session_config = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    session_config = json.load(f)
            except json.JSONDecodeError:
                pass
        
        session_config['spatial'] = spatial_config
        with open(config_path, 'w') as f:
            json.dump(session_config, f, indent=4)
            
        return jsonify({'success': True, 'config': spatial_config})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export_trt', methods=['POST'])
def export_trt():
    """Trigger TRT model export and stream status updates."""
    import subprocess
    import sys
    
    try:
        data = request.json
        batch_size = data.get('batch_size', 1)
        spatial_mode = data.get('spatial_mode', 'whole')
        pre_proc = data.get('pre_proc', 'default')
        
        name_root = f"dinov3_{spatial_mode}_{pre_proc}_bs{batch_size}"
        
        def generate():
            yield json.dumps({'title': 'Exporting...', 'message': 'Initializing build pipeline...', 'progress': 5}) + '\n'
            
            # Paths and Environment
            export_script = BASE_DIR / 'python' / 'spatial_anomaly_scripts' / 'export_dinov3_onnx_dynamic.py'
            trt_script = BASE_DIR / 'python' / 'spatial_anomaly_scripts' / 'jetson_deploy' / 'convert_onnx_to_trt.py'
            onnx_path = ENGINE_DIR / f'{name_root}.onnx'
            engine_path = ENGINE_DIR / f'{name_root}.engine'
            weights_path = MODELS_DIR / 'dinov3_vits16_pretrain_lvd1689m.pth'
            repo_path = BASE_DIR / 'python' / 'dinov3'

            def run_and_stream(cmd, step_name, start_progress, end_progress, cwd):
                print(f"Running {step_name}: {' '.join(cmd)}")
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(cwd)
                )
                for line in process.stdout:
                    if line.strip():
                        yield json.dumps({
                            'title': f'{step_name} Logs', 
                            'message': line.strip(), 
                            'progress': start_progress + (end_progress - start_progress) // 2
                        }) + '\n'
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)

            try:
                # Step 1: Export ONNX
                cmd_onnx = [
                    sys.executable, str(export_script),
                    '--output_path', str(onnx_path),
                    '--batch_size', str(batch_size),
                    '--weights_path', str(weights_path),
                    '--repo_path', str(repo_path),
                    '--fixed_batch'
                ]
                for update in run_and_stream(cmd_onnx, "ONNX Export", 5, 50, export_script.parent):
                    yield update
                
                # Step 2: Convert ONNX to TensorRT using trtexec
                cmd_trt = [
                    'trtexec',
                    f'--onnx={onnx_path}',
                    f'--saveEngine={engine_path}'
                ]
                for update in run_and_stream(cmd_trt, "TRT Compilation", 50, 100, BASE_DIR):
                    yield update
                
                yield json.dumps({
                    'title': 'Success!', 
                    'message': f'TRT Engine created: {engine_path.name}', 
                    'progress': 100,
                    'success': True
                }) + '\n'

            except subprocess.CalledProcessError as e:
                yield json.dumps({'title': 'Export Failed', 'message': f'Process failed with code {e.returncode}. Check the logs above.', 'progress': 0}) + '\n'
            except Exception as e:
                yield json.dumps({'title': 'Export Error', 'message': str(e), 'progress': 0}) + '\n'

        return app.response_class(stream_with_context(generate()), mimetype='application/json')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Anomaly Detection API Server")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  POST /api/configure_preprocess")
    print("  POST /api/preview_mask")
    print("  POST /api/extract_features")
    print("  POST /api/save_memory_bank")
    print("  POST /api/load_memory_bank")
    print("  GET  /api/list_memory_banks")
    print("  POST /api/detect_anomalies")
    print("  GET  /api/download_results/<session_id>")
    print("  GET  /api/get_image/<session_id>/<filename>")
    print("  POST /api/save_session")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
