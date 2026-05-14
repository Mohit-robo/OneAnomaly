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

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2

from feature_extractor import FeatureExtractor
from memory_bank import MemoryBank
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

# Global Preprocessing State
preprocess_config = {"mode": "thresholding"}
averaged_bg_reference: Optional[np.ndarray] = None

# Paths
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
MEMORY_BANK_DIR = BASE_DIR / 'memory_banks'

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MEMORY_BANK_DIR.mkdir(exist_ok=True)


def initialize_models():
    """Initialize ML models on first request."""
    global feature_extractor, memory_bank, anomaly_detector
    
    if feature_extractor is None:
        print("Initializing feature extractor...")
        feature_extractor = FeatureExtractor(
            model_name='dinov3_vits16',
            device='cuda',
            backend='pytorch'
        )
        
        print("Initializing memory bank...")
        memory_bank = MemoryBank(
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
    try:
        test_file = request.files.get('test_image')
        if not test_file:
            return jsonify({'error': 'No test image provided'}), 400
            
        config_data = request.form.get('config')
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
                    local_reference = build_averaged_reference(images)
                except Exception:
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
            
        # Save output grid
        out_path = test_dir / f"grid_{test_file.filename}.png"
        cv2.imwrite(str(out_path), grid)
        
        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/extract_features', methods=['POST'])
def extract_features():
    """
    Extract features from uploaded good images.
    
    Expected form data:
        - files: Multiple image files OR
        - zip_file: Single zip file containing images
    """
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
                    file.save(temp_dir / file.filename)
        
        else:
            return jsonify({'error': 'No files provided'}), 400
        
        # Preprocess images before extraction
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

        # Extract features
        features, image_paths = feature_extractor.extract_from_folder(str(temp_dir))
        
        # Add to memory bank
        memory_bank.add_features(features, apply_coreset=True)
        
        return jsonify({
            'success': True,
            'num_images': len(image_paths),
            'num_features': features.shape[0],
            'num_features_after_coreset': memory_bank.feature_count(),
            'feature_dim': features.shape[1]
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/save_memory_bank', methods=['POST'])
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
            'filepath': str(filepath),
            'num_features': memory_bank.feature_count()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/load_memory_bank', methods=['POST'])
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


@app.route('/list_memory_banks', methods=['GET'])
def list_memory_banks():
    """List all saved memory banks."""
    try:
        banks = [f.name for f in MEMORY_BANK_DIR.glob('*.pkl')]
        return jsonify({
            'memory_banks': banks
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect_anomalies', methods=['POST'])
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
                    file.save(temp_dir / file.filename)
        
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

        # Process images
        results = anomaly_detector.detect_from_folder(
            str(temp_dir),
            str(session_dir)
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'session_id': session_id,
            'is_single_image': is_single_image,
            'results': results,
            'output_dir': str(session_dir)
        }
        
        # If single image, include image data
        if is_single_image:
            image_name = list(results.keys())[0]
            overlay_path = session_dir / f"{Path(image_name).stem}_overlay.png"
            
            if overlay_path.exists():
                response_data['overlay_image'] = str(overlay_path)
        
        return jsonify(response_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download_results/<session_id>', methods=['GET'])
def download_results(session_id):
    """Download results as a zip file."""
    try:
        session_dir = OUTPUT_DIR / 'sessions' / session_id
        
        if not session_dir.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        # Create zip file
        zip_path = OUTPUT_DIR / f"{session_id}_results.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in session_dir.rglob('*'):
                if file.is_file():
                    zipf.write(file, file.relative_to(session_dir))
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"{session_id}_results.zip"
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_image/<session_id>/<filename>', methods=['GET'])
def get_image(session_id, filename):
    """Get a specific image from session results."""
    try:
        image_path = OUTPUT_DIR / 'sessions' / session_id / filename
        
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path, mimetype='image/png')
    
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
    print("  POST /extract_features")
    print("  POST /save_memory_bank")
    print("  POST /load_memory_bank")
    print("  GET  /list_memory_banks")
    print("  POST /detect_anomalies")
    print("  GET  /download_results/<session_id>")
    print("  GET  /get_image/<session_id>/<filename>")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
