"""
Flask API Server (Cloud Side)
====================================================
Responsibilities:
  - Serve the static frontend (public/)
  - Accept user uploads and proxy inference requests to the local Edge Gateway
  - Persist session configs to Amazon S3 (or local FS in dev mode)

This container performs ZERO GPU work. All heavy ML inference is
delegated to the local Gateway via GATEWAY_URL.
"""

import os
import sys
import io
import shutil
import zipfile
import base64
import random
import traceback
import json
import time
from pathlib import Path
from typing import Optional
from queue import Queue

from flask import Flask, request, jsonify, send_file, stream_with_context
from flask_cors import CORS
import numpy as np
import cv2
import requests
import boto3

# ---------------------------------------------------------------------------
# AWS Config
# ---------------------------------------------------------------------------
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "")
s3_client = boto3.client("s3") if S3_BUCKET else None

# ---------------------------------------------------------------------------
# Gateway Config
# ---------------------------------------------------------------------------
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")

# ---------------------------------------------------------------------------
# Preprocessing utilities (used only for preview_mask - CPU side only)
# ---------------------------------------------------------------------------
from src.test_thresholding import apply_threshold_mask
from src.test_thresholding import build_result_grid as build_thresh_grid
from src.test_bg_subtraction import apply_mask as apply_bg_mask
from src.test_bg_subtraction import build_result_grid as build_bg_grid
from src.test_bg_subtraction import build_averaged_reference, load_images_from_dir

# ---------------------------------------------------------------------------
# Paths (local / dev fallback)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'outputs'
BG_REF_PATH = UPLOAD_DIR / 'averaged_bg_reference.npy'

# Create local directories (used in dev mode; cloud uses /tmp equivalent)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(BASE_DIR / 'public'), static_url_path='')
CORS(app)

# ---------------------------------------------------------------------------
# In-process Session State
# ---------------------------------------------------------------------------
current_session_name = "default_session"
preprocess_config = {"mode": "thresholding"}
averaged_bg_reference: Optional[np.ndarray] = None
spatial_config = {
    "spatial_mode": "manual",
    "regions": [],
    "grid_ratios": {"x": 50, "y": 50}
}

# ---------------------------------------------------------------------------
# Background reference helpers
# ---------------------------------------------------------------------------
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

averaged_bg_reference = load_bg_ref()

# ---------------------------------------------------------------------------
# Log Capture (web terminal)
# ---------------------------------------------------------------------------
class LogCapture:
    def __init__(self):
        self.terminal = sys.stdout
        self.log_queue = []
        self.start_time = time.time()

    def write(self, message):
        self.terminal.write(message)
        if 'GET /api/logs' in message or 'GET /health' in message:
            return
        if message.strip():
            self.log_queue.append({
                'time': time.strftime('%H:%M:%S'),
                'message': message
            })
            if len(self.log_queue) > 1000:
                self.log_queue.pop(0)

    def flush(self):
        self.terminal.flush()

    def get_logs(self, after_index=0):
        return self.log_queue[after_index:], len(self.log_queue)

    def clear(self):
        self.log_queue = []

log_capture = LogCapture()
sys.stdout = log_capture
sys.stderr = log_capture

# ===========================================================================
# Routes
# ===========================================================================

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    gateway_ok = False
    try:
        r = requests.get(f"{GATEWAY_URL}/health", timeout=2)
        gateway_ok = r.status_code == 200
    except Exception:
        pass
    return jsonify({
        'status': 'healthy',
        'gateway_url': GATEWAY_URL,
        'gateway_reachable': gateway_ok,
        's3_configured': s3_client is not None,
    })


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get logs since a specific index."""
    try:
        after_index = int(request.args.get('after', 0))
        logs, current_index = log_capture.get_logs(after_index)
        return jsonify({'logs': logs, 'next_index': current_index})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
                except Exception as e:
                    return jsonify({'error': f"Failed to build reference: {str(e)}"}), 400

        return jsonify({'success': True, 'config': preprocess_config})
    except Exception as e:
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

        test_dir = UPLOAD_DIR / 'temp_preview'
        test_dir.mkdir(exist_ok=True)
        test_path = test_dir / test_file.filename
        test_file.save(test_path)

        image_bgr = cv2.imread(str(test_path))
        if image_bgr is None:
            return jsonify({'error': 'Invalid test image format'}), 400

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

            if local_reference is None:
                return jsonify({'error': 'No averaged background reference available.'}), 400

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
        else:
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

        if return_result:
            out_path = test_dir / f"masked_{test_file.filename}.jpg"
            cv2.imwrite(str(out_path), masked_img)
        else:
            out_path = test_dir / f"grid_{test_file.filename}.png"
            cv2.imwrite(str(out_path), grid)

        return send_file(out_path, mimetype='image/png')
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/configure_spatial', methods=['POST'])
def configure_spatial():
    """Receive and store the active spatial partitioning configuration."""
    global spatial_config
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'error': 'No json config provided or invalid json.'}), 400

        spatial_config.update(data)

        config_path = BASE_DIR / 'pipeline_config.json'
        session_config_file = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    session_config_file = json.load(f)
            except json.JSONDecodeError:
                pass

        session_config_file['spatial'] = spatial_config
        with open(config_path, 'w') as f:
            json.dump(session_config_file, f, indent=4)

        return jsonify({'success': True, 'config': spatial_config})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract_features', methods=['POST'])
def extract_features():
    def generate():
        yield json.dumps({'title': 'Extracting...', 'message': 'Initializing extraction...', 'progress': 0}) + '\n'

        try:
            temp_dir = UPLOAD_DIR / 'good'
            temp_dir.mkdir(exist_ok=True)

            for item in temp_dir.iterdir():
                if item.is_file(): item.unlink()
                elif item.is_dir(): shutil.rmtree(item)

            if 'zip_file' in request.files:
                zip_file = request.files['zip_file']
                zip_path = temp_dir / 'upload.zip'
                zip_file.save(zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                zip_path.unlink()
            elif 'files' in request.files:
                for file in request.files.getlist('files'):
                    if file.filename:
                        save_path = temp_dir / file.filename
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        file.save(save_path)
            else:
                yield json.dumps({'error': 'No files provided'}) + '\n'
                return

            image_files = [
                item for item in temp_dir.rglob('*')
                if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ]

            num_shots_str = request.form.get('num_shots')
            if num_shots_str and num_shots_str.strip():
                num_shots = int(num_shots_str)
                if 0 < num_shots < len(image_files):
                    image_files = random.sample(image_files, num_shots)

            yield json.dumps({'title': 'Preprocessing', 'message': f'Sending {len(image_files)} images to Gateway...', 'progress': 10}) + '\n'

            config_payload = {"phase1": preprocess_config, "phase2": spatial_config}
            requests.post(f"{GATEWAY_URL}/sync_session", json={
                "session_name": current_session_name,
                "config": config_payload
            }, timeout=10)

            yield json.dumps({'title': 'Extraction', 'message': 'Gateway indexing features...', 'progress': 50}) + '\n'

            images_b64 = []
            for item in image_files:
                with open(item, "rb") as f:
                    images_b64.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8'))

            resp = requests.post(f"{GATEWAY_URL}/build_memory_bank", json={
                "session_name": current_session_name,
                "images_b64": images_b64
            }, timeout=300)

            if resp.status_code != 200:
                raise Exception(f"Gateway Error: {resp.text}")

            result_data = resp.json()
            yield json.dumps({
                'success': True,
                'num_images': len(image_files),
                'num_features_after_coreset': result_data.get('n_features_saved', 0),
                'progress': 100
            }) + '\n'

        except Exception as e:
            traceback.print_exc()
            yield json.dumps({'error': str(e)}) + '\n'

    return app.response_class(stream_with_context(generate()), mimetype='application/json')


@app.route('/api/save_memory_bank', methods=['POST'])
def save_memory_bank():
    """
    Rename/alias a memory bank on the gateway side.
    The gateway owns the memory_banks/ directory; we proxy the instruction.
    """
    try:
        data = request.json
        bank_name = data.get('filename')
        if not bank_name:
            return jsonify({'error': 'Filename is required'}), 400

        # Proxy rename request to gateway
        resp = requests.post(f"{GATEWAY_URL}/rename_memory_bank", json={
            "session_name": current_session_name,
            "new_name": bank_name
        }, timeout=15)

        if resp.status_code == 200:
            return jsonify({'success': True, 'message': f'Memory bank saved as {bank_name}'})
        else:
            return jsonify({'error': resp.text}), resp.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Gateway unreachable. Ensure the local gateway is running.'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_memory_bank', methods=['POST'])
def load_memory_bank():
    """
    Instruct the gateway to pre-load a specific memory bank into its cache.
    The gateway automatically loads banks on the first infer call, so this
    endpoint is a convenience for eager loading.
    """
    try:
        data = request.json or {}
        bank_name = data.get('bank_name', current_session_name)

        resp = requests.post(f"{GATEWAY_URL}/load_memory_bank", json={
            "session_name": bank_name
        }, timeout=30)

        if resp.status_code == 200:
            return jsonify({'success': True, 'message': f'Memory bank {bank_name} loaded on gateway.'})
        else:
            return jsonify({'error': resp.text}), resp.status_code

    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Gateway unreachable.'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list_memory_banks', methods=['GET'])
def list_memory_banks():
    """List all saved memory banks by querying the gateway."""
    try:
        resp = requests.get(f"{GATEWAY_URL}/list_memory_banks", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            banks = data.get('banks', [])
            return jsonify({'banks': banks, 'memory_banks': banks})
        else:
            return jsonify({'error': resp.text}), resp.status_code
    except requests.exceptions.ConnectionError:
        return jsonify({'banks': [], 'memory_banks': [], 'warning': 'Gateway unreachable.'}), 200
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
        - threshold: Anomaly score threshold (default 0.5)
    """
    try:
        session_id = request.form.get('session_id', 'default_session')
        anomaly_threshold = float(request.form.get('threshold', 0.5))

        session_dir = OUTPUT_DIR / 'sessions' / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        for item in session_dir.iterdir():
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item)

        temp_dir = UPLOAD_DIR / 'temp_test'
        temp_dir.mkdir(exist_ok=True)

        for item in temp_dir.iterdir():
            if item.is_file(): item.unlink()
            elif item.is_dir(): shutil.rmtree(item)

        is_single_image = False

        if 'single_image' in request.files:
            is_single_image = True
            image_file = request.files['single_image']
            image_path = temp_dir / image_file.filename
            image_file.save(image_path)
        elif 'zip_file' in request.files:
            zip_file = request.files['zip_file']
            zip_path = temp_dir / 'upload.zip'
            zip_file.save(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            zip_path.unlink()
        elif 'files' in request.files:
            for file in request.files.getlist('files'):
                if file.filename:
                    save_path = temp_dir / file.filename
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    file.save(save_path)
        else:
            return jsonify({'error': 'No files provided'}), 400

        # Sync session config to gateway before inference
        config_payload = {"phase1": preprocess_config, "phase2": spatial_config}
        requests.post(f"{GATEWAY_URL}/sync_session", json={
            "session_name": current_session_name,
            "config": config_payload
        }, timeout=10)

        results_list = []
        max_score = 0.0

        for item in temp_dir.rglob('*'):
            if not (item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']):
                continue

            with open(item, "rb") as f:
                b64_str = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode('utf-8')

            resp = requests.post(f"{GATEWAY_URL}/infer", json={
                "session_name": current_session_name,
                "image_b64": b64_str,
                "phase": 4
            }, timeout=60)

            if resp.status_code != 200:
                print(f"Gateway infer error for {item.name}: {resp.text}")
                continue

            info = resp.json()
            score = info.get("score", 0.0)
            if score > max_score:
                max_score = score

            results_list.append({
                'filename': item.name,
                'anomaly_score': score,
                'is_anomaly': score > anomaly_threshold,
                'processed': True
            })

            overlay_b64 = info.get("stacked_b64") or info.get("heatmap_b64")
            if overlay_b64:
                stem = item.stem
                shutil.copy(item, session_dir / f"{stem}_source.png")

                b64_data = overlay_b64.split(',')[1] if ',' in overlay_b64 else overlay_b64
                overlay_data = base64.b64decode(b64_data)

                (session_dir / f"{stem}_overlay.png").write_bytes(overlay_data)
                (session_dir / f"torch_res_{stem}.png").write_bytes(overlay_data)

        response_data = {
            'success': True,
            'session_id': session_id,
            'num_images': len(results_list),
            'max_score': max_score,
            'results': results_list,
        }

        if is_single_image and results_list:
            stem = Path(results_list[0]['filename']).stem
            overlay_path = session_dir / f"{stem}_overlay.png"
            if overlay_path.exists():
                response_data['overlay_image'] = str(overlay_path)

        return jsonify(response_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_results/<session_id>', methods=['GET'])
def download_results(session_id):
    """Download results as a zip file."""
    try:
        session_dir = OUTPUT_DIR / 'sessions' / session_id

        if not session_dir.exists():
            return jsonify({'error': 'Session not found'}), 404

        zip_path = OUTPUT_DIR / f"{session_id}_results.zip"

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
            overlay_files = sorted(session_dir.glob('torch_res_*.png'))
            if not overlay_files:
                overlay_files = [f for f in session_dir.rglob('*') if f.is_file()]
            for file in overlay_files:
                zipf.write(file, file.name)

        return send_file(zip_path, as_attachment=True, download_name=f"{session_id}_overlays.zip")

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
    """Save the full pipeline session config (S3 in cloud, local FS in dev)."""
    try:
        payload = request.json
        if not payload:
            return jsonify({'error': 'No session data provided'}), 400

        session_id = payload.get('session_id') or ('session_' + str(int(time.time() * 1000)))
        filename = f"{session_id}_config.json"

        if s3_client:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=f"sessions/{filename}",
                Body=json.dumps(payload, indent=4)
            )
            print(f"Session config saved to S3: s3://{S3_BUCKET}/sessions/{filename}")
            return jsonify({'success': True, 'filename': filename, 'path': f"s3://{S3_BUCKET}/sessions/{filename}"})
        else:
            sessions_dir = OUTPUT_DIR / 'sessions'
            sessions_dir.mkdir(parents=True, exist_ok=True)
            save_path = sessions_dir / filename
            with open(save_path, 'w') as f:
                json.dump(payload, f, indent=4)
            print(f"Session config saved locally: {save_path}")
            return jsonify({'success': True, 'filename': filename, 'path': str(save_path)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===========================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("OneAnomaly Cloud API Server")
    print("="*60)
    print(f"\nGATEWAY_URL : {GATEWAY_URL}")
    print(f"S3_BUCKET   : {S3_BUCKET or '(local mode)'}")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /api/logs")
    print("  POST /api/configure_preprocess")
    print("  POST /api/preview_mask")
    print("  POST /api/configure_spatial")
    print("  POST /api/extract_features")
    print("  POST /api/save_memory_bank")
    print("  POST /api/load_memory_bank")
    print("  GET  /api/list_memory_banks")
    print("  POST /api/detect_anomalies")
    print("  GET  /api/download_results/<session_id>")
    print("  GET  /api/get_image/<session_id>/<filename>")
    print("  POST /api/save_session")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)
