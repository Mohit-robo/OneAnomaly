# Jetson TensorRT Deployment Package

This package contains everything needed to run DINOv3 anomaly detection on Jetson devices using TensorRT, **without requiring PyTorch**.

## System Requirements

- **Jetpack**: 5.1.2
- **TensorRT**: 8.5.x (included with Jetpack)
- **CUDA**: 11.4 (included with Jetpack)
- **Python**: 3.8+

## Installation

### 1. Install Python Dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Install FAISS GPU

For Jetson, you may need to build FAISS from source:

```bash
# Or try pre-built wheel if available
pip3 install faiss-gpu
```

### 3. Install CUDA Python Bindings

```bash
pip3 install cuda-python
```

## Usage

### Step 1: Convert ONNX to TensorRT Engine

First, transfer your ONNX model to the Jetson device, then convert it:

```bash
python3 convert_onnx_to_trt.py \
    --onnx_path /path/to/dinov3_vits16.onnx \
    --engine_path ./trt_engines/dinov3_vits16.trt \
    --fp16 \
    --workspace 4
```

**Arguments:**
- `--onnx_path`: Path to your ONNX model
- `--engine_path`: Where to save the TensorRT engine
- `--fp16`: Enable FP16 precision (recommended for Jetson)
- `--workspace`: Max workspace size in GB (default: 4)

### Step 2: Run Inference

```bash
python3 simple_inference_trt.py \
    --model_name dinov3_vits16 \
    --trt_path ./trt_engines/dinov3_vits16.trt \
    --train_dir /path/to/normal_images \
    --test_dir /path/to/test_images \
    --shots 500 \
    --k 5 \
    --sigma 8 \
    --vmax 0.15 \
    --output_dir ./results_trt
```

**Key Arguments:**
- `--trt_path`: Path to TensorRT engine file
- `--train_dir`: Folder with normal/good images for memory bank
- `--test_dir`: Folder with test images (good + anomalous)
- `--shots`: Number of normal images to use for memory bank
- `--k`: Number of neighbors for k-NN averaging (default: 1)
- `--sigma`: Gaussian smoothing sigma for heatmap (default: 4)
- `--vmax`: Fixed maximum for heatmap scaling (e.g., 0.15)
- `--save_bank`: Save memory bank to `.npy` file for reuse
- `--load_bank`: Load pre-computed memory bank

## File Structure

```
jetson_deploy/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── convert_onnx_to_trt.py      # ONNX to TensorRT conversion
├── simple_inference_trt.py     # Main inference script
└── src/
    ├── backbones_trt.py        # TensorRT wrapper with CUDA
    ├── utils.py                # Utility functions
    └── post_eval.py            # Evaluation metrics
```

## Performance Tips

1. **Use FP16**: Always enable `--fp16` for faster inference on Jetson
2. **Adjust Workspace**: Increase `--workspace` if you have more RAM available
3. **Memory Bank Caching**: Use `--save_bank` to save the memory bank and `--load_bank` to reuse it
4. **Batch Processing**: Process multiple images in the same run to amortize initialization costs

## Troubleshooting

### CUDA Initialization Error
- Ensure CUDA is properly installed: `nvidia-smi`
- Check TensorRT version: `dpkg -l | grep tensorrt`

### Out of Memory
- Reduce `--workspace` size
- Reduce `--shots` (number of normal images)
- Use FP16 precision

### Slow Inference
- Verify TensorRT engine was built with FP16
- Check GPU utilization: `tegrastats`
- Ensure FAISS is using GPU: Check for "Using FAISS GPU search" message

## Notes

- TensorRT engines are platform-specific and must be built on the target Jetson device
- The engine file cannot be transferred between different GPU architectures
- All preprocessing uses OpenCV/numpy (no PyTorch required)
- FAISS GPU is used for fast similarity search
