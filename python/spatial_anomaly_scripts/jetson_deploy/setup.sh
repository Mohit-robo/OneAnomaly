#!/bin/bash
# Quick setup script for Jetson deployment

echo "==================================="
echo "Jetson TensorRT Deployment Setup"
echo "==================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "WARNING: This doesn't appear to be a Jetson device"
    echo "TensorRT engines must be built on the target platform"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Check for TensorRT
echo ""
echo "Checking TensorRT installation..."
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: TensorRT not found. Please install Jetpack 5.1.2"
    exit 1
fi

# Check for CUDA
echo ""
echo "Checking CUDA installation..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "WARNING: nvidia-smi failed. CUDA may not be properly installed"
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p trt_engines
mkdir -p onnx_models
mkdir -p results_trt

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Copy your ONNX model to ./onnx_models/"
echo "2. Run: python3 convert_onnx_to_trt.py --onnx_path ./onnx_models/your_model.onnx"
echo "3. Run: python3 simple_inference_trt.py --trt_path ./trt_engines/your_model.trt --train_dir /path/to/normal --test_dir /path/to/test"
