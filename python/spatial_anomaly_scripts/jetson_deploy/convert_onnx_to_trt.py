#!/usr/bin/env python3
"""
Convert ONNX model to TensorRT engine for Jetpack 5.1.2
"""
import tensorrt as trt
import argparse
import os

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_from_onnx(onnx_path, engine_path, fp16_mode=True, workspace_size=4):
    """
    Build TensorRT engine from ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Where to save the TensorRT engine
        fp16_mode: Use FP16 precision (recommended for Jetson)
        workspace_size: Max workspace size in GB
    
    Returns:
        Path to saved engine
    """
    print(f"Building TensorRT engine from {onnx_path}...")
    print(f"  FP16 mode: {fp16_mode}")
    print(f"  Workspace size: {workspace_size}GB")
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("\nParsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f"✓ ONNX model parsed successfully")
    print(f"  Network inputs: {network.num_inputs}")
    print(f"  Network outputs: {network.num_outputs}")
    
    # Print input/output info
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        print(f"  Input {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
    
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        print(f"  Output {i}: {output_tensor.name}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))  # GB to bytes
    
    if fp16_mode and builder.platform_has_fast_fp16:
        print("\n✓ Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("\n⚠ FP16 not available or disabled, using FP32")
    
    # Build engine
    print("\nBuilding TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None
    
    print("✓ Engine built successfully")
    
    # Save engine
    print(f"\nSaving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✓ Engine saved successfully")
    print(f"  Engine size: {os.path.getsize(engine_path) / (1024**2):.2f} MB")
    
    return engine_path

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT")
    parser.add_argument("--onnx_path", type=str, required=True,
                        help="Path to ONNX model")
    parser.add_argument("--engine_path", type=str, default=None,
                        help="Output TensorRT engine path (default: auto-generated)")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 precision (default: True)")
    parser.add_argument("--workspace", type=int, default=4,
                        help="Max workspace size in GB (default: 4)")
    
    args = parser.parse_args()
    
    # Auto-generate engine path if not provided
    if args.engine_path is None:
        base_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
        engine_dir = os.path.join(os.path.dirname(args.onnx_path), "../trt_engines")
        os.makedirs(engine_dir, exist_ok=True)
        args.engine_path = os.path.join(engine_dir, f"{base_name}.trt")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.engine_path), exist_ok=True)
    
    # Build engine
    build_engine_from_onnx(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        fp16_mode=args.fp16,
        workspace_size=args.workspace
    )

if __name__ == "__main__":
    main()
