#!/usr/bin/env python3
"""
Export DINOv3 models to ONNX format
"""
import torch
import numpy as np
import argparse
from onnxsim import simplify
from src.backbones_v3 import DINOv3Wrapper

def export_dinov3_to_onnx(model_name, repo_path, weights_path, output_path, resolution=448):
    """
    Export DINOv3 model to ONNX format
    
    Args:
        model_name: Name of the DINOv3 model (e.g., 'dinov3_vits16')
        repo_path: Path to DINOv3 repository
        weights_path: Path to model weights
        output_path: Where to save the ONNX model
        resolution: Input resolution
    """
    print(f"Loading {model_name}...")
    wrapper = DINOv3Wrapper(
        model_name=model_name,
        device='cpu',  # Export on CPU for compatibility
        smaller_edge_size=resolution,
        repo_path=repo_path,
        weights_path=weights_path
    )
    
    model = wrapper.model
    model.eval()
    
    # Determine patch size and create dummy input
    ps = wrapper.patch_size
    # Create input that's divisible by patch size
    h = w = (resolution // ps) * ps
    
    print(f"Creating dummy input: ({1}, {3}, {h}, {w})")
    dummy_input = torch.randn(1, 3, h, w, dtype=torch.float32)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = model.get_intermediate_layers(
            dummy_input, 
            n=wrapper.extracted_layers, 
            reshape=False, 
            norm=True
        )
    print(f"  Output layers: {len(output)}")
    for i, out in enumerate(output):
        print(f"    Layer {wrapper.extracted_layers[i]}: {out.shape}")
    
    # Export to ONNX
    print(f"\nExporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # dynamic_axes={
        #     'input': {0: 'batch_size'},
        #     'output': {0: 'batch_size'}
        # },
        verbose=False
    )
    
    print(f"✓ Model exported successfully to {output_path}")
    
    # Verify the export
    print("\nVerifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    model_simp, check = simplify(
    onnx_model,
    overwrite_input_shapes={"input": [1, 3, resolution, resolution]}
    )

    assert check, "Simplification failed"
    onnx.save(model_simp, output_path)
    print(f"Success! Model saved to {output_path}")

    # Print model info
    print(f"\nModel Info:")
    print(f"  Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
    print(f"  Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Export DINOv3 to ONNX")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", 
                        help="DINOv3 model name")
    parser.add_argument("--repo_path", type=str, default="./dinov3",
                        help="Path to DINOv3 repository")
    parser.add_argument("--weights_path", type=str, 
                        default="./dino_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
                        help="Path to model weights")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output ONNX file path (default: auto-generated)")
    parser.add_argument("--resolution", type=int, default=448,
                        help="Input resolution")
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.output_path is None:
        args.output_path = f"./onnx_models/{args.model_name}.onnx"
    
    # Create output directory
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    export_dinov3_to_onnx(
        model_name=args.model_name,
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        output_path=args.output_path,
        resolution=args.resolution
    )

if __name__ == "__main__":
    main()
