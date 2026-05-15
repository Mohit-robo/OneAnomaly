#!/usr/bin/env python3
"""
Export DINOv3 models to ONNX format with dynamic batch size support
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from src.backbones_v3 import DINOv3Wrapper

class DINOv3ExportWrapper(nn.Module):
    """Wrapper to export DINOv3 model without mask requirements"""
    def __init__(self, model, extracted_layers, patch_size):
        super().__init__()
        self.model = model
        self.extracted_layers = extracted_layers
        self.patch_size = patch_size
        
        # Patch for ONNX export: vision_transformer.py does '0 * self.mask_token'
        # so it cannot be None. We use a simple Parameter to avoid prim.device conflicts.
        if hasattr(self.model, 'mask_token'):
            # Re-initialize as a zero parameter of the correct dimension
            embed_dim = getattr(self.model, 'embed_dim', self.model.cls_token.shape[-1])
            self.model.mask_token = nn.Parameter(torch.zeros(1, embed_dim), requires_grad=False)
    
    def forward(self, x):
        # Get intermediate layers
        outputs = self.model.get_intermediate_layers(
            x, 
            n=self.extracted_layers, 
            reshape=False, 
            norm=True
        )
        
        # Process and concatenate features
        B, _, H_img, W_img = x.shape
        target_h, target_w = H_img // self.patch_size, W_img // self.patch_size
        
        all_feats = []
        for out in outputs:
            N_out = out.shape[1]
            C_out = out.shape[2]
            
            # Infer spatial dimensions
            h_out = int(np.sqrt(N_out * (H_img / W_img)))
            w_out = N_out // h_out
            
            feat = out.reshape(B, h_out, w_out, C_out).permute(0, 3, 1, 2)
            
            if h_out != target_h or w_out != target_w:
                feat = torch.nn.functional.interpolate(
                    feat, size=(target_h, target_w), 
                    mode='bilinear', align_corners=False
                )
            
            all_feats.append(feat)
        
        # Concatenate and reshape
        combined = torch.cat(all_feats, dim=1)
        tokens = combined.flatten(2).permute(0, 2, 1)
        
        return tokens

def export_dinov3_to_onnx(model_name, repo_path, weights_path, output_path, resolution=448, dynamic_batch=True, batch_size=1):
    """
    Export DINOv3 model to ONNX format
    
    Args:
        model_name: Name of the DINOv3 model (e.g., 'dinov3_vits16')
        repo_path: Path to DINOv3 repository
        weights_path: Path to model weights
        output_path: Where to save the ONNX model
        resolution: Input resolution
        dynamic_batch: Enable dynamic batch size
        batch_size: Fixed batch size if dynamic_batch is False
    """
    print(f"Loading {model_name}...")
    wrapper = DINOv3Wrapper(
        model_name=model_name,
        device='cpu',
        smaller_edge_size=resolution,
        repo_path=repo_path,
        weights_path=weights_path
    )
    
    # Create export wrapper
    export_model = DINOv3ExportWrapper(
        wrapper.model,
        wrapper.extracted_layers,
        wrapper.patch_size
    )
    export_model.eval()
    
    # Determine patch size and create dummy input
    ps = wrapper.patch_size
    h = w = (resolution // ps) * ps
    
    export_batch = 1 if dynamic_batch else batch_size
    print(f"Creating dummy input: ({export_batch}, {3}, {h}, {w})")
    dummy_input = torch.randn(export_batch, 3, h, w, dtype=torch.float32)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = export_model(dummy_input)
    print(f"  Output shape: {output.shape}")
    
    # Test with batch size 2 if dynamic
    if dynamic_batch:
        print("\nTesting with batch size 2...")
        dummy_input_batch = torch.randn(2, 3, h, w, dtype=torch.float32)
        with torch.no_grad():
            output_batch = export_model(dummy_input_batch)
        print(f"  Output shape (batch=2): {output_batch.shape}")
    
    # Export to ONNX
    print(f"\nExporting to {output_path}...")
    print(f"  Dynamic batch size: {dynamic_batch}")
    if not dynamic_batch:
        print(f"  Fixed batch size: {batch_size}")
    
    export_kwargs = {
        'export_params': True,
        'opset_version': 18,
        'do_constant_folding': True,
        'input_names': ['input'],
        'output_names': ['output'],
        'verbose': False
    }
    
    if dynamic_batch:
        export_kwargs['dynamic_axes'] = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        **export_kwargs
    )
    
    print(f"✓ Model exported successfully to {output_path}")
    
    # Verify the export
    print("\nVerifying ONNX model...")
    import onnx
    import onnxruntime as ort
    
    onnx_model = onnx.load(output_path)
    # onnx.checker.check_model(onnx_model)
    print("✓ ONNX model loaded (skipping strict checker)")
    
    # Test ONNX Runtime inference
    print("\nTesting ONNX Runtime inference...")
    # Use CPU Provider for verification
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    
    # Test input
    input_name = session.get_inputs()[0].name
    dummy_np = dummy_input.numpy()
    output_ort = session.run(None, {input_name: dummy_np})[0]
    print(f"  ONNX Runtime output shape: {output_ort.shape}")
    
    # Test batch size 2 if dynamic
    if dynamic_batch:
        dummy_np_batch = dummy_input_batch.numpy()
        output_ort_batch = session.run(None, {input_name: dummy_np_batch})[0]
        print(f"  ONNX Runtime output shape (batch=2): {output_ort_batch.shape}")
        print("✓ Dynamic batch size working correctly")
    
    # Optional: Simplify
    try:
        from onnxsim import simplify
        print("\nSimplifying ONNX model...")
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, output_path)
            print(f"✓ Simplified model saved to {output_path}")
        else:
            print("⚠ Simplification check failed, using original model")
    except ImportError:
        print("⚠ onnx-simplifier not installed, skipping simplification")
    except Exception as e:
        print(f"⚠ Simplification failed: {e}")
        print("  Using original model (this is fine, simplification is optional)")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Export DINOv3 to ONNX")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", 
                        help="DINOv3 model name")
    parser.add_argument("--repo_path", type=str, default="./dinov3",
                        help="Path to DINOv3 repository")
    parser.add_argument("--weights_path", type=str, 
                        default="./dino_weights/dinov3_vits16_pretrain_lvd1689m.pth",
                        help="Path to model weights")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output ONNX file path (default: auto-generated)")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Input resolution")
    parser.add_argument("--dynamic_batch", action="store_true", default=False,
                        help="Enable dynamic batch size")
    parser.add_argument("--fixed_batch", action="store_true", default=True,
                        help="Use fixed batch size (default: True)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for fixed batch export")
    
    args = parser.parse_args()
    
    # Determine if dynamic batch should be used
    use_dynamic = args.dynamic_batch and not args.fixed_batch
    
    # Auto-generate output path if not provided
    if args.output_path is None:
        suffix = "_dynamic" if use_dynamic else f"_bs{args.batch_size}"
        args.output_path = f"./models/onnx_models/{args.model_name}{suffix}.onnx"
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    export_dinov3_to_onnx(
        model_name=args.model_name,
        repo_path=args.repo_path,
        weights_path=args.weights_path,
        output_path=args.output_path,
        resolution=args.resolution,
        dynamic_batch=use_dynamic,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
