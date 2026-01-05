"""
DINOv3 PyTorch Testing Script
Tests the DINOv3 ViT-S/16 distilled model for feature extraction.
This script processes a folder of images and saves feature visualizations.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm


class DINOv3FeatureExtractor:
    """Wrapper for DINOv3 model to extract patch-wise features."""
    
    def __init__(self, model_name='dinov3_vits16', device='cuda'):
        """
        Initialize DINOv3 model.
        
        Args:
            model_name: Name of the DINOv3 model variant
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_name = model_name
        
        print(f"Loading {model_name} model...")
        print("Note: You need to clone the DINOv3 repo and download weights first.")
        print("See: https://github.com/facebookresearch/dinov3")
        
        # Check if dinov3 repo exists
        repo_path = Path(__file__).parent / 'dinov3'
        if not repo_path.exists():
            raise FileNotFoundError(
                f"DINOv3 repository not found at {repo_path}. "
                "Please clone it: git clone https://github.com/facebookresearch/dinov3.git"
            )
        
        # Load model via torch.hub
        try:
            # Check if local weights file exists
            weights_path = Path(__file__).parent.parent / 'models' / f'{model_name}_pretrain_lvd1689m.pth'
            
            if weights_path.exists():
                print(f"Loading model with local weights: {weights_path}")
                # Use weights parameter to load pretrained weights directly
                self.model = torch.hub.load(
                    str(repo_path),
                    model_name,
                    source='local',
                    weights=str(weights_path),
                    pretrained=True
                )
                print("Model loaded successfully from local weights!")
            else:
                print(f"Local weights not found at: {weights_path}")
                print("Attempting to download from Meta servers...")
                print("Note: This requires authentication and may fail with 403 error.")
                print("If download fails, please follow instructions in SETUP_DINOV3.md")
                
                # Try to download (will likely fail with 403)
                self.model = torch.hub.load(
                    str(repo_path),
                    model_name,
                    source='local',
                    pretrained=True
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\n" + "="*60)
            print("MODEL LOADING FAILED")
            print("="*60)
            print("\nDINOv3 weights require manual download from Meta.")
            print("\nPlease follow these steps:")
            print("1. Visit: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
            print("2. Fill out the form to request access")
            print("3. Download the weights from the email link")
            print("4. Place the .pth file in: d:\\Mohit\\anomaly_app\\models\\")
            print("\nSee SETUP_DINOV3.md for detailed instructions.")
            print("="*60 + "\n")
            raise
        
        
        self.model.to(device)
        self.model.eval()
        
        # Get model info
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        
        print(f"Model loaded successfully!")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Embedding dimension: {self.embed_dim}")
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """
        Extract patch-wise features from an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            features: Numpy array of shape (num_patches, embed_dim)
            patch_grid: Tuple of (height, width) for patch grid
        """
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            # Get patch embeddings (excluding CLS token)
            output = self.model.forward_features(img_tensor)
            
            # output['x_norm_patchtokens'] contains patch tokens
            # Shape: (batch, num_patches, embed_dim)
            patch_features = output['x_norm_patchtokens']
            
            # Convert to numpy
            features = patch_features.cpu().numpy()[0]  # Remove batch dim
            
            # Calculate patch grid dimensions
            num_patches = features.shape[0]
            patches_per_side = int(np.sqrt(num_patches))
            patch_grid = (patches_per_side, patches_per_side)
        
        return features, patch_grid
    
    def visualize_features(self, features, patch_grid, output_path, num_components=3):
        """
        Visualize features using PCA and save to file.
        
        Args:
            features: Feature array of shape (num_patches, embed_dim)
            patch_grid: Tuple of (height, width) for patch grid
            output_path: Path to save visualization
            num_components: Number of PCA components to visualize
        """
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce dimensionality for visualization
        pca = PCA(n_components=num_components)
        features_pca = pca.fit_transform(features)
        
        # Normalize to [0, 1] for visualization
        features_pca = (features_pca - features_pca.min(axis=0)) / (
            features_pca.max(axis=0) - features_pca.min(axis=0) + 1e-8
        )
        
        # Reshape to grid
        h, w = patch_grid
        feature_maps = features_pca.reshape(h, w, num_components)
        
        # Create visualization
        fig, axes = plt.subplots(1, num_components + 1, figsize=(15, 4))
        
        # Show RGB visualization
        if num_components >= 3:
            axes[0].imshow(feature_maps)
            axes[0].set_title('RGB Composite')
            axes[0].axis('off')
            
            # Show individual components
            for i in range(num_components):
                axes[i + 1].imshow(feature_maps[:, :, i], cmap='viridis')
                axes[i + 1].set_title(f'Component {i + 1}')
                axes[i + 1].axis('off')
        else:
            for i in range(num_components):
                axes[i].imshow(feature_maps[:, :, i], cmap='viridis')
                axes[i].set_title(f'Component {i + 1}')
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test DINOv3 PyTorch model')
    parser.add_argument('--input_folder', type=str, required=True,
                       help='Path to folder containing test images')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Path to save output visualizations')
    parser.add_argument('--model', type=str, default='dinov3_vits16',
                       choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                       help='DINOv3 model variant to use')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process')
    
    args = parser.parse_args()
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Initialize feature extractor
    print("\n" + "="*60)
    print("DINOv3 PyTorch Feature Extraction Test")
    print("="*60 + "\n")
    
    extractor = DINOv3FeatureExtractor(
        model_name=args.model,
        device=args.device
    )
    
    # Get list of images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in Path(args.input_folder).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if args.max_images:
        image_files = image_files[:args.max_images]
    
    print(f"\nFound {len(image_files)} images to process\n")
    
    # Process images
    total_time = 0
    feature_stats = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Extract features
            start_time = time.time()
            features, patch_grid = extractor.extract_features(img_path)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            # Save visualization
            output_path = Path(args.output_folder) / f"{img_path.stem}_features.png"
            extractor.visualize_features(features, patch_grid, output_path)
            
            # Collect stats
            feature_stats.append({
                'image': img_path.name,
                'num_patches': features.shape[0],
                'feature_dim': features.shape[1],
                'time': elapsed
            })
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total images processed: {len(feature_stats)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(feature_stats):.3f}s")
    print(f"\nFeature dimensions: {feature_stats[0]['feature_dim']}")
    print(f"Number of patches per image: {feature_stats[0]['num_patches']}")
    print(f"Patch grid size: {patch_grid[0]}x{patch_grid[1]}")
    print(f"\nVisualizations saved to: {args.output_folder}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
