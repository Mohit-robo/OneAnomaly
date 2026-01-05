"""
Feature Extractor Module
Handles DINOv3 model loading and patch-wise feature extraction.
Supports both PyTorch and TensorRT backends.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2


class FeatureExtractor:
    """Extract patch-wise features using DINOv3."""
    
    def __init__(
        self,
        model_name: str = 'dinov3_vits16',
        device: str = 'cuda',
        backend: str = 'pytorch',
        model_path: Optional[str] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: DINOv3 model variant ('dinov3_vits16', 'dinov3_vitb16', etc.)
            device: Device to run inference on ('cuda' or 'cpu')
            backend: Backend to use ('pytorch' or 'tensorrt')
            model_path: Path to model weights or TensorRT engine
        """
        self.model_name = model_name
        self.device = device
        self.backend = backend
        self.model_path = model_path
        
        if backend == 'pytorch':
            self._load_pytorch_model()
        elif backend == 'tensorrt':
            self._load_tensorrt_model()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.image_size = 224
    
    def _load_pytorch_model(self):
        """Load PyTorch model."""
        print(f"Loading {self.model_name} with PyTorch backend...")
        
        # Find dinov3 repo
        repo_path = Path(__file__).parent / 'dinov3'
        if not repo_path.exists():
            raise FileNotFoundError(
                f"DINOv3 repository not found at {repo_path}. "
                "Clone it: git clone https://github.com/facebookresearch/dinov3.git"
            )
        
        # Load model
        try:
            # Check if local weights file exists
            weights_path = Path(__file__).parent.parent / 'models' / f'{self.model_name}_pretrain_lvd1689m.pth'
            
            if weights_path.exists():
                print(f"Loading from local weights: {weights_path.name}")
                # Use weights parameter to load pretrained weights directly
                self.model = torch.hub.load(
                    str(repo_path),
                    self.model_name,
                    source='local',
                    weights=str(weights_path),
                    pretrained=True
                )
            else:
                print(f"Local weights not found, attempting download...")
                print("Note: This may fail with 403 error. See SETUP_DINOV3.md")
                # Try to download (will likely fail with 403)
                self.model = torch.hub.load(
                    str(repo_path),
                    self.model_name,
                    source='local',
                    pretrained=True
                )
        except Exception as e:
            print(f"\nError loading model: {e}")
            print("\nPlease download weights manually. See SETUP_DINOV3.md for instructions.")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get model properties
        self.patch_size = self.model.patch_size
        self.embed_dim = self.model.embed_dim
        
        print(f"Model loaded: patch_size={self.patch_size}, embed_dim={self.embed_dim}")
    
    def _load_tensorrt_model(self):
        """Load TensorRT model."""
        # TODO: Implement TensorRT loading
        raise NotImplementedError("TensorRT backend not yet implemented")
    
    def extract_from_image(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract features from a single image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            features: Feature array of shape (num_patches, embed_dim)
            patch_grid: Tuple of (height, width) for patch grid
        """
        if self.backend == 'pytorch':
            return self._extract_pytorch(image)
        elif self.backend == 'tensorrt':
            return self._extract_tensorrt(image)
    
    def _extract_pytorch(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract features using PyTorch model with manual OpenCV preprocessing.
        Matches logic in dinov3_faiss.py
        """
        # Preprocessing matching the reference script
        # 1. Resize to 256x256 (IMG_SIZE in reference)
        # Note: Reference script uses cv2.INTER_LINEAR
        img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # 2. Normalize
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # 3. HWC to CHW and Add Batch Dim
        # (H, W, C) -> (C, H, W) -> (1, C, H, W)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device).float()
        
        # Extract features
        with torch.no_grad():
            output = self.model.forward_features(img_tensor)
            patch_features = output['x_norm_patchtokens']
            features = patch_features.cpu().numpy()[0]
            
        # L2 Normalize features (Critical for Coreset/FAISS)
        # Dimensions: (num_patches, embed_dim)
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norm + 1e-6)
        
        # Calculate patch grid
        num_patches = features.shape[0]
        patches_per_side = int(np.sqrt(num_patches))
        patch_grid = (patches_per_side, patches_per_side)
        
        return features, patch_grid
    
    def _extract_tensorrt(self, image: np.ndarray):
        """Extract features using TensorRT model."""
        # TODO: Implement TensorRT inference
        raise NotImplementedError("TensorRT backend not yet implemented")
    
    def extract_from_file(
        self,
        image_path: str
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract features from an image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            features: Feature array of shape (num_patches, embed_dim)
            patch_grid: Tuple of (height, width) for patch grid
        """
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.extract_from_image(image)
    
    def extract_from_folder(
        self,
        folder_path: str,
        extensions: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract features from all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            extensions: List of valid image extensions (default: common formats)
            
        Returns:
            all_features: Concatenated features from all images (N, embed_dim)
            image_paths: List of processed image paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Get image files recursively
        folder = Path(folder_path)
        image_files = [
            f for f in folder.rglob('*')
            if f.suffix.lower() in extensions and f.is_file()
        ]
        
        if not image_files:
            raise ValueError(f"No images found in {folder_path}")
        
        print(f"Extracting features from {len(image_files)} images...")
        
        # Extract features from each image
        all_features = []
        processed_paths = []
        
        for img_path in image_files:
            try:
                features, _ = self.extract_from_file(str(img_path))
                all_features.append(features)
                processed_paths.append(str(img_path))
            except Exception as e:
                print(f"Warning: Failed to process {img_path.name}: {e}")
                continue
        
        # Concatenate all features
        all_features = np.vstack(all_features)
        
        print(f"Extracted {all_features.shape[0]} patch features total")
        
        return all_features, processed_paths
    
    def get_feature_dimension(self) -> int:
        """Get the feature dimension."""
        return self.embed_dim
    
    def get_patch_size(self) -> int:
        """Get the patch size."""
        return self.patch_size
