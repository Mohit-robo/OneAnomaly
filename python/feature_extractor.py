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
            transforms.Resize(224),
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
        print(f"Loading {self.model_name} with TensorRT backend...")
        import sys
        
        trt_src_path = Path(__file__).parent / 'spatial_anomaly_scripts' / 'jetson_deploy' / 'src'
        if str(trt_src_path) not in sys.path:
            sys.path.insert(0, str(trt_src_path))
            
        from backbones_trt import DINOv3TensorRTWrapper
        
        engine_path = self.model_path
        if not engine_path:
            # Try default in models/
            engine_path = str(Path(__file__).parent.parent / 'models' / f'{self.model_name}.engine')
            
        if not os.path.exists(engine_path):
            # Fallback to models/engine_files/
            fallback_path = Path(__file__).parent.parent / 'models' / 'engine_files' / Path(engine_path).name
            if fallback_path.exists():
                engine_path = str(fallback_path)
            else:
                raise FileNotFoundError(f"TensorRT engine not found at {engine_path} or fallback {fallback_path}")
            
        self.model = DINOv3TensorRTWrapper(
            engine_path=engine_path,
            model_name=self.model_name,
            smaller_edge_size=256
        )
        self.patch_size = self.model.patch_size
        
        # Dynamically determine embed_dim and batch_size from model shape
        if hasattr(self.model, 'output_shape'):
            self.embed_dim = self.model.output_shape[-1]
        
        if hasattr(self.model, 'input_shape'):
            self.engine_batch_size = self.model.input_shape[0]
            print(f"Detected engine batch size: {self.engine_batch_size}")
        else:
            self.embed_dim = 384 if 'vits16' in self.model_name else 768
            
        self.engine_batch_size = self.model.input_shape[0] if hasattr(self.model, 'input_shape') else 1
        
        print(f"TensorRT model loaded: patch_size={self.patch_size}, embed_dim={self.embed_dim}")
    
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
        
    def extract_batch(
        self,
        images: List[np.ndarray]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract features from a batch of images efficiently.
        
        Args:
            images: List of input images as numpy arrays (H, W, C) in RGB format
            
        Returns:
            features: Feature array of shape (batch * num_patches, embed_dim)
            patch_grid: Tuple of (height, width) for patch grid of a single image
        """
        if not images:
            raise ValueError("Empty image sequence")
            
        if self.backend == 'pytorch':
            return self._extract_pytorch_batch(images)
        elif self.backend == 'tensorrt':
            return self._extract_tensorrt_batch(images)

    def _extract_tensorrt_batch(
        self,
        images: List[np.ndarray]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract features using TensorRT model for a batch."""
        tensors = []
        grid_size = None
        for image in images:
            # model.prepare_image handles resizing and normalization correctly
            img_tensor, grid = self.model.prepare_image(image)
            tensors.append(img_tensor)
            grid_size = grid

        # The TensorRT engine supports up to engine_batch_size (e.g. 1 or 2)
        batch_size_limit = getattr(self, 'engine_batch_size', 1)
        
        all_features = []
        for i in range(0, len(tensors), batch_size_limit):
            chunk = tensors[i : i + batch_size_limit]
            batch_tensor = np.concatenate(chunk, axis=0) # shape (B, 3, H, W)
            
            if hasattr(self.model, 'extract_features_batch') and batch_size_limit > 1:
                try:
                    feat_list = self.model.extract_features_batch(batch_tensor)
                    all_features.extend(feat_list)
                except Exception as e:
                    print(f"Warning: Batch extraction failed {e}. Falling back to single.")
                    for j in range(batch_tensor.shape[0]):
                        feat = self.model.extract_features(np.expand_dims(batch_tensor[j], axis=0))
                        all_features.append(feat)
            else:
                for j in range(batch_tensor.shape[0]):
                    feat = self.model.extract_features(np.expand_dims(batch_tensor[j], axis=0))
                    all_features.append(feat)
                    
        return np.vstack(all_features), grid_size

    def _extract_pytorch_batch(
        self,
        images: List[np.ndarray]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract features using PyTorch model for a batch."""
        tensors = []
        for image in images:
            img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            
            # (H, W, C) -> (C, H, W)
            tensors.append(torch.from_numpy(img.transpose(2, 0, 1)))
            
        # (N, C, H, W)
        batch_tensor = torch.stack(tensors).to(self.device).float()
        
        with torch.no_grad():
            output = self.model.forward_features(batch_tensor)
            patch_features = output['x_norm_patchtokens']
            # shape: (N, num_patches, embed_dim)
            features = patch_features.cpu().numpy()
            
        # Flatten across batch: (N * num_patches, embed_dim)
        features = features.reshape(-1, features.shape[-1])
        
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norm + 1e-6)
        
        num_patches = features.shape[0] // len(images)
        patches_per_side = int(np.sqrt(num_patches))
        patch_grid = (patches_per_side, patches_per_side)
        
        return features, patch_grid

    def _extract_tensorrt(self, image: np.ndarray):
        """Extract features using TensorRT model."""
        img_tensor, grid_size = self.model.prepare_image(image)
        features = self.model.extract_features(img_tensor)
        return features, grid_size
    
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
