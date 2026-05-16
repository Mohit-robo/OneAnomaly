"""
Memory Bank Module
Implements PatchCore-inspired memory bank for storing and querying features.
Uses FAISS for efficient similarity search.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import faiss
from sklearn.random_projection import SparseRandomProjection


class MemoryBank:
    """Memory bank for storing good image features with FAISS indexing."""
    
    def __init__(
        self,
        feature_dim: int = 384,
        use_gpu: bool = True,
        coreset_sampling_ratio: float = 1.0
    ):
        """
        Initialize memory bank.
        
        Args:
            feature_dim: Dimension of feature vectors
            use_gpu: Whether to use GPU for FAISS
            coreset_sampling_ratio: Ratio of features to keep via coreset sampling
        """
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu
        self.coreset_sampling_ratio = coreset_sampling_ratio
        
        self.features = None
        self.index = None
        self.is_fitted = False
    
    def add_features(
        self,
        features: np.ndarray,
        apply_coreset: bool = True
    ):
        """
        Add features to memory bank.
        
        Args:
            features: Feature array of shape (N, feature_dim)
            apply_coreset: Whether to apply coreset subsampling
        """
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {features.shape[1]}"
            )
        
        print(f"Adding {features.shape[0]} features to memory bank...")
        
        # Apply coreset subsampling if requested
        if apply_coreset and self.coreset_sampling_ratio < 1.0:
            features = self._coreset_subsample(features)
            print(f"After coreset sampling: {features.shape[0]} features")
        
        # Store features
        if self.features is None:
            self.features = features.astype(np.float32)
        else:
            self.features = np.vstack([self.features, features]).astype(np.float32)
        
        # Rebuild FAISS index
        self._build_index()
        
        print(f"Memory bank now contains {self.features.shape[0]} features")
    
    def _coreset_subsample(self, features: np.ndarray) -> np.ndarray:
        """
        Apply greedy coreset subsampling to reduce feature count.
        
        This is inspired by PatchCore's approach to select a diverse
        subset of features that best represent the full distribution.
        
        Args:
            features: Input features of shape (N, feature_dim)
            
        Returns:
            Subsampled features
        """
        num_samples = int(len(features) * self.coreset_sampling_ratio)
        num_samples = max(1, num_samples)  # At least 1 sample
        
        if num_samples >= len(features):
            return features
        
        # Use approximate greedy coreset selection
        # Start with a random sample
        selected_indices = [np.random.randint(0, len(features))]
        
        # Greedily select samples that are farthest from already selected
        selected_indices = [np.random.randint(0, len(features))]
        
        # Initialize min_distances to that of the first selected point
        # Compute L2 norm (euclidean distance)
        first_selected = features[selected_indices[0]]
        min_distances = np.linalg.norm(features - first_selected, axis=1)
        
        # Iteratively select the farthest point
        for _ in range(num_samples - 1):
            # Select the point with maximum minimum distance
            # (ignoring already selected ones effectively since their dist is 0)
            next_idx = np.argmax(min_distances)
            
            # If the max distance is very small, we might have selected all unique points
            if min_distances[next_idx] < 1e-6:
                break
                
            selected_indices.append(next_idx)
            
            # Update minimum distances with the new point
            new_distances = np.linalg.norm(features - features[next_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)
        
        return features[selected_indices]
    
    def _build_index(self):
        """Build FAISS index for similarity search."""
        if self.features is None or len(self.features) == 0:
            raise ValueError("No features to build index from")
        
        print("Building FAISS index...")
        
        # Create FAISS index (Inner Product = Cosine Similarity for normalized vectors)
        print("Using IndexFlatIP (Cosine Similarity)")
        self.index = faiss.IndexFlatIP(self.feature_dim)
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Add features to index
        self.index.add(self.features)
        self.is_fitted = True
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def query(
        self,
        query_features: np.ndarray,
        k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query memory bank for nearest neighbors.
        
        Args:
            query_features: Query features of shape (N, feature_dim)
            k: Number of nearest neighbors to retrieve
            
        Returns:
            distances: Distances to k nearest neighbors (N, k)
            indices: Indices of k nearest neighbors (N, k)
        """
        if not self.is_fitted:
            raise ValueError("Memory bank not fitted. Add features first.")
        
        if query_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {query_features.shape[1]}"
            )
        
        # Ensure float32
        query_features = query_features.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_features, k)
        
        return distances, indices
    
    def save(self, filepath: str):
        """
        Save memory bank to disk.
        
        Args:
            filepath: Path to save file (.pkl)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted memory bank")
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        data = {
            'features': self.features,
            'feature_dim': self.feature_dim,
            'coreset_sampling_ratio': self.coreset_sampling_ratio
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Memory bank saved to {filepath}")
        print(f"  Features: {self.features.shape}")
    
    def load(self, filepath: str):
        """
        Load memory bank from disk.
        
        Args:
            filepath: Path to saved file (.pkl)
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Loading memory bank from {filepath}...")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.features = data['features']
        self.feature_dim = data['feature_dim']
        self.coreset_sampling_ratio = data.get('coreset_sampling_ratio', 0.1)
        
        # Rebuild index
        self._build_index()
        
        print(f"Memory bank loaded successfully")
        print(f"  Features: {self.features.shape}")
    
    def feature_count(self) -> int:
        """Get number of features in memory bank."""
        if self.features is None:
            return 0
        return len(self.features)
    
    def clear(self):
        """Clear all features from memory bank."""
        self.features = None
        self.index = None
        self.is_fitted = False
        print("Memory bank cleared")

class SpatialMemoryBank:
    """Manages multiple memory banks for spatial regions."""
    
    def __init__(
        self,
        num_regions: int = 1,
        feature_dim: int = 384,
        use_gpu: bool = True,
        coreset_sampling_ratio: float = 1.0
    ):
        self.num_regions = num_regions
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.regions_coords = [] # List of [x, y, w, h]
        
        self.banks = {
            i: MemoryBank(feature_dim, use_gpu, coreset_sampling_ratio)
            for i in range(num_regions)
        }
    
    @property
    def is_fitted(self) -> bool:
        """Check if any of the underlying banks are fitted."""
        return any(bank.is_fitted for bank in self.banks.values())
        
    def add_features_batch(self, features_batch: np.ndarray, apply_coreset: bool = True):
        """
        Add features from a batch of regions.
        features_batch shape: (num_images * num_regions, feature_dim) or similar
        We assume features are ordered: [img1_reg1, img1_reg2, ..., img2_reg1, img2_reg2, ...]
        """
        if len(features_batch) % self.num_regions != 0:
            raise ValueError(f"Batch size {len(features_batch)} is not divisible by num_regions {self.num_regions}")
            
        num_images = len(features_batch) // self.num_regions
        
        # features_batch ordering: [img1_reg1_p1...pP, img1_reg2_p1...pP, ..., imgN_regR_p1...pP]
        # We need to reshape safely to (num_images, num_regions, num_patches, feature_dim)
        
        # Total patches across all regions and images
        total_items = features_batch.shape[0]
        num_patches = total_items // (num_images * self.num_regions)
        
        features_reshaped = features_batch.reshape(num_images, self.num_regions, num_patches, -1)
        
        for region_idx in range(self.num_regions):
            print(f"\nProcessing Memory Bank for Region {region_idx}...")
            region_features = features_reshaped[:, region_idx, :, :]
            # Flatten patches for all images in this region -> (num_images * num_patches, feature_dim)
            region_features_flat = region_features.reshape(-1, self.feature_dim)
            self.banks[region_idx].add_features(region_features_flat, apply_coreset)
            
    def query(self, features_batch: np.ndarray, k: int = 1):
        """
        Query the memory bank for a batch of regions.
        features_batch shape: (num_regions, num_patches_per_region, feature_dim)
        """
        if features_batch.shape[0] != self.num_regions:
            raise ValueError(f"Input features have {features_batch.shape[0]} regions, but bank has {self.num_regions}")
            
        all_distances = []
        all_indices = []
        
        for i in range(self.num_regions):
            region_features = features_batch[i] # (num_patches_per_region, feature_dim)
            dist, idx = self.banks[i].query(region_features, k=k)
            all_distances.append(dist)
            all_indices.append(idx)
            
        # Stack back (num_regions, num_patches_per_region, k)
        return np.array(all_distances), np.array(all_indices)
            
    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'num_regions': self.num_regions,
            'regions_coords': self.regions_coords,
            'feature_dim': self.feature_dim,
            'coreset_sampling_ratio': self.coreset_sampling_ratio,
            'banks_data': {}
        }
        for idx, bank in self.banks.items():
            if bank.is_fitted:
                data['banks_data'][idx] = bank.features
                
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"SpatialMemoryBank saved to {filepath} with {self.num_regions} regions.")

    def load(self, filepath: str):
        if not Path(filepath).exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.num_regions = data.get('num_regions', 1)
        self.regions_coords = data.get('regions_coords', [])
        self.feature_dim = data.get('feature_dim', 384)
        self.coreset_sampling_ratio = data.get('coreset_sampling_ratio', 1.0)
        
        self.banks = {}
        for i in range(self.num_regions):
            bank = MemoryBank(self.feature_dim, self.use_gpu, self.coreset_sampling_ratio)
            features = data.get('banks_data', {}).get(i)
            if features is not None:
                bank.features = features
                bank._build_index()
            self.banks[i] = bank
            
        print(f"SpatialMemoryBank loaded from {filepath} with {self.num_regions} regions.")
        
    def feature_count(self):
        return sum(bank.feature_count() for bank in self.banks.values())