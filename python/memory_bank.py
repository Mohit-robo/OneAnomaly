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

if __name__ == "__main__":
    
    mb = MemoryBank()

    features = np.random.random((100, 384))
    mb.add_features(features)

    mb.save("test_memory_bank.pkl")

    mb2 = MemoryBank()
    mb2.load("test_memory_bank.pkl")

    print('Success' if mb.feature_count() == mb2.feature_count() else 'Failure')