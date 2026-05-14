import cv2
import numpy as np
from sklearn.decomposition import PCA
import onnxruntime as ort

class DINOv3ONNXWrapper:
    """
    ONNX Runtime wrapper for DINOv3 models - Pure numpy implementation
    """
    def __init__(self, onnx_path, model_name, smaller_edge_size=448):
        self.onnx_path = onnx_path
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        
        # Set extracted layers and patch size based on model
        if 'vits16' in model_name:
            self.extracted_layers = [8, 11]  # Layer 9 and 12
            self.patch_size = 16
        else:
            self.extracted_layers = [1, 2]  # Stage 2 and Stage 3
            self.patch_size = 8
        
        print(f"Loading ONNX model from {onnx_path}")
        print(f"  Extracted layers: {self.extracted_layers}")
        print(f"  Patch size: {self.patch_size}")
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        print(f"  Execution providers: {self.session.get_providers()}")
    
    def prepare_image(self, img):
        """
        Prepare image for inference - Pure numpy/OpenCV
        
        Args:
            img: Either a file path (str) or numpy array (H, W, 3) in RGB
        
        Returns:
            image_array: Preprocessed numpy array (1, 3, H, W)
            grid_size: Tuple (grid_h, grid_w)
        """
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            # Assumed to be RGB from the inference script
            pass
        
        # For ONNX with fixed input size, we need to resize to exact dimensions
        # Calculate target size that's divisible by patch_size
        ps = self.patch_size
        target_size = (self.smaller_edge_size // ps) * ps
        
        # Resize to exact target dimensions
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        
        # Normalization (ImageNet defaults)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        image_array = np.expand_dims(img, axis=0).astype(np.float32)
        
        grid_size = (target_size // ps, target_size // ps)
        return image_array, grid_size
    
    def extract_features(self, image_array):
        """
        Extract features using ONNX Runtime
        
        Args:
            image_array: Preprocessed numpy array (B, 3, H, W) or (1, 3, H, W)
        
        Returns:
            features: Numpy array (B, N, C) or (N, C) for single image
        """
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: image_array})
        
        # Process outputs
        output = outputs[0]  # (B, N, C)
        
        # Return as-is for batch, squeeze for single image
        if output.shape[0] == 1:
            features = output.squeeze(0)  # (N, C)
        else:
            features = output  # (B, N, C)
        
        return features.astype(np.float32)
    
    def extract_features_batch(self, image_arrays):
        """
        Extract features from a batch of images
        
        Args:
            image_arrays: List of preprocessed numpy arrays [(1, 3, H, W), ...]
        
        Returns:
            features_list: List of numpy arrays [(N, C), ...]
        """
        # Stack into batch
        batch = np.concatenate(image_arrays, axis=0)  # (B, 3, H, W)
        
        # Run inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: batch})
        
        # Process outputs
        output = outputs[0]  # (B, N, C)
        
        # Split back into list
        features_list = [output[i] for i in range(output.shape[0])]
        
        return features_list
    
    def compute_background_mask(self, features, grid_size, threshold=10.0, masking_type=True):
        """
        Compute background mask using PCA - Pure numpy
        
        Args:
            features: Numpy array (N, C)
            grid_size: Tuple (grid_h, grid_w)
            threshold: PCA threshold
            masking_type: If True, return foreground mask
        
        Returns:
            mask: Boolean numpy array (N,)
        """
        if not masking_type:
            return np.ones(features.shape[0], dtype=bool)
        
        # PCA
        pca = PCA(n_components=1)
        pca_features = pca.fit_transform(features)
        pca_features = pca_features.reshape(grid_size)
        
        # Threshold
        foreground_mask = pca_features > threshold
        return foreground_mask.flatten()
    
    def get_embedding_visualization(self, features, grid_size):
        """
        Create PCA visualization - Pure numpy
        
        Args:
            features: Numpy array (N, C)
            grid_size: Tuple (grid_h, grid_w)
        
        Returns:
            vis: RGB numpy array (H, W, 3)
        """
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features)
        
        # Reshape to grid
        pca_grid = pca_features.reshape(grid_size[0], grid_size[1], 3)
        
        # Normalize to [0, 255]
        for i in range(3):
            channel = pca_grid[:, :, i]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            pca_grid[:, :, i] = channel
        
        vis = (pca_grid * 255).astype(np.uint8)
        return vis
