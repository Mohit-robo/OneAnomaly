import cv2
import numpy as np
from sklearn.decomposition import PCA
import tensorrt as trt
import cuda.bindings.runtime as cudart

class DINOv3TensorRTWrapper:
    """
    TensorRT wrapper for DINOv3 models - Pure numpy with CUDA integration
    """
    def __init__(self, engine_path, model_name, smaller_edge_size=448):
        self.engine_path = engine_path
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        
        # Set extracted layers and patch size based on model
        if 'vits16' in model_name:
            self.extracted_layers = [8, 11]  # Layer 9 and 12
            self.patch_size = 16
        else:
            self.extracted_layers = [1, 2]  # Stage 2 and Stage 3
            self.patch_size = 8
        
        print(f"Loading TensorRT engine from {engine_path}")
        print(f"  Extracted layers: {self.extracted_layers}")
        print(f"  Patch size: {self.patch_size}")
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output bindings
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.output_shape = None
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = shape
                print(f"  Input: {name}, shape: {shape}, dtype: {dtype}")
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name
                self.output_shape = shape
                print(f"  Output: {name}, shape: {shape}, dtype: {dtype}")
        
        # Allocate CUDA buffers
        self._allocate_buffers()
        
        print(f"✓ TensorRT engine loaded successfully")
    
    def _allocate_buffers(self):
        """Allocate CUDA device memory for input/output"""
        # Calculate sizes
        self.input_size = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_size = np.prod(self.output_shape) * np.dtype(np.float32).itemsize
        
        # Allocate device memory
        err, self.d_input = cudart.cudaMalloc(self.input_size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to allocate input device memory: {err}")
        
        err, self.d_output = cudart.cudaMalloc(self.output_size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to allocate output device memory: {err}")
        
        # Create CUDA stream
        err, self.stream = cudart.cudaStreamCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to create CUDA stream: {err}")
    
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
        
        # For TensorRT with fixed input size, resize to exact compiled engine dimensions
        ps = self.patch_size
        if hasattr(self, 'input_shape') and self.input_shape and len(self.input_shape) >= 4:
            target_h = self.input_shape[-2]
            target_w = self.input_shape[-1]
        else:
            target_h = (self.smaller_edge_size // ps) * ps
            target_w = target_h
            
        # Resize to exact target dimensions
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        # Normalization (ImageNet defaults)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        image_array = np.expand_dims(img, axis=0).astype(np.float32)
        
        grid_size = (target_h // ps, target_w // ps)
        return image_array, grid_size
    
    def split_image(self, image, top_ratio=0.4, top_center_ratio=0.6):
        """
        Split image into top and bottom regions with horizontal cropping for top region
        
        Args:
            image: RGB image (H, W, 3)
            top_ratio: Ratio of top region (default: 0.4 for 40%)
            top_center_ratio: Horizontal center ratio for top region (default: 0.6 for center 60%)
        
        Returns:
            top_region: Top portion of image (center cropped horizontally)
            bottom_region: Bottom portion of image (full width)
            split_point: Y-coordinate of split
        """
        h, w = image.shape[:2]
        split_point = int(h * top_ratio)
        
        # Extract top region (0 to split_point)
        top_full = image[:split_point, :, :]
        
        # Crop top region to center horizontally (center 60%)
        left_crop = int(w * (1 - top_center_ratio) / 2)  # 20% from left
        right_crop = w - left_crop  # 20% from right
        top_region = top_full[:, left_crop:right_crop, :]
        
        # Bottom region (full width)
        bottom_region = image[split_point:, :, :]
        
        return top_region, bottom_region, split_point
    
    def prepare_image_spatial(self, img, top_ratio=0.4, top_center_ratio=0.6):
        """
        Prepare image for spatial partitioning inference with batch processing
        
        Args:
            img: Either a file path (str) or numpy array (H, W, 3) in RGB
            top_ratio: Ratio of top region (default: 0.4 for 40%)
            top_center_ratio: Horizontal center ratio for top region (default: 0.6 for center 60%)
        
        Returns:
            batch_array: Preprocessed numpy array (2, 3, H, W) - batch of [top, bottom]
            grid_sizes: List of tuples [(grid_h_top, grid_w_top), (grid_h_bottom, grid_w_bottom)]
            regions: Tuple (top_region, bottom_region) - original cropped regions
            split_point: Y-coordinate of split in original image
        """
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            # Assumed to be RGB from the inference script
            pass
        
        # Split image into top and bottom regions
        top_region, bottom_region, split_point = self.split_image(img, top_ratio, top_center_ratio)
        
        # Prepare both regions
        ps = self.patch_size
        target_size = (self.smaller_edge_size // ps) * ps
        
        # Process top region
        top_resized = cv2.resize(top_region, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        top_normalized = top_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        top_normalized = (top_normalized - mean) / std
        top_chw = np.transpose(top_normalized, (2, 0, 1))
        
        # Process bottom region
        bottom_resized = cv2.resize(bottom_region, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        bottom_normalized = bottom_resized.astype(np.float32) / 255.0
        bottom_normalized = (bottom_normalized - mean) / std
        bottom_chw = np.transpose(bottom_normalized, (2, 0, 1))
        
        # Stack into batch (2, 3, H, W)
        batch_array = np.stack([top_chw, bottom_chw], axis=0).astype(np.float32)
        
        # Grid sizes (same for both since resized to same dimensions)
        grid_size = (target_size // ps, target_size // ps)
        grid_sizes = [grid_size, grid_size]
        
        return batch_array, grid_sizes, (top_region, bottom_region), split_point
    
    def extract_features(self, image_array):
        """
        Extract features using TensorRT
        
        Args:
            image_array: Preprocessed numpy array (1, 3, H, W)
        
        Returns:
            features: Numpy array (N, C) where N = grid_h * grid_w
        """
        # Check if engine requires fixed batch size > 1
        engine_batch_size = self.input_shape[0]
        current_batch_size = image_array.shape[0]
        
        # Pad input if necessary
        if engine_batch_size > current_batch_size:
            pad_width = ((0, engine_batch_size - current_batch_size), (0, 0), (0, 0), (0, 0))
            input_tensor = np.pad(image_array, pad_width, mode='constant')
        else:
            input_tensor = image_array
            
        input_tensor = np.ascontiguousarray(input_tensor)
        
        # Copy input to device
        err, = cudart.cudaMemcpyAsync(
            self.d_input, 
            input_tensor.ctypes.data, 
            self.input_size,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to copy input to device: {err}")
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, self.d_input)
        self.context.set_tensor_address(self.output_name, self.d_output)
        
        # Execute inference
        success = self.context.execute_async_v3(self.stream)
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Allocate host memory for output
        output_array = np.empty(self.output_shape, dtype=np.float32)
        
        # Copy output to host
        err, = cudart.cudaMemcpyAsync(
            output_array.ctypes.data,
            self.d_output,
            self.output_size,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to copy output to host: {err}")
        
        # Synchronize stream
        err, = cudart.cudaStreamSynchronize(self.stream)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to synchronize stream: {err}")
        
        # Slice output if we padded
        if engine_batch_size > current_batch_size:
            output_array = output_array[:current_batch_size]
            
        # Remove batch dimension
        features = output_array.squeeze(0)  # (N, C)
        
        # L2 normalize features for FP16 robustness
        # This helps reduce numerical differences between FP16 and FP32
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)
        
        return features.astype(np.float32)
    
    def extract_features_batch(self, image_array):
        """
        Extract features using TensorRT with batch support (batch size 2)
        
        Args:
            image_array: Preprocessed numpy array (B, 3, H, W) where B is batch size (1 or 2)
        
        Returns:
            features_list: List of numpy arrays, each (N, C) where N = grid_h * grid_w
                          Returns [features_0] for batch size 1
                          Returns [features_0, features_1] for batch size 2
        """
        batch_size = image_array.shape[0]
        
        # Ensure contiguous array
        image_array = np.ascontiguousarray(image_array)
        
        # Calculate batch sizes
        batch_input_size = batch_size * np.prod(self.input_shape[1:]) * np.dtype(np.float32).itemsize
        batch_output_size = batch_size * np.prod(self.output_shape[1:]) * np.dtype(np.float32).itemsize
        
        # Allocate temporary batch buffers
        err, d_input_batch = cudart.cudaMalloc(batch_input_size)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to allocate batch input device memory: {err}")
        
        err, d_output_batch = cudart.cudaMalloc(batch_output_size)
        if err != cudart.cudaError_t.cudaSuccess:
            cudart.cudaFree(d_input_batch)
            raise RuntimeError(f"Failed to allocate batch output device memory: {err}")
        
        try:
            # Copy input to device
            err, = cudart.cudaMemcpyAsync(
                d_input_batch,
                image_array.ctypes.data,
                batch_input_size,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream
            )
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to copy batch input to device: {err}")
            
            # Set tensor addresses
            self.context.set_tensor_address(self.input_name, d_input_batch)
            self.context.set_tensor_address(self.output_name, d_output_batch)
            
            # Execute inference
            success = self.context.execute_async_v3(self.stream)
            if not success:
                raise RuntimeError("TensorRT batch inference failed")
            
            # Allocate host memory for output
            output_shape_batch = (batch_size,) + self.output_shape[1:]
            output_array = np.empty(output_shape_batch, dtype=np.float32)
            
            # Copy output to host
            err, = cudart.cudaMemcpyAsync(
                output_array.ctypes.data,
                d_output_batch,
                batch_output_size,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self.stream
            )
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"Failed to copy batch output to host: {err}")
            
            # Synchronize stream
            cudart.cudaStreamSynchronize(self.stream)
            
            # Process each batch item
            features_list = []
            for i in range(batch_size):
                features = output_array[i]  # (N, C)
                
                # L2 normalize features
                norms = np.linalg.norm(features, axis=1, keepdims=True)
                features = features / (norms + 1e-8)
                
                features_list.append(features.astype(np.float32))
            
            return features_list
            
        finally:
            # Clean up batch buffers
            cudart.cudaFree(d_input_batch)
            cudart.cudaFree(d_output_batch)
    
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
    
    def __del__(self):
        """Cleanup CUDA resources"""
        try:
            if hasattr(self, 'd_input'):
                cudart.cudaFree(self.d_input)
            if hasattr(self, 'd_output'):
                cudart.cudaFree(self.d_output)
            if hasattr(self, 'stream'):
                cudart.cudaStreamDestroy(self.stream)
        except:
            pass
