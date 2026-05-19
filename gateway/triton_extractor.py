"""
Triton Feature Extractor
Connects to Triton inference server to extract DINOv3 features.
"""

import numpy as np
import cv2
import tritonclient.grpc as grpcclient

class TritonFeatureExtractor:
    def __init__(self, url="localhost:8001", model_name="dinov3_encoder"):
        self.url = url
        self.model_name = model_name
        self.client = grpcclient.InferenceServerClient(url=url)
        self.image_size = 224
        
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        # HWC -> CHW -> BCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0) # Shape: (1, 3, 224, 224)
        return img
    
    def extract_from_image(self, image: np.ndarray) -> tuple:
        return self.extract_batch([image])
    
    def extract_batch(self, images: list) -> tuple:
        if not images:
            raise ValueError("Empty image sequence")
            
        batch_size = len(images)
        tensors = [self._preprocess(img) for img in images]
        batch_tensor = np.vstack(tensors) # Shape: (B, 3, 224, 224)
        
        inputs = [grpcclient.InferInput('input', batch_tensor.shape, 'FP32')]
        inputs[0].set_data_from_numpy(batch_tensor.astype(np.float32))
        
        outputs = [grpcclient.InferRequestedOutput('output')]
        
        response = self.client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
        
        features = response.as_numpy('output')
        
        # Reshape if output is (B, N, C)
        if len(features.shape) == 3:
            b, n, c = features.shape
            features = features.reshape(b*n, c)
            
        # L2 Normalize
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norm + 1e-6)
        
        num_patches = features.shape[0] // batch_size
        patches_per_side = int(np.sqrt(num_patches))
        patch_grid = (patches_per_side, patches_per_side)
        
        return features, patch_grid
