import cv2
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
import numpy as np
from src.backbones import VisionTransformerWrapper

class DINOv3Wrapper(VisionTransformerWrapper):
    def __init__(self, model_name, device, smaller_edge_size=448, half_precision=False, repo_path=None, weights_path=None):
        self.repo_path = repo_path
        self.weights_path = weights_path
        
        # Select layers for concatenation
        if 'vits16' in model_name:
            self.extracted_layers = [8, 11] # Layer 9 and 12
            self.patch_size = 16
        else:
            self.extracted_layers = [1, 2] # Stage 2 (stride 8) and Stage 3 (stride 16)
            self.patch_size = 8 # Use the finer stride for memory bank
            
        super().__init__(model_name, device, smaller_edge_size, half_precision)

    def load_model(self):
        if not self.repo_path or not self.weights_path:
            raise ValueError("DINOv3 requires repo_path and weights_path")
        
        # Determine entry point based on model_name
        # Possible models: dinov3_convnext_tiny, dinov3_vits16
        entry_point = self.model_name
        
        print(f"Loading {entry_point} from {self.repo_path} with weights {self.weights_path}")
        
        if 'vits16' in entry_point:
            self.patch_size = 16
            model = torch.hub.load(
                self.repo_path,
                entry_point,
                source='local',
                weights=self.weights_path
            )
        else:
            # ConvNeXt architectures
            self.patch_size = 16 # We'll use stride 16 (Stage 3)
            model = torch.hub.load(
                self.repo_path,
                entry_point,
                source='local',
                weights=self.weights_path,
                patch_size=None 
            )
            
        model.eval()
        return model.to(self.device).float()

    def prepare_image(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, np.ndarray):
            # Assumed to be RGB from the simple_inference script
            pass
            
        # Resize maintaining aspect ratio (smaller edge to smaller_edge_size)
        h, w = img.shape[:2]
        if h < w:
            new_h = self.smaller_edge_size
            new_w = int(w * (self.smaller_edge_size / h))
        else:
            new_w = self.smaller_edge_size
            new_h = int(h * (self.smaller_edge_size / w))
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Normalization (ImageNet defaults)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # HWC to CHW and to Tensor
        image_tensor = torch.from_numpy(img).permute(2, 0, 1)
        
        height, width = image_tensor.shape[1:]
        ps = self.patch_size
        cropped_width, cropped_height = width - width % ps, height - height % ps
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // ps, cropped_width // ps)
        return image_tensor, grid_size

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            if self.half_precision:
                image_batch = image_batch.half()
            
            # get_intermediate_layers returns a tuple of results for each layer index in n
            # Each result is (B, N, C) where N=H*W/stride^2
            outputs = self.model.get_intermediate_layers(image_batch, n=self.extracted_layers, reshape=False, norm=True)
            
            # For each layer, we might need to upscale to the common grid size
            B, _, H_img, W_img = image_batch.shape
            target_h, target_w = H_img // self.patch_size, W_img // self.patch_size
            
            all_feats = []
            for out in outputs:
                # out is (B, N, C)
                N_out = out.shape[1]
                C_out = out.shape[2]
                
                # Infer spatial dimensions
                h_out = int(np.sqrt(N_out * (H_img / W_img)))
                w_out = N_out // h_out
                
                feat = out.reshape(B, h_out, w_out, C_out).permute(0, 3, 1, 2) # (B, C, H, W)
                
                if h_out != target_h or w_out != target_w:
                    feat = torch.nn.functional.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                all_feats.append(feat)
            
            # Concatenate features from all layers along the channel dimension
            combined = torch.cat(all_feats, dim=1)
            # Reshape back to (N, C)
            tokens = combined.flatten(2).permute(0, 2, 1).squeeze(0)
            
        return tokens.cpu().numpy()

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None, normalize=True):
        pca = PCA(n_components=3, svd_solver='randomized')
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        if normalize:
            normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            return normalized_tokens
        else:
            return reduced_tokens

    def compute_background_mask(self, img_features, grid_size, threshold = 10, masking_type = False, kernel_size = 3, border = 0.2):
        pca = PCA(n_components=1, svd_solver='randomized')
        first_pc = pca.fit_transform(img_features.astype(np.float32))
        if masking_type == True:
            mask = first_pc > threshold
            m = mask.reshape(grid_size)[int(grid_size[0] * border):int(grid_size[0] * (1-border)), int(grid_size[1] * border):int(grid_size[1] * (1-border))]
            if m.sum() <=  m.size * 0.35:
                mask = - first_pc > threshold
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)).astype(bool)
        elif masking_type == False:
            mask = np.ones_like(first_pc, dtype=bool)
        return mask.squeeze()
