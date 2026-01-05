import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import faiss 
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from pathlib import Path

# --- Configuration ---
IMG_SIZE = 256
PATCH_SIZE = 16 
NUM_PATCHES = IMG_SIZE // PATCH_SIZE
GOOD_DIR = "../Data/demo/trainCopy/good"
DAMAGED_DIR = "../Data/demo/trainCopy/bad"
OUTPUT_DIR = "results/anomalies_torch_amber_31_12/"
K_NEIGHBORS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load DINOv3
repo_path = Path(__file__).parent / 'dinov3'
weights_path = Path(__file__).parent.parent / 'models' / 'dinov3_vits16_pretrain_lvd1689m.pth'

print(f"Loading DINOv3 on {device}...")
model = torch.hub.load(str(repo_path), 'dinov3_vits16', 
                       source='local',
                        weights=str(weights_path),
                    pretrained=True).to(device)
model.eval()

def preprocess_opencv(img_path):
    """
    Replaces PIL/Torchvision transforms with OpenCV to ensure 
    identical bit-stream results for the TensorRT engine.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Use INTER_LINEAR to match Torchvision default
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # HWC to CHW and Add Batch Dim
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return img_t

def get_patch_features(img_path):
    img_t = preprocess_opencv(img_path)
    with torch.no_grad():
        # DINOv3 feature dict
        features = model.forward_features(img_t)["x_norm_patchtokens"]
    # Return as float32 for FAISS
    return features.squeeze(0).cpu().numpy().astype('float32')

# 2. Build the FAISS CPU Index
print(f"Extracting bank features from {GOOD_DIR}...")
good_files = [os.path.join(GOOD_DIR, f) for f in os.listdir(GOOD_DIR) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

all_patches = []
for f in tqdm(good_files):
    all_patches.append(get_patch_features(f))

memory_bank = np.vstack(all_patches)
faiss.normalize_L2(memory_bank)

index = faiss.IndexFlatIP(memory_bank.shape[1])
index.add(memory_bank)
print(f"FAISS Index ready with {index.ntotal} patches.")

# 3. Process Damaged Images
damaged_files = [f for f in os.listdir(DAMAGED_DIR) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in tqdm(damaged_files):
    path = os.path.join(DAMAGED_DIR, filename)
    feat_query = get_patch_features(path)
    
    faiss.normalize_L2(feat_query)
    similarities, _ = index.search(feat_query, K_NEIGHBORS)
    
    # Distance Map (1 - Cosine Similarity)
    avg_sim = np.mean(similarities, axis=1)
    dist_map = (1.0 - avg_sim).reshape(NUM_PATCHES, NUM_PATCHES)

    # --- Post-Processing ---
    upscaled = cv2.resize(dist_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    smoothed = gaussian_filter(upscaled, sigma=2)
    
    # Normalization for visualization
    smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8)

    # --- Visualization (BGR for saving via CV2 or RGB for PLT) ---
    original_bgr = cv2.imread(path)
    original_bgr = cv2.resize(original_bgr, (IMG_SIZE, IMG_SIZE))
    
    heatmap = cv2.applyColorMap((smoothed * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 0.7, heatmap, 0.3, 0)

    # Save Side-by-Side Plot using Matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Anomaly Heatmap")
    for ax in axes: ax.axis('off')
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"torch_res_{filename}"))
    plt.close()

print(f"Inference complete. Results in {OUTPUT_DIR}")