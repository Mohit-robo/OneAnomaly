import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import faiss
import random

from src.backbones_onnx import DINOv3ONNXWrapper
from src.utils import dists2map, augment_image
from src.post_eval import mean_top1p

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Folder with normal images.")
    parser.add_argument("--test_dir", type=str, required=True, help="Folder with test images.")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", help="DINOv3 model name (dinov3_convnext_tiny or dinov3_vits16).")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to ONNX model file.")
    parser.add_argument("--output_dir", type=str, default="inference_results_onnx", help="Where to save visual results.")
    parser.add_argument("--shots", type=int, default=5, help="Number of normal images to use.")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--mask_threshold", type=float, default=10.0)
    parser.add_argument("--no_mask", action="store_true", help="Disable foreground masking.")
    parser.add_argument("--save_bank", type=str, default=None, help="Save memory bank.")
    parser.add_argument("--load_bank", type=str, default=None, help="Load memory bank.")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Percentage of images to test.")
    parser.add_argument("--debug", action="store_true", help="Show extra visualization (PCA).")
    parser.add_argument("--augment", type=str, default=None, help="Augmentation to apply to normal images during bank building (e.g. hflip).")
    parser.add_argument("--k", type=int, default=1, help="Number of neighbors for k-NN averaging.")
    parser.add_argument("--sigma", type=int, default=4, help="Gaussian smoothing sigma for the anomaly map.")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed maximum value for heatmap scaling (e.g. 0.15).")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ONNX model: {args.onnx_path}...")
    # Initialize the ONNX Wrapper
    model = DINOv3ONNXWrapper(
        onnx_path=args.onnx_path,
        model_name=args.model_name,
        smaller_edge_size=args.resolution
    )

    # 1. Build Memory Bank
    if args.load_bank:
        bank_path = args.load_bank if args.load_bank.endswith('.npy') else f"{args.load_bank}.npy"
        print(f"Loading memory bank from {bank_path}")
        features_ref = np.load(bank_path).astype('float32')
    else:
        print(f"Building memory bank using {args.shots} images from {args.train_dir}")
        train_images = [f for f in sorted(os.listdir(args.train_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:args.shots]
        
        features_ref_list = []
        # No torch.inference_mode() needed for ONNX
        for img_name in tqdm(train_images, desc="Extracting normal features"):
                img_path = os.path.join(args.train_dir, img_name)
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                
                if args.augment:
                    aug_images = augment_image(image, augmentation=args.augment)
                else:
                    aug_images = [image]
                
                for aug_img in aug_images:
                    image_tensor, grid_size = model.prepare_image(aug_img)
                    features = model.extract_features(image_tensor)
                    
                    if not args.no_mask:
                        mask = model.compute_background_mask(features, grid_size, threshold=args.mask_threshold, masking_type=True)
                        features = features[mask]
                    
                    features_ref_list.append(features)

        features_ref = np.concatenate(features_ref_list, axis=0).astype('float32')
        
        if args.save_bank:
            print(f"Saving memory bank to {args.save_bank}")
            np.save(args.save_bank, features_ref)
            
    features_ref = np.ascontiguousarray(features_ref)
    faiss.normalize_L2(features_ref)
    
    # FAISS Index
    try:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])
        print("Using FAISS GPU search.")
    except:
        index = faiss.IndexFlatL2(features_ref.shape[1])
        print("Using FAISS CPU search.")
    
    index.add(features_ref)

    # 2. Run Inference
    print(f"Finding and sampling images in {args.test_dir} (Ratio: {args.sample_ratio})")
    all_test_images = []
    folder_breakdown = {}
    for root, dirs, files in os.walk(args.test_dir):
        folder_images = [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        if not folder_images:
            continue
        
        if args.sample_ratio < 1.0:
            num_samples = max(1, int(len(folder_images) * args.sample_ratio))
            sampled = random.sample(folder_images, num_samples)
            all_test_images.extend(sampled)
            folder_breakdown[os.path.relpath(root, args.test_dir)] = (len(folder_images), num_samples)
        else:
            all_test_images.extend(folder_images)
            folder_breakdown[os.path.relpath(root, args.test_dir)] = (len(folder_images), len(folder_images))

    print(f"Sampling Summary (Ratio: {args.sample_ratio}):")
    for f_name, (total, sampled) in folder_breakdown.items():
        print(f" - {f_name}: Picked {sampled} out of {total} images")

    # No torch.inference_mode() needed for ONNX
    for img_path in tqdm(all_test_images, desc="Detecting anomalies"):
        img_name = os.path.basename(img_path)
        image_test = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image_tensor, grid_size = model.prepare_image(image_test)
        features_test = model.extract_features(image_tensor)
        
        # Masking for test image
        if not args.no_mask:
            mask_test = model.compute_background_mask(features_test, grid_size, threshold=args.mask_threshold, masking_type=True)
        else:
            mask_test = np.ones(features_test.shape[0], dtype=bool)

        # Similarity Search
        features_test_masked = np.ascontiguousarray(features_test[mask_test])
        faiss.normalize_L2(features_test_masked)
        distances, _ = index.search(features_test_masked, k=args.k)
        
        if args.k > 1:
            distances = distances.mean(axis=1)
            
        distances = distances / 2 # L2 distance to cosine
        
        # Map distances back to grid
        output_distances = np.zeros(mask_test.shape)
        output_distances[mask_test] = distances.squeeze()
        anomaly_map = dists2map(output_distances.reshape(grid_size), image_test.shape, sigma=args.sigma)
        
        # 3. Visualization
        cols = 4 if args.debug else 3
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
        
        # Original Image
        axes[0].imshow(image_test)
        axes[0].set_title("Original Image (ONNX)")
        axes[0].axis('off')
        
        curr_ax = 1
        if args.debug:
            # PCA Visualization
            pca_vis = model.get_embedding_visualization(features_test, grid_size)
            axes[curr_ax].imshow(pca_vis)
            axes[curr_ax].set_title("Feature PCA")
            axes[curr_ax].axis('off')
            curr_ax += 1

        # Heatmap
        vmax = args.vmax if args.vmax is not None else anomaly_map.max()
        im = axes[curr_ax].imshow(anomaly_map, cmap='jet', vmin=0, vmax=vmax)
        axes[curr_ax].set_title("Anomaly Heatmap")
        axes[curr_ax].axis('off')
        plt.colorbar(im, ax=axes[curr_ax], fraction=0.046, pad=0.04)
        curr_ax += 1
        
        # Overlay
        # Use the same vmax for consistency if provided
        if args.vmax is not None:
            norm_map = np.clip(anomaly_map / args.vmax, 0, 1)
        else:
            norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        
        heatmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_test, 0.6, heatmap_color, 0.4, 0)
        
        axes[curr_ax].imshow(overlay)
        axes[curr_ax].set_title(f"Overlay (Score: {mean_top1p(output_distances):.4f})")
        axes[curr_ax].axis('off')
        
        plt.tight_layout()
        
        relative_path = os.path.relpath(img_path, args.test_dir)
        save_path = os.path.join(args.output_dir, relative_path)
        save_path = os.path.splitext(save_path)[0] + ".png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()

    print(f"\nInference completed! Results saved to: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
