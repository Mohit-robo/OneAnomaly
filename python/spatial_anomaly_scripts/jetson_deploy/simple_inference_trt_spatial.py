#!/usr/bin/env python3
"""
Spatial Partitioning Inference with TensorRT for Jetson
Processes top and bottom regions separately with batch inference
"""
import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import faiss

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from backbones_trt import DINOv3TensorRTWrapper
from utils import dists2map, augment_image
from post_eval import mean_top1p

def parse_args():
    parser = argparse.ArgumentParser(description="Spatial Partitioning Anomaly Detection with TensorRT")
    parser.add_argument("--engine_path", type=str, required=True, help="Path to TensorRT engine file")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", help="Model name")
    parser.add_argument("--train_dir", type=str, required=True, help="Training images directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Test images directory")
    parser.add_argument("--output_dir", type=str, default="./output_spatial_trt", help="Output directory")
    
    # Spatial partitioning parameters
    parser.add_argument("--top_ratio", type=float, default=0.4, help="Ratio for top region (default: 0.4)")
    parser.add_argument("--top_center_ratio", type=float, default=0.6, help="Horizontal center ratio for top region (default: 0.6)")
    
    # Memory bank parameters
    parser.add_argument("--shots", type=int, default=None, help="Number of training samples to use")
    parser.add_argument("--augment", type=str, default=None, choices=['flip', 'rotate', 'both'], 
                       help="Augmentation strategy for memory bank")
    parser.add_argument("--save_bank_top", type=str, default=None, help="Save top memory bank to file")
    parser.add_argument("--save_bank_bottom", type=str, default=None, help="Save bottom memory bank to file")
    parser.add_argument("--load_bank_top", type=str, default=None, help="Load top memory bank from file")
    parser.add_argument("--load_bank_bottom", type=str, default=None, help="Load bottom memory bank from file")
    
    # Inference parameters
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors")
    parser.add_argument("--sigma", type=int, default=4, help="Gaussian smoothing sigma")
    parser.add_argument("--no_mask", action="store_true", help="Disable foreground masking")
    parser.add_argument("--mask_threshold", type=float, default=10.0, help="PCA threshold for masking")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Ratio of test images to sample")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed maximum for heatmap scaling")
    parser.add_argument("--debug", action="store_true", help="Show debug visualizations")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading TensorRT engine: {args.engine_path}...")
    print(f"Using spatial partitioning: Top {int(args.top_ratio*100)}% (center {int(args.top_center_ratio*100)}% width) / Bottom {int((1-args.top_ratio)*100)}% (full width)")
    
    model = DINOv3TensorRTWrapper(
        engine_path=args.engine_path,
        model_name=args.model_name,
        smaller_edge_size=448
    )
    
    # 1. Build Memory Banks
    if args.load_bank_top and args.load_bank_bottom:
        print(f"Loading memory banks from {args.load_bank_top} and {args.load_bank_bottom}")
        features_ref_top = np.load(args.load_bank_top)
        features_ref_bottom = np.load(args.load_bank_bottom)
        print(f"  Top bank: {features_ref_top.shape}")
        print(f"  Bottom bank: {features_ref_bottom.shape}")
    else:
        print(f"Building memory banks from {args.train_dir}")
        train_images = [os.path.join(args.train_dir, f) for f in sorted(os.listdir(args.train_dir))
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        
        if args.shots:
            train_images = train_images[:args.shots]
        
        print(f"  Using {len(train_images)} training images")
        
        features_ref_top = []
        features_ref_bottom = []
        
        for img_path in tqdm(train_images, desc="Building memory banks"):
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
            # Augmentation
            if args.augment:
                aug_images = augment_image(image, augmentation=args.augment)
            else:
                aug_images = [image]
            
            for aug_img in aug_images:
                # NOTE: TensorRT engine must support batch size 2 for this to work
                # If engine has batch size 1, we need to process regions separately
                try:
                    # Try batch processing (requires batch size 2 engine)
                    batch_array, grid_sizes, regions, _ = model.prepare_image_spatial(
                        aug_img, args.top_ratio, args.top_center_ratio
                    )
                    
                    # ✅ Use batch inference
                    # This returns [features_top, features_bottom]
                    features_list = model.extract_features_batch(batch_array)
                    features_top = features_list[0]
                    features_bottom = features_list[1]
                    
                except Exception as e:
                    print(f"Warning: Batch processing failed ({e}), falling back to sequential")
                    # Fallback: process regions separately
                    top_region, bottom_region, _ = model.split_image(aug_img, args.top_ratio, args.top_center_ratio)
                    
                    top_tensor, grid_size_top = model.prepare_image(top_region)
                    features_top = model.extract_features(top_tensor)
                    
                    bottom_tensor, grid_size_bottom = model.prepare_image(bottom_region)
                    features_bottom = model.extract_features(bottom_tensor)
                
                # Apply masking
                if not args.no_mask:
                    grid_size = grid_sizes[0] if 'grid_sizes' in locals() else grid_size_top
                    mask_top = model.compute_background_mask(features_top, grid_size,
                                                            threshold=args.mask_threshold, masking_type=True)
                    mask_bottom = model.compute_background_mask(features_bottom, grid_size,
                                                               threshold=args.mask_threshold, masking_type=True)
                    features_ref_top.append(features_top[mask_top])
                    features_ref_bottom.append(features_bottom[mask_bottom])
                else:
                    features_ref_top.append(features_top)
                    features_ref_bottom.append(features_bottom)
        
        features_ref_top = np.vstack(features_ref_top)
        features_ref_bottom = np.vstack(features_ref_bottom)
        
        print(f"  Top memory bank: {features_ref_top.shape}")
        print(f"  Bottom memory bank: {features_ref_bottom.shape}")
        
        # Save memory banks
        if args.save_bank_top:
            print(f"Saving top memory bank to {args.save_bank_top}")
            np.save(args.save_bank_top, features_ref_top)
        if args.save_bank_bottom:
            print(f"Saving bottom memory bank to {args.save_bank_bottom}")
            np.save(args.save_bank_bottom, features_ref_bottom)
    
    # Normalize memory banks (CRITICAL for correct distance calculation)
    print("Normalizing memory banks...")
    features_ref_top = np.ascontiguousarray(features_ref_top)
    features_ref_bottom = np.ascontiguousarray(features_ref_bottom)
    faiss.normalize_L2(features_ref_top)
    faiss.normalize_L2(features_ref_bottom)
    
    # FAISS Indexes
    try:
        res = faiss.StandardGpuResources()
        index_top = faiss.GpuIndexFlatL2(res, features_ref_top.shape[1])
        index_bottom = faiss.GpuIndexFlatL2(res, features_ref_bottom.shape[1])
        print("✓ Using FAISS GPU search for both regions")
    except:
        index_top = faiss.IndexFlatL2(features_ref_top.shape[1])
        index_bottom = faiss.IndexFlatL2(features_ref_bottom.shape[1])
        print("✓ Using FAISS CPU search for both regions")
    
    index_top.add(features_ref_top)
    index_bottom.add(features_ref_bottom)
    
    # 2. Run Inference
    print(f"Finding and sampling images in {args.test_dir} (Ratio: {args.sample_ratio})")
    all_test_images = []
    folder_breakdown = {}
    
    for root, dirs, files in os.walk(args.test_dir):
        folder_images = [os.path.join(root, f) for f in sorted(files) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        if not folder_images:
            continue
        
        folder_name = os.path.basename(root)
        num_to_sample = max(1, int(len(folder_images) * args.sample_ratio))
        sampled = folder_images[:num_to_sample]
        all_test_images.extend(sampled)
        folder_breakdown[folder_name] = len(sampled)
    
    print(f"  Total test images: {len(all_test_images)}")
    for folder, count in folder_breakdown.items():
        print(f"    {folder}: {count}")
    
    # Process test images
    for img_path in tqdm(all_test_images, desc="Detecting anomalies"):
        img_name = os.path.basename(img_path)
        image_test = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Split test image and prepare batch
        batch_array, grid_sizes, regions, split_point = model.prepare_image_spatial(
            image_test, args.top_ratio, args.top_center_ratio
        )
        top_region, bottom_region = regions
        
        # Extract features (process separately for batch size 1 engines)
        top_tensor = np.expand_dims(batch_array[0], axis=0)
        features_test_top = model.extract_features(top_tensor)
        grid_size_top = grid_sizes[0]
        
        bottom_tensor = np.expand_dims(batch_array[1], axis=0)
        features_test_bottom = model.extract_features(bottom_tensor)
        grid_size_bottom = grid_sizes[1]
        
        # Process top region
        if not args.no_mask:
            mask_test_top = model.compute_background_mask(features_test_top, grid_size_top,
                                                         threshold=args.mask_threshold, masking_type=True)
        else:
            mask_test_top = np.ones(features_test_top.shape[0], dtype=bool)
        
        features_test_top_masked = np.ascontiguousarray(features_test_top[mask_test_top])
        faiss.normalize_L2(features_test_top_masked)  # Normalize test features
        distances_top, _ = index_top.search(features_test_top_masked, k=args.k)
        
        if args.k > 1:
            distances_top = distances_top.mean(axis=1)
        
        # Convert L2 distance to cosine similarity distance
        distances_top = distances_top / 2.0
        
        output_distances_top = np.zeros(mask_test_top.shape)
        output_distances_top[mask_test_top] = distances_top.squeeze()
        
        # Process bottom region
        if not args.no_mask:
            mask_test_bottom = model.compute_background_mask(features_test_bottom, grid_size_bottom,
                                                            threshold=args.mask_threshold, masking_type=True)
        else:
            mask_test_bottom = np.ones(features_test_bottom.shape[0], dtype=bool)
        
        features_test_bottom_masked = np.ascontiguousarray(features_test_bottom[mask_test_bottom])
        faiss.normalize_L2(features_test_bottom_masked)  # Normalize test features
        distances_bottom, _ = index_bottom.search(features_test_bottom_masked, k=args.k)
        
        if args.k > 1:
            distances_bottom = distances_bottom.mean(axis=1)
        
        # Convert L2 distance to cosine similarity distance
        distances_bottom = distances_bottom / 2.0
        
        output_distances_bottom = np.zeros(mask_test_bottom.shape)
        output_distances_bottom[mask_test_bottom] = distances_bottom.squeeze()
        
        # Create distance maps for each region
        anomaly_map_top_cropped = dists2map(output_distances_top.reshape(grid_size_top), 
                                            top_region.shape, sigma=args.sigma)
        anomaly_map_bottom = dists2map(output_distances_bottom.reshape(grid_size_bottom),
                                       bottom_region.shape, sigma=args.sigma)
        
        # Get original image dimensions
        img_h, img_w = image_test.shape[:2]
        top_h = int(img_h * args.top_ratio)
        bottom_h = img_h - top_h
        
        # For top region: create full-width map with padding
        left_crop_pixels = int(img_w * (1 - args.top_center_ratio) / 2)
        right_crop_pixels = img_w - left_crop_pixels
        center_width = right_crop_pixels - left_crop_pixels
        
        # Resize the cropped top anomaly map
        anomaly_map_top_cropped_resized = cv2.resize(anomaly_map_top_cropped,
                                                     (center_width, top_h),
                                                     interpolation=cv2.INTER_LINEAR)
        
        # Create full-width top map with zeros (blue) on sides
        anomaly_map_top_full = np.zeros((top_h, img_w), dtype=np.float32)
        anomaly_map_top_full[:, left_crop_pixels:right_crop_pixels] = anomaly_map_top_cropped_resized
        
        # Resize bottom anomaly map
        anomaly_map_bottom_resized = cv2.resize(anomaly_map_bottom,
                                                (img_w, bottom_h),
                                                interpolation=cv2.INTER_LINEAR)
        
        # Combine maps
        anomaly_map_combined = np.vstack([anomaly_map_top_full, anomaly_map_bottom_resized])
        
        # 3. Visualization
        cols = 5 if args.debug else 4
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
        
        # Original Image with ROI lines
        axes[0].imshow(image_test)
        # Horizontal split line
        axes[0].axhline(y=split_point, color='yellow', linestyle='--', linewidth=2)
        # Vertical crop lines for top region
        img_height = image_test.shape[0]
        img_width = image_test.shape[1]
        left_crop_x = int(img_width * (1 - args.top_center_ratio) / 2)
        right_crop_x = img_width - left_crop_x
        # Draw vertical lines in top region only (axes coordinates: 0=bottom, 1=top)
        y_bottom_of_top_region = 1.0 - (split_point / img_height)
        axes[0].axvline(x=left_crop_x, ymin=y_bottom_of_top_region, ymax=1.0, 
                       color='cyan', linestyle='--', linewidth=2)
        axes[0].axvline(x=right_crop_x, ymin=y_bottom_of_top_region, ymax=1.0, 
                       color='cyan', linestyle='--', linewidth=2)
        axes[0].set_title(f"Original (Top: {int(args.top_ratio*100)}% center {int(args.top_center_ratio*100)}%)")
        axes[0].axis('off')
        
        curr_ax = 1
        
        # Top region heatmap (with padding)
        vmax = args.vmax if args.vmax is not None else max(anomaly_map_top_full.max(), anomaly_map_bottom.max())
        im_top = axes[curr_ax].imshow(anomaly_map_top_full, cmap='jet', vmin=0, vmax=vmax)
        axes[curr_ax].set_title(f"Top (Center {int(args.top_center_ratio*100)}% Width Only)")
        axes[curr_ax].axis('off')
        plt.colorbar(im_top, ax=axes[curr_ax], fraction=0.046, pad=0.04)
        curr_ax += 1
        
        # Bottom region heatmap
        im_bottom = axes[curr_ax].imshow(anomaly_map_bottom, cmap='jet', vmin=0, vmax=vmax)
        axes[curr_ax].set_title("Bottom (Full Width)")
        axes[curr_ax].axis('off')
        plt.colorbar(im_bottom, ax=axes[curr_ax], fraction=0.046, pad=0.04)
        curr_ax += 1
        
        # Combined heatmap overlay
        if args.vmax is not None:
            norm_map = np.clip(anomaly_map_combined / args.vmax, 0, 1)
        else:
            norm_map = (anomaly_map_combined - anomaly_map_combined.min()) / \
                      (anomaly_map_combined.max() - anomaly_map_combined.min() + 1e-8)
        
        heatmap_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_test, 0.6, heatmap_color, 0.4, 0)
        
        # Draw split line and crop lines on overlay
        cv2.line(overlay, (0, split_point), (overlay.shape[1], split_point), (255, 255, 0), 2)
        cv2.line(overlay, (left_crop_x, 0), (left_crop_x, split_point), (0, 255, 255), 2)
        cv2.line(overlay, (right_crop_x, 0), (right_crop_x, split_point), (0, 255, 255), 2)
        
        axes[curr_ax].imshow(overlay)
        combined_score = (mean_top1p(output_distances_top) + mean_top1p(output_distances_bottom)) / 2
        axes[curr_ax].set_title(f"Combined Overlay (Score: {combined_score:.4f})")
        axes[curr_ax].axis('off')
        curr_ax += 1
        
        # Debug: PCA visualization
        if args.debug:
            pca_vis_top = model.get_embedding_visualization(features_test_top, grid_size_top)
            pca_vis_bottom = model.get_embedding_visualization(features_test_bottom, grid_size_bottom)
            # Combine PCA visualizations
            pca_combined = np.vstack([
                cv2.resize(pca_vis_top, (img_w, top_h)),
                cv2.resize(pca_vis_bottom, (img_w, bottom_h))
            ])
            axes[curr_ax].imshow(pca_combined)
            axes[curr_ax].set_title("PCA Visualization")
            axes[curr_ax].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(img_name)[0]}_spatial.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
