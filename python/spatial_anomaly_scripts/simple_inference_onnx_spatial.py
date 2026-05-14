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

def split_image(image, top_ratio=0.4, top_center_ratio=0.6):
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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Folder with normal images.")
    parser.add_argument("--test_dir", type=str, required=True, help="Folder with test images.")
    parser.add_argument("--model_name", type=str, default="dinov3_vits16", help="DINOv3 model name.")
    parser.add_argument("--onnx_path", type=str, required=True, help="Path to ONNX model file.")
    parser.add_argument("--output_dir", type=str, default="inference_results_onnx_spatial", help="Where to save visual results.")
    parser.add_argument("--shots", type=int, default=5, help="Number of normal images to use.")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--mask_threshold", type=float, default=10.0)
    parser.add_argument("--no_mask", action="store_true", help="Disable foreground masking.")
    parser.add_argument("--save_bank_top", type=str, default=None, help="Save top memory bank.")
    parser.add_argument("--save_bank_bottom", type=str, default=None, help="Save bottom memory bank.")
    parser.add_argument("--load_bank_top", type=str, default=None, help="Load top memory bank.")
    parser.add_argument("--load_bank_bottom", type=str, default=None, help="Load bottom memory bank.")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Percentage of images to test.")
    parser.add_argument("--debug", action="store_true", help="Show extra visualization.")
    parser.add_argument("--augment", type=str, default=None, help="Augmentation (e.g. hflip).")
    parser.add_argument("--k", type=int, default=1, help="Number of neighbors for k-NN averaging.")
    parser.add_argument("--sigma", type=int, default=4, help="Gaussian smoothing sigma.")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed maximum for heatmap scaling.")
    parser.add_argument("--top_ratio", type=float, default=0.4, help="Ratio for top region (default: 0.4)")
    parser.add_argument("--top_center_ratio", type=float, default=0.6, help="Horizontal center ratio for top region (default: 0.6 for center 60%)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ONNX model: {args.onnx_path}...")
    print(f"Using spatial partitioning: Top {int(args.top_ratio*100)}% (center {int(args.top_center_ratio*100)}% width) / Bottom {int((1-args.top_ratio)*100)}% (full width)")
    
    model = DINOv3ONNXWrapper(
        onnx_path=args.onnx_path,
        model_name=args.model_name,
        smaller_edge_size=args.resolution
    )

    # 1. Build Memory Banks (Top and Bottom)
    if args.load_bank_top and args.load_bank_bottom:
        print(f"Loading top memory bank from {args.load_bank_top}")
        features_ref_top = np.load(args.load_bank_top).astype('float32')
        print(f"Loading bottom memory bank from {args.load_bank_bottom}")
        features_ref_bottom = np.load(args.load_bank_bottom).astype('float32')
    else:
        print(f"Building memory banks using {args.shots} images from {args.train_dir}")
        train_images = [f for f in sorted(os.listdir(args.train_dir)) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:args.shots]
        
        features_ref_top_list = []
        features_ref_bottom_list = []
        
        for img_name in tqdm(train_images, desc="Extracting normal features"):
            img_path = os.path.join(args.train_dir, img_name)
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
            if args.augment:
                aug_images = augment_image(image, augmentation=args.augment)
            else:
                aug_images = [image]
            
            for aug_img in aug_images:
                # Split image into top and bottom
                top_region, bottom_region, _ = split_image(aug_img, args.top_ratio, args.top_center_ratio)
                
                # Prepare both regions
                image_tensor_top, grid_size_top = model.prepare_image(top_region)
                image_tensor_bottom, grid_size_bottom = model.prepare_image(bottom_region)
                
                # Batch process both regions together
                features_batch = model.extract_features_batch([image_tensor_top, image_tensor_bottom])
                features_top = features_batch[0]  # (N_top, C)
                features_bottom = features_batch[1]  # (N_bottom, C)
                
                if not args.no_mask:
                    mask_top = model.compute_background_mask(features_top, grid_size_top, 
                                                            threshold=args.mask_threshold, masking_type=True)
                    features_top = features_top[mask_top]
                    
                    mask_bottom = model.compute_background_mask(features_bottom, grid_size_bottom,
                                                               threshold=args.mask_threshold, masking_type=True)
                    features_bottom = features_bottom[mask_bottom]
                
                features_ref_top_list.append(features_top)
                features_ref_bottom_list.append(features_bottom)

        features_ref_top = np.concatenate(features_ref_top_list, axis=0).astype('float32')
        features_ref_bottom = np.concatenate(features_ref_bottom_list, axis=0).astype('float32')
        
        print(f"Top memory bank size: {features_ref_top.shape}")
        print(f"Bottom memory bank size: {features_ref_bottom.shape}")
        
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

    for img_path in tqdm(all_test_images, desc="Detecting anomalies"):
        img_name = os.path.basename(img_path)
        image_test = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # Split test image
        top_region, bottom_region, split_point = split_image(image_test, args.top_ratio, args.top_center_ratio)
        
        # Prepare both regions
        image_tensor_top, grid_size_top = model.prepare_image(top_region)
        image_tensor_bottom, grid_size_bottom = model.prepare_image(bottom_region)
        
        # Batch process both regions together (single inference call)
        features_batch = model.extract_features_batch([image_tensor_top, image_tensor_bottom])
        features_test_top = features_batch[0]  # (N_top, C)
        features_test_bottom = features_batch[1]  # (N_bottom, C)
        
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
        # The cropped region should be placed in the center with zeros on sides
        left_crop_pixels = int(img_w * (1 - args.top_center_ratio) / 2)
        right_crop_pixels = img_w - left_crop_pixels
        center_width = right_crop_pixels - left_crop_pixels
        
        # Resize the cropped top anomaly map to match the actual top region height
        anomaly_map_top_cropped_resized = cv2.resize(anomaly_map_top_cropped,
                                                     (center_width, top_h),
                                                     interpolation=cv2.INTER_LINEAR)
        
        # Create full-width top map with zeros (blue) on sides
        anomaly_map_top_full = np.zeros((top_h, img_w), dtype=np.float32)
        anomaly_map_top_full[:, left_crop_pixels:right_crop_pixels] = anomaly_map_top_cropped_resized
        
        # Resize bottom anomaly map to match dimensions
        anomaly_map_bottom_resized = cv2.resize(anomaly_map_bottom,
                                                (img_w, bottom_h),
                                                interpolation=cv2.INTER_LINEAR)
        
        # Now combine (both have same width and proper alignment)
        anomaly_map_combined = np.vstack([anomaly_map_top_full, anomaly_map_bottom_resized])
        
        # 3. Visualization
        cols = 5 if args.debug else 4
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
        
        # Original Image with ROI lines
        axes[0].imshow(image_test)
        # Horizontal split line
        axes[0].axhline(y=split_point, color='yellow', linestyle='--', linewidth=2)
        # Vertical crop lines for top region (showing center 60%)
        img_height = image_test.shape[0]
        img_width = image_test.shape[1]
        left_crop_x = int(img_width * (1 - args.top_center_ratio) / 2)
        right_crop_x = img_width - left_crop_x
        # Only draw vertical lines in top region
        # ymin/ymax are in axes coordinates: 0=bottom, 1=top
        # Top region is from ymin=(1 - split_point/height) to ymax=1
        y_bottom_of_top_region = 1.0 - (split_point / img_height)
        axes[0].axvline(x=left_crop_x, ymin=y_bottom_of_top_region, ymax=1.0, 
                       color='cyan', linestyle='--', linewidth=2)
        axes[0].axvline(x=right_crop_x, ymin=y_bottom_of_top_region, ymax=1.0, 
                       color='cyan', linestyle='--', linewidth=2)
        axes[0].set_title(f"Original (Top: {int(args.top_ratio*100)}% center {int(args.top_center_ratio*100)}%)")
        axes[0].axis('off')
        
        curr_ax = 1
        
        # Top region heatmap (with padding to show full width)
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
        
        # Draw split line on overlay
        cv2.line(overlay, (0, split_point), (overlay.shape[1], split_point), (255, 255, 0), 2)
        # Draw vertical crop lines in top region only
        cv2.line(overlay, (left_crop_x, 0), (left_crop_x, split_point), (0, 255, 255), 2)
        cv2.line(overlay, (right_crop_x, 0), (right_crop_x, split_point), (0, 255, 255), 2)
        
        axes[curr_ax].imshow(overlay)
        combined_score = (mean_top1p(output_distances_top) + mean_top1p(output_distances_bottom)) / 2
        axes[curr_ax].set_title(f"Combined Overlay (Score: {combined_score:.4f})")
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
