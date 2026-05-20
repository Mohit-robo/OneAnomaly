"""
Anomaly Detector Module
Performs anomaly detection using similarity search and generates heatmaps.
"""

from typing import Tuple, Optional
from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent Tkinter thread errors
import matplotlib.pyplot as plt


class AnomalyDetector:
    """Detect anomalies using memory bank similarity search."""
    
    def __init__(
        self,
        memory_bank,
        feature_extractor,
        k_neighbors: int = 3,
        gaussian_sigma: float = 2.0
    ):
        """
        Initialize anomaly detector.
        
        Args:
            memory_bank: MemoryBank instance with good features
            feature_extractor: FeatureExtractor instance
            k_neighbors: Number of nearest neighbors for anomaly scoring
            gaussian_sigma: Sigma for gaussian smoothing of anomaly maps
        """
        self.memory_bank = memory_bank
        self.feature_extractor = feature_extractor
        self.k_neighbors = k_neighbors
        self.gaussian_sigma = gaussian_sigma
    
    def detect_anomalies(
        self,
        image: np.ndarray,
        return_heatmap: bool = True,
        alpha: float = 0.5,
        regions: Optional[list] = None
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect anomalies in an image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            return_heatmap: Whether to generate anomaly heatmap
            
        Returns:
            anomaly_score: Overall anomaly score for the image
            anomaly_map: Anomaly heatmap (256, 256) if return_heatmap=True
            overlay: Heatmap overlaid on original image if return_heatmap=True
        """
        # Extract features from test image

        # Determine if we use spatial regions
        if regions is None:
            regions = getattr(self.memory_bank, 'regions_coords', None)
        num_regions = len(regions) if regions else getattr(self.memory_bank, 'num_regions', 1)

        if regions and num_regions > 1:
            # SPATIAL MODE
            # 1. Crop image into regions
            crops = []
            for reg in regions:
                if isinstance(reg, dict):
                    x, y, w, h = int(reg['x']), int(reg['y']), int(reg['w']), int(reg['h'])
                else:
                    x, y, w, h = int(reg[0]), int(reg[1]), int(reg[2]), int(reg[3])
                crop = image[y:y+h, x:x+w]
                crops.append(crop)
            
            # 2. Extract features in batch (returns a 2D 588x768 array if 3 regions x 196 patches)
            features_flat, patch_grid = self.feature_extractor.extract_batch(crops)
            
            # Reshape it to correctly match Regional banks: (num_regions, num_patches_per_region, feature_dim)
            features_per_region = features_flat.reshape(num_regions, -1, features_flat.shape[-1])
            
            # 3. Query spatial bank
            distances, _ = self.memory_bank.query(features_per_region, k=self.k_neighbors)
            # distances shape: (num_regions, num_patches_per_region, k)
            
            # 4. Calculate scores
            if self.k_neighbors == 1:
                similarities = distances[:, :, 0]
            else:
                similarities = distances.mean(axis=2)
            
            patch_scores = 1.0 - similarities # (num_regions, num_patches_per_region)
            
            # 5. Build global anomaly map by stitching regions
            # We assume patch_grid is same for all regions
            h_grid, w_grid = patch_grid
            full_score_map = np.zeros(image.shape[:2], dtype=np.float32)
            
            for i, reg in enumerate(regions):
                if isinstance(reg, dict):
                    x, y, w, h = int(reg['x']), int(reg['y']), int(reg['w']), int(reg['h'])
                else:
                    x, y, w, h = int(reg[0]), int(reg[1]), int(reg[2]), int(reg[3])

                if w <= 0 or h <= 0: continue
                
                reg_scores = patch_scores[i].reshape(h_grid, w_grid)
                
                # Safety check for empty patches
                if reg_scores.size == 0: continue
                
                # Resize regional scores to regional pixel size
                try:
                    reg_map = cv2.resize(reg_scores, (w, h), interpolation=cv2.INTER_CUBIC)
                    # Bounds check for stitching
                    y_end = min(y + h, image.shape[0])
                    x_end = min(x + w, image.shape[1])
                    full_score_map[y:y_end, x:x_end] = reg_map[:(y_end-y), :(x_end-x)]
                except Exception as e:
                    print(f"Warning: Failed to resize/stitch region {i}: {e}")
                
            # image_score = mean_top_1_percent(patch_scores)
            n_top = max(1, int(patch_scores.size * 0.01))
            anomaly_score = float(np.sort(patch_scores.flatten())[-n_top:].mean())
            
            score_map = full_score_map
        else:
            # NON-SPATIAL MODE (Standard)
            features, patch_grid = self.feature_extractor.extract_from_image(image)
            distances, _ = self.memory_bank.query(features, k=self.k_neighbors)
            
            if self.k_neighbors == 1:
                similarities = distances[:, 0]
            else:
                similarities = distances.mean(axis=1)
                
            patch_scores = 1.0 - similarities
            
            # image_score = mean_top_1_percent(patch_scores)
            n_top = max(1, int(patch_scores.size * 0.01))
            anomaly_score = float(np.sort(patch_scores.flatten())[-n_top:].mean())
            
            h, w = patch_grid
            score_map = patch_scores.reshape(h, w)
            score_map = cv2.resize(score_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Common Post-processing
        if self.gaussian_sigma > 0:
            score_map = gaussian_filter(score_map, sigma=self.gaussian_sigma)
        
        # Robust Normalization for Heatmap
        # We want to stretch the values between a 'normal' baseline and an 'anomaly' ceiling.
        # This prevents normal images from lighting up due to min-max scaling of the whole map.
        v_min = score_map.min()
        v_max = score_map.max()
        
        # We normalize such that scores above 0.3 start becoming significant.
        # This matches the threshold usually used for Cosine Distances with DINO.
        norm_map = np.clip((score_map - 0.0) / (max(v_max, 0.4) - 0.0), 0.0, 1.0)
        score_map = norm_map
        
        # Create overlay
        overlay = self._create_overlay(image, score_map, alpha=alpha)
        
        return anomaly_score, score_map, overlay
    
    def _create_overlay(
        self,
        image: np.ndarray,
        anomaly_map: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create heatmap overlay on original image.
        
        Args:
            image: Original image (H, W, C) in RGB
            anomaly_map: Anomaly heatmap (H, W) in [0, 1]
            alpha: Blending factor for overlay
            
        Returns:
            Overlay image (H, W, C) in RGB
        """
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Create colored heatmap (red = high anomaly)
        heatmap_colored = cv2.applyColorMap(
            (anomaly_map * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        # Reference uses 1-alpha original, alpha heatmap
        overlay = cv2.addWeighted(
            image,
            1.0 - alpha,
            heatmap_colored,
            alpha,
            0
        )
        
        return overlay
    
    def detect_from_file(
        self,
        image_path: str,
        save_output: bool = False,
        output_dir: Optional[str] = None,
        alpha: float = 0.5,
        regions: Optional[list] = None
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect anomalies in an image file.
        
        Args:
            image_path: Path to input image
            save_output: Whether to save output images
            output_dir: Directory to save outputs
            
        Returns:
            anomaly_score: Overall anomaly score
            anomaly_map: Anomaly heatmap
            overlay: Overlay image
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect anomalies
        score, heatmap, overlay = self.detect_anomalies(image, return_heatmap=True, alpha=alpha, regions=regions)
        
        # Save outputs if requested
        if save_output and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            
            # Save source image (for side-by-side view)
            source_path = output_path / f"{image_name}_source.png"
            cv2.imwrite(str(source_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Save heatmap
            heatmap_path = output_path / f"{image_name}_heatmap.png"
            cv2.imwrite(
                str(heatmap_path),
                (heatmap * 255).astype(np.uint8)
            )
            
            # Save overlay
            overlay_path = output_path / f"{image_name}_overlay.png"
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_path), overlay_bgr)
            
            # Save original with score annotation
            annotated = image.copy()
            cv2.putText(
                annotated,
                f"Anomaly Score: {score:.4f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2
            )
            annotated_path = output_path / f"{image_name}_annotated.png"
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(annotated_path), annotated_bgr)
            
            # Save Side-by-Side Plot (Matching reference script)
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(image)
                axes[0].set_title("Original")
                axes[1].imshow(overlay)
                axes[1].set_title("Anomaly Heatmap")
                for ax in axes: ax.axis('off')
                
                plt.savefig(str(output_path / f"torch_res_{image_name}.png"))
                plt.close()
            except Exception as e:
                print(f"Warning: Could not save matplotlib plot: {e}")
        
        return score, heatmap, overlay
    
    def detect_from_folder(
        self,
        folder_path: str,
        output_dir: str,
        extensions: Optional[list] = None,
        alpha: float = 0.5
    ) -> dict:
        """
        Detect anomalies in all images in a folder.
        
        Args:
            folder_path: Path to folder with test images
            output_dir: Directory to save outputs
            extensions: Valid image extensions
            
        Returns:
            Dictionary with results for each image
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Get image files recursively
        folder = Path(folder_path)
        image_files = [
            f for f in folder.rglob('*')
            if f.suffix.lower() in extensions and f.is_file()
        ]
        
        if not image_files:
            raise ValueError(f"No images found in {folder_path}")
        
        print(f"Processing {len(image_files)} images for anomaly detection...")
        
        results = {}
        
        for i, img_path in enumerate(image_files):
            try:

                print(f"Processing image {i+1}/{len(image_files)}: {img_path.name}")
                
                score, heatmap, overlay = self.detect_from_file(
                    str(img_path),
                    save_output=True,
                    output_dir=output_dir,
                    alpha=alpha
                )
                
                results[img_path.name] = {
                    'anomaly_score': score,
                    'processed': True
                }
                
                # print(f"  {img_path.name}: score={score:.4f}")
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                results[img_path.name] = {
                    'anomaly_score': None,
                    'processed': False,
                    'error': str(e)
                }
        
        return results

if __name__ == "__main__":
    import argparse
    import sys
    from feature_extractor import FeatureExtractor
    from memory_bank import MemoryBank
    
    def main():
        parser = argparse.ArgumentParser(description="Test Anomaly Detector Standalone")
        parser.add_argument("--good_images", type=str, required=True, help="Path to folder containing good/normal images")
        parser.add_argument("--input", type=str, required=True, help="Path to test image or folder")
        parser.add_argument("--output", type=str, default="debug_results", help="Directory to save results")
        parser.add_argument("--save_bank", type=str, help="Path to save the generated memory bank (.pkl)")
        parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
        
        args = parser.parse_args()
        
        print(f"Setting up test environment...")
        
        # 1. Initialize Feature Extractor
        print("Loading Feature Extractor...")
        try:
            fe = FeatureExtractor(
                model_name='dinov3_vits16',
                device=args.device,
                backend='pytorch'
            )
        except Exception as e:
            print(f"Error loading feature extractor: {e}")
            return
            
        # 2. Build Memory Bank On-the-Fly
        print(f"Building Memory Bank from {args.good_images}...")
        try:
            # Initialize empty bank
            mb = MemoryBank(
                feature_dim=fe.get_feature_dimension(),
                use_gpu=True,
                coreset_sampling_ratio=1.0 # Use all features
            )
            
            # Extract features from good folder
            features, image_paths = fe.extract_from_folder(args.good_images)
            print(f"Extracted {features.shape[0]} features from {len(image_paths)} good images")
            
            # Add to bank
            mb.add_features(features)
            
            # SAVE the bank if requested
            if args.save_bank:
                print(f"Saving memory bank to {args.save_bank}...")
                mb.save(args.save_bank)
            
        except Exception as e:
            print(f"Error building memory bank: {e}")
            return

        # 3. Initialize Anomaly Detector
        detector = AnomalyDetector(
            memory_bank=mb,
            feature_extractor=fe,
            k_neighbors=1,
            gaussian_sigma=4.0
        )
        
        # 4. Run Detection
        input_path = Path(args.input)
        
        if input_path.is_file():
            print(f"Processing single image: {input_path}")
            score, _, _ = detector.detect_from_file(
                str(input_path),
                save_output=True,
                output_dir=args.output
            )
            print(f"Done. Anomaly Score: {score:.4f}")
            print(f"Results saved to {args.output}")
            
        elif input_path.is_dir():
            print(f"Processing folder: {input_path}")
            detector.detect_from_folder(
                str(input_path),
                output_dir=args.output
            )
            print(f"Done. Results saved to {args.output}")
            
        else:
            print(f"Error: Input path not found: {input_path}")

    main()
