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
        return_heatmap: bool = True
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
        # Force resize to 256x256 to match reference script logic
        # This ensures sigma=2.0 smoothing has the same effect
        if image.shape[:2] != (256, 256):
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
            
        # Extract features from test image

        # Extract features from test image
        features, patch_grid = self.feature_extractor.extract_from_image(image)
        
        # Query memory bank for nearest neighbors
        distances, _ = self.memory_bank.query(features, k=self.k_neighbors)
        
        # Anomaly score is the distance to nearest neighbor
        # For k > 1, we can use mean or max distance
        # Calculate Cosine Distance (1 - Similarity)
        # distances from IndexFlatIP are cosine similarities (dot product of normalized vectors)
        if self.k_neighbors == 1:
            similarities = distances[:, 0]
        else:
            similarities = distances.mean(axis=1)
            
        # Convert similarity to distance (1 - sim)
        # Sim is in [-1, 1], so distance in [0, 2]
        patch_scores = 1.0 - similarities
        
        # Overall image anomaly score (max patch score)
        anomaly_score = float(patch_scores.max())
        
        # DEBUG: Print score stats
        # print(f"DEBUG: Patch Scores (1-Sim) - Min: {patch_scores.min():.4f}, Max: {patch_scores.max():.4f}, Mean: {patch_scores.mean():.4f}")
        
        if not return_heatmap:
            return anomaly_score, None, None
        
        # Generate anomaly heatmap
        # Reference logic: Upscale (CUBIC) -> Smooth -> Normalize
        
        target_size = image.shape[:2]
        
        # 1. Reshape scores to grid
        h, w = patch_grid
        score_map = patch_scores.reshape(h, w)
        
        # 2. Upscale to target size using Cubic interpolation
        score_map = cv2.resize(
            score_map,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        # 3. Apply gaussian smoothing
        if self.gaussian_sigma > 0:
            score_map = gaussian_filter(score_map, sigma=self.gaussian_sigma)
        
        # 4. Normalize to [0, 1]
        score_map = (score_map - score_map.min()) / (
            score_map.max() - score_map.min() + 1e-8
        )
        
        # Create overlay
        overlay = self._create_overlay(image, score_map)
        
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
        # Reference uses 0.7 original, 0.3 heatmap
        overlay = cv2.addWeighted(
            image,
            0.7,
            heatmap_colored,
            0.3,
            0
        )
        
        return overlay
    
    def detect_from_file(
        self,
        image_path: str,
        save_output: bool = False,
        output_dir: Optional[str] = None
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
        score, heatmap, overlay = self.detect_anomalies(image, return_heatmap=True)
        
        # Save outputs if requested
        if save_output and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            image_name = Path(image_path).stem
            
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
        extensions: Optional[list] = None
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
                    output_dir=output_dir
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
