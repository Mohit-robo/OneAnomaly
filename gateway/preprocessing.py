import sys
import os
from pathlib import Path
import numpy as np

# Add python dir to path so we can import existing src
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / 'python'))

from src.test_thresholding import apply_threshold_mask
from src.test_bg_subtraction import apply_mask as apply_bg_mask

def apply_preprocessing(image_bgr, config, bg_reference=None):
    mode = config.get("mode", "none")
    if mode == "thresholding":
        params = config.get("threshold_params", {})
        mask, masked_img, _ = apply_threshold_mask(
            image_bgr,
            channel=params.get("channel", "gray"),
            thresh_min=params.get("thresh_min", 30),
            thresh_max=params.get("thresh_max", 220),
            morph_open=params.get("morph_open", 3),
            morph_close=params.get("morph_close", 5)
        )
        return mask, masked_img
    elif mode == "bg_subtraction":
        if bg_reference is None:
            # Fallback to thresholding if no reference
            mask, masked_img, _ = apply_threshold_mask(image_bgr)
            return mask, masked_img
            
        params = config.get("bg_sub_params", {})
        mask, masked_img, _, _, _ = apply_bg_mask(
            image_bgr,
            bg_reference,
            diff_thresh=params.get("diff_thresh", 25),
            morph_open=params.get("morph_open", 3),
            morph_close=params.get("morph_close", 7),
            fill_holes=params.get("fill_holes", True),
            min_component_area=params.get("min_component", 500)
        )
        return mask, masked_img
    else: # none or fallback
        # just return image and all-1 mask
        return np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255, image_bgr
