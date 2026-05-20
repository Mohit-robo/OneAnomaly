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
    """
    Apply preprocessing to an image based on the given config dict.
    
    Config keys (flat, as sent from the frontend):
      mode: 'thresholding' | 'average' | 'none'
      channel, thresh_min, thresh_max, morph_open, morph_close  (thresholding)
      diff_threshold, fill_holes, min_component_ratio            (average/bg_subtraction)
    
    Also supports nested threshold_params / bg_sub_params for backward compat.
    """
    mode = config.get("mode", "none")
    print(f"[preprocessing] mode={mode}, config_keys={list(config.keys())}")

    if mode == "thresholding":
        # Support both flat keys (frontend) and nested threshold_params (legacy)
        nested = config.get("threshold_params", {})
        channel   = config.get("channel",    nested.get("channel",    "gray"))
        thresh_min = config.get("thresh_min", nested.get("thresh_min", 30))
        thresh_max = config.get("thresh_max", nested.get("thresh_max", 220))
        morph_open = config.get("morph_open", nested.get("morph_open", 3))
        morph_close = config.get("morph_close", nested.get("morph_close", 5))

        print(f"[preprocessing] thresholding: channel={channel} min={thresh_min} max={thresh_max} open={morph_open} close={morph_close}")
        mask, masked_img, _ = apply_threshold_mask(
            image_bgr,
            channel=channel,
            thresh_min=thresh_min,
            thresh_max=thresh_max,
            morph_open=morph_open,
            morph_close=morph_close
        )
        return mask, masked_img

    elif mode in ("average", "bg_subtraction"):
        if bg_reference is None:
            print("[preprocessing] bg_reference not available, falling back to thresholding")
            mask, masked_img, _ = apply_threshold_mask(image_bgr)
            return mask, masked_img

        # Support both flat keys (frontend) and nested bg_sub_params (legacy)
        nested = config.get("bg_sub_params", {})
        diff_thresh         = config.get("diff_threshold",      nested.get("diff_thresh", 25))
        morph_open          = config.get("morph_open",          nested.get("morph_open", 3))
        morph_close         = config.get("morph_close",         nested.get("morph_close", 7))
        fill_holes          = config.get("fill_holes",          nested.get("fill_holes", True))
        min_component_area  = config.get("min_component_ratio", nested.get("min_component", 500))

        print(f"[preprocessing] bg_subtraction: diff_thresh={diff_thresh}")
        mask, masked_img, _, _, _ = apply_bg_mask(
            image_bgr,
            bg_reference,
            diff_thresh=diff_thresh,
            morph_open=morph_open,
            morph_close=morph_close,
            fill_holes=fill_holes,
            min_component_area=min_component_area
        )
        return mask, masked_img

    else:  # 'none' or unknown — pass through
        print("[preprocessing] mode=none, returning original image")
        return np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255, image_bgr
