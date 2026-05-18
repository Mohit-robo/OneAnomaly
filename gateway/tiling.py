import numpy as np
import cv2

def crop_regions(image_bgr: np.ndarray, regions: list) -> list:
    """
    Given an image and a list of region dictionaries/tuples,
    return a list of cropped image arrays.
    
    If regions is empty, returns the original image as a single element list.
    """
    if not regions:
        return [image_bgr]
        
    crops = []
    h_img, w_img = image_bgr.shape[:2]
    
    for reg in regions:
        if isinstance(reg, dict):
            x, y, w, h = reg['x'], reg['y'], reg['width'], reg['height']
        else:
            x, y, w, h = reg
            
        # Ensure bounds
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))
        
        crop = image_bgr[y:y+h, x:x+w]
        crops.append(crop)
        
    return crops

def build_global_score_map(patch_scores: np.ndarray, regions: list, image_shape: tuple, patch_grid: tuple) -> np.ndarray:
    """
    Stitch regional spatial scores back to the global image map.
    
    Args:
        patch_scores: (num_regions, num_patches)
        regions: list of regions [x, y, w, h] or list of dicts
        image_shape: (H, W, C)
        patch_grid: (grid_h, grid_w)
    """
    h_grid, w_grid = patch_grid
    full_score_map = np.zeros(image_shape[:2], dtype=np.float32)
    
    for i, reg in enumerate(regions):
        if isinstance(reg, dict):
            x, y, w, h = reg['x'], reg['y'], reg['width'], reg['height']
        else:
            x, y, w, h = reg
            
        if w <= 0 or h <= 0: continue
        
        reg_scores = patch_scores[i].reshape(h_grid, w_grid)
        
        if reg_scores.size == 0: continue
        
        try:
            reg_map = cv2.resize(reg_scores, (w, h), interpolation=cv2.INTER_CUBIC)
            y_end = min(y + h, image_shape[0])
            x_end = min(x + w, image_shape[1])
            full_score_map[y:y_end, x:x_end] = reg_map[:(y_end-y), :(x_end-x)]
        except Exception as e:
            print(f"Warning: Failed to resize/stitch region {i}: {e}")
            
    return full_score_map
