"""
Stage 1 — Phase 1.1: Thresholding-Based Background Masking
===========================================================
Local test script to validate intensity thresholding logic.

Modes:
  --interactive   Open a live OpenCV window with trackbars to tune parameters.
  --batch         Process all images in input_dir and save outputs to output_dir.

Usage:
  # Interactive preview on a single image
  python tests/test_thresholding.py --image path/to/image.jpg --interactive

  # Batch process a folder
  python tests/test_thresholding.py --input_dir path/to/images --output_dir path/to/output --batch

  # Batch with explicit params
  python tests/test_thresholding.py --input_dir dataset/good --output_dir outputs/thresh_test \
      --channel gray --thresh_min 30 --thresh_max 220 --batch
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core thresholding logic
# ---------------------------------------------------------------------------

CHANNEL_OPTIONS = ["gray", "red", "green", "blue", "saturation", "value"]

def get_channel(image_bgr: np.ndarray, channel: str) -> np.ndarray:
    """
    Extract a single channel from a BGR image for thresholding.

    Args:
        image_bgr: BGR image (H, W, 3)
        channel: One of 'gray', 'red', 'green', 'blue', 'saturation', 'value'

    Returns:
        Single-channel image (H, W) uint8
    """
    if channel == "gray":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif channel == "red":
        return image_bgr[:, :, 2]
    elif channel == "green":
        return image_bgr[:, :, 1]
    elif channel == "blue":
        return image_bgr[:, :, 0]
    elif channel in ("saturation", "value"):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        return hsv[:, :, 1] if channel == "saturation" else hsv[:, :, 2]
    else:
        raise ValueError(f"Unknown channel: {channel}. Choose from {CHANNEL_OPTIONS}")


def apply_threshold_mask(
    image_bgr: np.ndarray,
    channel: str = "gray",
    thresh_min: int = 30,
    thresh_max: int = 220,
    morph_open: int = 3,
    morph_close: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a binary foreground mask via intensity thresholding.

    Steps:
      1. Extract channel
      2. Apply in-range threshold → binary mask
      3. Morphological open (remove noise) then close (fill holes)
      4. Apply mask to original image

    Args:
        image_bgr   : Input BGR image
        channel     : Channel to threshold
        thresh_min  : Lower intensity bound (inclusive)
        thresh_max  : Upper intensity bound (inclusive)
        morph_open  : Kernel size for morphological opening (0 = skip)
        morph_close : Kernel size for morphological closing (0 = skip)

    Returns:
        mask        : Binary mask (H, W) uint8, 255 = foreground
        masked_img  : Original image with background zeroed (H, W, 3) BGR
        channel_img : The extracted channel image for visual inspection
    """
    ch = get_channel(image_bgr, channel)

    # Threshold
    mask = cv2.inRange(ch, thresh_min, thresh_max)

    # Morphological cleanup
    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Apply mask
    masked_img = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    channel_img = cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)

    return mask, masked_img, channel_img


def build_result_grid(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    masked_img: np.ndarray,
    channel_img: np.ndarray,
    params: dict,
) -> np.ndarray:
    """
    Build a 2×2 display grid:
      [Original]  [Channel]
      [Mask]      [Masked Result]

    Annotates param values on the grid for readability.
    """
    H, W = image_bgr.shape[:2]
    target_h, target_w = 480, 640

    def _resize(img):
        return cv2.resize(img, (target_w, target_h))

    orig_r     = _resize(image_bgr)
    ch_r       = _resize(channel_img)
    mask_r     = _resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    masked_r   = _resize(masked_img)

    # Annotation bar
    def _annotate(img, text, color=(255, 255, 255)):
        out = img.copy()
        cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return out

    coverage = (mask > 0).sum() / mask.size * 100
    orig_r   = _annotate(orig_r,   f"Original  [{W}x{H}]")
    ch_r     = _annotate(ch_r,     f"Channel: {params['channel']}")
    mask_r   = _annotate(mask_r,   f"Mask  coverage={coverage:.1f}%")
    masked_r = _annotate(masked_r, f"Masked result")

    top    = np.hstack([orig_r, ch_r])
    bottom = np.hstack([mask_r, masked_r])
    grid   = np.vstack([top, bottom])

    return grid


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(image_path: str):
    """Open an OpenCV window with trackbars for live parameter tuning."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        sys.exit(1)

    WIN = "Thresholding Stage 1.1 | Q=quit S=save"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 1000)

    # Pump the Qt event loop until the window is confirmed visible.
    # getWindowProperty returns -1 when the window doesn't exist yet.
    # This is more reliable than a fixed waitKey sleep on all Qt builds.
    placeholder = np.zeros((100, 1280, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Initializing...", (440, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
    cv2.imshow(WIN, placeholder)
    for _ in range(20):          # up to ~200 ms of event-loop pumping
        cv2.waitKey(10)
        prop = cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE)
        if prop >= 1.0:
            break

    # Now the window handle is guaranteed to exist — safe to add trackbars
    cv2.createTrackbar("Channel 0-5",  WIN, 0,   5,   lambda _: None)
    cv2.createTrackbar("Thresh Min",   WIN, 30,  255, lambda _: None)
    cv2.createTrackbar("Thresh Max",   WIN, 220, 255, lambda _: None)
    cv2.createTrackbar("Morph Open",   WIN, 3,   31,  lambda _: None)
    cv2.createTrackbar("Morph Close",  WIN, 5,   31,  lambda _: None)

    save_dir = Path(image_path).parent / "thresh_interactive_output"
    save_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Interactive mode — {Path(image_path).name}")
    print(f"[INFO] Channel options: {' | '.join(f'{i}={c}' for i, c in enumerate(CHANNEL_OPTIONS))}")
    print(f"[INFO] Press S to save current result, Q to quit.\n")

    last_grid = None
    while True:
        ch_idx      = cv2.getTrackbarPos("Channel 0-5",  WIN)
        thresh_min  = cv2.getTrackbarPos("Thresh Min",   WIN)
        thresh_max  = cv2.getTrackbarPos("Thresh Max",   WIN)
        morph_open  = cv2.getTrackbarPos("Morph Open",   WIN)
        morph_close = cv2.getTrackbarPos("Morph Close",  WIN)

        thresh_max = max(thresh_max, thresh_min + 1)     # clamp so min < max

        channel = CHANNEL_OPTIONS[ch_idx]
        params  = dict(channel=channel, thresh_min=thresh_min,
                       thresh_max=thresh_max, morph_open=morph_open, morph_close=morph_close)

        mask, masked_img, channel_img = apply_threshold_mask(image_bgr, **params)
        grid = build_result_grid(image_bgr, mask, masked_img, channel_img, params)
        last_grid = (grid, mask, masked_img, params)

        cv2.imshow(WIN, grid)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            stem = Path(image_path).stem
            cv2.imwrite(str(save_dir / f"{stem}_grid.png"), grid)
            cv2.imwrite(str(save_dir / f"{stem}_mask.png"), mask)
            cv2.imwrite(str(save_dir / f"{stem}_masked.png"), masked_img)
            print(f"[SAVED] Results written to {save_dir}")

    cv2.destroyAllWindows()
    if last_grid:
        _, _, _, params = last_grid
        print("\n[FINAL PARAMS] — copy these into batch mode:")
        print(f"  --channel {params['channel']} --thresh_min {params['thresh_min']} "
              f"--thresh_max {params['thresh_max']} "
              f"--morph_open {params['morph_open']} --morph_close {params['morph_close']}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

def run_batch(args):
    """Process all images in input_dir and save multi-panel outputs."""
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in EXTS])
    if not image_paths:
        print(f"[ERROR] No images found in {input_dir}")
        sys.exit(1)

    params = dict(
        channel=args.channel,
        thresh_min=args.thresh_min,
        thresh_max=args.thresh_max,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
    )

    print(f"\n[Batch Thresholding] {len(image_paths)} images")
    print(f"  Params: {params}")
    print(f"  Output: {output_dir}\n")

    coverages = []
    for i, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [SKIP] Cannot load: {img_path.name}")
            continue

        mask, masked_img, channel_img = apply_threshold_mask(image_bgr, **params)
        grid = build_result_grid(image_bgr, mask, masked_img, channel_img, params)

        # Preserve relative sub-folder structure
        rel  = img_path.relative_to(input_dir)
        out  = (output_dir / rel).with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out), grid)

        if args.save_masks:
            mask_out = (output_dir / "masks" / rel).with_suffix(".png")
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_out), mask)

        coverage = (mask > 0).sum() / mask.size * 100
        coverages.append(coverage)
        print(f"  [{i+1:>3}/{len(image_paths)}] {img_path.name:<40} coverage={coverage:5.1f}%")

    if coverages:
        print(f"\n[Summary]")
        print(f"  Images processed : {len(coverages)}")
        print(f"  Coverage — mean  : {np.mean(coverages):.1f}%")
        print(f"  Coverage — min   : {np.min(coverages):.1f}%")
        print(f"  Coverage — max   : {np.max(coverages):.1f}%")
        low = [image_paths[i].name for i, c in enumerate(coverages) if c < 10]
        if low:
            print(f"  [WARNING] Low coverage (<10%) images: {low}")
    print(f"\n[DONE] Results saved to {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1.1 — Thresholding Background Masking Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--interactive", action="store_true",
                      help="Live trackbar preview (requires --image)")
    mode.add_argument("--batch", action="store_true",
                      help="Batch process folder (requires --input_dir, --output_dir)")

    # Inputs
    p.add_argument("--image",     type=str, help="Single image path (for --interactive)")
    p.add_argument("--input_dir", type=str, help="Folder of images (for --batch)")
    p.add_argument("--output_dir",type=str, default="outputs/thresh_test", help="Output folder")

    # Threshold params
    p.add_argument("--channel",    type=str,   default="gray",
                   choices=CHANNEL_OPTIONS, help="Channel to threshold")
    p.add_argument("--thresh_min", type=int,   default=30,  help="Min intensity (0-255)")
    p.add_argument("--thresh_max", type=int,   default=220, help="Max intensity (0-255)")
    p.add_argument("--morph_open", type=int,   default=3,   help="Morph open kernel size (0=off)")
    p.add_argument("--morph_close",type=int,   default=5,   help="Morph close kernel size (0=off)")

    # Extra
    p.add_argument("--save_masks", action="store_true",
                   help="Also save raw binary masks alongside grid images")

    return p.parse_args()


def main():
    args = parse_args()

    if args.interactive:
        if not args.image:
            print("[ERROR] --interactive requires --image")
            sys.exit(1)
        run_interactive(args.image)
    elif args.batch:
        if not args.input_dir:
            print("[ERROR] --batch requires --input_dir")
            sys.exit(1)
        run_batch(args)


if __name__ == "__main__":
    main()
