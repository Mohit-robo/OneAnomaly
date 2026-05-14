"""
Stage 1 — Phase 1.2: Averaged Background Subtraction
=====================================================
Builds a per-pixel mean from a folder of EMPTY background images
(no part present), subtracts it from test images to isolate the foreground.

Post-processing:
  morph open  → remove salt-and-pepper noise
  morph close → fill small holes
  refine_mask → drop small blobs + flood-fill enclosed holes

Usage:
  # Interactive — tune params live on a single image
  python tests/test_bg_subtraction.py \\
      --ref_dir ../dataset/metal_plate/bg \\
      --input_dir ../dataset/metal_plate/test/scratches/ \\
      --image ../dataset/metal_plate/test/scratches/001.png \\
      --interactive

  # Batch — process a whole folder
  python tests/test_bg_subtraction.py \\
      --ref_dir ../dataset/metal_plate/bg \\
      --input_dir ../dataset/metal_plate/test/scratches/ \\
      --diff_threshold 85 --morph_open 5 --morph_close 7 \\
      --output_dir outputs/bg_sub_test --save_masks
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_images_from_dir(folder: Path) -> list[np.ndarray]:
    """Load all images from a directory as BGR numpy arrays."""
    paths  = sorted([p for p in folder.rglob("*") if p.suffix.lower() in EXTS])
    images = [cv2.imread(str(p)) for p in paths]
    images = [img for img in images if img is not None]
    if not images:
        raise ValueError(f"No valid images found in {folder}")
    return images


def build_averaged_reference(images: list[np.ndarray]) -> np.ndarray:
    """
    Compute per-pixel mean across a list of BGR images.
    All images are resized to the dimensions of the first image.

    Returns:
        Reference image (H, W, 3) float32
    """
    h, w    = images[0].shape[:2]
    resized = [cv2.resize(img, (w, h)).astype(np.float32) for img in images]
    return np.mean(resized, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def apply_mask(
    image_bgr: np.ndarray,
    reference: np.ndarray,
    diff_threshold: float = 85.0,
    morph_open: int = 5,
    morph_close: int = 7,
    fill_holes: bool = True,
    min_component_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subtract averaged background from image, threshold to get foreground mask.

    Args:
        image_bgr           : Input BGR image
        reference           : Averaged background (BGR float32)
        diff_threshold      : Per-pixel max-channel diff threshold for foreground
        morph_open          : Morph open kernel — removes noise speckle
        morph_close         : Morph close kernel — fills small holes
        fill_holes          : Flood-fill enclosed holes in the mask
        min_component_ratio : Drop blobs smaller than this fraction of the
                              largest blob (removes stray noise outside object)

    Returns:
        mask       : Binary mask (H, W) uint8, 255 = foreground
        masked_img : Image with background zeroed
    """
    h, w   = image_bgr.shape[:2]
    ref_rs = cv2.resize(reference, (w, h))
    diff   = np.abs(image_bgr.astype(np.float32) - ref_rs)
    mask   = (diff.max(axis=2) > diff_threshold).astype(np.uint8) * 255

    if morph_open > 0:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    mask = refine_mask(mask, fill_holes=fill_holes,
                       min_component_ratio=min_component_ratio)
    return mask, cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def refine_mask(
    mask: np.ndarray,
    fill_holes: bool = True,
    min_component_ratio: float = 0.1,
) -> np.ndarray:
    """
    Remove small blobs and fill enclosed holes (black patches inside object).

    1. Drop connected components whose area < min_component_ratio × largest area.
    2. Flood-fill from image border → fill pixels enclosed by foreground.
    """
    if mask.max() == 0:
        return mask

    # Step 1 — drop small components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas    = stats[1:, cv2.CC_STAT_AREA]
    min_area = int(areas.max() * min_component_ratio)
    refined  = np.zeros_like(mask)
    for lbl, area in enumerate(areas, start=1):
        if area >= min_area:
            refined[labels == lbl] = 255
    mask = refined

    # Step 2 — fill enclosed holes
    if fill_holes and mask.max() > 0:
        flooded      = mask.copy()
        flood_canvas = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(flooded, flood_canvas, (0, 0), 255)
        mask = cv2.bitwise_or(mask, cv2.bitwise_not(flooded))

    return mask


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def build_result_grid(
    image_bgr: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    masked_img: np.ndarray,
    params: dict,
    label: str = "",
) -> np.ndarray:
    """2×2 grid: [Original | Reference] / [Mask | Masked Result] + param strip."""
    TH, TW = 480, 640

    def _r(img):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.resize(img, (TW, TH))

    def _a(img, txt, col=(0, 240, 100)):
        out = img.copy()
        cv2.putText(out, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(out, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1)
        return out

    H, W = image_bgr.shape[:2]
    cov  = (mask > 0).sum() / mask.size * 100
    grid = np.vstack([
        np.hstack([_a(_r(image_bgr),                              f"Original [{W}x{H}]"),
                   _a(_r(reference.astype(np.uint8)),             "Averaged Reference")]),
        np.hstack([_a(_r(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), f"Mask  cov={cov:.1f}%"),
                   _a(_r(masked_img),                             "Masked Result")]),
    ])
    strip = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
    pstr  = (f"diff_thr={params['diff_threshold']}  open={params['morph_open']}  "
             f"close={params['morph_close']}  {label}")
    cv2.putText(strip, pstr, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    return np.vstack([grid, strip])


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(image_path: str, reference: np.ndarray):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[ERROR] Cannot load: {image_path}")
        sys.exit(1)

    WIN = "BG Subtraction | Q=quit  S=save"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 1000)

    # Pump Qt event loop until window handle is ready (avoids NULL-handle crash)
    ph = np.zeros((100, 1280, 3), dtype=np.uint8)
    cv2.putText(ph, "Initializing...", (440, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
    cv2.imshow(WIN, ph)
    for _ in range(20):
        cv2.waitKey(10)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1.0:
            break

    cv2.createTrackbar("Diff Threshold", WIN, 85,  255, lambda _: None)
    cv2.createTrackbar("Morph Open",     WIN, 5,   31,  lambda _: None)
    cv2.createTrackbar("Morph Close",    WIN, 7,   31,  lambda _: None)
    cv2.createTrackbar("Fill Holes 0/1", WIN, 1,   1,   lambda _: None)
    cv2.createTrackbar("Min Comp %",     WIN, 10,  100, lambda _: None)

    save_dir = Path(image_path).parent / "bgsub_output"
    save_dir.mkdir(exist_ok=True)
    print(f"\n[INFO] {Path(image_path).name}  |  S=save  Q=quit\n")

    last = {}
    while True:
        diff_thr    = cv2.getTrackbarPos("Diff Threshold", WIN)
        morph_open  = cv2.getTrackbarPos("Morph Open",     WIN)
        morph_close = cv2.getTrackbarPos("Morph Close",    WIN)
        fill_holes  = bool(cv2.getTrackbarPos("Fill Holes 0/1", WIN))
        min_ratio   = cv2.getTrackbarPos("Min Comp %", WIN) / 100.0

        params = dict(diff_threshold=diff_thr, morph_open=morph_open, morph_close=morph_close)
        mask, masked_img = apply_mask(
            image_bgr, reference,
            diff_threshold=float(diff_thr),
            morph_open=morph_open, morph_close=morph_close,
            fill_holes=fill_holes, min_component_ratio=min_ratio,
        )
        last = {**params, "fill_holes": fill_holes, "min_ratio": min_ratio}

        cv2.imshow(WIN, build_result_grid(image_bgr, reference, mask, masked_img, params))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            stem = Path(image_path).stem
            cv2.imwrite(str(save_dir / f"{stem}_grid.png"),   build_result_grid(image_bgr, reference, mask, masked_img, params))
            cv2.imwrite(str(save_dir / f"{stem}_mask.png"),   mask)
            cv2.imwrite(str(save_dir / f"{stem}_masked.png"), masked_img)
            print(f"[SAVED] {save_dir}")

    cv2.destroyAllWindows()
    fh = "--fill_holes" if last.get("fill_holes") else "--no_fill_holes"
    print("\n[FINAL PARAMS]:")
    print(f"  --diff_threshold {last.get('diff_threshold', 85)} "
          f"--morph_open {last.get('morph_open', 5)} "
          f"--morph_close {last.get('morph_close', 7)} "
          f"{fh} --min_component_ratio {last.get('min_ratio', 0.1):.2f}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def run_batch(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in EXTS])
    if not image_paths:
        print(f"[ERROR] No images in {input_dir}")
        sys.exit(1)

    params = dict(diff_threshold=args.diff_threshold,
                  morph_open=args.morph_open, morph_close=args.morph_close)

    print(f"\n[Batch BG Subtraction — AVERAGE]  {len(image_paths)} images")
    print(f"  Params : {params}")
    print(f"  Output : {output_dir}\n")

    print(f"[INFO] Building reference from: {args.ref_dir}")
    reference = build_averaged_reference(load_images_from_dir(Path(args.ref_dir)))
    cv2.imwrite(str(output_dir / "averaged_reference.png"), reference.astype(np.uint8))
    print(f"[INFO] Reference saved → {output_dir / 'averaged_reference.png'}\n")

    coverages = []
    for i, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [SKIP] {img_path.name}")
            continue

        mask, masked_img = apply_mask(
            image_bgr, reference,
            diff_threshold=args.diff_threshold,
            morph_open=args.morph_open, morph_close=args.morph_close,
            fill_holes=(not args.no_fill_holes),
            min_component_ratio=args.min_component_ratio,
        )
        rel = img_path.relative_to(input_dir)
        out = (output_dir / rel).with_suffix(".png")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), build_result_grid(image_bgr, reference, mask, masked_img,
                                                params, label=img_path.name))
        if args.save_masks:
            mo = (output_dir / "masks" / rel).with_suffix(".png")
            mo.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mo), mask)

        cov = (mask > 0).sum() / mask.size * 100
        coverages.append(cov)
        print(f"  [{i+1:>3}/{len(image_paths)}] {img_path.name:<40} coverage={cov:5.1f}%")

    if coverages:
        print(f"\n[Summary]  mean={np.mean(coverages):.1f}%  "
              f"min={np.min(coverages):.1f}%  max={np.max(coverages):.1f}%")
        low = [image_paths[i].name for i, c in enumerate(coverages) if c < 5]
        if low:
            print(f"  [WARNING] Very low coverage (<5%): {low}")
    print(f"\n[DONE] → {output_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1.2 — Averaged Background Subtraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ref_dir",    required=True,
                   help="Folder of EMPTY background images (no part) for averaging")
    p.add_argument("--input_dir",  required=True,
                   help="Folder of test images to process")
    p.add_argument("--output_dir", default="outputs/bg_sub_test",
                   help="Output folder (default: outputs/bg_sub_test)")
    p.add_argument("--image",       help="Single image for --interactive preview")
    p.add_argument("--interactive", action="store_true",
                   help="Open live trackbar preview")
    p.add_argument("--diff_threshold",      type=float, default=85.0,
                   help="Pixel diff threshold (default: 85)")
    p.add_argument("--morph_open",          type=int,   default=5,
                   help="Morph open kernel size (default: 5)")
    p.add_argument("--morph_close",         type=int,   default=7,
                   help="Morph close kernel size (default: 7)")
    p.add_argument("--no_fill_holes",       action="store_true",
                   help="Disable flood-fill hole removal")
    p.add_argument("--min_component_ratio", type=float, default=0.1,
                   help="Min blob keep ratio vs largest blob (default: 0.1)")
    p.add_argument("--save_masks",          action="store_true",
                   help="Save raw binary masks alongside result grids")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Building averaged reference from: {args.ref_dir}")
    reference = build_averaged_reference(load_images_from_dir(Path(args.ref_dir)))
    print(f"[INFO] Reference ready.\n")

    if args.interactive:
        if not args.image:
            print("[ERROR] --interactive requires --image")
            sys.exit(1)
        run_interactive(args.image, reference)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
