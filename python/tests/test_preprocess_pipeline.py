"""
Stage 1 — Combined Preprocessing Pipeline Test
===============================================
Chains Phase 1.1 (thresholding) and Phase 1.2 (BG subtraction) together
and feeds the resulting masks into the existing anomaly detection pipeline.

This is the end-to-end local validation script for Stage 1. It:
  1. Builds a memory bank from "good" images with preprocessing applied
  2. Runs anomaly detection on "test" images with the same preprocessing
  3. Saves a 4-panel result: [Original | Preprocessed | Heatmap | Overlay]

Usage:
  # Thresholding preprocess → anomaly detection
  python tests/test_preprocess_pipeline.py \
      --preprocess thresh \
      --good_dir dataset/good --test_dir dataset/test \
      --output_dir outputs/pipeline_test \
      --onnx_path models/dinov3_vits16.onnx

  # Averaged BG subtraction → anomaly detection
  python tests/test_preprocess_pipeline.py \
      --preprocess average \
      --good_dir dataset/good --test_dir dataset/test \
      --output_dir outputs/pipeline_test \
      --onnx_path models/dinov3_vits16.onnx

  # Template BG subtraction → anomaly detection
  python tests/test_preprocess_pipeline.py \
      --preprocess template --template path/to/blank.jpg \
      --good_dir dataset/good --test_dir dataset/test \
      --output_dir outputs/pipeline_test \
      --onnx_path models/dinov3_vits16.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Path setup: resolve sibling test modules + spatial_anomaly_scripts/src ----
_TESTS_DIR      = Path(__file__).resolve().parent
_PYTHON_DIR     = _TESTS_DIR.parent
_SPATIAL_DIR    = _PYTHON_DIR / "spatial_anomaly_scripts"

for _p in [str(_TESTS_DIR), str(_PYTHON_DIR), str(_SPATIAL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_thresholding   import apply_threshold_mask, CHANNEL_OPTIONS   # noqa: E402
from test_bg_subtraction import build_averaged_reference, mask_by_template, load_images_from_dir  # noqa: E402

# Dynamically load src.* from spatial_anomaly_scripts to avoid static-analysis false-positives.
import importlib.util as _ilu

def _load_spatial_module(name: str):
    """Load a module from spatial_anomaly_scripts/src by name."""
    spec = _ilu.spec_from_file_location(
        name, _SPATIAL_DIR / "src" / f"{name.split('.')[-1]}.py"
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

_backbones_onnx = _load_spatial_module("src.backbones_onnx")
_utils          = _load_spatial_module("src.utils")
_post_eval      = _load_spatial_module("src.post_eval")

DINOv3ONNXWrapper = _backbones_onnx.DINOv3ONNXWrapper
dists2map         = _utils.dists2map
mean_top1p        = _post_eval.mean_top1p

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Preprocessing dispatcher
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Stateful wrapper around Stage 1 preprocessing.
    Call build_reference() once, then apply() on each image.
    """

    def __init__(self, args):
        self.mode        = args.preprocess
        self.args        = args
        self.reference   = None   # averaged/template BGR float32
        self._built      = False

    def build_reference(self, good_images: list[np.ndarray]):
        """
        Pre-compute any reference that needs the full good-image set.
        Must be called before apply().
        """
        if self.mode == "average":
            print("[Preprocessor] Building averaged reference from good images...")
            self.reference = build_averaged_reference(good_images)
            print(f"  Reference shape: {self.reference.shape}")

        elif self.mode == "template":
            ref_path = self.args.template
            if not ref_path:
                raise ValueError("--mode template requires --template")
            ref = cv2.imread(ref_path)
            if ref is None:
                raise FileNotFoundError(f"Cannot load template: {ref_path}")
            self.reference = ref.astype(np.float32)
            print(f"[Preprocessor] Template loaded: {ref_path}")

        elif self.mode == "thresh":
            print(f"[Preprocessor] Thresholding — channel={self.args.channel}  "
                  f"min={self.args.thresh_min}  max={self.args.thresh_max}")

        else:
            raise ValueError(f"Unknown preprocess mode: {self.mode}")

        self._built = True

    def apply(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply preprocessing to a single BGR image.

        Returns:
            mask        : Binary mask (H, W) uint8, 255 = foreground
            preprocessed: Image with background zeroed (H, W, 3) BGR
        """
        if not self._built:
            raise RuntimeError("Call build_reference() before apply()")

        a = self.args
        if self.mode == "thresh":
            mask, preprocessed, _ = apply_threshold_mask(
                image_bgr,
                channel=a.channel,
                thresh_min=a.thresh_min,
                thresh_max=a.thresh_max,
                morph_open=a.morph_open,
                morph_close=a.morph_close,
            )
        elif self.mode in ("average", "template"):
            mask, preprocessed = mask_by_template(
                image_bgr, self.reference,
                diff_threshold=a.diff_threshold,
                morph_open=a.morph_open,
                morph_close=a.morph_close,
            )
        else:
            # No masking — pass through
            mask        = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
            preprocessed = image_bgr.copy()

        return mask, preprocessed


# ---------------------------------------------------------------------------
# Feature extraction helpers (ONNX)
# ---------------------------------------------------------------------------

def extract_features_onnx(
    model: "DINOv3ONNXWrapper",
    image_rgb: np.ndarray,
    mask: np.ndarray = None,
    mask_threshold: float = 10.0,
    apply_mask: bool = True,
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Prepare image, extract ONNX features, optionally apply BG mask.

    Args:
        model         : DINOv3ONNXWrapper instance
        image_rgb     : RGB image (H, W, 3)
        mask          : Binary foreground mask (H, W), 255=fg. If None, uses PCA mask.
        mask_threshold: PCA threshold for background detection (if mask is None)
        apply_mask    : Whether to mask features at all

    Returns:
        features_masked : (N_fg, C) masked features
        feat_mask       : (N_total,) boolean mask over patch grid
        grid_size       : (gh, gw) patch grid dimensions
    """
    image_tensor, grid_size = model.prepare_image(image_rgb)
    features = model.extract_features(image_tensor)  # (N, C)

    if not apply_mask:
        feat_mask = np.ones(features.shape[0], dtype=bool)
    elif mask is not None:
        # Resize binary mask to patch grid and flatten
        gh, gw = grid_size
        mask_resized = cv2.resize(mask, (gw, gh), interpolation=cv2.INTER_NEAREST)
        feat_mask = (mask_resized.flatten() > 0)
    else:
        # Fall back to PCA-based background detection
        feat_mask = model.compute_background_mask(
            features, grid_size, threshold=mask_threshold, masking_type=True
        )

    features_masked = np.ascontiguousarray(features[feat_mask]).astype(np.float32)
    return features_masked, feat_mask, grid_size


# ---------------------------------------------------------------------------
# FAISS index helper
# ---------------------------------------------------------------------------

def build_faiss_index(features: np.ndarray) -> faiss.Index:
    """Build a GPU-first (fallback CPU) FAISS L2 index."""
    features = np.ascontiguousarray(features.astype(np.float32))
    faiss.normalize_L2(features)

    try:
        res   = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features.shape[1])
        print("  FAISS: using GPU index")
    except Exception:
        index = faiss.IndexFlatL2(features.shape[1])
        print("  FAISS: using CPU index")

    index.add(features)
    return index


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_result_figure(
    image_rgb: np.ndarray,
    preprocessed_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    overlay_rgb: np.ndarray,
    score: float,
    output_path: Path,
    vmax: float = None,
):
    """Save a 4-panel result figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(preprocessed_rgb)
    axes[1].set_title("Preprocessed (mask applied)")
    axes[1].axis("off")

    vmax_val = vmax if vmax else anomaly_map.max()
    im = axes[2].imshow(anomaly_map, cmap="jet", vmin=0, vmax=vmax_val)
    axes[2].set_title("Anomaly Heatmap")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(overlay_rgb)
    axes[3].set_title(f"Overlay  (score={score:.4f})")
    axes[3].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    good_dir = Path(args.good_dir)
    test_dir = Path(args.test_dir)

    # ── Load images ──────────────────────────────────────────────────────────
    good_paths = sorted([p for p in good_dir.rglob("*") if p.suffix.lower() in EXTS])
    test_paths = sorted([p for p in test_dir.rglob("*") if p.suffix.lower() in EXTS])

    if not good_paths:
        print(f"[ERROR] No good images found in {good_dir}")
        sys.exit(1)
    if not test_paths:
        print(f"[ERROR] No test images found in {test_dir}")
        sys.exit(1)

    if args.shots:
        good_paths = good_paths[:args.shots]

    print(f"\n[Pipeline] Preprocess mode : {args.preprocess.upper()}")
    print(f"[Pipeline] Good images     : {len(good_paths)}")
    print(f"[Pipeline] Test images     : {len(test_paths)}")

    good_images_bgr = [cv2.imread(str(p)) for p in good_paths]
    good_images_bgr = [img for img in good_images_bgr if img is not None]

    # ── Build preprocessor ───────────────────────────────────────────────────
    preprocessor = Preprocessor(args)
    preprocessor.build_reference(good_images_bgr)

    # ── Load ONNX model ──────────────────────────────────────────────────────
    print(f"\n[Model] Loading ONNX: {args.onnx_path}")
    model = DINOv3ONNXWrapper(
        onnx_path=args.onnx_path,
        model_name=args.model_name,
        smaller_edge_size=args.resolution,
    )

    # ── Build memory bank ────────────────────────────────────────────────────
    print(f"\n[Memory Bank] Extracting features from {len(good_images_bgr)} good images...")
    t0 = time.perf_counter()

    all_features = []
    for i, (path, img_bgr) in enumerate(zip(good_paths, good_images_bgr)):
        mask, preprocessed_bgr = preprocessor.apply(img_bgr)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

        feats, _, _ = extract_features_onnx(
            model, preprocessed_rgb,
            mask=mask,
            mask_threshold=args.mask_threshold,
            apply_mask=(not args.no_mask),
        )
        if feats.shape[0] > 0:
            all_features.append(feats)
        print(f"  [{i+1}/{len(good_images_bgr)}] {path.name}  → {feats.shape[0]} features")

    all_features = np.concatenate(all_features, axis=0)
    build_time   = time.perf_counter() - t0
    print(f"\n  Memory bank: {all_features.shape}  (built in {build_time:.1f}s)")

    if args.save_bank:
        np.save(args.save_bank, all_features)
        print(f"  Saved to: {args.save_bank}")

    # ── Build FAISS index ────────────────────────────────────────────────────
    print("\n[FAISS] Building index...")
    index = build_faiss_index(all_features)
    print(f"  Index: {index.ntotal} vectors  dim={all_features.shape[1]}")

    # ── Run detection on test images ─────────────────────────────────────────
    print(f"\n[Detection] Running on {len(test_paths)} test images...\n")

    scores    = []
    latencies = []

    for i, test_path in enumerate(test_paths):
        img_bgr = cv2.imread(str(test_path))
        if img_bgr is None:
            print(f"  [SKIP] {test_path.name}")
            continue

        t1 = time.perf_counter()

        mask, preprocessed_bgr = preprocessor.apply(img_bgr)
        image_rgb       = cv2.cvtColor(img_bgr,          cv2.COLOR_BGR2RGB)
        preprocessed_rgb = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

        feats_test, feat_mask, grid_size = extract_features_onnx(
            model, preprocessed_rgb,
            mask=mask,
            mask_threshold=args.mask_threshold,
            apply_mask=(not args.no_mask),
        )

        # FAISS search
        feats_test_norm = np.ascontiguousarray(feats_test.astype(np.float32))
        faiss.normalize_L2(feats_test_norm)
        distances, _ = index.search(feats_test_norm, k=args.k)

        if args.k > 1:
            distances = distances.mean(axis=1)
        distances = distances / 2.0  # L2 → cosine

        # Map back to grid
        output_dists = np.zeros(feat_mask.shape[0], dtype=np.float32)
        output_dists[feat_mask] = distances.squeeze()

        anomaly_map = dists2map(output_dists.reshape(grid_size), image_rgb.shape, sigma=args.sigma)
        score       = mean_top1p(output_dists)

        # Overlay
        vmax_val  = args.vmax if args.vmax else anomaly_map.max()
        norm_map  = np.clip(anomaly_map / (vmax_val + 1e-8), 0, 1)
        heat_bgr  = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat_rgb  = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        overlay   = cv2.addWeighted(image_rgb, 0.6, heat_rgb, 0.4, 0)

        latency = (time.perf_counter() - t1) * 1000
        scores.append(score)
        latencies.append(latency)

        # Save result figure
        rel     = test_path.relative_to(test_dir)
        out_fig = (out_dir / rel).with_suffix(".png")
        save_result_figure(
            image_rgb, preprocessed_rgb, anomaly_map, overlay, score, out_fig, vmax=args.vmax
        )

        status = "ANOMALY" if score > args.threshold else "OK"
        print(f"  [{i+1:>3}/{len(test_paths)}] {test_path.name:<40} "
              f"score={score:.4f}  [{status}]  lat={latency:.1f}ms")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[Results Summary]")
    print(f"  Threshold (τ)      : {args.threshold}")
    print(f"  Total images       : {len(scores)}")
    if scores:
        anomalies = sum(1 for s in scores if s > args.threshold)
        print(f"  Anomalous images   : {anomalies}  ({anomalies/len(scores)*100:.0f}%)")
        print(f"  Score — mean       : {np.mean(scores):.4f}")
        print(f"  Score — max        : {np.max(scores):.4f}")
        print(f"  Score — min        : {np.min(scores):.4f}")
        print(f"  Latency — mean     : {np.mean(latencies):.1f}ms")
        print(f"  Latency — max      : {np.max(latencies):.1f}ms")
    print(f"  Output             : {out_dir.resolve()}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Stage 1 — Preprocessing + Anomaly Detection Pipeline Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    p.add_argument("--onnx_path", type=str, required=True,
                   help="Path to DINOv3 ONNX model file")
    p.add_argument("--good_dir",  type=str, required=True,
                   help="Folder of normal/good training images")
    p.add_argument("--test_dir",  type=str, required=True,
                   help="Folder of test images")

    # Preprocess
    p.add_argument("--preprocess", type=str, default="thresh",
                   choices=["thresh", "template", "average", "none"],
                   help="Preprocessing mode (default: thresh)")
    p.add_argument("--template",      type=str, help="Blank template image (for --preprocess template)")

    # Thresholding params (Phase 1.1)
    p.add_argument("--channel",    type=str, default="gray", choices=CHANNEL_OPTIONS)
    p.add_argument("--thresh_min", type=int, default=30)
    p.add_argument("--thresh_max", type=int, default=220)

    # BG subtraction params (Phase 1.2)
    p.add_argument("--diff_threshold", type=float, default=25.0)

    # Shared morph params
    p.add_argument("--morph_open",  type=int, default=3)
    p.add_argument("--morph_close", type=int, default=5)

    # Model
    p.add_argument("--model_name", type=str, default="dinov3_vits16")
    p.add_argument("--resolution", type=int, default=448)

    # Detection
    p.add_argument("--shots",     type=int,   default=None,
                   help="Limit number of good images (default: all)")
    p.add_argument("--k",         type=int,   default=1,
                   help="k-NN neighbors for scoring")
    p.add_argument("--sigma",     type=int,   default=4,
                   help="Gaussian smoothing sigma on heatmap")
    p.add_argument("--vmax",      type=float, default=None,
                   help="Fixed heatmap max (None = auto)")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Anomaly score threshold τ (default: 0.1)")
    p.add_argument("--no_mask",   action="store_true",
                   help="Disable feature masking during FAISS search")
    p.add_argument("--mask_threshold", type=float, default=10.0,
                   help="PCA threshold for feature masking")

    # Output
    p.add_argument("--output_dir", type=str, default="outputs/pipeline_test")
    p.add_argument("--save_bank",  type=str, default=None,
                   help="Save memory bank features as .npy")

    return p.parse_args()


if __name__ == "__main__":
    main()
