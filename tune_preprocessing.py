"""
Hyperparameter tuning for the image preprocessing pipeline.

Performs a grid search over preprocessing parameters and evaluates each
combination against ground truth OCR text (from ground_truth JSON files)
using Character Error Rate (CER).

The best parameter set is printed and optionally saved to best_params.json
for use in batch_processor.py and pipeline/ocr.py.

Tunable parameters:
  blur_kernel      : Gaussian blur kernel size {3, 5, 7}
  threshold_block  : Adaptive threshold block size {11, 15, 21, 31}
  threshold_c      : Adaptive threshold constant {2, 5, 8}
  deskew_tolerance : Max skew angle to auto-correct {0.5, 1.0, 2.0}
  use_otsu         : Global Otsu vs adaptive thresholding {True, False}

Usage:
    python tune_preprocessing.py --ground-truth ground_truth/ --image-folder images/
    python tune_preprocessing.py --ground-truth ground_truth/ --image-folder images/ --save
"""

import argparse
import json
import os
import sys
import time
from itertools import product

import cv2

from pipeline.ocr import _call_vision_api
from pipeline.preprocessing import preprocess_image


PARAM_GRID = {
    "blur_kernel":      [3, 5, 7],
    "threshold_block":  [11, 15, 21, 31],
    "threshold_c":      [2, 5, 8],
    "deskew_tolerance": [0.5, 1.0, 2.0],
    "use_otsu":         [False, True],
}

BEST_PARAMS_FILE = "best_params.json"


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

def _cer(reference: str, hypothesis: str) -> float:
    ref = list(reference.strip())
    hyp = list(hypothesis.strip())
    if not ref:
        return 0.0 if not hyp else 1.0
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref)][len(hyp)] / len(ref)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def _get_ocr_text_for_params(image_path: str, params: dict) -> str:
    """Preprocess image with given params, then call Vision API."""
    processed = preprocess_image(image_path, params=params)
    _, encoded = cv2.imencode(".png", processed)
    image_bytes = encoded.tobytes()
    try:
        return _call_vision_api(image_bytes)
    except Exception as e:
        return f"ERROR: {e}"


def tune(
    ground_truth_dir: str,
    image_folder: str,
    max_combinations: int = 50,
    save_best: bool = False,
) -> dict:
    """
    Grid search over preprocessing parameters.
    For each parameter combination, run OCR on ground truth images and
    compute average CER against the reference OCR text in each JSON.
    Returns the best parameter dict.
    """
    gt_files = [f for f in os.listdir(ground_truth_dir) if f.endswith(".json")]
    if not gt_files:
        print(f"No ground truth JSON files found in {ground_truth_dir}")
        return {}

    # Load ground truth OCR references
    gt_samples = []
    for gf in sorted(gt_files):
        with open(os.path.join(ground_truth_dir, gf), encoding="utf-8") as f:
            gt = json.load(f)
        file_name = gt.get("file_name", gf)
        ref_text = gt.get("ocr_text", "")
        if not ref_text:
            continue
        img_path = os.path.join(image_folder, file_name)
        if not os.path.exists(img_path):
            print(f"  Image not found: {img_path}, skipping.")
            continue
        gt_samples.append({"image_path": img_path, "ref_text": ref_text})

    if not gt_samples:
        print("No valid ground truth samples with both ocr_text and images found.")
        return {}

    # Enumerate parameter combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_combinations = list(product(*values))

    # Limit combinations if too many
    if len(all_combinations) > max_combinations:
        import random
        random.seed(0)
        all_combinations = random.sample(all_combinations, max_combinations)
        print(f"Sampling {max_combinations} combinations from {len(list(product(*values)))} total.")
    else:
        print(f"Testing {len(all_combinations)} parameter combinations.")

    print(f"Using {len(gt_samples)} ground truth samples.\n")

    results = []
    best_cer = float("inf")
    best_params = {}

    for combo_idx, combo in enumerate(all_combinations, 1):
        params = dict(zip(keys, combo))
        total_cer = 0.0

        for sample in gt_samples:
            ocr_text = _get_ocr_text_for_params(sample["image_path"], params)
            if ocr_text.startswith("ERROR"):
                total_cer += 1.0
            else:
                total_cer += _cer(sample["ref_text"], ocr_text)

        avg_cer = total_cer / len(gt_samples)
        results.append({"params": params, "avg_cer": round(avg_cer, 6)})

        if avg_cer < best_cer:
            best_cer = avg_cer
            best_params = params.copy()

        print(
            f"  [{combo_idx:>3}/{len(all_combinations)}] "
            f"blur={params['blur_kernel']} "
            f"block={params['threshold_block']} "
            f"c={params['threshold_c']} "
            f"deskew={params['deskew_tolerance']} "
            f"otsu={params['use_otsu']} "
            f"→ CER={avg_cer:.4f}"
            + (" ← BEST" if avg_cer == best_cer else "")
        )

    # Sort and display top 5
    results.sort(key=lambda x: x["avg_cer"])
    print(f"\n=== Top 5 Parameter Combinations ===")
    for i, r in enumerate(results[:5], 1):
        p = r["params"]
        print(
            f"  #{i} CER={r['avg_cer']:.6f} | "
            f"blur={p['blur_kernel']} block={p['threshold_block']} "
            f"c={p['threshold_c']} deskew={p['deskew_tolerance']} otsu={p['use_otsu']}"
        )

    print(f"\nBest params (CER={best_cer:.6f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    if save_best:
        with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print(f"\nBest parameters saved to {BEST_PARAMS_FILE}")
        print("Use them in batch_processor.py via --params-file or load directly in pipeline/ocr.py")

    return best_params


def load_best_params(path: str = BEST_PARAMS_FILE) -> dict:
    """Load previously saved best parameters."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for manuscript image preprocessing."
    )
    parser.add_argument("--ground-truth", required=True,
                        help="Directory containing ground truth JSON files with ocr_text fields")
    parser.add_argument("--image-folder", required=True,
                        help="Directory containing source images referenced in ground truth")
    parser.add_argument("--max-combinations", type=int, default=50,
                        help="Max parameter combinations to test (default: 50)")
    parser.add_argument("--save", action="store_true",
                        help=f"Save best parameters to {BEST_PARAMS_FILE}")
    args = parser.parse_args()

    if not os.path.isdir(args.ground_truth):
        print(f"Error: '{args.ground_truth}' is not a directory.")
        sys.exit(1)
    if not os.path.isdir(args.image_folder):
        print(f"Error: '{args.image_folder}' is not a directory.")
        sys.exit(1)

    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS not set. OCR calls will fail.")

    tune(
        ground_truth_dir=args.ground_truth,
        image_folder=args.image_folder,
        max_combinations=args.max_combinations,
        save_best=args.save,
    )


if __name__ == "__main__":
    main()
