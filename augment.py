from __future__ import annotations
"""
Data augmentation for Hindi manuscript images.

Generates multiple synthetic variants of each original scan to:
  - Stress-test the preprocessing and OCR pipeline
  - Create augmented training/evaluation data
  - Measure pipeline robustness across scan quality conditions

Augmentation techniques:
  1. Rotation        — simulate imperfect scanning alignment
  2. Brightness      — simulate varying lighting / faded ink
  3. Contrast        — aged paper vs fresh scan variation
  4. Gaussian noise  — scanner sensor noise
  5. Blur            — simulate out-of-focus scans

Usage:
    python augment.py <image_folder> <output_folder> [--variants 5]
    python augment.py images/ augmented/ --variants 8
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np

SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

AUG_CONFIGS = [
    {"name": "rot_2",    "rotation": 2,   "brightness": 0,  "contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "rot_neg2", "rotation": -2,  "brightness": 0,  "contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "rot_5",    "rotation": 5,   "brightness": 0,  "contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "rot_neg5", "rotation": -5,  "brightness": 0,  "contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "rot_10",   "rotation": 10,  "brightness": 0,  "contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "bright_p20", "rotation": 0, "brightness": 50, "contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "bright_n20", "rotation": 0, "brightness": -50,"contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "bright_p40", "rotation": 0, "brightness": 80, "contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "bright_n40", "rotation": 0, "brightness": -80,"contrast": 1.0, "noise_sigma": 0,  "blur_k": 0},
    {"name": "contrast_low",  "rotation": 0, "brightness": 0, "contrast": 0.7, "noise_sigma": 0, "blur_k": 0},
    {"name": "contrast_high", "rotation": 0, "brightness": 0, "contrast": 1.5, "noise_sigma": 0, "blur_k": 0},
    {"name": "noise_5",  "rotation": 0, "brightness": 0, "contrast": 1.0, "noise_sigma": 5,  "blur_k": 0},
    {"name": "noise_15", "rotation": 0, "brightness": 0, "contrast": 1.0, "noise_sigma": 15, "blur_k": 0},
    {"name": "noise_25", "rotation": 0, "brightness": 0, "contrast": 1.0, "noise_sigma": 25, "blur_k": 0},
    {"name": "blur_3",   "rotation": 0, "brightness": 0, "contrast": 1.0, "noise_sigma": 0,  "blur_k": 3},
    {"name": "blur_5",   "rotation": 0, "brightness": 0, "contrast": 1.0, "noise_sigma": 0,  "blur_k": 5},
    {"name": "combined_mild", "rotation": 2, "brightness": 20, "contrast": 0.9, "noise_sigma": 5, "blur_k": 3},
    {"name": "combined_hard", "rotation": -4, "brightness": -30, "contrast": 0.75, "noise_sigma": 15, "blur_k": 5},
]


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _adjust_brightness_contrast(img: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    if brightness == 0 and contrast == 1.0:
        return img
    result = img.astype(np.float32)
    result = result * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return img
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def augment_image(img: np.ndarray, config: dict) -> np.ndarray:
    """Apply one augmentation config to an image."""
    result = _rotate(img, config.get("rotation", 0))
    result = _adjust_brightness_contrast(
        result, config.get("brightness", 0), config.get("contrast", 1.0)
    )
    result = _add_gaussian_noise(result, config.get("noise_sigma", 0))
    result = _blur(result, config.get("blur_k", 0))
    return result


# ---------------------------------------------------------------------------
# Batch augmentation
# ---------------------------------------------------------------------------

def augment_folder(
    input_folder: str,
    output_folder: str,
    max_variants: int = 5,
    seed: int = 42,
) -> int:
    """
    Generate up to max_variants augmented copies of each image in input_folder.
    Saves results to output_folder. Returns total images generated.
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_folder, exist_ok=True)

    images = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ]
    if not images:
        print(f"No supported images found in {input_folder}")
        return 0

    configs_to_use = AUG_CONFIGS[:max_variants]
    total_generated = 0

    for fname in sorted(images):
        src = os.path.join(input_folder, fname)
        img = cv2.imread(src)
        if img is None:
            print(f"  Cannot read {fname}, skipping.")
            continue

        base, ext = os.path.splitext(fname)

        for cfg in configs_to_use:
            aug = augment_image(img, cfg)
            out_name = f"{base}_aug_{cfg['name']}{ext}"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, aug)
            total_generated += 1

        print(f"  {fname} → {len(configs_to_use)} variants")

    print(f"\nGenerated {total_generated} augmented images in {output_folder}")
    return total_generated


def augment_single(
    image_path: str,
    output_folder: str,
    max_variants: int = 5,
) -> list[str]:
    """Augment a single image. Returns list of output paths."""
    os.makedirs(output_folder, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    base = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    configs_to_use = AUG_CONFIGS[:max_variants]
    paths = []

    for cfg in configs_to_use:
        aug = augment_image(img, cfg)
        out_path = os.path.join(output_folder, f"{base}_aug_{cfg['name']}{ext}")
        cv2.imwrite(out_path, aug)
        paths.append(out_path)

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented variants of manuscript images for robustness testing."
    )
    parser.add_argument("image_folder", help="Folder containing original images")
    parser.add_argument("output_folder", help="Folder to save augmented images")
    parser.add_argument("--variants", type=int, default=5,
                        help=f"Augmentation variants per image (1-{len(AUG_CONFIGS)}, default 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not os.path.isdir(args.image_folder):
        print(f"Error: '{args.image_folder}' is not a directory.")
        sys.exit(1)

    n = min(args.variants, len(AUG_CONFIGS))
    print(f"Augmenting images from {args.image_folder}")
    print(f"Variants per image: {n}")
    print(f"Output: {args.output_folder}\n")

    augment_folder(args.image_folder, args.output_folder, max_variants=n, seed=args.seed)


if __name__ == "__main__":
    main()
