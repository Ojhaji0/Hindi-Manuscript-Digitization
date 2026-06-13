"""
Custom image preprocessing pipeline for Hindi manuscript scans.
Applied before OCR to improve text clarity and recognition accuracy.

Pipeline stages:
  1. Grayscale conversion
  2. Denoising (Gaussian blur)
  3. Adaptive binarization (Otsu / adaptive Gaussian threshold)
  4. Deskewing (Hough line transform)
  5. Border crop
"""

import cv2
import numpy as np


DEFAULT_PARAMS = {
    "blur_kernel": 3,          # Gaussian blur kernel size (must be odd)
    "threshold_block": 15,     # Adaptive threshold block size (must be odd)
    "threshold_c": 5,          # Adaptive threshold constant subtracted from mean
    "deskew_tolerance": 1.0,   # Max angle (degrees) to correct without user confirmation
    "border_margin": 10,       # Pixels to crop from each edge after binarization
    "use_otsu": False,         # True = global Otsu, False = adaptive Gaussian (better for aged paper)
}


def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(gray, (k, k), 0)


def binarize(blurred: np.ndarray, use_otsu: bool = False,
             block_size: int = 15, c: int = 5) -> np.ndarray:
    if use_otsu:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        bs = block_size if block_size % 2 == 1 else block_size + 1
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs, c
        )
    return binary


def _compute_skew_angle(binary: np.ndarray) -> float:
    """Estimate skew angle using Hough line transform."""
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = np.degrees(theta) - 90
        if abs(angle_deg) < 45:
            angles.append(angle_deg)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(binary: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
    angle = _compute_skew_angle(binary)
    if abs(angle) < tolerance:
        return binary
    h, w = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def crop_borders(binary: np.ndarray, margin: int = 10) -> np.ndarray:
    h, w = binary.shape
    m = max(0, margin)
    return binary[m: h - m, m: w - m]


def preprocess_image(image_path: str, params: dict = None) -> np.ndarray:
    """
    Run the full 5-step preprocessing pipeline on a manuscript image.
    Returns a binary (black text on white background) numpy array.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    img = load_image(image_path)
    gray = to_grayscale(img)
    blurred = denoise(gray, kernel_size=p["blur_kernel"])
    binary = binarize(blurred, use_otsu=p["use_otsu"],
                      block_size=p["threshold_block"], c=p["threshold_c"])
    deskewed = deskew(binary, tolerance=p["deskew_tolerance"])
    cropped = crop_borders(deskewed, margin=p["border_margin"])
    return cropped


def save_preprocessed(image_path: str, output_path: str, params: dict = None) -> str:
    """Preprocess and save result to output_path. Returns output_path."""
    result = preprocess_image(image_path, params)
    cv2.imwrite(output_path, result)
    return output_path
