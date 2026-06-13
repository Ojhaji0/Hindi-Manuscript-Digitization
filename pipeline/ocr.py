"""
OCR module: Google Cloud Vision API wrapper with hash-based response caching.

Caching avoids redundant API calls when re-processing images (debugging,
re-runs). Cache is stored in a local SQLite file (cache.db).
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
import time

import cv2
import numpy as np

try:
    from google.cloud import vision
    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

from .preprocessing import preprocess_image

CACHE_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache.db")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _init_cache(db_path: str = CACHE_DB) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS ocr_cache (
               image_hash TEXT PRIMARY KEY,
               ocr_text   TEXT NOT NULL,
               cached_at  REAL NOT NULL
           )"""
    )
    conn.commit()
    conn.close()


def _hash_file(image_path: str) -> str:
    h = hashlib.md5()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_get(image_hash: str, db_path: str = CACHE_DB) -> str | None:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT ocr_text FROM ocr_cache WHERE image_hash = ?", (image_hash,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _cache_set(image_hash: str, text: str, db_path: str = CACHE_DB) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO ocr_cache (image_hash, ocr_text, cached_at) VALUES (?, ?, ?)",
        (image_hash, text, time.time()),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# OCR core
# ---------------------------------------------------------------------------

def _call_vision_api(image_bytes: bytes) -> str:
    if not _VISION_AVAILABLE:
        raise RuntimeError("google-cloud-vision is not installed.")
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    context = vision.ImageContext(language_hints=["hi"])
    response = client.document_text_detection(image=image, image_context=context)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    if response.text_annotations:
        return response.text_annotations[0].description
    return ""


def detect_text(
    image_path: str,
    use_preprocessing: bool = True,
    preprocess_params: dict = None,
    use_cache: bool = True,
    cache_db: str = CACHE_DB,
) -> str:
    """
    Extract text from a manuscript image using Google Cloud Vision.

    Steps:
      1. Hash original file for cache lookup.
      2. If cache hit, return stored text immediately (avoids API call).
      3. Optionally preprocess the image (grayscale → denoise → binarize → deskew → crop).
      4. Call Vision API on preprocessed bytes.
      5. Store result in cache.

    Returns raw OCR text string.
    """
    _init_cache(cache_db)

    image_hash = _hash_file(image_path)

    if use_cache:
        cached = _cache_get(image_hash, cache_db)
        if cached is not None:
            return cached

    if use_preprocessing:
        processed_array = preprocess_image(image_path, params=preprocess_params)
        _, encoded = cv2.imencode(".png", processed_array)
        image_bytes = encoded.tobytes()
    else:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

    text = _call_vision_api(image_bytes)

    if use_cache:
        _cache_set(image_hash, text, cache_db)

    return text


def detect_text_batch(
    image_paths: list[str],
    use_preprocessing: bool = True,
    preprocess_params: dict = None,
    use_cache: bool = True,
    cache_db: str = CACHE_DB,
    max_workers: int = 4,
) -> dict[str, str]:
    """
    Process multiple images in parallel using a thread pool.
    Returns dict mapping image_path -> ocr_text.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}

    def _process(path):
        return path, detect_text(
            path,
            use_preprocessing=use_preprocessing,
            preprocess_params=preprocess_params,
            use_cache=use_cache,
            cache_db=cache_db,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process, p): p for p in image_paths}
        for future in as_completed(futures):
            path, text = future.result()
            results[path] = text

    return results
