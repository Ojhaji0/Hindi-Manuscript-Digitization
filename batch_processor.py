"""
Batch processing engine for Hindi manuscript digitization.

Processes an entire folder of manuscript images end-to-end:
  OCR (with preprocessing + caching) → NLP extraction → SQLite database

Features:
  - Resume capability: skips images already in the processing log
  - Parallel Vision API calls (ThreadPoolExecutor, I/O-bound)
  - Per-image timing for benchmarking
  - Processing log CSV for audit trail

Usage:
    python batch_processor.py <image_folder> [--workers 4] [--no-cache] [--no-preprocess]
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    from google.cloud import language_v1
    _NLP_AVAILABLE = True
except ImportError:
    _NLP_AVAILABLE = False

try:
    from hindu_calendar import HinduCalendar as _HC
    _HC_INSTANCE = _HC(method="hindi", city="auto")
except Exception:
    _HC_INSTANCE = None

from pipeline.ocr import detect_text
from pipeline.nlp import run_nlp_pipeline
from pipeline.database import write_records, build_indexes, init_db

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
LOG_FILE = "processing_log.csv"
LOG_FIELDS = [
    "file_name", "timestamp", "status", "records_extracted",
    "processing_time_sec", "error_message",
]


# ---------------------------------------------------------------------------
# Processing log helpers
# ---------------------------------------------------------------------------

def _load_log(log_path: str) -> set[str]:
    """Return set of already-processed filenames from the log."""
    done = set()
    if not os.path.exists(log_path):
        return done
    with open(log_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "success":
                done.add(row["file_name"])
    return done


def _append_log(log_path: str, row: dict) -> None:
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Single-image processor (called inside thread pool)
# ---------------------------------------------------------------------------

def _process_one(
    image_path: str,
    nlp_client,
    hc_instance,
    preprocess_params: dict,
    use_preprocessing: bool,
    use_cache: bool,
) -> dict:
    start = time.perf_counter()
    file_name = os.path.basename(image_path)
    try:
        # Stage 1 – OCR
        raw_text = detect_text(
            image_path,
            use_preprocessing=use_preprocessing,
            preprocess_params=preprocess_params,
            use_cache=use_cache,
        )
        if not raw_text.strip():
            return {
                "file_name": file_name,
                "timestamp": datetime.now().isoformat(),
                "status": "empty",
                "records_extracted": 0,
                "processing_time_sec": round(time.perf_counter() - start, 3),
                "error_message": "No text detected",
            }

        # Stages 2–5 – NLP pipeline
        records = run_nlp_pipeline(
            raw_text,
            file_name=file_name,
            nlp_client=nlp_client,
            hc_instance=hc_instance,
        )

        # Write to database
        write_records(records, file_name=file_name)

        elapsed = round(time.perf_counter() - start, 3)
        return {
            "file_name": file_name,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "records_extracted": len(records),
            "processing_time_sec": elapsed,
            "error_message": "",
        }

    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 3)
        return {
            "file_name": file_name,
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "records_extracted": 0,
            "processing_time_sec": elapsed,
            "error_message": str(exc)[:300],
        }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    image_folder: str,
    workers: int = 4,
    use_preprocessing: bool = True,
    preprocess_params: dict = None,
    use_cache: bool = True,
    log_path: str = LOG_FILE,
    credentials_path: str = None,
) -> dict:
    """
    Process all supported images in image_folder.
    Returns summary stats dict.
    """
    if credentials_path and os.path.exists(credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    init_db()

    # Collect images
    all_images = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not all_images:
        print(f"No supported images found in {image_folder}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

    # Skip already processed
    done_set = _load_log(log_path)
    to_process = [p for p in all_images if os.path.basename(p) not in done_set]
    skipped = len(all_images) - len(to_process)

    print(f"Found {len(all_images)} images. "
          f"Skipping {skipped} already processed. "
          f"Processing {len(to_process)}.")

    if not to_process:
        return {"total": len(all_images), "success": 0, "failed": 0, "skipped": skipped}

    # Initialize API clients
    nlp_client = None
    if _NLP_AVAILABLE and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            nlp_client = language_v1.LanguageServiceClient()
        except Exception:
            pass

    hc_instance = _HC_INSTANCE

    # Parallel processing
    stats = {"total": len(all_images), "success": 0, "failed": 0, "skipped": skipped}
    batch_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                path, nlp_client, hc_instance,
                preprocess_params or {}, use_preprocessing, use_cache,
            ): path
            for path in to_process
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            _append_log(log_path, result)

            status_icon = "✓" if result["status"] == "success" else "✗"
            print(
                f"  [{i}/{len(to_process)}] {status_icon} {result['file_name']} "
                f"— {result['records_extracted']} records "
                f"({result['processing_time_sec']}s)"
            )

            if result["status"] == "success":
                stats["success"] += 1
            else:
                stats["failed"] += 1
                if result["error_message"]:
                    print(f"    ERROR: {result['error_message']}")

    # Build database indexes after bulk insert
    print("\nBuilding database indexes...")
    build_indexes()

    total_elapsed = round(time.perf_counter() - batch_start, 2)
    avg_time = round(total_elapsed / max(len(to_process), 1), 3)

    print(f"\n{'='*50}")
    print(f"Batch complete in {total_elapsed}s")
    print(f"  Success : {stats['success']}")
    print(f"  Failed  : {stats['failed']}")
    print(f"  Skipped : {stats['skipped']}")
    print(f"  Avg time/image: {avg_time}s")
    print(f"  Log: {log_path}")
    print(f"  DB : manuscripts.db")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch process Hindi manuscript images into a searchable database."
    )
    parser.add_argument("image_folder", help="Folder containing manuscript images")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel Vision API workers (default: 4)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable Vision API response cache")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip image preprocessing (use raw image for OCR)")
    parser.add_argument("--credentials", default=None,
                        help="Path to Google Cloud service account JSON")
    parser.add_argument("--log", default=LOG_FILE,
                        help=f"Path to processing log CSV (default: {LOG_FILE})")
    args = parser.parse_args()

    if not os.path.isdir(args.image_folder):
        print(f"Error: '{args.image_folder}' is not a directory.")
        sys.exit(1)

    run_batch(
        image_folder=args.image_folder,
        workers=args.workers,
        use_preprocessing=not args.no_preprocess,
        use_cache=not args.no_cache,
        log_path=args.log,
        credentials_path=args.credentials,
    )


if __name__ == "__main__":
    main()
