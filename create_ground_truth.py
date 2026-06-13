"""
Ground truth annotation helper for Hindi manuscript pipeline.

Instead of annotating from scratch, this script:
  1. Runs the OCR pipeline on your image(s)
  2. Extracts what it can automatically
  3. Creates a pre-filled JSON file with the pipeline's best guess
  4. You open the JSON, CORRECT the mistakes, and save

This is much faster than manual annotation — you're correcting, not typing.

Usage:
    # Single image → creates ground_truth/image00015.json
    python create_ground_truth.py image00015.jpg

    # Entire folder → creates one JSON per image
    python create_ground_truth.py images/ --batch

    # With output folder
    python create_ground_truth.py images/ --batch --output-dir ground_truth/

    # Without calling Vision API (if no credentials yet)
    python create_ground_truth.py image00015.jpg --no-ocr
"""

import argparse
import json
import os
import sys

from pipeline.nlp import run_nlp_pipeline, clean_text
from pipeline.ocr import detect_text

SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
OUTPUT_DIR = "ground_truth"

FIELDS_TO_KEEP = [
    "Given Name",
    "Relation",
    "Gender",
    "Caste",
    "Subcaste",
    "From Which Place",
    "Date of Ritual (Gregorian)",
]


def _generate_for_image(
    image_path: str,
    output_dir: str,
    use_ocr: bool = True,
    overwrite: bool = False,
) -> str:
    """
    Generate a pre-filled ground truth JSON for one image.
    Returns path to the created JSON file.
    """
    file_name = os.path.basename(image_path)
    base = os.path.splitext(file_name)[0]
    out_path = os.path.join(output_dir, f"{base}.json")

    if os.path.exists(out_path) and not overwrite:
        print(f"  SKIP {file_name} → {out_path} already exists (use --overwrite to replace)")
        return out_path

    ocr_text = ""
    records = []

    if use_ocr:
        print(f"  Running OCR on {file_name}...", end=" ", flush=True)
        try:
            ocr_text = detect_text(image_path, use_preprocessing=True, use_cache=True)
            if not ocr_text.strip():
                print("no text detected.")
            else:
                print(f"{len(ocr_text)} chars.")
                records = run_nlp_pipeline(ocr_text, file_name=file_name)
        except Exception as e:
            print(f"\n  WARNING: OCR/NLP failed — {e}")
            print(f"  Creating empty template. Fill in manually.")
    else:
        print(f"  Creating empty template for {file_name} (--no-ocr mode)")

    # Build pre-filled record list — only keep fields that need human verification
    clean_records = []
    for rec in records:
        entry = {}
        for field in FIELDS_TO_KEEP:
            entry[field] = rec.get(field, "")
        # Flag if low confidence so annotator pays more attention
        if rec.get("flagged") or rec.get("confidence", 1.0) < 0.6:
            entry["_NEEDS_REVIEW"] = True
        clean_records.append(entry)

    # If no records extracted, add one empty template record
    if not clean_records:
        clean_records = [{field: "" for field in FIELDS_TO_KEEP}]

    # Instructions comment (stored as first record with special key, removed before evaluation)
    output = {
        "_instructions": (
            "STEP 1: Replace 'ocr_text' with the exact text visible in the image (copy what you read). "
            "STEP 2: In 'records', correct any wrong field values. "
            "STEP 3: Add missing people as new record objects. "
            "STEP 4: Delete the '_instructions' and '_NEEDS_REVIEW' keys before saving final version. "
            "Leave a field as empty string \"\" if info is absent from image — do not guess."
        ),
        "file_name": file_name,
        "ocr_text": ocr_text,
        "records": clean_records,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  Created → {out_path}  ({len(clean_records)} record(s) pre-filled)")
    return out_path


def _clean_for_evaluation(json_path: str) -> None:
    """Remove helper keys (_instructions, _NEEDS_REVIEW) before evaluation."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_instructions", None)
    for rec in data.get("records", []):
        rec.pop("_NEEDS_REVIEW", None)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Cleaned {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-filled ground truth JSON files from manuscript images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to a single image file OR a folder (use with --batch)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all images in the input folder"
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"Where to save JSON files (default: {OUTPUT_DIR}/)"
    )
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="Skip Vision API — create empty templates only (useful if no credentials yet)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing JSON files (default: skip existing)"
    )
    parser.add_argument(
        "--clean", metavar="JSON_FILE",
        help="Remove _instructions/_NEEDS_REVIEW keys from a finished annotation file"
    )
    args = parser.parse_args()

    # Clean mode
    if args.clean:
        if not os.path.exists(args.clean):
            print(f"File not found: {args.clean}")
            sys.exit(1)
        _clean_for_evaluation(args.clean)
        return

    use_ocr = not args.no_ocr

    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: '{args.input}' is not a directory. Use --batch with a folder path.")
            sys.exit(1)
        images = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in SUPPORTED
        ])
        if not images:
            print(f"No supported images found in {args.input}")
            sys.exit(1)
        print(f"Found {len(images)} images. Generating ground truth templates...\n")
        for img in images:
            _generate_for_image(img, args.output_dir, use_ocr=use_ocr, overwrite=args.overwrite)
        print(f"\nDone. Open the JSON files in {args.output_dir}/ and correct any mistakes.")
    else:
        if not os.path.isfile(args.input):
            print(f"Error: '{args.input}' is not a file. For folders, add --batch flag.")
            sys.exit(1)
        _generate_for_image(args.input, args.output_dir, use_ocr=use_ocr, overwrite=args.overwrite)
        print(f"\nNext: open the JSON file and correct any wrong fields.")

    print()
    print("After correcting all JSON files, run:")
    print(f"  python evaluate.py accuracy --ground-truth {args.output_dir}/ --image-folder <images-folder>")


if __name__ == "__main__":
    main()
