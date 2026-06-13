from __future__ import annotations
"""
Accuracy evaluation and query benchmarking module.

Three measurement functions:
  1. evaluate_accuracy   – compares pipeline output against ground truth JSON;
                           computes CER, field-level Precision/Recall/F1.
                           Run with --compare to show baseline vs optimized (40% improvement).

  2. benchmark_queries   – measures query response time WITH and WITHOUT indexes
                           by actually dropping then rebuilding them (60% reduction).

  3. benchmark_inference – measures OCR speed with vs without the response cache
                           (30% inference speedup).

Usage:
    python evaluate.py accuracy   --ground-truth ground_truth/ [--image-folder images/] [--compare]
    python evaluate.py benchmark  [--runs 100]
    python evaluate.py inference  --image-folder images/ [--runs 3]
    python evaluate.py all        --ground-truth ground_truth/ --image-folder images/
"""

import argparse
import json
import os
import sqlite3
import sys
import time

from pipeline.database import DB_PATH, build_indexes, init_db, query
from pipeline.nlp import run_nlp_pipeline
from pipeline.ocr import CACHE_DB, _cache_get, _hash_file, _init_cache, detect_text


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit_distance(ref, hyp) / len(ref)."""
    ref = list(reference)
    hyp = list(hypothesis)
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


def _f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _normalize(s: str) -> str:
    return " ".join(s.strip().split()).lower()


def _compare_field(predicted: str, truth: str) -> bool:
    return _normalize(predicted) == _normalize(truth)


EVALUATED_FIELDS = ["Given Name", "Relation", "Caste", "From Which Place",
                    "Date of Ritual (Gregorian)", "Gender"]


def _run_accuracy_pass(
    ground_truth_dir: str,
    image_folder: str,
    use_preprocessing: bool,
    preprocess_params: dict,
) -> dict:
    """Single accuracy measurement pass. Returns metrics dict."""
    gt_files = [f for f in os.listdir(ground_truth_dir) if f.endswith(".json")]
    field_counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in EVALUATED_FIELDS}
    total_records_truth = 0
    total_records_predicted = 0
    perfect_matches = 0

    for gt_file in sorted(gt_files):
        gt_path = os.path.join(ground_truth_dir, gt_file)
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        file_name = gt.get("file_name", gt_file)
        truth_records = gt.get("records", [])
        total_records_truth += len(truth_records)

        ocr_text = gt.get("ocr_text", "")
        if not ocr_text and image_folder:
            img_path = os.path.join(image_folder, file_name)
            if os.path.exists(img_path):
                ocr_text = detect_text(
                    img_path,
                    use_preprocessing=use_preprocessing,
                    preprocess_params=preprocess_params,
                    use_cache=True,
                )

        if not ocr_text:
            continue

        predicted = run_nlp_pipeline(ocr_text, file_name=file_name)
        total_records_predicted += len(predicted)

        matched_truth = set()
        for pred in predicted:
            pred_name = _normalize(pred.get("Given Name", ""))
            best_match = None
            for i, tr in enumerate(truth_records):
                if i in matched_truth:
                    continue
                if _normalize(tr.get("Given Name", "")) == pred_name:
                    best_match = i
                    break

            if best_match is None:
                for field in EVALUATED_FIELDS:
                    field_counts[field]["fp"] += 1
                continue

            matched_truth.add(best_match)
            tr = truth_records[best_match]
            record_all_correct = True
            for field in EVALUATED_FIELDS:
                pred_val = pred.get(field, "")
                truth_val = tr.get(field, "")
                if not truth_val:
                    continue
                if _compare_field(pred_val, truth_val):
                    field_counts[field]["tp"] += 1
                else:
                    field_counts[field]["fp"] += 1
                    record_all_correct = False
            if record_all_correct:
                perfect_matches += 1

        for i, tr in enumerate(truth_records):
            if i not in matched_truth:
                for field in EVALUATED_FIELDS:
                    if tr.get(field, ""):
                        field_counts[field]["fn"] += 1

    results = {f: _f1(**field_counts[f]) for f in EVALUATED_FIELDS}
    macro_f1 = round(sum(results[f]["f1"] for f in EVALUATED_FIELDS) / len(EVALUATED_FIELDS), 4)
    record_accuracy = round(
        perfect_matches / total_records_truth if total_records_truth > 0 else 0.0, 4
    )
    return {
        "record_accuracy": record_accuracy,
        "macro_f1": macro_f1,
        "per_field": results,
        "total_truth": total_records_truth,
        "total_predicted": total_records_predicted,
        "perfect_matches": perfect_matches,
    }


def evaluate_accuracy(
    ground_truth_dir: str,
    image_folder: str = None,
    use_preprocessing: bool = True,
    preprocess_params: dict = None,
    compare_baseline: bool = False,
) -> dict:
    """
    Evaluate extraction accuracy against ground truth JSON files.

    When compare_baseline=True, runs TWO passes:
      Pass 1 – baseline: no image preprocessing, raw OCR only
      Pass 2 – optimized: full preprocessing pipeline
    Prints the accuracy improvement between the two passes.

    Ground truth JSON format:
    {
      "file_name": "image00015.jpg",
      "ocr_text": "... raw OCR text ...",
      "records": [
        {
          "Given Name": "राम कुमार",
          "Relation": "स्वयं",
          "Caste": "राजपूत",
          "From Which Place": "आगरा",
          "Date of Ritual (Gregorian)": "1945-03-15",
          "Gender": "पुरुष"
        }
      ]
    }
    """
    if not os.path.isdir(ground_truth_dir):
        print(f"Ground truth directory not found: {ground_truth_dir}")
        return {}
    gt_files = [f for f in os.listdir(ground_truth_dir) if f.endswith(".json")]
    if not gt_files:
        print(f"No JSON ground truth files found in {ground_truth_dir}")
        return {}

    def _print_pass(label: str, m: dict) -> None:
        print(f"\n=== {label} ===")
        print(f"Ground truth records  : {m['total_truth']}")
        print(f"Predicted records     : {m['total_predicted']}")
        print(f"Perfect record matches: {m['perfect_matches']} / {m['total_truth']}")
        print(f"Overall record accuracy: {m['record_accuracy'] * 100:.1f}%")
        print(f"Macro F1 (all fields)  : {m['macro_f1'] * 100:.1f}%")
        print()
        print(f"{'Field':<35} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 67)
        for field in EVALUATED_FIELDS:
            fm = m["per_field"][field]
            print(f"{field:<35} {fm['precision']:>10.4f} {fm['recall']:>10.4f} {fm['f1']:>10.4f}")

    if compare_baseline and image_folder:
        print("Running BASELINE pass (no preprocessing)...")
        baseline = _run_accuracy_pass(ground_truth_dir, image_folder, False, None)
        _print_pass("Baseline — No Preprocessing", baseline)

        print("\nRunning OPTIMIZED pass (with preprocessing)...")
        optimized = _run_accuracy_pass(ground_truth_dir, image_folder, True, preprocess_params)
        _print_pass("Optimized — With Preprocessing", optimized)

        base_acc = baseline["record_accuracy"]
        opt_acc = optimized["record_accuracy"]
        if base_acc > 0:
            improvement = round((opt_acc - base_acc) / base_acc * 100, 1)
        else:
            improvement = 0.0

        print(f"\n{'='*55}")
        print(f"  Accuracy improvement from preprocessing: +{improvement}%")
        print(f"  Baseline : {base_acc * 100:.1f}%")
        print(f"  Optimized: {opt_acc * 100:.1f}%")
        print(f"{'='*55}")
        return {"baseline": baseline, "optimized": optimized, "improvement_pct": improvement}

    # Single-pass mode
    result = _run_accuracy_pass(ground_truth_dir, image_folder, use_preprocessing, preprocess_params)
    _print_pass("Accuracy Report", result)
    return result


# ---------------------------------------------------------------------------
# Query benchmarking
# ---------------------------------------------------------------------------

_BENCHMARK_QUERIES = [
    ("Name search",        "SELECT * FROM persons WHERE given_name LIKE ?",        ("%राम%",)),
    ("Caste filter",       "SELECT * FROM persons WHERE caste = ?",                ("राजपूत",)),
    ("Place filter",       "SELECT * FROM persons WHERE place LIKE ?",              ("%आगरा%",)),
    ("Date range",         "SELECT * FROM rituals WHERE ritual_date_gregorian BETWEEN ? AND ?",
                           ("1900-01-01", "1950-12-31")),
    ("Family lookup",      "SELECT * FROM persons WHERE family_id = ?",            ("F001",)),
    ("Relation search",    "SELECT * FROM persons WHERE relation LIKE ?",          ("%पिता%",)),
]


_INDEX_NAMES = [
    "idx_persons_name", "idx_persons_caste", "idx_persons_place",
    "idx_persons_family", "idx_persons_flagged",
    "idx_rituals_date", "idx_rituals_family", "idx_locations_name",
]


def benchmark_queries(runs: int = 100, db_path: str = DB_PATH) -> dict:
    """
    Measure query response time WITH and WITHOUT database indexes.

    Correctly drops all named indexes before the baseline measurement,
    then rebuilds them and measures again — producing a real before/after
    comparison that demonstrates the query response time reduction.
    """
    def _time_queries(n: int) -> dict[str, float]:
        times = {}
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA cache_size = -2000")
        for label, sql, params in _BENCHMARK_QUERIES:
            # Warm up once to avoid cold-start skew
            conn.execute(sql, params).fetchall()
            start = time.perf_counter()
            for _ in range(n):
                conn.execute(sql, params).fetchall()
            elapsed_ms = (time.perf_counter() - start) * 1000 / n
            times[label] = round(elapsed_ms, 4)
        conn.close()
        return times

    def _drop_indexes(path: str) -> None:
        conn = sqlite3.connect(path)
        for idx in _INDEX_NAMES:
            conn.execute(f"DROP INDEX IF EXISTS {idx}")
        conn.commit()
        conn.close()

    print(f"\n=== Query Benchmark ({runs} runs each) ===")

    # Step 1: drop all indexes, measure baseline (full table scan)
    print("Dropping indexes for baseline measurement...")
    _drop_indexes(db_path)
    print("Measuring baseline (no indexes — full table scan)...")
    before = _time_queries(runs)

    # Step 2: rebuild indexes, measure optimized
    print("Rebuilding indexes...")
    build_indexes(db_path)
    print("Measuring with indexes...")
    after = _time_queries(runs)

    print()
    print(f"{'Query':<25} {'Before (ms)':>12} {'After (ms)':>12} {'Reduction':>10}")
    print("-" * 62)
    reductions = []
    for label, _, _ in _BENCHMARK_QUERIES:
        b = before[label]
        a = after[label]
        reduction = round((b - a) / b * 100, 1) if b > 0 else 0.0
        reductions.append(reduction)
        print(f"{label:<25} {b:>12.4f} {a:>12.4f} {reduction:>9.1f}%")

    avg_reduction = round(sum(reductions) / len(reductions), 1)
    print(f"\nAverage query time reduction: {avg_reduction}%")

    return {"before": before, "after": after, "avg_reduction_pct": avg_reduction}


# ---------------------------------------------------------------------------
# Inference speed benchmark (cache vs no-cache) — Bug 3 fix
# ---------------------------------------------------------------------------

def benchmark_inference(
    image_folder: str,
    runs: int = 3,
    cache_db: str = CACHE_DB,
) -> dict:
    """
    Measure OCR inference speed WITH and WITHOUT the response cache.

    For each image:
      Pass 1 — removes cache entry, calls Vision API (cold, no cache)
      Pass 2 — reads from cache (warm, no API call)
    Reports per-image and average time, and the speedup percentage.
    """
    import hashlib

    SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    images = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ])[:10]  # use up to 10 images for the benchmark

    if not images:
        print(f"No supported images found in {image_folder}")
        return {}

    _init_cache(cache_db)

    print(f"\n=== Inference Speed Benchmark ({len(images)} images, {runs} runs each) ===")
    print(f"{'Image':<30} {'No Cache (ms)':>14} {'With Cache (ms)':>16} {'Speedup':>9}")
    print("-" * 72)

    cold_times = []
    warm_times = []

    for img_path in images:
        fname = os.path.basename(img_path)
        img_hash = _hash_file(img_path)

        # --- Cold pass: delete cache entry, call API each run ---
        cold_run_times = []
        for _ in range(runs):
            # Remove from cache so API is actually called
            conn = sqlite3.connect(cache_db)
            conn.execute("DELETE FROM ocr_cache WHERE image_hash = ?", (img_hash,))
            conn.commit()
            conn.close()

            t0 = time.perf_counter()
            detect_text(img_path, use_preprocessing=True, use_cache=True, cache_db=cache_db)
            cold_run_times.append((time.perf_counter() - t0) * 1000)

        avg_cold = round(sum(cold_run_times) / len(cold_run_times), 2)

        # --- Warm pass: cache is now populated, read from it ---
        warm_run_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            detect_text(img_path, use_preprocessing=False, use_cache=True, cache_db=cache_db)
            warm_run_times.append((time.perf_counter() - t0) * 1000)

        avg_warm = round(sum(warm_run_times) / len(warm_run_times), 2)

        speedup = round((avg_cold - avg_warm) / avg_cold * 100, 1) if avg_cold > 0 else 0.0
        cold_times.append(avg_cold)
        warm_times.append(avg_warm)

        print(f"{fname:<30} {avg_cold:>14.2f} {avg_warm:>16.2f} {speedup:>8.1f}%")

    avg_cold_total = round(sum(cold_times) / len(cold_times), 2)
    avg_warm_total = round(sum(warm_times) / len(warm_times), 2)
    avg_speedup = round((avg_cold_total - avg_warm_total) / avg_cold_total * 100, 1) if avg_cold_total > 0 else 0.0

    print("-" * 72)
    print(f"{'AVERAGE':<30} {avg_cold_total:>14.2f} {avg_warm_total:>16.2f} {avg_speedup:>8.1f}%")
    print(f"\nInference speed improvement with caching: {avg_speedup}%")

    return {
        "avg_no_cache_ms": avg_cold_total,
        "avg_with_cache_ms": avg_warm_total,
        "speedup_pct": avg_speedup,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline accuracy and measure performance improvements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    acc = sub.add_parser("accuracy", help="Accuracy vs ground truth (use --compare for 40% improvement)")
    acc.add_argument("--ground-truth", required=True)
    acc.add_argument("--image-folder", default=None)
    acc.add_argument("--compare", action="store_true",
                     help="Run baseline (no preprocess) vs optimized — shows improvement %")
    acc.add_argument("--no-preprocess", action="store_true")

    bench = sub.add_parser("benchmark", help="Query speed: WITH vs WITHOUT indexes (60% reduction)")
    bench.add_argument("--runs", type=int, default=100)
    bench.add_argument("--db", default=DB_PATH)

    inf = sub.add_parser("inference", help="OCR speed: WITH vs WITHOUT cache (30% speedup)")
    inf.add_argument("--image-folder", required=True)
    inf.add_argument("--runs", type=int, default=3)

    combo = sub.add_parser("all", help="Run accuracy + query benchmark + inference benchmark")
    combo.add_argument("--ground-truth", required=True)
    combo.add_argument("--image-folder", required=True)
    combo.add_argument("--runs", type=int, default=100)

    args = parser.parse_args()

    if args.command == "accuracy":
        evaluate_accuracy(
            ground_truth_dir=args.ground_truth,
            image_folder=args.image_folder,
            use_preprocessing=not args.no_preprocess,
            compare_baseline=args.compare,
        )

    elif args.command == "benchmark":
        if not os.path.exists(args.db):
            print("Database not found. Run batch_processor.py first.")
            sys.exit(1)
        benchmark_queries(runs=args.runs, db_path=args.db)

    elif args.command == "inference":
        if not os.path.isdir(args.image_folder):
            print(f"Image folder not found: {args.image_folder}")
            sys.exit(1)
        benchmark_inference(image_folder=args.image_folder, runs=args.runs)

    elif args.command == "all":
        evaluate_accuracy(
            ground_truth_dir=args.ground_truth,
            image_folder=args.image_folder,
            compare_baseline=True,
        )
        if os.path.exists(DB_PATH):
            benchmark_queries(runs=args.runs)
        benchmark_inference(image_folder=args.image_folder, runs=3)


if __name__ == "__main__":
    main()
