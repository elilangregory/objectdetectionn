"""
Local validation — mirrors competition sandbox checks and scoring.

Usage:
    venv/bin/python local_validate.py submission_nr2.zip
    venv/bin/python local_validate.py submission_nr2.zip --n 20   # quick test
    venv/bin/python local_validate.py submission_nr2.zip --n 0    # all images
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import numpy as np
from pathlib import Path

# ── Limits (from competition docs) ────────────────────────────────────────────
LIMITS = {
    "max_uncompressed_mb": 420,
    "max_files":           1000,
    "max_py_files":        10,
    "max_weight_files":    3,
    "max_weight_mb":       420,
    "allowed_extensions":  {".py", ".json", ".yaml", ".yml", ".cfg",
                            ".pt", ".pth", ".onnx", ".safetensors", ".npy"},
    "weight_extensions":   {".pt", ".pth", ".onnx", ".safetensors", ".npy"},
}

BLOCKED_IMPORTS = [
    "import os", "import sys", "import subprocess", "import socket",
    "import ctypes", "import builtins", "import importlib",
    "import pickle", "import marshal", "import shelve", "import shutil",
    "import yaml", "import requests", "import urllib", "import http",
    "import multiprocessing", "import threading", "import signal", "import gc",
    "import code", "import codeop", "import pty",
]
BLOCKED_CALLS = ["exec(", "compile(", "__import__("]
BLOCKED_CALLS_REGEX = [r"(?<!\.)(?<!\w)eval\("]  # match eval( but not .eval( or COCOeval(


def check_zip(zip_path):
    errors   = []
    warnings = []

    print(f"\n{'='*55}")
    print(f"  ZIP VALIDATION: {zip_path}")
    print(f"{'='*55}")

    if not Path(zip_path).exists():
        print(f"  ERROR: file not found: {zip_path}")
        return False, errors

    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.infolist()
        names   = [e.filename for e in entries]

        # 1. run.py at root
        if "run.py" not in names:
            errors.append("run.py not found at root (most common error — don't nest in subfolder)")
        else:
            print("  [OK] run.py found at root")

        # 2. File count
        n_files = len(names)
        if n_files > LIMITS["max_files"]:
            errors.append(f"Too many files: {n_files} > {LIMITS['max_files']}")
        else:
            print(f"  [OK] File count: {n_files}")

        # 3. Allowed extensions
        bad_exts = []
        for name in names:
            if name.endswith("/"):   # directory entry
                continue
            ext = Path(name).suffix.lower()
            if ext and ext not in LIMITS["allowed_extensions"]:
                bad_exts.append(name)
        if bad_exts:
            errors.append(f"Disallowed file types: {bad_exts[:5]}")
        else:
            print(f"  [OK] All file types allowed")

        # 4. Python file count
        py_files = [n for n in names if n.endswith(".py")]
        if len(py_files) > LIMITS["max_py_files"]:
            errors.append(f"Too many .py files: {len(py_files)} > {LIMITS['max_py_files']}")
        else:
            print(f"  [OK] Python files: {len(py_files)}")

        # 5. Weight file count
        weight_files = [n for n in names
                        if Path(n).suffix.lower() in LIMITS["weight_extensions"]]
        if len(weight_files) > LIMITS["max_weight_files"]:
            errors.append(f"Too many weight files: {len(weight_files)} > {LIMITS['max_weight_files']}")
        else:
            print(f"  [OK] Weight files: {len(weight_files)}  {weight_files}")

        # 6. Uncompressed size
        total_bytes = sum(e.file_size for e in entries)
        total_mb    = total_bytes / 1e6
        if total_mb > LIMITS["max_uncompressed_mb"]:
            errors.append(f"Uncompressed size {total_mb:.1f} MB > {LIMITS['max_uncompressed_mb']} MB")
        else:
            print(f"  [OK] Uncompressed size: {total_mb:.1f} MB")

        # 7. Weight size total
        weight_bytes = sum(e.file_size for e in entries
                           if Path(e.filename).suffix.lower() in LIMITS["weight_extensions"])
        weight_mb = weight_bytes / 1e6
        if weight_mb > LIMITS["max_weight_mb"]:
            errors.append(f"Weight files total {weight_mb:.1f} MB > {LIMITS['max_weight_mb']} MB")
        else:
            print(f"  [OK] Weight size total: {weight_mb:.1f} MB")

        # 8. Scan Python files for blocked imports/calls
        print("\n  Scanning Python files for blocked patterns ...")
        for name in py_files:
            content = zf.read(name).decode("utf-8", errors="replace")
            for blocked in BLOCKED_IMPORTS + BLOCKED_CALLS:
                if blocked in content:
                    errors.append(f"Blocked pattern '{blocked}' in {name}")
            for pattern in BLOCKED_CALLS_REGEX:
                if re.search(pattern, content):
                    errors.append(f"Blocked pattern '{pattern}' in {name}")
        if not any("Blocked" in e for e in errors):
            print(f"  [OK] No blocked patterns found")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    ✗ {e}")
        return False, errors
    else:
        print(f"\n  All checks passed!")
        return True, []


def run_submission(zip_path, images_dir, annotations_path, n_images, python_bin):
    print(f"\n{'='*55}")
    print(f"  RUNNING SUBMISSION")
    print(f"{'='*55}")

    tmp = Path(tempfile.mkdtemp(prefix="nmiai_val_"))
    try:
        # Extract zip
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        print(f"  Extracted to: {tmp}")

        # Check run.py exists
        run_py = tmp / "run.py"
        if not run_py.exists():
            print("  ERROR: run.py not found after extraction")
            return None

        # Set up input dir (symlink or copy limited images)
        input_dir  = tmp / "input_images"
        output_dir = tmp / "output"
        output_dir.mkdir()
        input_dir.mkdir()

        images = sorted(
            p for p in Path(images_dir).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if n_images:
            images = images[:n_images]
        print(f"  Input images: {len(images)}")

        for img in images:
            (input_dir / img.name).symlink_to(img.resolve())

        output_path = output_dir / "predictions.json"

        # Run
        cmd = [python_bin, str(run_py),
               "--input",  str(input_dir),
               "--output", str(output_path)]
        print(f"  Running: python run.py --input ... --output ...")
        print()
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(tmp), capture_output=False, text=True)
        elapsed = time.time() - t0

        print(f"\n  Exit code: {result.returncode}")
        print(f"  Time:      {elapsed:.1f}s")

        if elapsed > 300:
            print(f"  WARNING: {elapsed:.0f}s exceeds 300s sandbox timeout!")

        if result.returncode != 0:
            print("  ERROR: run.py failed")
            return None

        if not output_path.exists():
            print("  ERROR: predictions.json not written")
            return None

        with open(output_path) as f:
            preds = json.load(f)

        print(f"  Predictions: {len(preds)}")

        # Validate prediction format
        format_errors = []
        for i, p in enumerate(preds[:5]):
            for key in ["image_id", "category_id", "bbox", "score"]:
                if key not in p:
                    format_errors.append(f"Missing '{key}' in prediction {i}")
            if "bbox" in p and len(p["bbox"]) != 4:
                format_errors.append(f"bbox must have 4 elements, got {len(p['bbox'])} in prediction {i}")
        if format_errors:
            print(f"  Format errors: {format_errors}")
        else:
            print(f"  [OK] Prediction format valid")
            print(f"  Sample: {preds[0]}")

        return preds, elapsed

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]);  y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def compute_ap(predictions, gt_by_image, match_category=False, iou_thr=0.5):
    preds_sorted   = sorted(predictions, key=lambda p: p["score"], reverse=True)
    evaluated_ids  = {p["image_id"] for p in predictions}
    n_gt = sum(len(gt_by_image.get(i, [])) for i in evaluated_ids)
    if n_gt == 0:
        return 0.0

    matched = {}
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for i, pred in enumerate(preds_sorted):
        img_id  = pred["image_id"]
        gt_anns = gt_by_image.get(img_id, [])
        if img_id not in matched:
            matched[img_id] = set()
        best_iou, best_idx = 0.0, -1
        for j, gt in enumerate(gt_anns):
            if j in matched[img_id]: continue
            if match_category and pred["category_id"] != gt["category_id"]: continue
            iou = box_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou: best_iou = iou; best_idx = j
        if best_iou >= iou_thr and best_idx >= 0:
            tp[i] = 1; matched[img_id].add(best_idx)
        else:
            fp[i] = 1

    cum_tp    = np.cumsum(tp)
    cum_fp    = np.cumsum(fp)
    recall    = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p = precision[recall >= thr]
        ap += (np.max(p) if len(p) else 0.0)
    return ap / 11.0


def score(predictions, annotations_path):
    print(f"\n{'='*55}")
    print(f"  SCORING")
    print(f"{'='*55}")

    with open(annotations_path) as f:
        ann_data = json.load(f)

    gt_by_image = {}
    for ann in ann_data["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    evaluated_ids = {p["image_id"] for p in predictions}
    n_gt_total = sum(len(gt_by_image.get(i, [])) for i in evaluated_ids)
    print(f"  Evaluated images: {len(evaluated_ids)}")
    print(f"  GT boxes (evaluated images): {n_gt_total}")
    print(f"  Predictions: {len(predictions)}")

    det_map = compute_ap(predictions, gt_by_image, match_category=False)
    cls_map = compute_ap(predictions, gt_by_image, match_category=True)
    final   = 0.7 * det_map + 0.3 * cls_map

    print(f"\n  Detection mAP@0.5:      {det_map:.4f}")
    print(f"  Classification mAP@0.5: {cls_map:.4f}")
    print(f"  Final score:            {final:.4f}  (0.7×det + 0.3×cls)")
    return det_map, cls_map, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip",            help="Path to submission zip")
    parser.add_argument("--annotations",  default="train1/annotations.json")
    parser.add_argument("--images",       default="train1/images")
    parser.add_argument("--n",            type=int, default=20,
                        help="Number of images to test (0=all)")
    parser.add_argument("--python",       default=str(Path(sys.executable)),
                        help="Python binary to use")
    parser.add_argument("--skip-run",     action="store_true",
                        help="Only validate zip structure, don't run")
    args = parser.parse_args()

    zip_path = Path(args.zip)

    # Step 1: validate zip
    ok, errors = check_zip(zip_path)
    if not ok:
        print("\nFix errors before submitting.")
        sys.exit(1)

    if args.skip_run:
        print("\nSkipping run (--skip-run)")
        sys.exit(0)

    # Step 2: run
    result = run_submission(
        zip_path, args.images, args.annotations,
        args.n, args.python
    )
    if result is None:
        print("\nSubmission failed to run.")
        sys.exit(1)

    preds, elapsed = result

    # Step 3: score
    det_map, cls_map, final = score(preds, args.annotations)

    print(f"\n{'='*55}")
    print(f"  SUMMARY")
    print(f"{'='*55}")
    print(f"  Zip:          {zip_path.name}")
    print(f"  Images:       {args.n if args.n else 'all'}")
    print(f"  Runtime:      {elapsed:.1f}s {'⚠️ > 300s!' if elapsed > 300 else '✓'}")
    print(f"  Final score:  {final:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
