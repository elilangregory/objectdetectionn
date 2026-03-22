"""
Local validation — simulates the sandbox environment.

Runs submission/run.py against a folder of images, times it,
and validates the output format.

Usage:
    python local_validation.py
    python local_validation.py --images train/images --zip submission.zip
"""

import argparse
import json
import time
import zipfile
import shutil
import subprocess
from pathlib import Path


def validate_predictions(predictions_path, images_dir):
    with open(predictions_path) as f:
        preds = json.load(f)

    print(f"\n=== Output validation ===")
    print(f"Total predictions: {len(preds)}")

    if not preds:
        print("WARNING: No predictions produced")
        return

    # Check required fields
    required = {"image_id", "category_id", "bbox", "score"}
    missing_fields = [i for i, p in enumerate(preds) if not required.issubset(p.keys())]
    if missing_fields:
        print(f"ERROR: {len(missing_fields)} predictions missing required fields")
    else:
        print("Fields: OK (image_id, category_id, bbox, score all present)")

    # Check bbox format [x, y, w, h]
    bad_bbox = [p for p in preds if len(p["bbox"]) != 4]
    if bad_bbox:
        print(f"ERROR: {len(bad_bbox)} predictions with invalid bbox length")
    else:
        print("Bbox format: OK (all have 4 values)")

    # Check score range
    bad_score = [p for p in preds if not (0.0 <= p["score"] <= 1.0)]
    if bad_score:
        print(f"ERROR: {len(bad_score)} predictions with score outside [0, 1]")
    else:
        print("Scores: OK (all in [0, 1])")

    # Check category_id range
    bad_cat = [p for p in preds if not (0 <= p["category_id"] <= 356)]
    if bad_cat:
        print(f"WARNING: {len(bad_cat)} predictions with category_id outside [0, 356]")
    else:
        print("Category IDs: OK (all in [0, 356])")

    # Stats
    image_ids = set(p["image_id"] for p in preds)
    n_images  = len(list(Path(images_dir).glob("*.*")))
    print(f"\nImages with predictions: {len(image_ids)} / {n_images}")
    print(f"Avg predictions per image: {len(preds) / max(len(image_ids), 1):.1f}")


def run(args):
    images_dir  = Path(args.images)
    output_dir  = Path("local_val_output")
    extract_dir = Path("local_val_submission")
    predictions = output_dir / "predictions.json"

    # Clean up previous run
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    output_dir.mkdir()
    extract_dir.mkdir()

    # Extract zip
    zip_path = Path(args.zip)
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Verify run.py is at root
    run_py = extract_dir / "run.py"
    if not run_py.exists():
        print("ERROR: run.py not found at root of zip — this will fail in sandbox")
        return

    print(f"run.py found at root: OK")
    print(f"Files in zip:")
    for f in sorted(extract_dir.iterdir()):
        size = f.stat().st_size / 1e6
        print(f"  {f.name:<35} {size:.1f}MB")

    # Run and time it
    print(f"\n=== Running submission (timeout: 300s) ===")
    if args.n:
        print(f"NOTE: testing on {args.n} images only (use --n 0 for all)")
    print(f"NOTE: MPS (local) is ~10-20x slower than L4 CUDA (sandbox)\n")

    # Collect images and optionally limit
    all_images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.n:
        test_images = all_images[:args.n]
        test_dir = Path("local_val_images")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir()
        for p in test_images:
            shutil.copy2(p, test_dir / p.name)
        run_input = test_dir.resolve()
    else:
        run_input = images_dir.resolve()

    start = time.time()
    timed_out = False

    try:
        result = subprocess.run(
            [
                "venv/bin/python", str(run_py),
                "--input",  str(run_input),
                "--output", str(predictions.resolve()),
            ],
            timeout=300,
        )
        returncode = result.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = -1

    elapsed = time.time() - start
    n_images = args.n if args.n else len(all_images)

    print(f"\n=== Timing ===")
    if timed_out:
        print(f"TIMEOUT after 300s on {n_images} images (MPS)")
        per_image = 300 / n_images
    else:
        print(f"Elapsed: {elapsed:.1f}s on {n_images} images")
        per_image = elapsed / n_images

    # Extrapolate to sandbox (L4 ~15x faster than MPS)
    speedup = 15
    est_sandbox = (per_image * len(all_images)) / speedup
    print(f"Estimated sandbox time (L4, {len(all_images)} images): ~{est_sandbox:.0f}s")
    if est_sandbox > 270:
        print("WARNING: may be tight on sandbox — consider reducing TTA")
    else:
        print("Sandbox timing: looks OK")

    if timed_out or returncode != 0:
        if not timed_out:
            print(f"\nERROR: run.py exited with code {returncode}")
        return

    # Validate output
    if not predictions.exists():
        print("ERROR: predictions.json was not created")
        return

    validate_predictions(predictions, images_dir)
    print(f"\nPredictions saved to: {predictions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="train/images",
                        help="Folder of images to run inference on")
    parser.add_argument("--zip",    default="submission.zip",
                        help="Path to your submission zip")
    parser.add_argument("--n",      type=int, default=10,
                        help="Number of images to test on (0 = all)")
    args = parser.parse_args()
    run(args)
