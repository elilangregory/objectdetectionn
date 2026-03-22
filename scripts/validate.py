"""
Validation script for best_v1.pt

Runs inference on val images and saves annotated outputs so you can
visually inspect how the model is detecting products.

Usage:
    python validate.py
    python validate.py --model best_v1.pt --n 10 --conf 0.25
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def validate(model_path, val_dir, out_dir, n_images, conf):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)

    val_images = sorted(Path(val_dir).glob("*.*"))[:n_images]
    if not val_images:
        print(f"No images found in {val_dir}")
        return

    print(f"Running inference on {len(val_images)} images (conf={conf})\n")

    total_detections = 0

    for img_path in val_images:
        results = model.predict(str(img_path), conf=conf, device="mps", verbose=False)
        result = results[0]

        n_det = len(result.boxes)
        total_detections += n_det
        print(f"{img_path.name}: {n_det} detections")

        # Save annotated image
        annotated = result.plot()
        out_path = out_dir / img_path.name
        import cv2
        cv2.imwrite(str(out_path), annotated)

    print(f"\nTotal detections: {total_detections} across {len(val_images)} images")
    print(f"Avg per image: {total_detections / len(val_images):.1f}")
    print(f"\nAnnotated images saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="best_v1.pt")
    parser.add_argument("--val-dir", default="dataset/images/val")
    parser.add_argument("--out",     default="validation_output")
    parser.add_argument("--n",       type=int,   default=10)
    parser.add_argument("--conf",    type=float, default=0.25)
    args = parser.parse_args()

    validate(args.model, args.val_dir, args.out, args.n, args.conf)
