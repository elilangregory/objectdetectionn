"""
Full pipeline: clean -> convert -> split -> train

Usage:
    python run.py
    python run.py --epochs 100 --model yolov8s.pt --imgsz 640
"""

import argparse
from pathlib import Path

from clean_annotations import clean
from coco_to_yolo import convert
from prepare_dataset import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="train/annotations.json")
    parser.add_argument("--images",     default="train/images")
    parser.add_argument("--clean-out",  default="train/annotations_clean.json")
    parser.add_argument("--labels",     default="train/labels")
    parser.add_argument("--dataset",    default="dataset")
    parser.add_argument("--val-split",  type=float, default=0.2)
    parser.add_argument("--model",      default="yolov8s.pt")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--batch",      type=int,   default=8)
    args = parser.parse_args()

    print("\n=== Step 1/4: Clean annotations ===")
    clean(args.input, args.images, args.clean_out)

    print("\n=== Step 2/4: Convert to YOLO format ===")
    convert(args.clean_out, args.labels)

    print("\n=== Step 3/4: Prepare dataset (train/val split) ===")
    prepare(args.images, args.labels, args.dataset, args.val_split, seed=42)

    data_yaml = str(Path(args.dataset).resolve() / "data.yaml")

    print("\n=== Step 4/4: Train ===\n")
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="mps",
    )


if __name__ == "__main__":
    main()
