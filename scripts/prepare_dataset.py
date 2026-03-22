"""
Prepare dataset for YOLO training.

Reorganizes into the structure YOLO expects:
    dataset/
        images/train/   images/val/
        labels/train/   labels/val/

Then writes data.yaml.

Usage:
    python prepare_dataset.py \
        --images   train/images \
        --labels   train/labels \
        --out      dataset \
        --val-split 0.2
"""

import argparse
import random
import shutil
from pathlib import Path


def prepare(images_dir, labels_dir, out_dir, val_split, seed):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)

    # Collect images that have a matching label file
    image_files = [
        p for p in sorted(images_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and (labels_dir / (p.stem + ".txt")).exists()
    ]

    if not image_files:
        print("No images with matching labels found.")
        return

    random.seed(seed)
    random.shuffle(image_files)
    split_idx = int(len(image_files) * (1 - val_split))
    train_imgs = image_files[:split_idx]
    val_imgs   = image_files[split_idx:]

    print(f"Total: {len(image_files)}  |  Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

    for split, files in [("train", train_imgs), ("val", val_imgs)]:
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            shutil.copy2(img_path, img_out / img_path.name)
            lbl_path = labels_dir / (img_path.stem + ".txt")
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)

    # Read class names from classes.txt
    classes_file = labels_dir / "classes.txt"
    with open(classes_file) as f:
        names = [line.strip() for line in f if line.strip()]

    # Write data.yaml
    yaml_path = out_dir / "data.yaml"
    abs_out = out_dir.resolve()
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_out}\n")
        f.write(f"train: images/train\n")
        f.write(f"val:   images/val\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"names:\n")
        for name in names:
            f.write(f"  - \"{name}\"\n")

    print(f"Wrote {yaml_path}")
    print(f"\nRun training with:")
    print(f"  yolo train model=yolov8n.pt data={yaml_path.resolve()} epochs=50 imgsz=640")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",    default="train/images")
    parser.add_argument("--labels",    default="train/labels")
    parser.add_argument("--out",       default="dataset")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    prepare(args.images, args.labels, args.out, args.val_split, args.seed)
