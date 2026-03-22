"""
Convert COCO annotations to YOLO format.

YOLO expects one .txt per image (same name, different extension) with lines:
    <class_index> <cx> <cy> <w> <h>   (all normalized 0-1)

Also writes a classes.txt mapping index -> category name.

Usage:
    python coco_to_yolo.py \
        --input  train/annotations_clean.json \
        --output train/labels
"""

import json
import argparse
from pathlib import Path


def convert(ann_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path) as f:
        data = json.load(f)

    # Build lookups
    images = {img["id"]: img for img in data["images"]}

    # Sort categories by id so index is stable
    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_index = {cat["id"]: i for i, cat in enumerate(categories)}

    # Write classes.txt
    classes_file = out_dir / "classes.txt"
    with open(classes_file, "w") as f:
        for cat in categories:
            f.write(cat["name"] + "\n")
    print(f"Wrote {classes_file}")

    # Group annotations by image
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Write one .txt per image
    for img_id, img in images.items():
        stem = Path(img["file_name"]).stem
        label_file = out_dir / f"{stem}.txt"
        iw, ih = img["width"], img["height"]

        lines = []
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / iw
            cy = (y + h / 2) / ih
            nw = w / iw
            nh = h / ih
            cls = cat_id_to_index[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_file, "w") as f:
            f.write("\n".join(lines))

    print(f"Wrote {len(images)} label files to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="train/annotations_clean.json")
    parser.add_argument("--output", default="train/labels")
    args = parser.parse_args()
    convert(args.input, args.output)
