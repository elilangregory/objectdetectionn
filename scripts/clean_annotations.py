"""
Clean up a COCO-format annotations.json file.

Checks performed:
  1. Remove images whose file doesn't exist on disk
  2. Remove annotations with missing image_id or category_id
  3. Remove annotations with invalid bboxes (zero/negative area, out of bounds)
  4. Remove duplicate annotation IDs (keep first occurrence)
  5. Remove categories with no remaining annotations
  6. Re-index all IDs to be contiguous (images, annotations, categories)

Usage:
    python clean_annotations.py \
        --input  train/annotations.json \
        --images train/images \
        --output train/annotations_clean.json
"""

import json
import argparse
import os
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {path}")


def clean(ann_path, images_dir, out_path):
    data = load(ann_path)

    images = data["images"]
    categories = data["categories"]
    annotations = data["annotations"]

    stats = {}

    # 1. Remove images missing on disk
    existing_files = {p.name for p in Path(images_dir).iterdir() if p.is_file()}
    before = len(images)
    images = [img for img in images if img["file_name"] in existing_files]
    stats["images_missing_on_disk"] = before - len(images)

    valid_image_ids = {img["id"] for img in images}
    valid_category_ids = {cat["id"] for cat in categories}

    # 2. Remove annotations with unknown image_id or category_id
    before = len(annotations)
    annotations = [
        a for a in annotations
        if a["image_id"] in valid_image_ids and a["category_id"] in valid_category_ids
    ]
    stats["annotations_invalid_ref"] = before - len(annotations)

    # 3. Remove annotations with invalid bboxes
    def bbox_valid(ann, img_lookup):
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            return False
        img = img_lookup.get(ann["image_id"])
        if img and (x < 0 or y < 0 or x + w > img["width"] or y + h > img["height"]):
            return False
        return True

    img_lookup = {img["id"]: img for img in images}
    before = len(annotations)
    annotations = [a for a in annotations if bbox_valid(a, img_lookup)]
    stats["annotations_invalid_bbox"] = before - len(annotations)

    # 4. Remove duplicate annotation IDs (keep first)
    before = len(annotations)
    seen_ids = set()
    deduped = []
    for a in annotations:
        if a["id"] not in seen_ids:
            seen_ids.add(a["id"])
            deduped.append(a)
    annotations = deduped
    stats["annotations_duplicate_ids"] = before - len(annotations)

    # 5. Remove categories with no remaining annotations
    used_category_ids = {a["category_id"] for a in annotations}
    before = len(categories)
    categories = [c for c in categories if c["id"] in used_category_ids]
    stats["categories_unused"] = before - len(categories)

    # 6. Re-index IDs to be contiguous
    # Images
    img_id_map = {old["id"]: new_id for new_id, old in enumerate(images, start=1)}
    for img in images:
        img["id"] = img_id_map[img["id"]]

    # Categories
    cat_id_map = {old["id"]: new_id for new_id, old in enumerate(categories, start=1)}
    for cat in categories:
        cat["id"] = cat_id_map[cat["id"]]

    # Annotations
    for i, ann in enumerate(annotations, start=1):
        ann["id"] = i
        ann["image_id"] = img_id_map[ann["image_id"]]
        ann["category_id"] = cat_id_map[ann["category_id"]]

    print("\n=== Cleanup summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v} removed")
    print(f"\nFinal counts:")
    print(f"  images:      {len(images)}")
    print(f"  categories:  {len(categories)}")
    print(f"  annotations: {len(annotations)}")

    save({"images": images, "categories": categories, "annotations": annotations}, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="train/annotations.json")
    parser.add_argument("--images", default="train/images")
    parser.add_argument("--output", default="train/annotations_clean.json")
    args = parser.parse_args()
    clean(args.input, args.images, args.output)
