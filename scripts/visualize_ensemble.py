"""
Visualize ensemble pipeline (nr6) on test images.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python scripts/visualize_ensemble.py --n 25
"""

import argparse
import json
import random
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import importlib.util


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "run_nr6", "submission_nr6/run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base_dir = Path("submission_nr6")
    return mod.SubmissionPipeline(base_dir)


def shorten(name, max_len=28):
    return name[:max_len] + "…" if len(name) > max_len else name


def main(args):
    with open(args.annotations) as f:
        ann_data = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in ann_data["categories"]}

    print("Loading ensemble pipeline ...")
    pipeline = load_pipeline()

    image_paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.n and args.n < len(image_paths):
        random.seed(args.seed)
        selected = random.sample(image_paths, args.n)
    else:
        selected = image_paths

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_det = 0
    for img_path in selected:
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\nSkipping {img_path.name}: {e}")
            continue
        iw, ih = img_pil.size
        print(f"\n{img_path.name}  {iw}x{ih}")

        preds = pipeline.predict_image(img_path)
        if not preds:
            print("  No detections")
            continue
        total_det += len(preds)

        scale = min(1.0, 2000 / max(iw, ih))
        disp_w, disp_h = int(iw * scale), int(ih * scale)
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (disp_w, disp_h))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, scale * 0.6)
        thickness = max(1, int(scale * 2))

        for pred in preds:
            x, y, w, h = pred["bbox"]
            sx1, sy1 = int(x * scale), int(y * scale)
            sx2, sy2 = int((x + w) * scale), int((y + h) * scale)
            score = pred["score"]
            cat_id = pred["category_id"]

            if score > 0.6:
                color = (0, 200, 0)
            elif score > 0.3:
                color = (0, 200, 255)
            else:
                color = (0, 80, 255)

            cv2.rectangle(bgr, (sx1, sy1), (sx2, sy2), color, thickness)
            name = shorten(cat_id_to_name.get(cat_id, f"id={cat_id}"))
            label = f"{name} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            ty = max(sy1 - 4, th + 2)
            cv2.rectangle(bgr, (sx1, ty - th - 2), (sx1 + tw + 2, ty + 2), color, -1)
            cv2.putText(bgr, label, (sx1 + 1, ty), font, font_scale, (0, 0, 0), 1)

        cv2.putText(bgr, f"{img_path.name}  {iw}x{ih}  det={len(preds)}",
                    (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(bgr, "ENSEMBLE: multiclass YOLO + DINOv2 centroid",
                    (10, 55), font, 0.45, (200, 200, 200), 1)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), bgr)
        print(f"  Detections: {len(preds)} -> {out_path}")

    print(f"\nDone. {len(selected)} images, {total_det} total detections -> {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="train1/annotations.json")
    parser.add_argument("--images", default="test_no_annotations")
    parser.add_argument("--out", default="output_test_vis_ensemble")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
