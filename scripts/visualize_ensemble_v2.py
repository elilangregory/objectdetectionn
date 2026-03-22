"""
Visualize ensemble v2: centroid-primary, multiclass-as-validator.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python scripts/visualize_ensemble_v2.py --n 25
"""

import argparse
import json
import random
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm
import onnxruntime as ort
import importlib.util

# Load teammate's pipeline for detection + classification A
spec = importlib.util.spec_from_file_location("run_nr6", "submission_nr6/run.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

BATCH = 32
TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]),
]


def load_dinov2(path, device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                               pretrained=False, dynamic_img_size=True)
    state = torch.load(str(path), map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def embed_all(model, images, device):
    out = []
    for i in range(0, len(images), BATCH):
        batch_pil = images[i:i+BATCH]
        aug = []
        for t in TTA_TRANSFORMS:
            with torch.no_grad():
                b = torch.stack([t(x) for x in batch_pil]).to(device)
                e = F.normalize(model(b), dim=-1)
            aug.append(e)
        out.append(F.normalize(torch.stack(aug).mean(0), dim=-1).cpu().numpy())
    return np.concatenate(out)


def shorten(name, max_len=28):
    return name[:max_len] + "…" if len(name) > max_len else name


def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open(args.annotations) as f:
        ann_data = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in ann_data["categories"]}

    # Load teammate's pipeline (for detection + classifier A)
    print("Loading multiclass pipeline ...")
    mc_pipeline = mod.SubmissionPipeline(Path("submission_nr6"))
    # Override detection confidence if specified
    if args.det_conf is not None:
        for det in mc_pipeline.detectors:
            det.conf = args.det_conf
        print(f"Detection conf overridden to {args.det_conf}")

    # Load our DINOv2 + centroids
    print("Loading DINOv2 ...")
    dinov2 = load_dinov2("dinov2_finetuned.pt", device)

    print("Loading centroids ...")
    with open("submission_nr6/centroids.json") as f:
        cent_data = json.load(f)
    centroid_embs = np.array(cent_data["centroids"], dtype=np.float32)
    centroid_ids = cent_data["category_ids"]

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
    total_agree = 0
    total_disagree_keep = 0
    total_dropped = 0

    for img_path in selected:
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\nSkipping {img_path.name}: {e}")
            continue
        iw, ih = img_pil.size
        print(f"\n{img_path.name}  {iw}x{ih}")

        # Step 1: Detection via teammate's YOLO
        image = Image.open(img_path).convert("RGB")
        detections = mc_pipeline._run_detectors(image)
        if not detections:
            print("  No detections")
            continue

        # Step 2: Crop with context
        crops = mc_pipeline._crop_detections(image, detections)

        # Step 3: Get multiclass predictions
        results_A = mc_pipeline._classify_A(crops)

        # Step 4: Get centroid predictions
        crop_embs = embed_all(dinov2, crops, device)
        sims = crop_embs @ centroid_embs.T
        results_B = []
        for row in sims:
            best_idx = int(np.argmax(row))
            results_B.append((centroid_ids[best_idx], float(row[best_idx])))

        # Step 5: Ensemble — centroid primary, multiclass validator
        scale = min(1.0, 2000 / max(iw, ih))
        disp_w, disp_h = int(iw * scale), int(ih * scale)
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (disp_w, disp_h))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, scale * 0.6)
        thickness = max(1, int(scale * 2))

        kept = 0
        for det, (cat_A, conf_A), (cat_B, sim_B) in zip(detections, results_A, results_B):
            x1, y1, x2, y2 = det["box"].tolist()
            det_score = det["score"]

            agree = (cat_A == cat_B)

            if agree:
                # Both agree — high confidence, keep
                category_id = cat_B  # centroid's pick (same as A)
                color = (0, 200, 0)  # green
                status = "AGREE"
                total_agree += 1
            elif sim_B >= args.centroid_thresh:
                # Disagree but centroid is confident — trust centroid
                category_id = cat_B
                color = (0, 200, 255)  # yellow
                status = "CENTROID"
                total_disagree_keep += 1
            else:
                # Disagree and centroid is weak — drop
                total_dropped += 1
                continue

            kept += 1
            sx1, sy1 = int(x1 * scale), int(y1 * scale)
            sx2, sy2 = int(x2 * scale), int(y2 * scale)

            cv2.rectangle(bgr, (sx1, sy1), (sx2, sy2), color, thickness)
            name = shorten(cat_id_to_name.get(category_id, f"id={category_id}"))
            label = f"{name} ({sim_B:.2f})"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            ty = max(sy1 - 4, th + 2)
            cv2.rectangle(bgr, (sx1, ty - th - 2), (sx1 + tw + 2, ty + 2), color, -1)
            cv2.putText(bgr, label, (sx1 + 1, ty), font, font_scale, (0, 0, 0), 1)

        total_det += kept
        cv2.putText(bgr, f"{img_path.name}  {iw}x{ih}  kept={kept}/{len(detections)}",
                    (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(bgr, f"GREEN=agree  YELLOW=centroid-only  centroid_thresh={args.centroid_thresh}",
                    (10, 55), font, 0.45, (200, 200, 200), 1)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), bgr)
        print(f"  Kept: {kept}/{len(detections)} → {out_path}")

    print(f"\nDone. {len(selected)} images, {total_det} kept")
    print(f"  Agreed: {total_agree}  Centroid-only: {total_disagree_keep}  Dropped: {total_dropped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="train1/annotations.json")
    parser.add_argument("--images", default="test_no_annotations")
    parser.add_argument("--out", default="output_test_vis_ensemble_v2")
    parser.add_argument("--centroid-thresh", dest="centroid_thresh", type=float, default=0.50)
    parser.add_argument("--det-conf", dest="det_conf", type=float, default=None,
                        help="Override detection confidence (default: use manifest)")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
