"""
Visualize detection + classification on unlabeled images.
Saves annotated images to output directory.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python scripts/visualize_unlabeled.py
    venv/bin/python scripts/visualize_unlabeled.py --n 10 --images test_no_annotations
"""

import argparse
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import timm

BATCH_SIZE = 64
ANGLES = ["main", "front", "back", "left", "right"]
TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
]


def load_dinov2(weights_path, device, model_name="vit_small_patch14_dinov2.lvd142m"):
    model = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)
    state = torch.load(str(weights_path), map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def embed_all(model, crops, device):
    all_embs = []
    for i in range(0, len(crops), BATCH_SIZE):
        batch_pil = crops[i:i + BATCH_SIZE]
        aug_embs = []
        for t in TTA_TRANSFORMS:
            batch = torch.stack([t(img) for img in batch_pil]).to(device)
            with torch.no_grad():
                emb = model(batch)
            aug_embs.append(F.normalize(emb, dim=-1))
        avg = F.normalize(torch.stack(aug_embs).mean(dim=0), dim=-1)
        all_embs.append(avg.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def build_ref_db(product_images_dir, ann_path, dinov2, device):
    with open(ann_path) as f:
        data = json.load(f)
    with open(Path(product_images_dir) / "metadata.json") as f:
        meta = json.load(f)

    name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                    for p in meta["products"]}
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}

    ref_images, ref_cat_ids = [], []
    for cat in data["categories"]:
        if cat["id"] == 0:
            continue
        code = name_to_code.get(cat["name"].strip().lower())
        if not code:
            continue
        for angle in ANGLES:
            for ext in ["jpg", "jpeg", "png"]:
                p = Path(product_images_dir) / code / f"{angle}.{ext}"
                if p.exists():
                    try:
                        ref_images.append(Image.open(p).convert("RGB"))
                        ref_cat_ids.append(cat["id"])
                    except Exception:
                        pass
                    break

    print(f"  {len(ref_images)} ref images, {len(set(ref_cat_ids))} products")
    embeddings = embed_all(dinov2, ref_images, device)
    return embeddings, ref_cat_ids, cat_id_to_name


def make_centroids(ref_embs, ref_cat_ids):
    cat_ids_unique = sorted(set(ref_cat_ids))
    centroids, centroid_ids = [], []
    for cid in cat_ids_unique:
        mask = [i for i, c in enumerate(ref_cat_ids) if c == cid]
        centroid = ref_embs[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
        centroid_ids.append(cid)
    return np.array(centroids), centroid_ids


def classify_linear(crop_embs, linear_path, device):
    import torch.nn as nn
    state = torch.load(linear_path, map_location=device)
    num_classes = state["bias"].shape[0]
    in_dim = state["weight"].shape[1]
    linear = nn.Linear(in_dim, num_classes)
    linear.load_state_dict(torch.load(linear_path, map_location=device))
    linear.eval().to(device)
    embs_t = torch.from_numpy(crop_embs).to(device)
    with torch.no_grad():
        probs = torch.softmax(linear(embs_t), dim=1)
        cats = probs.argmax(dim=1).cpu().numpy()
        confs = probs.max(dim=1).values.cpu().numpy()
    return [(int(c), float(s)) for c, s in zip(cats, confs)]


def classify_crops(crop_embs, ref_embs, ref_cat_ids, use_centroids=False):
    if use_centroids:
        centroid_embs, centroid_ids = make_centroids(ref_embs, ref_cat_ids)
        sims = crop_embs @ centroid_embs.T
        results = []
        for row in sims:
            best_idx = int(np.argmax(row))
            results.append((centroid_ids[best_idx], float(row[best_idx])))
        return results

    sims = crop_embs @ ref_embs.T
    results = []
    for sim_row in sims:
        best = {}
        for sim, cat_id in zip(sim_row, ref_cat_ids):
            if cat_id not in best or sim > best[cat_id]:
                best[cat_id] = float(sim)
        best_cat = int(max(best, key=best.get))
        best_sim = best[best_cat]
        results.append((best_cat, best_sim))
    return results


def shorten(name, max_len=28):
    return name[:max_len] + "…" if len(name) > max_len else name


def run(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading YOLO ...")
    yolo = YOLO(args.yolo)

    print("Loading DINOv2 ...")
    dinov2 = load_dinov2(args.dinov2, device, model_name=args.model)

    print("Building reference DB ...")
    ref_embs, ref_cat_ids, cat_id_to_name = build_ref_db(
        args.product_images, args.annotations, dinov2, device)

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

    total_detections = 0
    for img_path in selected:
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\nSkipping {img_path.name}: {e}")
            continue
        iw, ih = img_pil.size
        print(f"\n{img_path.name}  {iw}x{ih}")

        results = yolo.predict(img_pil, imgsz=1280, conf=args.conf,
                               device=device, verbose=False)
        boxes  = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        crops, meta = [], []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(iw, int(x2)), min(ih, int(y2))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crops.append(img_pil.crop((x1, y1, x2, y2)))
            meta.append((x1, y1, x2, y2, float(score)))

        if not crops:
            print("  No detections")
            continue

        total_detections += len(crops)

        crop_embs = embed_all(dinov2, crops, device)
        if args.linear:
            classifications = classify_linear(crop_embs, args.linear, device)
        else:
            classifications = classify_crops(crop_embs, ref_embs, ref_cat_ids,
                                             use_centroids=args.centroid)

        # Draw
        scale = min(1.0, 2000 / max(iw, ih))
        disp_w, disp_h = int(iw * scale), int(ih * scale)
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (disp_w, disp_h))

        font      = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, scale * 0.6)
        thickness  = max(1, int(scale * 2))

        for (x1, y1, x2, y2, det_score), (cat_id, sim) in zip(meta, classifications):
            sx1, sy1 = int(x1*scale), int(y1*scale)
            sx2, sy2 = int(x2*scale), int(y2*scale)

            if sim < args.cls_conf:
                continue

            if sim > 0.6:
                color = (0, 200, 0)
            elif sim > 0.45:
                color = (0, 200, 255)
            else:
                color = (0, 80, 255)

            cv2.rectangle(bgr, (sx1, sy1), (sx2, sy2), color, thickness)

            name = shorten(cat_id_to_name.get(cat_id, f"id={cat_id}"))
            label = f"{name} ({sim:.2f})"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            ty = max(sy1 - 4, th + 2)
            cv2.rectangle(bgr, (sx1, ty - th - 2), (sx1 + tw + 2, ty + 2), color, -1)
            cv2.putText(bgr, label, (sx1 + 1, ty), font, font_scale, (0, 0, 0), 1)

        cv2.putText(bgr, f"{img_path.name}  {iw}x{ih}  det={len(crops)}",
                    (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(bgr, "GREEN=confident  YELLOW=moderate  RED=uncertain",
                    (10, 55), font, 0.45, (200, 200, 200), 1)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), bgr)
        print(f"  Detections: {len(crops)} → {out_path}")

    print(f"\nDone. {len(selected)} images, {total_detections} total detections → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo",           default="weights/bestv2.pt")
    parser.add_argument("--dinov2",         default="dinov2_finetuned.pt")
    parser.add_argument("--annotations",    default="train1/annotations.json")
    parser.add_argument("--images",         default="test_no_annotations")
    parser.add_argument("--product-images", dest="product_images",
                                            default="NM_NGD_product_images")
    parser.add_argument("--out",            default="output_test_vis")
    parser.add_argument("--conf",           type=float, default=0.25)
    parser.add_argument("--cls-conf",       dest="cls_conf", type=float, default=0.60)
    parser.add_argument("--centroid",       action="store_true",
                        help="Use category centroids instead of individual refs")
    parser.add_argument("--linear",         default=None,
                        help="Path to linear_head.pt (overrides centroid/ref matching)")
    parser.add_argument("--model",          default="vit_small_patch14_dinov2.lvd142m",
                        help="timm model name")
    parser.add_argument("--n",              type=int,   default=0,
                        help="Images to visualize (0=all)")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()
    run(args)
