"""
Classification validation script.

1. Builds a DINOv2 reference embedding database from NM_NGD_product_images
2. Runs YOLO detection on images
3. For each detected crop, finds nearest product via cosine similarity
4. Saves annotated images showing predicted product name + confidence

First run will download DINOv2 weights (~22MB) via torch.hub.

Usage:
    python classify_validate.py
    python classify_validate.py --n 5 --conf 0.25 --angles main front
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# ── DINOv2 setup ──────────────────────────────────────────────────────────────

def load_dinov2(device):
    print("Loading DINOv2 ViT-S/14 ...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval().to(device)
    return model

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Fix 2 — TTA augmentations applied to query crops
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
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
]

def embed(model, images, device, batch_size=64):
    """images: list of PIL Images → (N, D) numpy array"""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([TRANSFORM(img) for img in images[i:i+batch_size]]).to(device)
        with torch.no_grad():
            emb = model(batch)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


# ── Reference database ────────────────────────────────────────────────────────

def build_reference_db(product_images_dir, ann_path, angles, dinov2, device):
    print(f"Building reference database (angles: {angles}) ...")

    with open(ann_path) as f:
        data = json.load(f)
    with open(Path(product_images_dir) / "metadata.json") as f:
        meta = json.load(f)

    # name → product_code
    name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                    for p in meta["products"]}

    ref_images = []
    ref_category_ids = []
    ref_labels = []  # for display

    for cat in data["categories"]:
        cat_id = cat["id"]
        name = cat["name"].strip().lower()
        code = name_to_code.get(name)
        if not code:
            continue

        found_any = False
        for angle in angles:
            for ext in ["jpg", "jpeg", "png"]:
                img_path = Path(product_images_dir) / code / f"{angle}.{ext}"
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        ref_images.append(img)
                        ref_category_ids.append(cat_id)
                        ref_labels.append(cat["name"])
                        found_any = True
                    except Exception:
                        pass
                    break

        if not found_any:
            pass  # category has no reference images, will fall back to unknown

    print(f"  {len(ref_images)} reference images for {len(set(ref_category_ids))} products")

    embeddings = embed(dinov2, ref_images, device)
    category_ids = np.array(ref_category_ids)
    labels = ref_labels

    return embeddings, category_ids, labels


# ── Classification ─────────────────────────────────────────────────────────────

def embed_with_tta(model, crop_pil, device):
    """Fix 2: embed crop with multiple augmentations, return averaged embedding."""
    tensors = torch.stack([t(crop_pil) for t in TTA_TRANSFORMS]).to(device)
    with torch.no_grad():
        embs = model(tensors)
    embs = F.normalize(embs, dim=-1)
    avg = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
    return avg.cpu().numpy()[0]


def classify_crop(crop_pil, ref_embeddings, ref_category_ids, ref_labels,
                  dinov2, device, threshold=0.5, margin=0.05):
    # Fix 2: use TTA-averaged embedding instead of single forward pass
    emb = embed_with_tta(dinov2, crop_pil, device)
    sims = ref_embeddings @ emb  # cosine similarity (already normalized)

    # Best score per unique category
    best_sim = {}
    best_label = {}
    for sim, cat_id, label in zip(sims, ref_category_ids, ref_labels):
        if cat_id not in best_sim or sim > best_sim[cat_id]:
            best_sim[cat_id] = sim
            best_label[cat_id] = label

    sorted_cats = sorted(best_sim, key=best_sim.get, reverse=True)
    top_cat   = sorted_cats[0]
    top_sim   = best_sim[top_cat]
    top_label = best_label[top_cat]

    # Fix 1: require clear margin over second-best
    if len(sorted_cats) > 1:
        second_sim = best_sim[sorted_cats[1]]
        if top_sim - second_sim < margin:
            return 355, "unknown_product", float(top_sim)

    if top_sim < threshold:
        return 355, "unknown_product", float(top_sim)

    return int(top_cat), top_label, float(top_sim)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    dinov2 = load_dinov2(device)

    ref_embeddings, ref_category_ids, ref_labels = build_reference_db(
        args.product_images, args.annotations, args.angles, dinov2, device
    )

    model = YOLO(args.model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(Path(args.images).glob("*.*"))[:args.n]
    print(f"\nRunning on {len(images)} images ...\n")

    for img_path in images:
        results = model.predict(str(img_path), conf=args.conf, device=device, verbose=False)
        result = results[0]
        img_bgr = cv2.imread(str(img_path))

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        print(f"{img_path.name}: {len(boxes)} detections")

        for box, det_conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            # Clamp to image bounds
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = img_bgr[y1:y2, x1:x2]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

            cat_id, label, sim = classify_crop(
                crop_pil, ref_embeddings, ref_category_ids, ref_labels,
                dinov2, device, threshold=args.threshold, margin=args.margin
            )

            # Draw box
            color = (0, 200, 0) if cat_id != 355 else (0, 0, 200)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

            # Label: short name + similarity score
            short_label = label[:30]
            text = f"{short_label} ({sim:.2f})"
            cv2.putText(img_bgr, text, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), img_bgr)

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="best_v1.pt")
    parser.add_argument("--images",         default="train/images")
    parser.add_argument("--annotations",    default="train/annotations.json")
    parser.add_argument("--product-images", default="NM_NGD_product_images")
    parser.add_argument("--out",            default="classify_output")
    parser.add_argument("--n",              type=int,   default=5)
    parser.add_argument("--conf",           type=float, default=0.25)
    parser.add_argument("--threshold",      type=float, default=0.5,
                        help="Min cosine similarity to classify (else unknown_product)")
    parser.add_argument("--margin",         type=float, default=0.05,
                        help="Min gap between top-1 and top-2 similarity (else unknown_product)")
    parser.add_argument("--angles",         nargs="+",
                        default=["main", "front", "back", "left", "right"],
                        help="Reference image angles to use")
    args = parser.parse_args()
    run(args)
