"""
Build submission_nr6:
  Detection:      third_medium_best.onnx (teammate's medium YOLO, 960px)
  Classification: ensemble of teammate's dual-head classifier + DINOv2 centroid matching

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python submission_nr6/build.py
"""

import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

_pil_open = Image.open

from torchvision import transforms

Image.open = _pil_open

import timm

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent

# Source files
YOLO_ONNX_SRC     = ROOT / "workspaces" / "multiclass" / "third_medium_best.onnx"
CLASSIFIER_SRC     = ROOT / "workspaces" / "multiclass" / "classifier.onnx"
REF_PROTO_SRC      = ROOT / "workspaces" / "multiclass" / "reference_prototypes.json"
DINO_SRC           = ROOT / "dinov2_finetuned.pt"

# Data paths for centroid building
ANN_PATH   = ROOT / "train1" / "annotations.json"
TRAIN_IMGS = ROOT / "train1" / "images"
PROD_IMGS  = ROOT / "NM_NGD_product_images"

ANGLES = ["main", "front", "back", "left", "right"]
MAX_TRAIN_CROPS = 10

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


def load_dinov2(weights_path, device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                               pretrained=False, dynamic_img_size=True)
    state = torch.load(str(weights_path), map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def embed(model, images, device, batch_size=64):
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch_pil = images[i:i+batch_size]
        aug_embs = []
        for t in TTA_TRANSFORMS:
            batch = torch.stack([t(img) for img in batch_pil]).to(device)
            with torch.no_grad():
                emb = model(batch)
            aug_embs.append(F.normalize(emb, dim=-1))
        avg = F.normalize(torch.stack(aug_embs).mean(dim=0), dim=-1)
        all_embs.append(avg.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Copy weight files
    print("\nCopying weight files ...")

    yolo_dst = HERE / "third_medium_best.onnx"
    shutil.copy2(YOLO_ONNX_SRC, yolo_dst)
    print(f"  YOLO ONNX:       {yolo_dst.stat().st_size / 1e6:.1f} MB")

    cls_dst = HERE / "classifier.onnx"
    shutil.copy2(CLASSIFIER_SRC, cls_dst)
    print(f"  Classifier ONNX: {cls_dst.stat().st_size / 1e6:.1f} MB")

    dino_dst = HERE / "dinov2_weights.pt"
    shutil.copy2(DINO_SRC, dino_dst)
    print(f"  DINOv2 weights:  {dino_dst.stat().st_size / 1e6:.1f} MB")

    ref_proto_dst = HERE / "reference_prototypes.json"
    shutil.copy2(REF_PROTO_SRC, ref_proto_dst)
    print(f"  Ref prototypes:  {ref_proto_dst.stat().st_size / 1e6:.1f} MB")

    # 2. Build centroids.json
    print("\nBuilding DINOv2 centroids ...")
    dinov2 = load_dinov2(dino_dst, device)

    with open(ANN_PATH) as f:
        data = json.load(f)
    with open(PROD_IMGS / "metadata.json") as f:
        meta = json.load(f)

    name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                    for p in meta["products"]}

    # Studio reference images
    ref_images, ref_cat_ids = [], []
    for cat in data["categories"]:
        cat_id = cat["id"]
        if cat_id == 0:
            continue
        code = name_to_code.get(cat["name"].strip().lower())
        if not code:
            continue
        for angle in ANGLES:
            for ext in ["jpg", "jpeg", "png"]:
                p = PROD_IMGS / code / f"{angle}.{ext}"
                if p.exists():
                    try:
                        ref_images.append(Image.open(p).convert("RGB"))
                        ref_cat_ids.append(cat_id)
                    except Exception:
                        pass
                    break
    print(f"  Studio: {len(ref_images)} images for {len(set(ref_cat_ids))} products")

    # Add real shelf crops from train1 (up to MAX_TRAIN_CROPS per category)
    img_id_to_path = {}
    for img_info in data["images"]:
        p = TRAIN_IMGS / img_info["file_name"]
        if p.exists():
            img_id_to_path[img_info["id"]] = p

    by_cat = {}
    for ann in data["annotations"]:
        if ann["category_id"] == 0:
            continue
        # Exclude bad annotation image
        if ann["image_id"] == 295:
            continue
        by_cat.setdefault(ann["category_id"], []).append(ann)

    loaded_imgs = {}
    for cat_id, anns in by_cat.items():
        count = 0
        for ann in anns:
            if count >= MAX_TRAIN_CROPS:
                break
            img_path = img_id_to_path.get(ann["image_id"])
            if not img_path:
                continue
            if ann["image_id"] not in loaded_imgs:
                try:
                    loaded_imgs[ann["image_id"]] = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
            img = loaded_imgs[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            ref_images.append(img.crop((x1, y1, x2, y2)))
            ref_cat_ids.append(cat_id)
            count += 1

    print(f"  Total:  {len(ref_images)} images for {len(set(ref_cat_ids))} categories")

    # Embed all refs
    embeddings = embed(dinov2, ref_images, device)

    # Compute centroids (mean embedding per category, L2-normalized)
    print("\nComputing category centroids ...")
    cat_ids_unique = sorted(set(ref_cat_ids))
    centroids = []
    centroid_cat_ids = []
    for cid in cat_ids_unique:
        mask = [i for i, c in enumerate(ref_cat_ids) if c == cid]
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid.tolist())
        centroid_cat_ids.append(cid)
    print(f"  {len(centroids)} category centroids, dim={len(centroids[0])}")

    # Save as JSON
    centroids_path = HERE / "centroids.json"
    with open(centroids_path, "w") as f:
        json.dump({"category_ids": centroid_cat_ids, "centroids": centroids}, f)
    print(f"  Centroids JSON: {centroids_path.stat().st_size / 1e6:.2f} MB")

    # 3. Summary
    print("\n=== Submission contents ===")
    total = 0
    for fn in sorted(HERE.iterdir()):
        if fn.suffix in {".onnx", ".pt", ".json", ".py"}:
            size = fn.stat().st_size / 1e6
            total += size
            print(f"  {fn.name:<35} {size:.1f} MB")
    print(f"  {'TOTAL':<35} {total:.1f} MB  ({'OK' if total < 420 else 'OVER LIMIT'})")

    print(f"\nTo zip:")
    print(f"  cd {HERE} && zip -r ../submission_nr6.zip . -x '.*' '__MACOSX/*' 'build.py'")


if __name__ == "__main__":
    main()
