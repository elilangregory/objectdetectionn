"""
Builds the submission directory ready to zip and upload.

Run this once locally whenever you have a new model or want to rebuild embeddings.

Usage:
    python build_submission.py
    python build_submission.py --yolo best_v1.pt
"""

import argparse
import json
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm


ANGLES = ["main", "front", "back", "left", "right"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_dinov2(weights_path, device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, dynamic_img_size=True)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def embed(model, images, device, batch_size=64):
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([TRANSFORM(img) for img in images[i:i + batch_size]]).to(device)
        with torch.no_grad():
            emb = model(batch)
        emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def build(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    sub_dir = Path(args.out)
    sub_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Copy YOLO weights ──────────────────────────────────────────────────
    yolo_src = Path(args.yolo)
    yolo_dst = sub_dir / "yolo_weights.pt"
    shutil.copy2(yolo_src, yolo_dst)
    print(f"Copied YOLO weights: {yolo_dst} ({yolo_src.stat().st_size / 1e6:.1f}MB)")

    # ── 2. Copy DINOv2 weights ────────────────────────────────────────────────
    dino_src = Path(args.dinov2)
    dino_dst = sub_dir / "dinov2_vits14.pt"
    shutil.copy2(dino_src, dino_dst)
    print(f"Copied DINOv2 weights: {dino_dst} ({dino_src.stat().st_size / 1e6:.1f}MB)")

    # ── 3. Build reference embeddings ─────────────────────────────────────────
    print("\nBuilding reference embeddings ...")
    dinov2 = load_dinov2(dino_dst, device)

    with open(args.annotations) as f:
        data = json.load(f)
    with open(Path(args.product_images) / "metadata.json") as f:
        meta = json.load(f)

    name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                    for p in meta["products"]}

    ref_images = []
    ref_category_ids = []

    for cat in data["categories"]:
        cat_id = cat["id"]
        name   = cat["name"].strip().lower()
        code   = name_to_code.get(name)
        if not code:
            continue
        for angle in ANGLES:
            for ext in ["jpg", "jpeg", "png"]:
                img_path = Path(args.product_images) / code / f"{angle}.{ext}"
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        ref_images.append(img)
                        ref_category_ids.append(cat_id)
                    except Exception:
                        pass
                    break

    print(f"  {len(ref_images)} reference images for {len(set(ref_category_ids))} products")
    embeddings = embed(dinov2, ref_images, device)

    emb_path = sub_dir / "ref_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"  Saved embeddings: {emb_path} ({emb_path.stat().st_size / 1e6:.1f}MB)")

    cat_map_path = sub_dir / "category_map.json"
    with open(cat_map_path, "w") as f:
        json.dump({"category_ids": ref_category_ids}, f)
    print(f"  Saved category map: {cat_map_path}")

    # ── 4. run.py is already in submission/ — no copy needed ─────────────────
    print(f"\nrun.py already in place")

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("\n=== Submission directory contents ===")
    total = 0
    for f in sorted(sub_dir.iterdir()):
        size = f.stat().st_size / 1e6
        total += size
        print(f"  {f.name:<30} {size:.1f}MB")
    print(f"  {'TOTAL':<30} {total:.1f}MB")
    print(f"\nTo zip and submit:")
    print(f"  cd {sub_dir} && zip -r ../submission.zip . -x '.*' '__MACOSX/*'")
    print(f"  unzip -l ../submission.zip | head -10  # verify run.py is at root")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo",           default="weights/bestv2.pt")
    parser.add_argument("--dinov2",         default="weights/dinov2_vits14.pt")
    parser.add_argument("--annotations",    default="train/annotations.json")
    parser.add_argument("--product-images", default="NM_NGD_product_images")
    parser.add_argument("--out",            default="submission")
    args = parser.parse_args()
    build(args)
