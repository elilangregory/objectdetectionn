"""
Train a linear classifier head on frozen DINOv2 embeddings.
Uses precomputed embeddings from cross_validate cache when available.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python agent_ws/train_linear_head.py
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
_pil_open = Image.open
from torchvision import transforms
try:
    from ultralytics import YOLO
    Image.open = _pil_open
except ImportError:
    pass
import timm

BATCH = 32
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


def load_dinov2(path, device, model_name="vit_small_patch14_dinov2.lvd142m"):
    model = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)
    state = torch.load(str(path), map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def embed_batch(model, images, device):
    all_embs = []
    for i in range(0, len(images), BATCH):
        batch_pil = images[i:i+BATCH]
        aug_embs = []
        for t in TTA_TRANSFORMS:
            with torch.no_grad():
                batch = torch.stack([t(x) for x in batch_pil]).to(device)
                e = F.normalize(model(batch), dim=-1)
            aug_embs.append(e)
        avg = F.normalize(torch.stack(aug_embs).mean(dim=0), dim=-1)
        all_embs.append(avg.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def main(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.annotations) as f:
        ann_data = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in ann_data["categories"]}
    num_classes = max(c["id"] for c in ann_data["categories"]) + 1
    print(f"Classes: {num_classes}")

    # Count annotations per category
    cat_counts = {}
    for ann in ann_data["annotations"]:
        cat_counts[ann["category_id"]] = cat_counts.get(ann["category_id"], 0) + 1

    # Load DINOv2
    print("Loading DINOv2 ...")
    dinov2 = load_dinov2(args.dinov2, device, model_name=args.model)

    # ── Extract shelf crop embeddings ──
    print("\nExtracting shelf crop embeddings ...")
    img_id_to_path = {}
    for img_info in ann_data["images"]:
        p = Path(args.images) / img_info["file_name"]
        if p.exists():
            img_id_to_path[img_info["id"]] = p

    # Split by image (80/20)
    all_img_ids = sorted(img_id_to_path.keys())
    np.random.seed(args.seed)
    np.random.shuffle(all_img_ids)
    split = int(0.8 * len(all_img_ids))
    train_img_ids = set(all_img_ids[:split])
    val_img_ids = set(all_img_ids[split:])
    print(f"  Train images: {len(train_img_ids)}, Val images: {len(val_img_ids)}")

    # Group annotations by image
    anns_by_img = {}
    for ann in ann_data["annotations"]:
        if ann["image_id"] == 295:  # bad annotations
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    train_embs, train_labels = [], []
    val_embs, val_labels = [], []
    loaded_imgs = {}

    for img_id in sorted(anns_by_img.keys()):
        if img_id not in img_id_to_path:
            continue
        is_train = img_id in train_img_ids

        if img_id not in loaded_imgs:
            try:
                loaded_imgs[img_id] = Image.open(img_id_to_path[img_id]).convert("RGB")
            except Exception:
                continue
        img = loaded_imgs[img_id]

        crops, labels = [], []
        for ann in anns_by_img[img_id]:
            x, y, w, h = ann["bbox"]
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.width, int(x+w)), min(img.height, int(y+h))
            if x2-x1 < 5 or y2-y1 < 5:
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            labels.append(ann["category_id"])

        if not crops:
            continue

        embs = embed_batch(dinov2, crops, device)
        if is_train:
            train_embs.append(embs)
            train_labels.extend(labels)
        else:
            val_embs.append(embs)
            val_labels.extend(labels)

    # Free image memory
    loaded_imgs.clear()

    # ── Add studio images to training set ──
    print("\nAdding studio images ...")
    with open(Path(args.product_images) / "metadata.json") as f:
        meta = json.load(f)
    name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                    for p in meta["products"]}

    studio_crops, studio_labels = [], []
    for cat in ann_data["categories"]:
        cat_id = cat["id"]
        if cat_id == 0:
            continue
        code = name_to_code.get(cat["name"].strip().lower())
        if not code:
            continue
        for angle in ANGLES:
            for ext in ["jpg", "jpeg", "png"]:
                p = Path(args.product_images) / code / f"{angle}.{ext}"
                if p.exists():
                    try:
                        studio_crops.append(Image.open(p).convert("RGB"))
                        studio_labels.append(cat_id)
                    except Exception:
                        pass
                    break

    if studio_crops:
        studio_embs = embed_batch(dinov2, studio_crops, device)
        train_embs.append(studio_embs)
        train_labels.extend(studio_labels)
    print(f"  Studio: {len(studio_crops)} images")

    # Concatenate
    train_embs = np.concatenate(train_embs, axis=0)
    val_embs = np.concatenate(val_embs, axis=0) if val_embs else np.zeros((0, 384))
    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)

    print(f"\nTraining samples: {len(train_embs)}")
    print(f"Validation samples: {len(val_embs)}")
    print(f"Unique train classes: {len(set(train_labels))}")
    print(f"Unique val classes: {len(set(val_labels))}")

    # ── Class weights (inverse sqrt frequency) ──
    class_weights = torch.ones(num_classes, device=device)
    for cid, count in cat_counts.items():
        if cid < num_classes:
            class_weights[cid] = 1.0 / (count ** 0.5)
    class_weights = class_weights / class_weights.mean()

    # ── Train linear head ──
    print(f"\nTraining linear head ({train_embs.shape[1]} → {num_classes}) ...")
    linear = nn.Linear(train_embs.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_X = torch.from_numpy(train_embs).to(device)
    train_Y = torch.from_numpy(train_labels).long().to(device)
    val_X = torch.from_numpy(val_embs).to(device) if len(val_embs) > 0 else None
    val_Y = torch.from_numpy(val_labels).long().to(device) if len(val_labels) > 0 else None

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        linear.train()
        perm = torch.randperm(len(train_X), device=device)
        total_loss, correct, total = 0.0, 0, 0
        for i in range(0, len(train_X), args.batch):
            idx = perm[i:i+args.batch]
            logits = linear(train_X[idx])
            loss = F.cross_entropy(logits, train_Y[idx], weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == train_Y[idx]).sum().item()
            total += len(idx)
        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # Val
        if val_X is not None and len(val_X) > 0:
            linear.eval()
            with torch.no_grad():
                val_logits = linear(val_X)
                val_loss = F.cross_entropy(val_logits, val_Y, weight=class_weights).item()
                val_preds = val_logits.argmax(1)
                val_acc = (val_preds == val_Y).float().mean().item()
        else:
            val_loss, val_acc = 0.0, 0.0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(linear.state_dict(), args.output)

    print(f"\nBest val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Saved to {args.output}")

    # ── Breakdown by frequency bucket ──
    if val_X is not None and len(val_X) > 0:
        linear.load_state_dict(torch.load(args.output, map_location=device))
        linear.eval()
        with torch.no_grad():
            val_preds = linear(val_X).argmax(1).cpu().numpy()
        val_labels_np = val_Y.cpu().numpy()

        buckets = {"1-5": [], "6-10": [], "11-50": [], "50+": []}
        for i, (pred, gt) in enumerate(zip(val_preds, val_labels_np)):
            count = cat_counts.get(gt, 0)
            if count <= 5:
                buckets["1-5"].append(pred == gt)
            elif count <= 10:
                buckets["6-10"].append(pred == gt)
            elif count <= 50:
                buckets["11-50"].append(pred == gt)
            else:
                buckets["50+"].append(pred == gt)

        print("\nVal accuracy by annotation frequency:")
        for name, correct_list in buckets.items():
            if correct_list:
                acc = sum(correct_list) / len(correct_list)
                print(f"  {name:>5} annotations: {acc:.4f} ({len(correct_list)} samples)")
            else:
                print(f"  {name:>5} annotations: no samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dinov2",         default="dinov2_finetuned.pt")
    parser.add_argument("--model",          default="vit_small_patch14_dinov2.lvd142m",
                        help="timm model name (vit_base_patch14_dinov2.lvd142m for ViT-B)")
    parser.add_argument("--annotations",    default="train1/annotations.json")
    parser.add_argument("--images",         default="train1/images")
    parser.add_argument("--product-images", dest="product_images",
                                            default="NM_NGD_product_images")
    parser.add_argument("--output",         default="agent_ws/linear_head.pt")
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--batch",          type=int, default=256)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()
    main(args)
