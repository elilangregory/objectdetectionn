"""
Fine-tune DINOv2 with InfoNCE contrastive loss.

Teaches DINOv2 that a shelf crop of product X and a reference photo
of product X should have similar embeddings (closes the domain gap).

Training pairs:
    Anchor:   shelf crop from training images (bbox annotation)
    Positive: random reference image of the same product
    Negatives: reference images of all other products in the batch

Usage:
    python finetune_dinov2.py
    python finetune_dinov2.py --epochs 20 --batch 128 --lr 2e-5

Output: dinov2_finetuned.pt  (drop-in replacement for dinov2_vits14.pt)
"""

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm


# ── Augmentations ─────────────────────────────────────────────────────────────

# Strong augmentation for shelf crops — simulate real-world shelf conditions
CROP_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Light augmentation for reference images — they're already clean
REF_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShelfCropDataset(Dataset):
    def __init__(self, ann_path, images_dir, product_images_dir,
                 angles=("main", "front", "back", "left", "right")):
        with open(ann_path) as f:
            data = json.load(f)
        with open(Path(product_images_dir) / "metadata.json") as f:
            meta = json.load(f)

        name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                        for p in meta["products"]}

        self.cat_to_refs = {}
        for cat in data["categories"]:
            cat_id = cat["id"]
            code   = name_to_code.get(cat["name"].strip().lower())
            if not code:
                continue
            paths = []
            for angle in angles:
                for ext in ["jpg", "jpeg", "png"]:
                    p = Path(product_images_dir) / code / f"{angle}.{ext}"
                    if p.exists():
                        paths.append(p)
                        break
            if paths:
                self.cat_to_refs[cat_id] = paths

        img_lookup = {img["id"]: img for img in data["images"]}
        self.samples = []
        for ann in data["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in self.cat_to_refs:
                continue
            img_info = img_lookup.get(ann["image_id"])
            if not img_info:
                continue
            img_path = Path(images_dir) / img_info["file_name"]
            if not img_path.exists():
                continue
            x, y, w, h = ann["bbox"]
            if w < 5 or h < 5:
                continue
            self.samples.append((img_path, (x, y, w, h), cat_id))

        print(f"Dataset: {len(self.samples)} crops from "
              f"{len(self.cat_to_refs)} categories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (x, y, w, h), cat_id = self.samples[idx]

        img    = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        crop   = img.crop((x1, y1, x2, y2))
        crop_t = CROP_TRANSFORM(crop)

        ref_path = random.choice(self.cat_to_refs[cat_id])
        ref      = Image.open(ref_path).convert("RGB")
        ref_t    = REF_TRANSFORM(ref)

        return crop_t, ref_t, cat_id


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def infonce_loss(crop_embs, ref_embs, category_ids, temperature=0.07):
    sims     = crop_embs @ ref_embs.T / temperature
    cat_ids  = category_ids.unsqueeze(1)
    pos_mask = (cat_ids == cat_ids.T).float()
    sims     = sims - sims.max(dim=1, keepdim=True).values.detach()
    exp_sims = torch.exp(sims)
    log_prob = sims - torch.log(exp_sims.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    return (-(log_prob * pos_mask).sum(dim=1) / pos_count).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = ShelfCropDataset(
        ann_path=args.annotations,
        images_dir=args.images,
        product_images_dir=args.product_images,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    print(f"Loading DINOv2 from {args.weights} ...")
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        dynamic_img_size=True,
    )
    state = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    # Freeze first 9 blocks, train last 3 + norm
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(f"blocks.{i}" in name for i in range(9, 12)) or "norm" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    # Mixed precision — faster on both MPS and CUDA
    use_amp  = device in ("cuda", "mps")
    dtype    = torch.bfloat16 if device == "mps" else torch.float16
    scaler   = torch.amp.GradScaler(enabled=(device == "cuda"))

    print(f"Mixed precision: {use_amp} ({dtype})")
    print(f"\nTraining for {args.epochs} epochs ...\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for step, (crops, refs, cat_ids) in enumerate(loader):
            crops   = crops.to(device, non_blocking=True)
            refs    = refs.to(device, non_blocking=True)
            cat_ids = cat_ids.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
                crop_embs = F.normalize(model(crops), dim=-1)
                ref_embs  = F.normalize(model(refs),  dim=-1)
                loss      = infonce_loss(crop_embs, ref_embs, cat_ids)

            optimizer.zero_grad()
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 20 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} "
                      f"step {step}/{len(loader)} "
                      f"loss={loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}\n")

        # Checkpoint after every epoch
        ckpt_path = Path(args.output).with_suffix(".ckpt.pt")
        torch.save({"epoch": epoch + 1, "state_dict": model.state_dict()}, ckpt_path)

    out_path = Path(args.output)
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1e6:.1f}MB)")
    print(f"\nNext: python build_submission.py --dinov2 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations",    default="../train/annotations_clean.json")
    parser.add_argument("--images",         default="../train/images")
    parser.add_argument("--product-images", dest="product_images",
                                            default="../NM_NGD_product_images")
    parser.add_argument("--weights",        default="../weights/dinov2_vits14.pt")
    parser.add_argument("--output",         default="../dinov2_finetuned.pt")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch",          type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-5)
    args = parser.parse_args()
    train(args)
