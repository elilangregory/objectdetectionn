"""
Fine-tune DINOv2 v2 — studio photos only, heavy augmentation, 448x448.

Key differences from v1:
- NO shelf crops (dirty annotations excluded)
- Anchor = main.jpg with 2 different heavy augmentations (main is what shelves look like)
- Positive = different angle (front/back/left/right) with light aug
- Projection head (2-layer MLP) during training, discarded at inference
- Symmetric InfoNCE loss (anchor→pos + pos→anchor)
- LR warmup over first 2 epochs
- Gradient clipping for MPS stability
- 448x448 input to capture fine details (logos, text)
- batch=8 to fit in 24GB with larger images
- Start from clean base weights (dinov2_vits14.pt)

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python scripts/finetune_dinov2_v2.py
    venv/bin/python scripts/finetune_dinov2_v2.py --epochs 20 --batch 8
"""

import argparse
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import json


IMGSZ = 448

# ── Augmentations ─────────────────────────────────────────────────────────────

SHELF_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMGSZ, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.RandomPerspective(distortion_scale=0.3)
    ], p=0.4),
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.4,
        hue=0.1,
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 3.0))
    ], p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

CLEAN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMGSZ + 32, IMGSZ + 32)),
    transforms.RandomCrop(IMGSZ),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Projection head ───────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """2-layer MLP: 384 → 512 → 128. Used only during training."""
    def __init__(self, in_dim=384, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class StudioDataset(Dataset):
    """
    Each sample yields 2 anchors + 1 positive:
    - anchor1, anchor2 = two DIFFERENT heavy augmentations of main.jpg
    - positive         = a different angle with light aug

    main.jpg prioritised as anchor — products on shelves face front.
    """
    def __init__(self, product_images_dir):
        self.cat_to_main  = {}
        self.cat_to_other = {}

    def load_from_annotations(self, ann_path, product_images_dir,
                               angles=("main", "front", "back", "left", "right")):
        with open(ann_path) as f:
            data = json.load(f)
        with open(Path(product_images_dir) / "metadata.json") as f:
            meta = json.load(f)

        name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                        for p in meta["products"]}

        for cat in data["categories"]:
            cat_id = cat["id"]
            if cat_id == 0:
                continue
            code = name_to_code.get(cat["name"].strip().lower())
            if not code:
                continue

            all_paths = []
            for angle in angles:
                for ext in ["jpg", "jpeg", "png"]:
                    p = Path(product_images_dir) / code / f"{angle}.{ext}"
                    if p.exists():
                        all_paths.append((angle, p))
                        break

            if not all_paths:
                continue

            main_path   = next((p for a, p in all_paths if a == "main"), all_paths[0][1])
            other_paths = [p for a, p in all_paths if p != main_path]

            self.cat_to_main[cat_id]  = main_path
            self.cat_to_other[cat_id] = other_paths

        self.cat_ids = list(self.cat_to_main.keys())
        n_with_other = sum(1 for v in self.cat_to_other.values() if v)
        print(f"Studio dataset: {len(self.cat_ids)} products")
        print(f"  With extra angles (strong positives): {n_with_other}")
        print(f"  Anchor-only (main used as positive too): {len(self.cat_ids) - n_with_other}")

    def __len__(self):
        return len(self.cat_ids)

    def __getitem__(self, idx):
        cat_id   = self.cat_ids[idx]
        main_img = Image.open(self.cat_to_main[cat_id]).convert("RGB")
        anchor1  = SHELF_TRANSFORM(main_img)
        anchor2  = SHELF_TRANSFORM(main_img)

        others = self.cat_to_other[cat_id]
        pos_path = random.choice(others) if others else self.cat_to_main[cat_id]
        pos_img  = Image.open(pos_path).convert("RGB")
        positive = CLEAN_TRANSFORM(pos_img)

        return anchor1, anchor2, positive, cat_id


# ── Losses ────────────────────────────────────────────────────────────────────

def infonce_loss(anchor_embs, pos_embs, category_ids, temperature=0.07):
    sims      = anchor_embs @ pos_embs.T / temperature
    cat_ids   = category_ids.unsqueeze(1)
    pos_mask  = (cat_ids == cat_ids.T).float()
    sims      = sims - sims.max(dim=1, keepdim=True).values.detach()
    exp_sims  = torch.exp(sims)
    log_prob  = sims - torch.log(exp_sims.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    return (-(log_prob * pos_mask).sum(dim=1) / pos_count).mean()


def symmetric_loss(a_embs, p_embs, cat_ids, temperature=0.07):
    """Average of anchor→positive and positive→anchor InfoNCE."""
    l1 = infonce_loss(a_embs, p_embs, cat_ids, temperature)
    l2 = infonce_loss(p_embs, a_embs, cat_ids, temperature)
    return (l1 + l2) / 2


# ── LR schedule with warmup ───────────────────────────────────────────────────

def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Input size: {IMGSZ}x{IMGSZ}")
    print(f"Batch size: {args.batch} products → {args.batch * 2} anchor views/step")

    dataset = StudioDataset(args.product_images)
    dataset.load_from_annotations(args.annotations, args.product_images)

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True,
    )

    print(f"\nLoading base DINOv2 from {args.weights} ...")
    backbone = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        dynamic_img_size=True,
    )
    state = torch.load(args.weights, map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    backbone.load_state_dict(state, strict=False)
    backbone = backbone.to(device)

    # Freeze first 9 blocks, train last 3 + norm
    for param in backbone.parameters():
        param.requires_grad = False
    for name, param in backbone.named_parameters():
        if any(f"blocks.{i}" in name for i in range(9, 12)) or "norm" in name:
            param.requires_grad = True

    proj_head = ProjectionHead(in_dim=384, hidden_dim=512, out_dim=128).to(device)

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad) + \
                sum(p.numel() for p in proj_head.parameters())
    total     = sum(p.numel() for p in backbone.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} backbone + projection head")

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, backbone.parameters())) +
        list(proj_head.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps  = args.epochs * len(loader)
    warmup_steps = 2 * len(loader)  # 2 epoch warmup

    use_amp = device in ("cuda", "mps")
    dtype   = torch.bfloat16 if device == "mps" else torch.float16
    scaler  = torch.amp.GradScaler(enabled=(device == "cuda"))

    print(f"Mixed precision: {use_amp} ({dtype})")
    print(f"LR warmup: {warmup_steps} steps (2 epochs)")
    print(f"\nTraining {args.epochs} epochs, {len(loader)} steps/epoch ...\n")

    global_step = 0

    for epoch in range(args.epochs):
        backbone.train()
        proj_head.train()
        total_loss = 0.0

        for step, (anc1, anc2, positives, cat_ids) in enumerate(loader):
            # Update LR manually (warmup + cosine)
            lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            anchors = torch.cat([anc1, anc2], dim=0).to(device, non_blocking=True)
            pos_rep = torch.cat([positives, positives], dim=0).to(device, non_blocking=True)
            ids_rep = torch.cat([cat_ids, cat_ids], dim=0).to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
                anc_proj = proj_head(backbone(anchors))
                pos_proj = proj_head(backbone(pos_rep))
                loss     = symmetric_loss(anc_proj, pos_proj, ids_rep)

            optimizer.zero_grad()
            if device == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(proj_head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(proj_head.parameters()), 1.0)
                optimizer.step()

            total_loss  += loss.item()
            global_step += 1

            if step % 20 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}  "
                      f"step {step}/{len(loader)}  "
                      f"loss={loss.item():.4f}  lr={lr:.2e}")

        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save backbone only (projection head discarded at inference)
        ckpt_dir = Path("checkpoints/dinov2_v2")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
        torch.save({"epoch": epoch + 1, "state_dict": backbone.state_dict()}, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}\n")

    out_path = Path(args.output)
    torch.save(backbone.state_dict(), out_path)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations",    default="train1/annotations.json")
    parser.add_argument("--product-images", dest="product_images",
                                            default="NM_NGD_product_images")
    parser.add_argument("--weights",        default="weights/dinov2_vits14.pt")
    parser.add_argument("--output",         default="dinov2_v2.pt")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch",          type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=1e-5)
    args = parser.parse_args()
    train(args)
