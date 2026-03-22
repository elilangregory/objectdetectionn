"""
Fine-tune DINOv2 v3 — shelf crops + studio refs with architectural improvements.

Merges v1's working data strategy with v2's training improvements:
  FROM V1: ShelfCropDataset (anchor=shelf crop, positive=studio ref photo)
  FROM V2: Projection head, symmetric InfoNCE, LR warmup + cosine, grad clip, AMP

NEW: Excludes image_id=295 (bad annotations), 224x224, starts from base weights.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python agent_ws/finetune_dinov2_v3.py
    venv/bin/python agent_ws/finetune_dinov2_v3.py --epochs 30 --batch 64
"""

import argparse
import json
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


# IDs to exclude from training (bad annotations)
EXCLUDE_IMAGE_IDS = {295}

IMGSZ = 224

# ── Augmentations ─────────────────────────────────────────────────────────────

# Strong augmentation for shelf crops — simulate real-world shelf conditions
CROP_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMGSZ, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Light augmentation for reference images — they're already clean
REF_TRANSFORM = transforms.Compose([
    transforms.Resize((IMGSZ + 32, IMGSZ + 32)),
    transforms.RandomCrop(IMGSZ),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Projection head (from v2, discarded at inference) ────────────────────────

class ProjectionHead(nn.Module):
    """2-layer MLP: 384 -> 512 -> 128. Used only during training."""
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


# ── Dataset (from v1: shelf crops as anchors, studio refs as positives) ──────

class ShelfCropDataset(Dataset):
    def __init__(self, ann_path, images_dir, product_images_dir,
                 exclude_image_ids=None,
                 angles=("main", "front", "back", "left", "right")):
        with open(ann_path) as f:
            data = json.load(f)
        with open(Path(product_images_dir) / "metadata.json") as f:
            meta = json.load(f)

        exclude_image_ids = exclude_image_ids or set()

        name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                        for p in meta["products"]}

        # Build category -> reference image paths
        self.cat_to_refs = {}
        for cat in data["categories"]:
            cat_id = cat["id"]
            code = name_to_code.get(cat["name"].strip().lower())
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

        # Build samples: (image_path, bbox, category_id)
        img_lookup = {img["id"]: img for img in data["images"]}
        self.samples = []
        n_excluded = 0
        for ann in data["annotations"]:
            cat_id = ann["category_id"]
            if cat_id not in self.cat_to_refs:
                continue
            img_id = ann["image_id"]
            if img_id in exclude_image_ids:
                n_excluded += 1
                continue
            img_info = img_lookup.get(img_id)
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
        if n_excluded:
            print(f"  Excluded {n_excluded} annotations from image_ids {exclude_image_ids}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, (x, y, w, h), cat_id = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        crop = img.crop((x1, y1, x2, y2))
        crop_t = CROP_TRANSFORM(crop)

        ref_path = random.choice(self.cat_to_refs[cat_id])
        ref = Image.open(ref_path).convert("RGB")
        ref_t = REF_TRANSFORM(ref)

        return crop_t, ref_t, cat_id


# ── Losses (from v2: symmetric InfoNCE) ─────────────────────────────────────

def infonce_loss(anchor_embs, pos_embs, category_ids, temperature=0.07):
    """Standard InfoNCE: anchor -> positive direction."""
    sims = anchor_embs @ pos_embs.T / temperature
    cat_ids = category_ids.unsqueeze(1)
    pos_mask = (cat_ids == cat_ids.T).float()
    sims = sims - sims.max(dim=1, keepdim=True).values.detach()
    exp_sims = torch.exp(sims)
    log_prob = sims - torch.log(exp_sims.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    return (-(log_prob * pos_mask).sum(dim=1) / pos_count).mean()


def symmetric_loss(a_embs, p_embs, cat_ids, temperature=0.07):
    """Average of anchor->positive and positive->anchor InfoNCE."""
    l1 = infonce_loss(a_embs, p_embs, cat_ids, temperature)
    l2 = infonce_loss(p_embs, a_embs, cat_ids, temperature)
    return (l1 + l2) / 2


# ── LR schedule with warmup (from v2) ───────────────────────────────────────

def get_lr(step, total_steps, warmup_steps, base_lr):
    """Linear warmup then cosine annealing."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else \
             "mps"  if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Input size: {IMGSZ}x{IMGSZ}")

    dataset = ShelfCropDataset(
        ann_path=args.annotations,
        images_dir=args.images,
        product_images_dir=args.product_images,
        exclude_image_ids=EXCLUDE_IMAGE_IDS,
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

    # Load base DINOv2 weights (NOT finetuned — start fresh)
    print(f"\nLoading base DINOv2 ({args.model}) from {args.weights} ...")
    backbone = timm.create_model(
        args.model,
        pretrained=False,
        dynamic_img_size=True,
    )
    state = torch.load(args.weights, map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    backbone.load_state_dict(state, strict=False)
    backbone = backbone.to(device)

    # Freeze all, then unfreeze last 3 blocks + norm
    n_blocks = len(backbone.blocks)
    unfreeze_from = n_blocks - 3
    for param in backbone.parameters():
        param.requires_grad = False
    for name, param in backbone.named_parameters():
        if any(f"blocks.{i}" in name for i in range(unfreeze_from, n_blocks)) or "norm" in name:
            param.requires_grad = True
    print(f"Blocks: {n_blocks}, unfreezing last 3 ({unfreeze_from}-{n_blocks-1}) + norm")

    embed_dim = backbone.embed_dim  # 384 for ViT-S, 768 for ViT-B
    print(f"Embedding dim: {embed_dim}")
    proj_head = ProjectionHead(in_dim=embed_dim, hidden_dim=512, out_dim=128).to(device)

    trainable_bb = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    trainable_ph = sum(p.numel() for p in proj_head.parameters())
    total = sum(p.numel() for p in backbone.parameters())
    print(f"Trainable params: {trainable_bb + trainable_ph:,} "
          f"({trainable_bb:,} backbone + {trainable_ph:,} proj head) / {total:,} total")

    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, backbone.parameters())) +
        list(proj_head.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    total_steps = args.epochs * len(loader)
    warmup_steps = 2 * len(loader)  # 2 epoch warmup

    # Mixed precision
    use_amp = device in ("cuda", "mps")
    dtype = torch.bfloat16 if device == "mps" else torch.float16
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    print(f"Mixed precision: {use_amp} ({dtype})")
    print(f"LR warmup: {warmup_steps} steps (2 epochs)")
    print(f"Gradient clipping: max_norm=1.0")
    print(f"\nTraining {args.epochs} epochs, {len(loader)} steps/epoch ...\n")

    # Checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(args.epochs):
        backbone.train()
        proj_head.train()
        total_loss = 0.0

        for step, (crops, refs, cat_ids) in enumerate(loader):
            # Update LR manually (warmup + cosine)
            lr = get_lr(global_step, total_steps, warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            crops = crops.to(device, non_blocking=True)
            refs = refs.to(device, non_blocking=True)
            cat_ids = cat_ids.to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
                crop_embs = proj_head(backbone(crops))
                ref_embs = proj_head(backbone(refs))
                loss = symmetric_loss(crop_embs, ref_embs, cat_ids)

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

            total_loss += loss.item()
            global_step += 1

            if step % 10 == 0:
                elapsed = (torch.cuda.Event if device == "cuda" else type(None))
                import time
                if not hasattr(train, '_start_time'):
                    train._start_time = time.time()
                mins = (time.time() - train._start_time) / 60
                eta_steps = (args.epochs * len(loader)) - global_step
                sps = global_step / max(mins * 60, 1)
                eta_mins = eta_steps / max(sps, 0.01) / 60
                print(f"Epoch {epoch+1}/{args.epochs}  "
                      f"step {step}/{len(loader)}  "
                      f"loss={loss.item():.4f}  lr={lr:.2e}  "
                      f"[{mins:.1f}min elapsed, ~{eta_mins:.0f}min remaining]")

        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}")

        # Save checkpoint every 2 epochs (backbone only — proj head discarded)
        if (epoch + 1) % 2 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pt"
            torch.save({"epoch": epoch + 1, "state_dict": backbone.state_dict()}, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        print()

    # Save final backbone weights
    out_path = Path(args.output)
    torch.save(backbone.state_dict(), out_path)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"\nNext: python build_submission.py --dinov2 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DINOv2 v3: shelf crops + studio refs, "
                    "projection head, symmetric InfoNCE, warmup + cosine LR")
    parser.add_argument("--annotations",    default="train1/annotations.json")
    parser.add_argument("--images",         default="train1/images")
    parser.add_argument("--product-images", dest="product_images",
                                            default="NM_NGD_product_images")
    parser.add_argument("--model",          default="vit_base_patch14_dinov2.lvd142m",
                        help="timm model name")
    parser.add_argument("--weights",        default="weights/dinov2_vitb14.pt",
                        help="Base DINOv2 weights (NOT finetuned)")
    parser.add_argument("--output",         default="dinov2_v3_vitb.pt")
    parser.add_argument("--ckpt-dir",       dest="ckpt_dir",
                                            default="agent_ws/checkpoints/dinov2_v3_vitb")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch",          type=int,   default=32,
                        help="Batch size (32 for ViT-B on MPS/24GB)")
    parser.add_argument("--lr",             type=float, default=1e-5)
    args = parser.parse_args()
    train(args)
