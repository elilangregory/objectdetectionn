"""
Compare two DINOv2 weights side by side.
Builds refs and scores each model, prints a summary table.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python scripts/compare_models.py
    venv/bin/python scripts/compare_models.py --n 0   # all images
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import timm

ANGLES = ["main", "front", "back", "left", "right"]
BATCH  = 32

TRANSFORM = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORM_FLIP = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(path, device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                               pretrained=False, dynamic_img_size=True)
    state = torch.load(str(path), map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def embed(model, images, device):
    out = []
    for i in range(0, len(images), BATCH):
        batch_pil = images[i:i+BATCH]
        with torch.no_grad():
            t1 = torch.stack([TRANSFORM(x) for x in batch_pil]).to(device)
            t2 = torch.stack([TRANSFORM_FLIP(x) for x in batch_pil]).to(device)
            e1 = F.normalize(model(t1), dim=-1)
            e2 = F.normalize(model(t2), dim=-1)
            avg = F.normalize((e1 + e2) / 2, dim=-1)
        out.append(avg.cpu().numpy())
    return np.concatenate(out, axis=0)


def build_refs(model, ann_path, product_images_dir, device):
    with open(ann_path) as f:
        data = json.load(f)
    with open(Path(product_images_dir) / "metadata.json") as f:
        meta = json.load(f)

    name_to_code = {p["product_name"].strip().lower(): p["product_code"]
                    for p in meta["products"]}

    images, cat_ids = [], []
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
                    images.append(Image.open(p).convert("RGB"))
                    cat_ids.append(cat["id"])
                    break

    print(f"  {len(images)} ref images, {len(set(cat_ids))} products")
    embs = embed(model, images, device)
    return embs, cat_ids


def classify(crop_embs, ref_embs, ref_cat_ids):
    sims = crop_embs @ ref_embs.T
    results = []
    for row in sims:
        best = {}
        for sim, cid in zip(row, ref_cat_ids):
            if cid not in best or sim > best[cid]:
                best[cid] = float(sim)
        top = int(max(best, key=best.get))
        results.append((top, best[top]))
    return results


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2]); y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0


def compute_ap(predictions, gt_by_image, match_category=False):
    preds = sorted(predictions, key=lambda p: p["score"], reverse=True)
    n_gt  = sum(len(gt_by_image.get(i, [])) for i in {p["image_id"] for p in preds})
    if n_gt == 0:
        return 0.0
    matched = {}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, pred in enumerate(preds):
        iid = pred["image_id"]
        gts = gt_by_image.get(iid, [])
        matched.setdefault(iid, set())
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched[iid]:
                continue
            if match_category and pred["category_id"] != g["category_id"]:
                continue
            iou = box_iou(pred["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou = iou; best_j = j
        if best_iou >= 0.5 and best_j >= 0:
            tp[i] = 1; matched[iid].add(best_j)
        else:
            fp[i] = 1
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    rec = cum_tp / n_gt
    pre = cum_tp / (cum_tp + cum_fp + 1e-9)
    ap = sum(np.max(pre[rec >= t]) if (rec >= t).any() else 0.0
             for t in np.linspace(0, 1, 11)) / 11.0
    return ap


def score_model(name, model, yolo, crops_meta, all_crops, ref_embs, ref_ids,
                gt_by_image, cls_conf):
    print(f"\n--- {name} ---")
    print("  Embedding crops ...")
    crop_embs = embed(model, all_crops, next(model.parameters()).device)
    results   = classify(crop_embs, ref_embs, ref_ids)

    preds, dropped = [], 0
    for (img_id, bbox, det_score), (cat_id, sim) in zip(crops_meta, results):
        if sim < cls_conf:
            dropped += 1
            continue
        preds.append({"image_id": img_id, "category_id": cat_id,
                      "bbox": bbox, "score": det_score})

    det_map = compute_ap(preds, gt_by_image, match_category=False)
    cls_map = compute_ap(preds, gt_by_image, match_category=True)
    final   = 0.7 * det_map + 0.3 * cls_map

    print(f"  Predictions: {len(preds)}  (dropped {dropped} below cls_conf={cls_conf})")
    print(f"  Detection  mAP: {det_map:.4f}")
    print(f"  Classif.   mAP: {cls_map:.4f}")
    print(f"  Final score:    {final:.4f}")
    return det_map, cls_map, final


def main(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.annotations) as f:
        ann_data = json.load(f)
    gt_by_image = {}
    for ann in ann_data["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    print("\nLoading YOLO ...")
    yolo = YOLO(args.yolo)

    image_paths = sorted(p for p in Path(args.images).iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.n:
        image_paths = image_paths[:args.n]
    print(f"Running on {len(image_paths)} images ...")

    # Detect once — shared between both models
    print("Detecting ...")
    all_crops, crops_meta = [], []
    for img_path in image_paths:
        img_id = int(img_path.stem.split("_")[-1])
        img    = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        res    = yolo.predict(img, imgsz=1280, conf=args.conf,
                              device=device, verbose=False)
        for box, score in zip(res[0].boxes.xyxy.cpu().numpy(),
                               res[0].boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = (max(0,int(box[0])), max(0,int(box[1])),
                               min(iw,int(box[2])), min(ih,int(box[3])))
            if x2-x1 < 5 or y2-y1 < 5:
                continue
            all_crops.append(img.crop((x1, y1, x2, y2)))
            crops_meta.append((img_id, [x1, y1, x2-x1, y2-y1], float(score)))
    print(f"Total crops: {len(all_crops)}")

    results = {}
    for label, weights in [("finetuned_v1", args.model_a),
                            ("epoch_016",    args.model_b)]:
        print(f"\nLoading {label} from {weights} ...")
        model = load_model(weights, device)
        print("Building refs ...")
        ref_embs, ref_ids = build_refs(model, args.annotations,
                                       args.product_images, device)
        det, cls, final = score_model(label, model, yolo, crops_meta, all_crops,
                                      ref_embs, ref_ids, gt_by_image, args.cls_conf)
        results[label] = (det, cls, final)

    print("\n" + "="*55)
    print(f"{'Model':<20} {'Det mAP':>10} {'Cls mAP':>10} {'Final':>10}")
    print("-"*55)
    for label, (det, cls, final) in results.items():
        print(f"{label:<20} {det:>10.4f} {cls:>10.4f} {final:>10.4f}")
    print("="*55)

    best = max(results, key=lambda k: results[k][2])
    print(f"\nWinner: {best}  (final={results[best][2]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a",        default="dinov2_finetuned.pt",
                                            help="Model A (old finetuned)")
    parser.add_argument("--model-b",        default="checkpoints/dinov2_v2/epoch_016.pt",
                                            help="Model B (new epoch_016)")
    parser.add_argument("--yolo",           default="weights/bestv2.pt")
    parser.add_argument("--annotations",    default="train1/annotations.json")
    parser.add_argument("--images",         default="train1/images")
    parser.add_argument("--product-images", dest="product_images",
                                            default="NM_NGD_product_images")
    parser.add_argument("--conf",           type=float, default=0.25)
    parser.add_argument("--cls-conf",       type=float, default=0.75)
    parser.add_argument("--n",              type=int,   default=20,
                                            help="Images to test (0=all)")
    args = parser.parse_args()
    main(args)
