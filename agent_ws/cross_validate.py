"""
5-fold cross-validation for detection + classification pipeline.
Uses studio-only refs to prevent data leakage.

Supports:
- Centroid matching vs individual ref matching
- Top-K weighted voting on individual refs
- Combined centroid + top-K mode
- cls_conf threshold sweep
- Configurable inference resolution

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python agent_ws/cross_validate.py
    venv/bin/python agent_ws/cross_validate.py --centroid --cls-conf-sweep
    venv/bin/python agent_ws/cross_validate.py --topk 10
    venv/bin/python agent_ws/cross_validate.py --centroid-topk --topk 10
    venv/bin/python agent_ws/cross_validate.py --imgsz 448
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
# Save original Image.open before ultralytics monkey-patches it
_pil_open = Image.open
from torchvision import transforms
from ultralytics import YOLO
# Restore original Image.open to avoid pi_heif errors
Image.open = _pil_open
import timm

ANGLES = ["main", "front", "back", "left", "right"]
BATCH = 32


def make_tta_transforms(imgsz):
    return [
        transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    ]


def load_model(path, device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                               pretrained=False, dynamic_img_size=True)
    state = torch.load(str(path), map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def embed(model, images, device, tta_transforms):
    out = []
    for i in range(0, len(images), BATCH):
        batch_pil = images[i:i+BATCH]
        aug_embs = []
        for t in tta_transforms:
            with torch.no_grad():
                batch = torch.stack([t(x) for x in batch_pil]).to(device)
                e = F.normalize(model(batch), dim=-1)
            aug_embs.append(e)
        avg = F.normalize(torch.stack(aug_embs).mean(dim=0), dim=-1)
        out.append(avg.cpu().numpy())
    return np.concatenate(out, axis=0)


def build_studio_refs(model, ann_path, product_images_dir, device, tta_transforms):
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
                    try:
                        images.append(Image.open(p).convert("RGB"))
                        cat_ids.append(cat["id"])
                    except Exception:
                        pass
                    break

    embs = embed(model, images, device, tta_transforms)
    return embs, cat_ids


def make_centroids(ref_embs, ref_cat_ids):
    """Compute mean embedding per category, L2-normalized."""
    cat_ids_unique = sorted(set(ref_cat_ids))
    centroids = []
    centroid_cat_ids = []
    for cid in cat_ids_unique:
        mask = [i for i, c in enumerate(ref_cat_ids) if c == cid]
        centroid = ref_embs[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
        centroid_cat_ids.append(cid)
    return np.array(centroids), centroid_cat_ids


def classify_topk(crop_embs, ref_embs, ref_cat_ids, k):
    """Top-K voting on individual refs, weighted by similarity."""
    sims = crop_embs @ ref_embs.T  # (n_crops, n_refs)
    ref_cat_arr = np.array(ref_cat_ids)
    results = []
    for row in sims:
        topk_idx = np.argsort(row)[-k:][::-1]
        topk_sims = row[topk_idx]
        topk_cats = ref_cat_arr[topk_idx]
        # Weighted vote by category
        cat_scores = {}
        for cid, s in zip(topk_cats, topk_sims):
            cid = int(cid)
            cat_scores[cid] = cat_scores.get(cid, 0.0) + float(s)
        winner = max(cat_scores, key=cat_scores.get)
        total_sim = float(np.sum(topk_sims))
        confidence = cat_scores[winner] / total_sim if total_sim > 0 else 0.0
        results.append((winner, confidence))
    return results


def classify(crop_embs, ref_embs, ref_cat_ids, use_centroids=False,
             topk=0, centroid_topk=False):
    if centroid_topk and topk > 0:
        # Combined mode: 0.5 * centroid_sim + 0.5 * topk_weighted_sim
        # Step 1: centroid match
        centroid_embs, centroid_ids = make_centroids(ref_embs, ref_cat_ids)
        cent_sims = crop_embs @ centroid_embs.T
        # Step 2: topk on individual refs
        ind_sims = crop_embs @ ref_embs.T
        ref_cat_arr = np.array(ref_cat_ids)
        results = []
        for cent_row, ind_row in zip(cent_sims, ind_sims):
            # Centroid top-1
            cent_best_idx = int(np.argmax(cent_row))
            cent_cat = centroid_ids[cent_best_idx]
            cent_sim = float(cent_row[cent_best_idx])
            # Top-K voting
            topk_idx = np.argsort(ind_row)[-topk:][::-1]
            topk_sims = ind_row[topk_idx]
            topk_cats = ref_cat_arr[topk_idx]
            cat_scores = {}
            for cid, s in zip(topk_cats, topk_sims):
                cid = int(cid)
                cat_scores[cid] = cat_scores.get(cid, 0.0) + float(s)
            topk_winner = max(cat_scores, key=cat_scores.get)
            total_topk_sim = float(np.sum(topk_sims))
            topk_conf = cat_scores[topk_winner] / total_topk_sim if total_topk_sim > 0 else 0.0
            # Combined: use centroid's category, blend scores
            # Final category = centroid's pick if centroid and topk agree, else topk winner
            # Score blend for the winning category
            if cent_cat == topk_winner:
                final_cat = cent_cat
                final_score = 0.5 * cent_sim + 0.5 * topk_conf
            else:
                # Both disagree — pick based on combined evidence
                # Get centroid sim for topk_winner
                topk_winner_cent_idx = None
                for ci, cid in enumerate(centroid_ids):
                    if cid == topk_winner:
                        topk_winner_cent_idx = ci
                        break
                topk_winner_cent_sim = float(cent_row[topk_winner_cent_idx]) if topk_winner_cent_idx is not None else 0.0
                # Get topk score for cent_cat
                cent_cat_topk_score = cat_scores.get(cent_cat, 0.0) / total_topk_sim if total_topk_sim > 0 else 0.0
                # Score both candidates
                cent_combined = 0.5 * cent_sim + 0.5 * cent_cat_topk_score
                topk_combined = 0.5 * topk_winner_cent_sim + 0.5 * topk_conf
                if cent_combined >= topk_combined:
                    final_cat = cent_cat
                    final_score = cent_combined
                else:
                    final_cat = topk_winner
                    final_score = topk_combined
            results.append((final_cat, final_score))
        return results
    elif topk > 0:
        return classify_topk(crop_embs, ref_embs, ref_cat_ids, topk)
    elif use_centroids:
        centroid_embs, centroid_ids = make_centroids(ref_embs, ref_cat_ids)
        sims = crop_embs @ centroid_embs.T
        results = []
        for row in sims:
            best_idx = int(np.argmax(row))
            results.append((centroid_ids[best_idx], float(row[best_idx])))
        return results
    else:
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
    n_gt = sum(len(gt_by_image.get(i, [])) for i in {p["image_id"] for p in preds})
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


def score_predictions(crops_meta, classifications, gt_by_image, cls_conf):
    preds = []
    dropped = 0
    for (img_id, bbox, det_score), (cat_id, sim) in zip(crops_meta, classifications):
        if sim < cls_conf:
            dropped += 1
            continue
        preds.append({"image_id": img_id, "category_id": cat_id,
                      "bbox": bbox, "score": det_score})
    det_map = compute_ap(preds, gt_by_image, match_category=False)
    cls_map = compute_ap(preds, gt_by_image, match_category=True)
    final = 0.7 * det_map + 0.3 * cls_map
    return det_map, cls_map, final, len(preds), dropped


def main(args):
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Inference resolution: {args.imgsz}px")
    if args.centroid_topk and args.topk > 0:
        match_mode = f"centroid-topk (k={args.topk})"
    elif args.topk > 0:
        match_mode = f"topk (k={args.topk})"
    elif args.centroid:
        match_mode = "centroid"
    else:
        match_mode = "individual refs"
    print(f"Matching: {match_mode}")

    tta_transforms = make_tta_transforms(args.imgsz)

    with open(args.annotations) as f:
        ann_data = json.load(f)
    gt_by_image = {}
    for ann in ann_data["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    # ── Cache logic: save/load embeddings to skip YOLO+DINOv2 on repeat runs ──
    cache_dir = Path("agent_ws/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"cv_{Path(args.dinov2).stem}_{args.imgsz}"
    cache_embs = cache_dir / f"{cache_key}_crop_embs.npy"
    cache_meta = cache_dir / f"{cache_key}_meta.json"
    cache_refs = cache_dir / f"{cache_key}_ref_embs.npy"
    cache_ref_ids = cache_dir / f"{cache_key}_ref_cat_ids.json"
    cache_img_ids = cache_dir / f"{cache_key}_img_ids.json"

    if cache_embs.exists() and cache_meta.exists() and cache_refs.exists() and not args.no_cache:
        print("Loading from cache ...")
        all_crop_embs = np.load(str(cache_embs))
        with open(cache_meta) as f:
            cached = json.load(f)
        all_meta_flat = [tuple(m) for m in cached["meta"]]
        ref_embs = np.load(str(cache_refs))
        with open(cache_ref_ids) as f:
            ref_cat_ids = json.load(f)
        with open(cache_img_ids) as f:
            cached_imgs = json.load(f)
        image_ids_ordered = cached_imgs["image_ids"]
        crop_idx = {int(k): tuple(v) for k, v in cached_imgs["crop_idx"].items()}
        print(f"  {len(all_crop_embs)} crop embeddings, {len(ref_embs)} ref embeddings")
    else:
        print("Loading YOLO ...")
        yolo = YOLO(args.yolo)

        print("Loading DINOv2 ...")
        model = load_model(args.dinov2, device)

        print("Building studio-only refs ...")
        ref_embs, ref_cat_ids = build_studio_refs(
            model, args.annotations, args.product_images, device, tta_transforms)
        print(f"  {len(ref_embs)} ref embeddings, {len(set(ref_cat_ids))} products")

        # Get all image paths
        image_paths = sorted(
            p for p in Path(args.images).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        print(f"Total images: {len(image_paths)}")

        # Detect all images once
        print("Detecting all images ...")
        all_crops_by_img = {}
        for img_path in image_paths:
            img_id = int(img_path.stem.split("_")[-1])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  Skipping {img_path.name}: {e}")
                all_crops_by_img[img_id] = ([], [])
                continue
            iw, ih = img.size
            res = yolo.predict(img, imgsz=1280, conf=args.conf, device=device, verbose=False)
            crops, meta = [], []
            for box, score in zip(res[0].boxes.xyxy.cpu().numpy(),
                                   res[0].boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = (max(0, int(box[0])), max(0, int(box[1])),
                                   min(iw, int(box[2])), min(ih, int(box[3])))
                if x2-x1 < 5 or y2-y1 < 5:
                    continue
                crops.append(img.crop((x1, y1, x2, y2)))
                meta.append((img_id, [x1, y1, x2-x1, y2-y1], float(score)))
            all_crops_by_img[img_id] = (crops, meta)
        print(f"Total crops: {sum(len(v[0]) for v in all_crops_by_img.values())}")

        # Embed all crops once
        print("Embedding all crops ...")
        all_crops_flat = []
        all_meta_flat = []
        image_ids_ordered = []
        for img_path in image_paths:
            img_id = int(img_path.stem.split("_")[-1])
            crops, meta = all_crops_by_img[img_id]
            all_crops_flat.extend(crops)
            all_meta_flat.extend(meta)
            image_ids_ordered.append(img_id)

        if all_crops_flat:
            all_crop_embs = embed(model, all_crops_flat, device, tta_transforms)
        else:
            all_crop_embs = np.zeros((0, ref_embs.shape[1]))

        # Build index: img_id → slice of crop embeddings
        crop_idx = {}
        pos = 0
        for img_path in image_paths:
            img_id = int(img_path.stem.split("_")[-1])
            n = len(all_crops_by_img[img_id][0])
            crop_idx[img_id] = (pos, pos + n)
            pos += n

        # Save cache
        print("Saving cache ...")
        np.save(str(cache_embs), all_crop_embs)
        with open(cache_meta, "w") as f:
            json.dump({"meta": all_meta_flat}, f)
        np.save(str(cache_refs), ref_embs)
        with open(cache_ref_ids, "w") as f:
            json.dump(ref_cat_ids, f)
        with open(cache_img_ids, "w") as f:
            json.dump({"image_ids": image_ids_ordered,
                        "crop_idx": {str(k): v for k, v in crop_idx.items()}}, f)
        print("Cache saved.")

    # Get image paths for fold splitting
    image_paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Total images: {len(image_paths)}")

    # Create 5 folds
    np.random.seed(args.seed)
    indices = np.random.permutation(len(image_paths))
    folds = np.array_split(indices, args.k)

    # Determine cls_conf values to sweep
    if args.cls_conf_sweep:
        cls_confs = [0.0, 0.3, 0.5, 0.6, 0.65, 0.70, 0.75, 0.80]
    else:
        cls_confs = [args.cls_conf]

    # Run cross-validation
    for cls_conf in cls_confs:
        fold_results = []

        for fold_i, fold_indices in enumerate(folds):
            val_img_ids = set()
            for idx in fold_indices:
                img_path = image_paths[idx]
                img_id = int(img_path.stem.split("_")[-1])
                val_img_ids.add(img_id)

            # Gather val crops
            val_meta = []
            val_emb_indices = []
            for img_id in val_img_ids:
                if img_id not in crop_idx:
                    continue
                start, end = crop_idx[img_id]
                val_meta.extend(all_meta_flat[start:end])
                val_emb_indices.extend(range(start, end))

            if not val_emb_indices:
                continue

            val_crop_embs = all_crop_embs[val_emb_indices]

            # Classify
            classifications = classify(val_crop_embs, ref_embs, ref_cat_ids,
                                       use_centroids=args.centroid,
                                       topk=args.topk,
                                       centroid_topk=args.centroid_topk)

            # Build GT for val images only
            val_gt = {iid: gt_by_image.get(iid, []) for iid in val_img_ids}

            det, cls, final, n_preds, n_dropped = score_predictions(
                val_meta, classifications, val_gt, cls_conf)
            fold_results.append((det, cls, final, len(val_img_ids), n_preds, n_dropped))

        # Summary
        avg_det = np.mean([r[0] for r in fold_results])
        avg_cls = np.mean([r[1] for r in fold_results])
        avg_final = np.mean([r[2] for r in fold_results])
        total_preds = sum(r[4] for r in fold_results)
        total_dropped = sum(r[5] for r in fold_results)

        if args.cls_conf_sweep:
            print(f"cls_conf={cls_conf:.2f}  det={avg_det:.4f}  cls={avg_cls:.4f}  "
                  f"final={avg_final:.4f}  preds={total_preds}  dropped={total_dropped}")
        else:
            print(f"\n{'='*60}")
            print(f"{'Fold':<8} {'Images':>8} {'Det mAP':>10} {'Cls mAP':>10} {'Final':>10}")
            print(f"{'-'*60}")
            for i, (det, cls, final, n_imgs, n_preds, n_dropped) in enumerate(fold_results):
                print(f"Fold {i+1:<3} {n_imgs:>8} {det:>10.4f} {cls:>10.4f} {final:>10.4f}")
            print(f"{'-'*60}")
            print(f"{'AVG':<8} {sum(r[3] for r in fold_results):>8} "
                  f"{avg_det:>10.4f} {avg_cls:>10.4f} {avg_final:>10.4f}")
            print(f"{'='*60}")
            print(f"Predictions: {total_preds}  Dropped: {total_dropped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo",           default="weights/bestv2.pt")
    parser.add_argument("--dinov2",         default="dinov2_finetuned.pt")
    parser.add_argument("--annotations",    default="train1/annotations.json")
    parser.add_argument("--images",         default="train1/images")
    parser.add_argument("--product-images", dest="product_images",
                                            default="NM_NGD_product_images")
    parser.add_argument("--conf",           type=float, default=0.25)
    parser.add_argument("--cls-conf",       dest="cls_conf", type=float, default=0.75)
    parser.add_argument("--cls-conf-sweep", dest="cls_conf_sweep", action="store_true")
    parser.add_argument("--centroid",       action="store_true",
                        help="Use category centroids instead of individual refs")
    parser.add_argument("--topk",           type=int, default=0,
                        help="Top-K voting on individual refs (0=disabled)")
    parser.add_argument("--centroid-topk",  dest="centroid_topk", action="store_true",
                        help="Combined centroid + top-K mode (requires --topk N)")
    parser.add_argument("--no-cache",       dest="no_cache", action="store_true",
                        help="Force recompute (ignore cached embeddings)")
    parser.add_argument("--imgsz",          type=int, default=224)
    parser.add_argument("--k",             type=int, default=5)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()
    main(args)
