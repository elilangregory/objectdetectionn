"""
Evaluate linear head vs centroid matching using cached CV embeddings.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python agent_ws/eval_linear_head.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


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


def main():
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    # Load cache
    cache_dir = Path("agent_ws/cache")
    cache_key = "cv_dinov2_finetuned_224"
    all_crop_embs = np.load(str(cache_dir / f"{cache_key}_crop_embs.npy"))
    with open(cache_dir / f"{cache_key}_meta.json") as f:
        all_meta_flat = [tuple(m) for m in json.load(f)["meta"]]
    ref_embs = np.load(str(cache_dir / f"{cache_key}_ref_embs.npy"))
    with open(cache_dir / f"{cache_key}_ref_cat_ids.json") as f:
        ref_cat_ids = json.load(f)
    with open(cache_dir / f"{cache_key}_img_ids.json") as f:
        cached_imgs = json.load(f)
    crop_idx = {int(k): tuple(v) for k, v in cached_imgs["crop_idx"].items()}

    print(f"Loaded: {len(all_crop_embs)} crops, {len(ref_embs)} refs")

    # Load GT
    with open("train1/annotations.json") as f:
        ann_data = json.load(f)
    gt_by_image = {}
    for ann in ann_data["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    # Load linear head
    num_classes = max(c["id"] for c in ann_data["categories"]) + 1
    linear = nn.Linear(all_crop_embs.shape[1], num_classes).to(device)
    linear.load_state_dict(torch.load("agent_ws/linear_head.pt", map_location=device))
    linear.eval()

    # Count annotations per category (for hybrid threshold)
    cat_counts = {}
    for ann in ann_data["annotations"]:
        cat_counts[ann["category_id"]] = cat_counts.get(ann["category_id"], 0) + 1

    # Centroid matching
    cat_ids_unique = sorted(set(ref_cat_ids))
    centroids = []
    for cid in cat_ids_unique:
        mask = [i for i, c in enumerate(ref_cat_ids) if c == cid]
        centroid = ref_embs[mask].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Classify all crops with each method
    embs_t = torch.from_numpy(all_crop_embs).to(device)

    # Method 1: Centroid
    cent_sims = all_crop_embs @ centroids.T
    centroid_results = []
    for row in cent_sims:
        best_idx = int(np.argmax(row))
        centroid_results.append((cat_ids_unique[best_idx], float(row[best_idx])))

    # Method 2: Linear head
    with torch.no_grad():
        logits = linear(embs_t)
        probs = torch.softmax(logits, dim=1)
        linear_cats = probs.argmax(dim=1).cpu().numpy()
        linear_confs = probs.max(dim=1).values.cpu().numpy()
    linear_results = [(int(c), float(s)) for c, s in zip(linear_cats, linear_confs)]

    # Method 3: Hybrid (linear for common, centroid for rare)
    hybrid_results = []
    for i, ((cent_cat, cent_sim), (lin_cat, lin_conf)) in enumerate(
            zip(centroid_results, linear_results)):
        lin_count = cat_counts.get(lin_cat, 0)
        if lin_count >= 50 and lin_conf > 0.3:
            hybrid_results.append((lin_cat, lin_conf))
        else:
            hybrid_results.append((cent_cat, cent_sim))

    # 5-fold CV evaluation
    image_paths = sorted(
        p for p in Path("train1/images").iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    folds = np.array_split(indices, 5)

    cls_confs = [0.0, 0.3, 0.5, 0.6]

    for method_name, results in [("centroid", centroid_results),
                                  ("linear", linear_results),
                                  ("hybrid", hybrid_results)]:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        for cls_conf in cls_confs:
            fold_scores = []
            for fold_indices in folds:
                val_img_ids = set()
                for idx in fold_indices:
                    img_id = int(image_paths[idx].stem.split("_")[-1])
                    val_img_ids.add(img_id)

                preds = []
                for img_id in val_img_ids:
                    if img_id not in crop_idx:
                        continue
                    start, end = crop_idx[img_id]
                    for j in range(start, end):
                        cat_id, sim = results[j]
                        if sim < cls_conf:
                            continue
                        meta = all_meta_flat[j]
                        preds.append({"image_id": meta[0], "category_id": cat_id,
                                      "bbox": meta[1], "score": meta[2]})

                val_gt = {iid: gt_by_image.get(iid, []) for iid in val_img_ids}
                det = compute_ap(preds, val_gt, match_category=False)
                cls = compute_ap(preds, val_gt, match_category=True)
                fold_scores.append((det, cls, 0.7*det + 0.3*cls))

            avg_det = np.mean([s[0] for s in fold_scores])
            avg_cls = np.mean([s[1] for s in fold_scores])
            avg_final = np.mean([s[2] for s in fold_scores])
            print(f"  cls_conf={cls_conf:.1f}  det={avg_det:.4f}  cls={avg_cls:.4f}  final={avg_final:.4f}")


if __name__ == "__main__":
    main()
