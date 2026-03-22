"""
Visualize nr7 pipeline: two-scale YOLO + centroid classification.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python scripts/visualize_nr7.py --n 25
    venv/bin/python scripts/visualize_nr7.py --n 25 --det-conf 0.10 --cls-conf 0.45
"""

import argparse
import json
import random
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
import timm
import onnxruntime as ort

BATCH = 32
TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ]),
]

DETECTORS = [
    {"path": "submission_nr7/third_medium_best.onnx", "imgsz": 960},
    {"path": "submission_nr7/second_small_best.onnx", "imgsz": 768},
]


def letterbox_image(image, target_size):
    w, h = image.size
    ratio = min(target_size / float(w), target_size / float(h))
    rw, rh = int(round(w * ratio)), int(round(h * ratio))
    pad_x, pad_y = (target_size - rw) / 2.0, (target_size - rh) / 2.0
    resized = image.resize((rw, rh), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(resized, (int(round(pad_x - 0.1)), int(round(pad_y - 0.1))))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...], ratio, pad_x, pad_y


def deletterbox_boxes(boxes, ratio, pad_x, pad_y, w, h):
    r = boxes.copy()
    r[:, [0, 2]] = (r[:, [0, 2]] - pad_x) / max(ratio, 1e-8)
    r[:, [1, 3]] = (r[:, [1, 3]] - pad_y) / max(ratio, 1e-8)
    r[:, 0] = np.clip(r[:, 0], 0, w); r[:, 2] = np.clip(r[:, 2], 0, w)
    r[:, 1] = np.clip(r[:, 1], 0, h); r[:, 3] = np.clip(r[:, 3], 0, h)
    return r


def cxcywh_to_xyxy(boxes):
    r = np.zeros_like(boxes)
    r[:, 0] = boxes[:, 0] - boxes[:, 2] / 2; r[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    r[:, 2] = boxes[:, 0] + boxes[:, 2] / 2; r[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return r


def iou_vector(box, boxes):
    if len(boxes) == 0: return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    a2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(a1 + a2 - inter, 1e-8, None)


def nms(boxes, scores, iou_thresh, max_det=300):
    if len(boxes) == 0: return np.zeros((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if len(keep) >= max_det or order.size == 1: break
        remaining = order[1:]
        overlaps = iou_vector(boxes[i], boxes[remaining])
        order = remaining[overlaps < iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def weighted_box_fusion(boxes_list, scores_list, iou_threshold):
    if not boxes_list: return []
    order = np.argsort(np.asarray(scores_list))[::-1]
    clusters = []
    for idx in order.tolist():
        box, score = boxes_list[idx], float(scores_list[idx])
        match, best_iou = None, 0.0
        for c in clusters:
            overlap = float(iou_vector(box, c["box"][None, :])[0])
            if overlap >= iou_threshold and overlap > best_iou:
                best_iou = overlap; match = c
        if match is None:
            clusters.append({"box": box.copy(), "weight": score, "scores": [score]})
        else:
            w = match["weight"] + score
            match["box"] = (match["box"] * match["weight"] + box * score) / max(w, 1e-8)
            match["weight"] = w; match["scores"].append(score)
    return [{"box": c["box"], "score": max(c["scores"])} for c in
            sorted(clusters, key=lambda c: max(c["scores"]), reverse=True)]


def detect_single(session, input_name, image, imgsz, det_conf):
    w, h = image.size
    array, ratio, pad_x, pad_y = letterbox_image(image, imgsz)
    output = session.run(None, {input_name: array})[0]
    if output.ndim == 3: output = output[0]
    if output.shape[0] <= 8 and output.shape[1] > output.shape[0]: output = output.T
    raw_boxes = output[:, :4].astype(np.float32)
    raw_scores = output[:, 4:].astype(np.float32)
    scores = raw_scores.max(axis=1) if raw_scores.shape[1] > 1 else raw_scores[:, 0]
    boxes = cxcywh_to_xyxy(raw_boxes)
    boxes = deletterbox_boxes(boxes, ratio, pad_x, pad_y, w, h)
    keep = scores >= det_conf
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0: return [], []
    selected = nms(boxes, scores, 0.7)
    return [boxes[i] for i in selected if (boxes[i][2]-boxes[i][0])>=2 and (boxes[i][3]-boxes[i][1])>=2], \
           [float(scores[i]) for i in selected if (boxes[i][2]-boxes[i][0])>=2 and (boxes[i][3]-boxes[i][1])>=2]


def load_dinov2(path, device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, dynamic_img_size=True)
    state = torch.load(str(path), map_location=device, weights_only=False)
    if "state_dict" in state: state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def embed_all(model, images, device):
    out = []
    for i in range(0, len(images), BATCH):
        batch_pil = images[i:i+BATCH]
        aug = []
        for t in TTA_TRANSFORMS:
            with torch.no_grad():
                b = torch.stack([t(x) for x in batch_pil]).to(device)
                e = F.normalize(model(b), dim=-1)
            aug.append(e)
        out.append(F.normalize(torch.stack(aug).mean(0), dim=-1).cpu().numpy())
    return np.concatenate(out)


def shorten(name, max_len=28):
    return name[:max_len] + "…" if len(name) > max_len else name


def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open(args.annotations) as f:
        ann_data = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in ann_data["categories"]}

    print("Loading detectors ...")
    providers = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
    detectors = []
    for cfg in DETECTORS:
        sess = ort.InferenceSession(cfg["path"], providers=providers)
        detectors.append((sess, sess.get_inputs()[0].name, cfg["imgsz"]))

    print("Loading DINOv2 ...")
    dinov2 = load_dinov2("dinov2_finetuned.pt", device)

    print("Loading centroids ...")
    with open("submission_nr7/centroids.json") as f:
        cent_data = json.load(f)
    centroid_embs = np.array(cent_data["centroids"], dtype=np.float32)
    centroid_ids = cent_data["category_ids"]

    image_paths = sorted(p for p in Path(args.images).iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.n and args.n < len(image_paths):
        random.seed(args.seed)
        selected = random.sample(image_paths, args.n)
    else:
        selected = image_paths

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_det = 0

    for img_path in selected:
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"\nSkipping {img_path.name}: {e}")
            continue
        iw, ih = img_pil.size
        print(f"\n{img_path.name}  {iw}x{ih}")

        # Two-scale detection + WBF
        all_boxes, all_scores = [], []
        for sess, input_name, imgsz in detectors:
            boxes, scores = detect_single(sess, input_name, img_pil, imgsz, args.det_conf)
            all_boxes.extend(boxes)
            all_scores.extend(scores)

        if not all_boxes:
            print("  No detections")
            continue

        fused = weighted_box_fusion(all_boxes, all_scores, 0.55)

        # Crop with context
        crops, det_meta = [], []
        for det in fused:
            x1, y1, x2, y2 = det["box"].tolist()
            w, h = x2 - x1, y2 - y1
            px, py = w * 0.12, h * 0.12
            left = max(0, int(math.floor(x1 - px)))
            top = max(0, int(math.floor(y1 - py)))
            right = min(iw, int(math.ceil(x2 + px)))
            bottom = min(ih, int(math.ceil(y2 + py)))
            if right - left < 5 or bottom - top < 5: continue
            crops.append(img_pil.crop((left, top, right, bottom)))
            det_meta.append(det)

        if not crops:
            print("  No valid crops")
            continue

        # Centroid classification
        crop_embs = embed_all(dinov2, crops, device)
        sims = crop_embs @ centroid_embs.T

        # Draw
        scale = min(1.0, 2000 / max(iw, ih))
        disp_w, disp_h = int(iw * scale), int(ih * scale)
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (disp_w, disp_h))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.35, scale * 0.6)
        thickness = max(1, int(scale * 2))

        kept = 0
        for i, det in enumerate(det_meta):
            best_idx = int(np.argmax(sims[i]))
            best_sim = float(sims[i][best_idx])
            if best_sim < args.cls_conf:
                continue

            kept += 1
            cat_id = centroid_ids[best_idx]
            x1, y1, x2, y2 = det["box"].tolist()
            sx1, sy1 = int(x1 * scale), int(y1 * scale)
            sx2, sy2 = int(x2 * scale), int(y2 * scale)

            if best_sim > 0.6:
                color = (0, 200, 0)
            elif best_sim > 0.45:
                color = (0, 200, 255)
            else:
                color = (0, 80, 255)

            cv2.rectangle(bgr, (sx1, sy1), (sx2, sy2), color, thickness)
            name = shorten(cat_id_to_name.get(cat_id, f"id={cat_id}"))
            label = f"{name} ({best_sim:.2f})"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
            ty = max(sy1 - 4, th + 2)
            cv2.rectangle(bgr, (sx1, ty - th - 2), (sx1 + tw + 2, ty + 2), color, -1)
            cv2.putText(bgr, label, (sx1 + 1, ty), font, font_scale, (0, 0, 0), 1)

        total_det += kept
        cv2.putText(bgr, f"{img_path.name}  {iw}x{ih}  kept={kept}/{len(det_meta)}",
                    (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(bgr, f"det={args.det_conf} cls={args.cls_conf}",
                    (10, 55), font, 0.45, (200, 200, 200), 1)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), bgr)
        print(f"  Kept: {kept}/{len(det_meta)} → {out_path}")

    print(f"\nDone. {len(selected)} images, {total_det} total kept → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="train1/annotations.json")
    parser.add_argument("--images", default="test_no_annotations")
    parser.add_argument("--out", default="output_test_vis_nr7")
    parser.add_argument("--det-conf", dest="det_conf", type=float, default=0.15)
    parser.add_argument("--cls-conf", dest="cls_conf", type=float, default=0.50)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
