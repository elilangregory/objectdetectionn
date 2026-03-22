"""
Submission nr7 — Two-scale YOLO detection (WBF) + DINOv2 centroid classification.
No multiclass classifier — pure centroid matching.

Called by sandbox as:
    python run.py --input /data/images --output /output/predictions.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

SCRIPT_DIR = Path(__file__).resolve().parent
CLS_CONF = 0.50  # centroid similarity threshold
DET_CONF = 0.10
DET_IOU = 0.7
MAX_DET = 300
WBF_IOU = 0.55
CROP_CONTEXT = 0.12
BATCH_SIZE = 64

TTA_TRANSFORMS = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]

DETECTORS = [
    {"path": "third_medium_best.onnx", "imgsz": 960},
    {"path": "second_small_best.onnx", "imgsz": 768},
]


# ---------------------------------------------------------------------------
# YOLO detection utilities
# ---------------------------------------------------------------------------
def letterbox_image(image, target_size):
    width, height = image.size
    ratio = min(target_size / float(width), target_size / float(height))
    rw = int(round(width * ratio))
    rh = int(round(height * ratio))
    pad_x = (target_size - rw) / 2.0
    pad_y = (target_size - rh) / 2.0
    resized = image.resize((rw, rh), Image.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(resized, (int(round(pad_x - 0.1)), int(round(pad_y - 0.1))))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...], ratio, pad_x, pad_y


def deletterbox_boxes(boxes, ratio, pad_x, pad_y, w, h):
    result = boxes.copy()
    result[:, [0, 2]] = (result[:, [0, 2]] - pad_x) / max(ratio, 1e-8)
    result[:, [1, 3]] = (result[:, [1, 3]] - pad_y) / max(ratio, 1e-8)
    result[:, 0] = np.clip(result[:, 0], 0, w)
    result[:, 2] = np.clip(result[:, 2], 0, w)
    result[:, 1] = np.clip(result[:, 1], 0, h)
    result[:, 3] = np.clip(result[:, 3], 0, h)
    return result


def cxcywh_to_xyxy(boxes):
    result = np.zeros_like(boxes)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def iou_vector(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    a2 = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(a1 + a2 - inter, 1e-8, None)


def nms(boxes, scores, iou_thresh, max_det):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if len(keep) >= max_det or order.size == 1:
            break
        remaining = order[1:]
        overlaps = iou_vector(boxes[i], boxes[remaining])
        order = remaining[overlaps < iou_thresh]
    return np.asarray(keep, dtype=np.int64)


def weighted_box_fusion(boxes_list, scores_list, iou_threshold):
    if not boxes_list:
        return []
    order = np.argsort(np.asarray(scores_list))[::-1]
    clusters = []
    for idx in order.tolist():
        box = boxes_list[idx]
        score = float(scores_list[idx])
        match = None
        best_iou = 0.0
        for cluster in clusters:
            overlap = float(iou_vector(box, cluster["box"][None, :])[0])
            if overlap >= iou_threshold and overlap > best_iou:
                best_iou = overlap
                match = cluster
        if match is None:
            clusters.append({"box": box.copy(), "weight": score, "scores": [score]})
        else:
            w = match["weight"] + score
            match["box"] = (match["box"] * match["weight"] + box * score) / max(w, 1e-8)
            match["weight"] = w
            match["scores"].append(score)
    return [{"box": c["box"], "score": max(c["scores"])} for c in
            sorted(clusters, key=lambda c: max(c["scores"]), reverse=True)]


def detect_single(session, input_name, image, imgsz):
    w, h = image.size
    array, ratio, pad_x, pad_y = letterbox_image(image, imgsz)
    output = session.run(None, {input_name: array})[0]
    if output.ndim == 3:
        output = output[0]
    if output.shape[0] <= 8 and output.shape[1] > output.shape[0]:
        output = output.T
    raw_boxes = output[:, :4].astype(np.float32)
    raw_scores = output[:, 4:].astype(np.float32)
    scores = raw_scores.max(axis=1) if raw_scores.shape[1] > 1 else raw_scores[:, 0]
    boxes = cxcywh_to_xyxy(raw_boxes)
    boxes = deletterbox_boxes(boxes, ratio, pad_x, pad_y, w, h)
    keep = scores >= DET_CONF
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return [], []
    selected = nms(boxes, scores, DET_IOU, MAX_DET)
    out_boxes, out_scores = [], []
    for i in selected.tolist():
        if (boxes[i][2] - boxes[i][0]) < 2 or (boxes[i][3] - boxes[i][1]) < 2:
            continue
        out_boxes.append(boxes[i])
        out_scores.append(float(scores[i]))
    return out_boxes, out_scores


# ---------------------------------------------------------------------------
# DINOv2 centroid classification
# ---------------------------------------------------------------------------
def load_dinov2(device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                               pretrained=False, dynamic_img_size=True)
    state = torch.load(str(SCRIPT_DIR / "dinov2_weights.pt"),
                       map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def embed_all(model, crops, device):
    out = []
    for i in range(0, len(crops), BATCH_SIZE):
        batch_pil = crops[i:i + BATCH_SIZE]
        aug = []
        for t in TTA_TRANSFORMS:
            with torch.no_grad():
                b = torch.stack([t(x) for x in batch_pil]).to(device)
                e = F.normalize(model(b), dim=-1)
            aug.append(e)
        out.append(F.normalize(torch.stack(aug).mean(0), dim=-1).cpu().numpy())
    return np.concatenate(out)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box.tolist()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load detectors
    print("Loading detectors ...")
    detectors = []
    for cfg in DETECTORS:
        sess = ort.InferenceSession(str(SCRIPT_DIR / cfg["path"]), providers=providers)
        detectors.append((sess, sess.get_inputs()[0].name, cfg["imgsz"]))

    # Load DINOv2 + centroids
    print("Loading DINOv2 ...")
    dinov2 = load_dinov2(device)

    print("Loading centroids ...")
    with (SCRIPT_DIR / "centroids.json").open() as f:
        cent_data = json.load(f)
    centroid_embs = np.array(cent_data["centroids"], dtype=np.float32)
    centroid_ids = cent_data["category_ids"]

    # Process images
    input_dir = Path(args.input)
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Found {len(image_paths)} images")

    all_predictions = []

    for img_path in image_paths:
        image_id = int(img_path.stem.split("_")[-1])
        image = Image.open(img_path).convert("RGB")
        iw, ih = image.size

        # Two-scale detection + WBF
        all_boxes, all_scores = [], []
        for sess, input_name, imgsz in detectors:
            boxes, scores = detect_single(sess, input_name, image, imgsz)
            all_boxes.extend(boxes)
            all_scores.extend(scores)

        if not all_boxes:
            continue

        fused = weighted_box_fusion(all_boxes, all_scores, WBF_IOU)

        # Crop detections with context padding
        crops = []
        det_meta = []
        for det in fused:
            x1, y1, x2, y2 = det["box"].tolist()
            w, h = x2 - x1, y2 - y1
            px, py = w * CROP_CONTEXT, h * CROP_CONTEXT
            left = max(0, int(math.floor(x1 - px)))
            top = max(0, int(math.floor(y1 - py)))
            right = min(iw, int(math.ceil(x2 + px)))
            bottom = min(ih, int(math.ceil(y2 + py)))
            if right - left < 5 or bottom - top < 5:
                continue
            crops.append(image.crop((left, top, right, bottom)))
            det_meta.append(det)

        if not crops:
            continue

        # Classify via centroid matching
        crop_embs = embed_all(dinov2, crops, device)
        sims = crop_embs @ centroid_embs.T

        for i, det in enumerate(det_meta):
            best_idx = int(np.argmax(sims[i]))
            best_sim = float(sims[i][best_idx])

            if best_sim < CLS_CONF:
                continue

            category_id = centroid_ids[best_idx]
            final_score = float(det["score"]) * (0.75 + 0.25 * best_sim)

            all_predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": xyxy_to_xywh(det["box"]),
                "score": round(min(max(final_score, 0.0), 1.0), 4),
            })

    print(f"Predictions: {len(all_predictions)}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(all_predictions, f)


if __name__ == "__main__":
    main()
