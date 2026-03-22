"""
Annotation helper: run YOLO + DINOv2 + linear head on test images,
produce visualizations and a JSON template with pre-filled predictions.
The user corrects mistakes (set "correct" to true/false/new_category_id)
instead of annotating from scratch.

Usage:
    cd /Users/elilangregory/Documents/NM_AI/object_detection
    venv/bin/python scripts/annotation_helper.py --n 10
    venv/bin/python scripts/annotation_helper.py --n 5 --seed 123

Output:
    output_annotate/          — numbered visualization images + contact sheets
    output_annotate/annotations_template.json
"""

import argparse
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path

# IMPORTANT: save PIL Image.open before ultralytics monkey-patches it (pi_heif issue)
from PIL import Image
_pil_open = Image.open

from ultralytics import YOLO

# Restore original PIL Image.open
Image.open = _pil_open

import timm
from torchvision import transforms


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 64

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_dinov2(weights_path, device):
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        dynamic_img_size=True,
    )
    state = torch.load(str(weights_path), map_location=device, weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def load_linear_head(path, embed_dim=384, num_classes=356, device="cpu"):
    linear = nn.Linear(embed_dim, num_classes)
    linear.load_state_dict(torch.load(str(path), map_location=device))
    return linear.eval().to(device)


# ---------------------------------------------------------------------------
# Embedding + classification
# ---------------------------------------------------------------------------
def embed_all(model, crops, device):
    """Embed a list of PIL crops with TTA and return (N, D) numpy array."""
    all_embs = []
    for i in range(0, len(crops), BATCH_SIZE):
        batch_pil = crops[i : i + BATCH_SIZE]
        aug_embs = []
        for t in TTA_TRANSFORMS:
            batch = torch.stack([t(img) for img in batch_pil]).to(device)
            with torch.no_grad():
                emb = model(batch)
            aug_embs.append(F.normalize(emb, dim=-1))
        avg = F.normalize(torch.stack(aug_embs).mean(dim=0), dim=-1)
        all_embs.append(avg.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def classify_linear(crop_embs, linear, device):
    """Return list of (category_id, confidence) using the linear head."""
    embs_t = torch.from_numpy(crop_embs).to(device)
    with torch.no_grad():
        probs = torch.softmax(linear(embs_t), dim=1)
        cats = probs.argmax(dim=1).cpu().numpy()
        confs = probs.max(dim=1).values.cpu().numpy()
    return [(int(c), float(s)) for c, s in zip(cats, confs)]


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def confidence_color_bgr(conf):
    """Green >0.7, yellow >0.4, red otherwise (BGR)."""
    if conf > 0.7:
        return (0, 200, 0)      # green
    elif conf > 0.4:
        return (0, 200, 255)    # yellow
    else:
        return (0, 0, 220)      # red


def shorten(name, max_len=30):
    return name[: max_len - 1] + "\u2026" if len(name) > max_len else name


def draw_annotated_image(img_pil, detections, cat_id_to_name):
    """
    Draw numbered bounding boxes with category + confidence on the image.
    detections: list of dicts with keys idx, x1, y1, x2, y2, det_conf, cat_id, cls_conf
    Returns BGR numpy array (scaled to max 2400px on longest side).
    """
    iw, ih = img_pil.size
    scale = min(1.0, 2400 / max(iw, ih))
    disp_w, disp_h = int(iw * scale), int(ih * scale)
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (disp_w, disp_h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, scale * 0.55)
    thickness = max(1, int(scale * 2))

    for det in detections:
        sx1 = int(det["x1"] * scale)
        sy1 = int(det["y1"] * scale)
        sx2 = int(det["x2"] * scale)
        sy2 = int(det["y2"] * scale)
        color = confidence_color_bgr(det["cls_conf"])
        idx = det["idx"]
        cat_name = shorten(cat_id_to_name.get(det["cat_id"], f"id={det['cat_id']}"))
        label = f"#{idx} {cat_name} ({det['cls_conf']:.2f})"

        # Draw box
        cv2.rectangle(bgr, (sx1, sy1), (sx2, sy2), color, thickness)

        # Label background + text
        (tw, th_text), _ = cv2.getTextSize(label, font, font_scale, 1)
        ty = max(sy1 - 4, th_text + 4)
        cv2.rectangle(bgr, (sx1, ty - th_text - 4), (sx1 + tw + 4, ty + 2), color, -1)
        cv2.putText(bgr, label, (sx1 + 2, ty - 1), font, font_scale, (0, 0, 0), 1,
                    cv2.LINE_AA)

        # Number circle on the box (top-left corner)
        cx, cy = sx1 + 12, sy1 + 12
        cv2.circle(bgr, (cx, cy), 12, (255, 255, 255), -1)
        cv2.circle(bgr, (cx, cy), 12, color, 2)
        num_str = str(idx)
        (nw, nh), _ = cv2.getTextSize(num_str, font, 0.45, 1)
        cv2.putText(bgr, num_str, (cx - nw // 2, cy + nh // 2), font, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)

    return bgr


def make_contact_sheet(crops, detections, cat_id_to_name, thumb_size=180, cols=6):
    """
    Create a contact sheet showing each numbered crop individually at readable size.
    Returns BGR numpy array.
    """
    n = len(crops)
    if n == 0:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    rows = math.ceil(n / cols)
    label_h = 40  # space for text below each thumbnail
    cell_w = thumb_size
    cell_h = thumb_size + label_h
    sheet_w = cols * cell_w
    sheet_h = rows * cell_h
    sheet = np.full((sheet_h, sheet_w, 3), 40, dtype=np.uint8)  # dark background

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (crop_pil, det) in enumerate(zip(crops, detections)):
        row = i // cols
        col = i % cols
        ox = col * cell_w
        oy = row * cell_h

        # Resize crop to fit thumbnail, maintaining aspect ratio
        cw, ch = crop_pil.size
        ratio = min(thumb_size / cw, thumb_size / ch)
        new_w, new_h = int(cw * ratio), int(ch * ratio)
        crop_bgr = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
        crop_resized = cv2.resize(crop_bgr, (new_w, new_h))

        # Center in cell
        pad_x = (thumb_size - new_w) // 2
        pad_y = (thumb_size - new_h) // 2
        sheet[oy + pad_y : oy + pad_y + new_h, ox + pad_x : ox + pad_x + new_w] = crop_resized

        # Color-coded border
        color = confidence_color_bgr(det["cls_conf"])
        cv2.rectangle(sheet, (ox + 1, oy + 1), (ox + thumb_size - 2, oy + thumb_size - 2),
                      color, 2)

        # Number badge
        idx_str = f"#{det['idx']}"
        cv2.rectangle(sheet, (ox, oy), (ox + 30, oy + 18), color, -1)
        cv2.putText(sheet, idx_str, (ox + 2, oy + 14), font, 0.4, (0, 0, 0), 1,
                    cv2.LINE_AA)

        # Label below thumbnail
        cat_name = shorten(cat_id_to_name.get(det["cat_id"], f"id={det['cat_id']}"), 22)
        label = f"{cat_name}"
        conf_label = f"conf={det['cls_conf']:.2f}"
        cv2.putText(sheet, label, (ox + 2, oy + thumb_size + 14), font, 0.32,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(sheet, conf_label, (ox + 2, oy + thumb_size + 30), font, 0.32,
                    color, 1, cv2.LINE_AA)

    return sheet


# ---------------------------------------------------------------------------
# Extract image_id from filename
# ---------------------------------------------------------------------------
def image_id_from_filename(name):
    """Extract numeric id from filenames like IMG_3123.jpeg or img_00042.jpg."""
    stem = Path(name).stem
    # Try to extract trailing digits
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    if digits:
        return int(digits)
    # Fallback: hash
    return abs(hash(stem)) % 1_000_000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load categories
    with open(args.annotations) as f:
        ann_data = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in ann_data["categories"]}
    categories_list = ann_data["categories"]

    # Load models
    print("Loading YOLO ...")
    yolo = YOLO(args.yolo)

    print("Loading DINOv2 ...")
    dinov2 = load_dinov2(args.dinov2, device)

    print("Loading linear head ...")
    linear = load_linear_head(args.linear, embed_dim=384, num_classes=356, device=device)

    # Select images
    image_paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if args.n and args.n < len(image_paths):
        random.seed(args.seed)
        selected = random.sample(image_paths, args.n)
    else:
        selected = image_paths
    selected.sort()
    print(f"Processing {len(selected)} images ...")

    # Output
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON template structures
    json_images = []
    json_annotations = []
    ann_id_counter = 1
    total_detections = 0

    for img_path in selected:
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Skipping {img_path.name}: {e}")
            continue

        iw, ih = img_pil.size
        img_id = image_id_from_filename(img_path.name)
        print(f"\n{img_path.name}  ({iw}x{ih})  image_id={img_id}")

        json_images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": iw,
            "height": ih,
        })

        # Run YOLO detection
        results = yolo.predict(img_pil, imgsz=1280, conf=args.det_conf,
                               device=device, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()

        # Extract crops
        crops = []
        raw_meta = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(iw, int(x2)), min(ih, int(y2))
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            crops.append(img_pil.crop((x1, y1, x2, y2)))
            raw_meta.append((x1, y1, x2, y2, float(score)))

        if not crops:
            print("  No detections")
            continue

        total_detections += len(crops)

        # Classify with DINOv2 + linear head
        crop_embs = embed_all(dinov2, crops, device)
        classifications = classify_linear(crop_embs, linear, device)

        # Build detection list
        detections = []
        for i, ((x1, y1, x2, y2, det_conf), (cat_id, cls_conf)) in enumerate(
            zip(raw_meta, classifications)
        ):
            det = {
                "idx": i + 1,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "det_conf": det_conf,
                "cat_id": cat_id,
                "cls_conf": cls_conf,
            }
            detections.append(det)

            # COCO-style bbox: [x, y, width, height]
            bw = x2 - x1
            bh = y2 - y1
            json_annotations.append({
                "id": ann_id_counter,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, bw, bh],
                "score": round(cls_conf, 4),
                "det_score": round(det_conf, 4),
                "category_name": cat_id_to_name.get(cat_id, f"id={cat_id}"),
                "correct": None,  # <-- user fills this: true / false / new_category_id
            })
            ann_id_counter += 1

        # Draw annotated full image
        vis_img = draw_annotated_image(img_pil, detections, cat_id_to_name)
        vis_path = out_dir / f"{img_path.stem}_annotated.jpg"
        cv2.imwrite(str(vis_path), vis_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Draw contact sheet of numbered crops
        sheet = make_contact_sheet(crops, detections, cat_id_to_name)
        sheet_path = out_dir / f"{img_path.stem}_crops.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])

        print(f"  {len(detections)} detections -> {vis_path.name} + {sheet_path.name}")

    # Write JSON template
    template = {
        "info": (
            "Annotation template. For each annotation, set 'correct' to: "
            "true (prediction is right), false (delete this detection), "
            "or a new category_id (integer) to fix the class."
        ),
        "images": json_images,
        "annotations": json_annotations,
        "categories": categories_list,
    }
    json_path = out_dir / "annotations_template.json"
    with open(json_path, "w") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(selected)} images, {total_detections} total detections")
    print(f"Visualizations : {out_dir}/")
    print(f"JSON template  : {json_path}")
    print(f"{'=' * 60}")
    print(
        "\nEdit annotations_template.json:\n"
        '  "correct": true    — prediction is correct\n'
        '  "correct": false   — delete this detection (false positive)\n'
        '  "correct": 42      — change category_id to 42\n'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate annotation templates from model predictions"
    )
    parser.add_argument("--yolo", default="weights/bestv2.pt",
                        help="YOLO weights path")
    parser.add_argument("--dinov2", default="dinov2_finetuned.pt",
                        help="DINOv2 fine-tuned weights")
    parser.add_argument("--linear", default="agent_ws/linear_head.pt",
                        help="Linear classifier head")
    parser.add_argument("--annotations", default="train1/annotations.json",
                        help="COCO annotations (for category names)")
    parser.add_argument("--images", default="test_no_annotations",
                        help="Directory of images to annotate")
    parser.add_argument("--out", default="output_annotate",
                        help="Output directory")
    parser.add_argument("--det-conf", dest="det_conf", type=float, default=0.25,
                        help="YOLO detection confidence threshold")
    parser.add_argument("--n", type=int, default=0,
                        help="Number of images to process (0 = all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image selection")
    args = parser.parse_args()
    run(args)
