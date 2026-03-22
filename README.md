# NM i AI — Object Detection & Classification

## Solution Overview

Two-stage pipeline for detecting and classifying grocery products on store shelves.

**Score: 0.8945** (0.7 × detection_mAP + 0.3 × classification_mAP)

### Detection
Two YOLO detectors run at different scales (960px and 768px) and fuse results with Weighted Box Fusion. This catches products at varying sizes and distances.

### Classification
Each detected crop is embedded with a fine-tuned DINOv2 ViT-S/14 using TTA (identity + horizontal flip). The embedding is matched via cosine similarity against 355 category centroids — one prototype vector per product, averaged from studio reference photos and training shelf crops. Predictions below cosine similarity 0.50 are dropped.

### Why centroid matching?
We tested multiple classification approaches: individual reference matching, linear classifier heads, and multiclass YOLO. Centroid matching scored best on our cross-validation AND looked most robust on unseen test images. It generalizes because it matches against domain-independent studio photos rather than memorizing training data.

## Build

### Prerequisites
```
Python 3.11+
torch, torchvision, timm==0.9.12, ultralytics, onnxruntime, numpy, Pillow, opencv-python
```

### Required data (not in git)
```
train1/images/                  # 248 training images
NM_NGD_product_images/          # Studio product photos (5 angles per product)
```

### Step 1: Fine-tune DINOv2
Contrastive learning: shelf crop ↔ studio photo pairs close the domain gap.
```bash
venv/bin/python scripts/finetune_dinov2.py --epochs 20 --batch 64
# Output: dinov2_finetuned.pt
```

### Step 2: Build centroids
Embed studio photos + training crops, average per category.
```bash
venv/bin/python scripts/build_centroids.py
cp centroids.json submission_nr7/
```

### Step 3: Zip and validate
```bash
cd submission_nr7
zip -r ../submission_nr7.zip . -x '.*' '__MACOSX/*'
cd ..
venv/bin/python local_validate.py submission_nr7.zip --n 5
```

## Submission contents

| File | Size | Purpose |
|------|------|---------|
| run.py | <1KB | Pipeline entry point |
| third_medium_best.onnx | 99MB | YOLO detector (960px) |
| second_small_best.onnx | 43MB | YOLO detector (768px) |
| dinov2_weights.pt | 84MB | Fine-tuned DINOv2 ViT-S/14 |
| centroids.json | 3MB | 355 category prototypes |
| **Total** | **229MB** | |

## Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| DET_CONF | 0.15 | Detection confidence threshold |
| CLS_CONF | 0.50 | Centroid similarity threshold |
| WBF_IOU | 0.55 | Box fusion IoU threshold |
| CROP_CONTEXT | 0.12 | Padding around crops (12%) |

## Evaluation

5-fold cross-validation with studio-only refs (no data leakage):
```bash
venv/bin/python agent_ws/cross_validate.py --centroid --cls-conf 0.60
```

Visualize predictions on unlabeled images:
```bash
venv/bin/python scripts/visualize_nr7.py --n 25 --seed 42
```
