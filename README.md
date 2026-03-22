# NM i AI — Object Detection & Classification

**Score: 0.8945** (0.7 × detection_mAP + 0.3 × classification_mAP)

## Recreate the submission

All weights are pre-built and included. Just clone and zip:

```bash
git clone https://github.com/elilangregory/objectdetectionn.git
cd objectdetectionn/submission_nr7
zip -r ../submission_nr7.zip . -x '.*' '__MACOSX/*'
```

`submission_nr7.zip` is the final submission. See [REPRODUCE.md](REPRODUCE.md) for validation and retraining instructions.

## Solution

Two-stage pipeline for detecting and classifying grocery products on store shelves.

### Step 1 — Detection

Two YOLO detectors run on each image at different scales (960px and 768px). Their predictions are merged with Weighted Box Fusion, which combines overlapping boxes from both scales into one refined bounding box per product. This catches products at varying sizes and distances.

### Step 2 — Classification

Each detected product is cropped (with 12% context padding) and embedded with a fine-tuned DINOv2 ViT-S/14 (224px, with TTA: identity + horizontal flip). The 384-dim embedding is matched via cosine similarity against 355 pre-computed category centroids. Each centroid is the average embedding of all studio reference photos and training shelf crops for that product category. The closest match becomes the prediction. If cosine similarity is below 0.50, the detection is dropped.

### Why centroid matching?

We tested multiple classification approaches:

- **Individual reference matching** — noisy, one bad studio angle flips the result
- **Linear classifier head** — 90.5% accuracy on training data but overfits, looks worse on unseen images
- **Multiclass YOLO** — strong on common products but unreliable on rare ones, sometimes confidently wrong

Centroid matching scored best on cross-validation and looked most robust on real store images we photographed ourselves. It generalizes because it matches against domain-independent studio product photos rather than memorizing training data patterns. Averaging multiple reference images per category into one centroid smooths out noise from bad angles or lighting.

### How we fine-tuned DINOv2

Base DINOv2 ViT-S/14 was fine-tuned with contrastive learning (InfoNCE loss). Each training pair consists of a shelf crop (anchor, from training annotations) and a studio reference photo (positive, from NM_NGD product images) of the same product. This teaches the model that a product on a shelf and the same product in a studio photo should have similar embeddings — bridging the domain gap between the two image types.

## Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| DET_CONF | 0.15 | Detection confidence threshold |
| CLS_CONF | 0.50 | Centroid cosine similarity threshold |
| WBF_IOU | 0.55 | Weighted Box Fusion IoU threshold |
| CROP_CONTEXT | 0.12 | Padding around detected crops |

## Submission contents

| File | Size | Purpose |
|------|------|---------|
| `run.py` | <1KB | Inference pipeline |
| `third_medium_best.onnx` | 99MB | YOLO detector (960px) |
| `second_small_best.onnx` | 43MB | YOLO detector (768px) |
| `dinov2_weights.pt` | 84MB | Fine-tuned DINOv2 ViT-S/14 |
| `centroids.json` | 3MB | 355 category prototypes |
| **Total** | **229MB** | |
