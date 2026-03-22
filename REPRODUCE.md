# Reproduce Final Submission (nr7)

All weights and pre-computed data are included in the repo. **No training required.** The `submission_nr7/` directory contains everything needed to create the exact submission that scored 0.8945.

## Create submission (no training, no data needed)

```bash
git clone https://github.com/elilangregory/objectdetectionn.git
cd objectdetectionn/submission_nr7
zip -r ../submission_nr7.zip . -x '.*' '__MACOSX/*'
```

Done. `submission_nr7.zip` is the final submission, ready to upload.

## What's inside submission_nr7/

All files are pre-built and included in the repo:

| File | Size | What it is |
|------|------|-----------|
| `run.py` | <1KB | Inference pipeline (entry point) |
| `third_medium_best.onnx` | 99MB | YOLO detector (960px scale) |
| `second_small_best.onnx` | 43MB | YOLO detector (768px scale) |
| `dinov2_weights.pt` | 84MB | Fine-tuned DINOv2 ViT-S/14 |
| `centroids.json` | 3MB | 355 pre-computed category prototypes |

Nothing else is needed. No Python environment, no training data, no extra downloads.

## Validate locally (optional)

Requires a Python environment and training images (not in git):

```bash
python -m venv venv
venv/bin/pip install torch torchvision timm==0.9.12 onnxruntime numpy Pillow opencv-python
venv/bin/python local_validate.py submission_nr7.zip --n 5
```

## Retrain from scratch (optional)

Only needed if you want to train new weights instead of using the provided ones.

Requires data not in git:
```
train1/images/              # 248 training images
NM_NGD_product_images/      # Studio product photos (5 angles per product)
```

```bash
# 1. Fine-tune DINOv2
venv/bin/python scripts/finetune_dinov2.py --epochs 20 --batch 64

# 2. Build centroids
venv/bin/python scripts/build_centroids.py
cp centroids.json submission_nr7/
cp dinov2_finetuned.pt submission_nr7/dinov2_weights.pt

# 3. Zip
cd submission_nr7
zip -r ../submission_nr7.zip . -x '.*' '__MACOSX/*'
```
