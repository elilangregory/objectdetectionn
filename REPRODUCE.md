# Reproduce Final Submission (nr7)

All weights are included in the repo. No training needed.

## Quick start

```bash
git clone https://github.com/elilangregory/objectdetectionn.git
cd objectdetectionn
python -m venv venv
venv/bin/pip install torch torchvision timm==0.9.12 onnxruntime numpy Pillow opencv-python
cd submission_nr7
zip -r ../submission_nr7.zip . -x '.*' '__MACOSX/*'
```

`submission_nr7.zip` is the final submission.

## Validate locally (needs training images)

```bash
cd ..
venv/bin/python local_validate.py submission_nr7.zip --n 5
```

## Retrain from scratch

Requires data not in git:
```
train1/images/              # 248 training images
NM_NGD_product_images/      # Studio product photos
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
