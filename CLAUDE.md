# NM i AI — Object Detection Agent

## Role
Autonomous competition agent. Think freely, write code, run experiments — no approval needed.
All dev work goes in `agent_ws/`. Only move files to root when explicitly told to.

## Competition
- Score: `0.7 × detection_mAP + 0.3 × classification_mAP`
- Sandbox: NVIDIA L4, CUDA 12.4, ultralytics==8.1.0, timm==0.9.12, no internet, 300s timeout, 420MB zip limit
- 6 submissions/day

## Stack
- `best_v1.pt` — YOLO single-class detector
- `dinov2_vits14.pt` — DINOv2 ViT-S/14 embeddings
- `submission/run.py` — sandbox entry point (forbidden: os, sys, subprocess, pickle, yaml)
- `build_submission.py` — rebuilds submission
- `local_validation.py` — times and validates locally (`--n 5` for quick check)

## Agent workspace rules
- Experiment freely in `agent_ws/` — create scripts, test ideas, iterate
- Use `venv/bin/python` for all commands
- When an experiment succeeds, report results and wait for instruction to promote to production
- Keep responses short — no summaries of what you just did

## Priorities (in order)
1. Fix ultralytics version mismatch — export YOLO to ONNX
2. Upgrade DINOv2 ViT-S → ViT-B
3. Fine-tune DINOv2 with contrastive loss
4. Add linear classifier head on top of DINOv2
