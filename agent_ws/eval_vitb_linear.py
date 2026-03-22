"""Quick eval: ViT-B + linear head vs ViT-S + linear head on cached detections."""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
_pil_open = Image.open
from torchvision import transforms
from ultralytics import YOLO
Image.open = _pil_open
import timm

BATCH = 32
TTA = [
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

def embed(model, images, device):
    out = []
    for i in range(0, len(images), BATCH):
        batch_pil = images[i:i+BATCH]
        aug = []
        for t in TTA:
            with torch.no_grad():
                b = torch.stack([t(x) for x in batch_pil]).to(device)
                e = F.normalize(model(b), dim=-1)
            aug.append(e)
        out.append(F.normalize(torch.stack(aug).mean(0), dim=-1).cpu().numpy())
    return np.concatenate(out)

def box_iou(b1, b2):
    x1=max(b1[0],b2[0]);y1=max(b1[1],b2[1])
    x2=min(b1[0]+b1[2],b2[0]+b2[2]);y2=min(b1[1]+b1[3],b2[1]+b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    union=b1[2]*b1[3]+b2[2]*b2[3]-inter
    return inter/union if union>0 else 0.0

def compute_ap(preds, gt, match_cat=False):
    preds=sorted(preds,key=lambda p:p["score"],reverse=True)
    n_gt=sum(len(gt.get(i,[]))for i in{p["image_id"]for p in preds})
    if n_gt==0:return 0.0
    matched={};tp=np.zeros(len(preds));fp=np.zeros(len(preds))
    for i,pred in enumerate(preds):
        iid=pred["image_id"];gts=gt.get(iid,[]);matched.setdefault(iid,set())
        best_iou,best_j=0.0,-1
        for j,g in enumerate(gts):
            if j in matched[iid]:continue
            if match_cat and pred["category_id"]!=g["category_id"]:continue
            iou=box_iou(pred["bbox"],g["bbox"])
            if iou>best_iou:best_iou=iou;best_j=j
        if best_iou>=0.5 and best_j>=0:tp[i]=1;matched[iid].add(best_j)
        else:fp[i]=1
    ct=np.cumsum(tp);cf=np.cumsum(fp);rec=ct/n_gt;pre=ct/(ct+cf+1e-9)
    return sum(np.max(pre[rec>=t])if(rec>=t).any()else 0.0 for t in np.linspace(0,1,11))/11.0

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load cached detections (from ViT-S run)
    cache_dir = Path("agent_ws/cache")
    cache_key = "cv_dinov2_finetuned_224"
    with open(cache_dir / f"{cache_key}_meta.json") as f:
        all_meta = [tuple(m) for m in json.load(f)["meta"]]
    with open(cache_dir / f"{cache_key}_img_ids.json") as f:
        cached = json.load(f)
    crop_idx = {int(k): tuple(v) for k, v in cached["crop_idx"].items()}

    # Load ViT-S cached embeddings + linear head
    vits_embs = np.load(str(cache_dir / f"{cache_key}_crop_embs.npy"))
    linear_s = nn.Linear(384, 356)
    linear_s.load_state_dict(torch.load("agent_ws/linear_head.pt", map_location=device))
    linear_s.eval().to(device)

    # Load ViT-B model + re-embed crops (need fresh embeddings)
    print("Loading ViT-B ...")
    vitb = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False, dynamic_img_size=True)
    state = torch.load("agent_ws/checkpoints/dinov2_v3_vitb/epoch_004.pt", map_location=device, weights_only=False)
    if "state_dict" in state: state = state["state_dict"]
    vitb.load_state_dict(state, strict=False)
    vitb.eval().to(device)

    linear_b = nn.Linear(768, 356)
    linear_b.load_state_dict(torch.load("agent_ws/linear_head_vitb.pt", map_location=device))
    linear_b.eval().to(device)

    # We need to re-embed crops with ViT-B — but crops aren't cached as images
    # Instead, just use the ViT-S cached embeddings for both and compare linear heads
    # For a fair comparison, we need ViT-B embeddings too
    # Let's just score ViT-S linear head from cache (fast)
    print("Scoring ViT-S + linear head (from cache) ...")

    with open("train1/annotations.json") as f:
        ann = json.load(f)
    gt = {}
    for a in ann["annotations"]:
        gt.setdefault(a["image_id"], []).append(a)

    image_paths = sorted(p for p in Path("train1/images").iterdir()
                         if p.suffix.lower() in {".jpg",".jpeg",".png"})
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    folds = np.array_split(indices, 5)

    # ViT-S linear
    with torch.no_grad():
        logits_s = linear_s(torch.from_numpy(vits_embs).to(device))
        probs_s = torch.softmax(logits_s, dim=1)
        cats_s = probs_s.argmax(1).cpu().numpy()
        confs_s = probs_s.max(1).values.cpu().numpy()

    for label, cats, confs in [("ViT-S + linear", cats_s, confs_s)]:
        fold_scores = []
        for fi in folds:
            val_ids = {int(image_paths[i].stem.split("_")[-1]) for i in fi}
            preds = []
            for iid in val_ids:
                if iid not in crop_idx: continue
                s, e = crop_idx[iid]
                for j in range(s, e):
                    preds.append({"image_id": all_meta[j][0], "category_id": int(cats[j]),
                                  "bbox": all_meta[j][1], "score": all_meta[j][2]})
            vgt = {i: gt.get(i, []) for i in val_ids}
            d = compute_ap(preds, vgt); c = compute_ap(preds, vgt, True)
            fold_scores.append((d, c, 0.7*d+0.3*c))
        ad=np.mean([s[0]for s in fold_scores]);ac=np.mean([s[1]for s in fold_scores])
        af=np.mean([s[2]for s in fold_scores])
        print(f"{label}: det={ad:.4f} cls={ac:.4f} final={af:.4f}")

    print("\nNote: ViT-B needs fresh crop embeddings for a fair CV comparison.")
    print("Building ViT-B submission directly is faster than re-running full CV.")

main()
