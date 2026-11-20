"""Offline 4x4 tile labeling with SAM (runs on the homeserver GPU).

The multicaptcha payload is ONE photo split 4x4; a passing round's picks are the exact true tile
set. SAM is class-agnostic, so the tile set is derived from the salient object's mask. This probe
tests mask-selection heuristics against the 101 exact grids.

  /home/eas2007arda/brain_env/bin/python sam_probe.py [n]
"""
import json
import sys

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

CKPT = "/home/eas2007arda/captcharesearch/sam_vit_b.pth"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def tiles_of_mask(mask, rows=4, min_frac=0.15):
    h, w = mask.shape
    th, tw = h / rows, w / rows
    out = set()
    for r in range(rows):
        for c in range(rows):
            y0, y1 = int(r * th), int((r + 1) * th)
            x0, x1 = int(c * tw), int((c + 1) * tw)
            frac = float(mask[y0:y1, x0:x1].mean()) if mask[y0:y1, x0:x1].size else 0.0
            if frac >= min_frac:
                out.add(r * rows + c)
    return out


def main():
    manifest = json.load(open("gt_exact/manifest.json", encoding="utf-8"))
    for m in manifest:
        m["image"] = str(m["image"]).replace("\\", "/")
    grids = [m for m in manifest if m["rows"] == 4][:N]
    sam = sam_model_registry["vit_b"](checkpoint=CKPT).to("cuda")
    gen = SamAutomaticMaskGenerator(sam, points_per_side=8, pred_iou_thresh=0.8,
                                    stability_score_thresh=0.9)
    heuristics = {"largest<=60%": 0, "largest5-60%": 0, "area*iou": 0, "center_containing": 0}
    total = 0
    for m in grids:
        img = cv2.cvtColor(cv2.imread(m["image"]), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        masks = gen.generate(img)
        exp = set(m["picks"])
        total += 1
        if not masks:
            continue
        area_lim = 0.6 * h * w
        cands = [x for x in masks if x["area"] <= area_lim]
        # H1: largest mask <= 60%
        if cands:
            best = max(cands, key=lambda x: x["area"])
            if tiles_of_mask(best["segmentation"]) == exp:
                heuristics["largest<=60%"] += 1
        # H2: largest in [5%, 60%]
        cands2 = [x for x in masks if 0.05 * h * w <= x["area"] <= area_lim]
        if cands2:
            best = max(cands2, key=lambda x: x["area"])
            if tiles_of_mask(best["segmentation"]) == exp:
                heuristics["largest5-60%"] += 1
        # H3: max area * predicted_iou
        best = max(cands or masks, key=lambda x: x["area"] * x["predicted_iou"])
        if tiles_of_mask(best["segmentation"]) == exp:
            heuristics["area*iou"] += 1
        # H4: mask containing the image centre
        cy, cx = h // 2, w // 2
        center = [x for x in masks if x["segmentation"][cy, cx]]
        if center:
            best = max(center, key=lambda x: x["area"])
            if tiles_of_mask(best["segmentation"]) == exp:
                heuristics["center_containing"] += 1
        print(f"{m['image']} exp={sorted(exp)} h1={sorted(tiles_of_mask(max(cands, key=lambda x: x['area'])['segmentation'])) if cands else None}", flush=True)
    print("heuristic exact:", heuristics, "of", total)


if __name__ == "__main__":
    main()