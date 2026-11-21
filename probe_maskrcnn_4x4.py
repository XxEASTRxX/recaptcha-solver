"""4x4 multicaptcha via COCO instance segmentation masks.

The 4x4 grid is ONE 450x450 photo split into 16 tiles; a tile counts when ANY part of the object
is visible (acceptance is exact).  A bounding box over-covers diagonal objects, which is why the
box->tile sweeps never found a separating threshold.  Mask R-CNN gives a per-instance MASK, so the
tile set is the tiles with >=T mask pixels.  Evaluated against the 101 exact 4x4 grids.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(ROOT, "gt_exact", "manifest.json")

COCO = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
        7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
        13: "bench", 17: "cat", 18: "dog"}
NORM = {"Bike": "bicycle", "bicycle": "bicycle", "bus": "bus", "Bus": "bus",
        "motorcycle": "motorcycle", "car": "car", "Car": "car",
        "traffic light": "traffic light", "Traffic light": "traffic light",
        "fire hydrant": "fire hydrant", "Fire hydrant": "fire hydrant"}


def main():
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    man = [x for x in json.load(open(MAN, encoding="utf-8")) if x["rows"] == 4][:rows]
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    tf = transforms.functional.pil_to_tensor
    out = []
    for i, g in enumerate(man):
        im = Image.open(os.path.join(ROOT, g["image"])).convert("RGB")
        x = tf(im).float() / 255
        with torch.no_grad():
            o = model([x])[0]
        want = NORM.get(g["label"], g["label"].lower())
        rec = {"image": g["image"], "label": want, "picks": g["picks"], "inst": []}
        for lab, sc, mask in zip(o["labels"].tolist(), o["scores"].tolist(), o["masks"]):
            if COCO.get(lab) != want or sc < 0.3:
                continue
            m = (mask[0].numpy() > 0.5)
            cov = []
            for r in range(4):
                for c in range(4):
                    y1, y2 = r * m.shape[0] // 4, (r + 1) * m.shape[0] // 4
                    x1, x2 = c * m.shape[1] // 4, (c + 1) * m.shape[1] // 4
                    cov.append(int(m[y1:y2, x1:x2].sum()))
            rec["inst"].append({"score": round(sc, 3), "cov": cov})
        out.append(rec)
        print(f"{i+1}/{len(man)} {want} gt={g['picks']} inst={len(rec['inst'])}", flush=True)
    json.dump(out, open(os.path.join(ROOT, "probe_maskrcnn_4x4.json"), "w"), indent=1)

    print("\nscore_thr pix_thr exact  tp fp fn")
    best = None
    for s_thr in (0.3, 0.5, 0.7):
        for pix in (0, 1, 3, 6, 12, 25, 50):
            ex = tp = fp = fn = 0
            for rec in out:
                sel = set()
                for inst in rec["inst"]:
                    if inst["score"] < s_thr:
                        continue
                    for t, c in enumerate(inst["cov"]):
                        if c >= pix:
                            sel.add(t)
                gt = set(rec["picks"])
                ex += (sel == gt)
                tp += len(sel & gt); fp += len(sel - gt); fn += len(gt - sel)
            print(f"{s_thr:9.1f} {pix:7d} {ex:5d}  {tp:3d} {fp:3d} {fn:3d}")
            if best is None or ex > best[0]:
                best = (ex, s_thr, pix)
    print("best", best)


if __name__ == "__main__":
    main()
