"""Class-aware SAM labeling: COCO detector box -> SAM box prompt -> precise object mask -> tiles.

SAM alone is class-agnostic (largest-mask heuristics 1/10 on the exact grids). The detector knows
the class but its boxes over/under-extend; SAM refines each box to the object's pixels. This probe
validates the hybrid against the exact 4x4 grids.

  /home/eas2007arda/brain_env/bin/python sam_detector_probe.py [n]
"""
import json
import sys

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_320_fpn)

CKPT = "/home/eas2007arda/captcharesearch/sam_vit_b.pth"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
NORM = {"Bike": "bicycle", "bicycle": "bicycle", "Bus": "bus", "bus": "bus",
        "Fire hydrant": "fire hydrant", "fire hydrant": "fire hydrant",
        "Pedestrian crossing": "pedestrian crossing", "car": "car", "Car": "car",
        "motorcycle": "motorcycle", "traffic light": "traffic light"}


def tiles_of_mask(mask, rows=4, min_frac=0.15):
    h, w = mask.shape
    th, tw = h / rows, w / rows
    out = set()
    for r in range(rows):
        for c in range(rows):
            y0, y1 = int(r * th), int((r + 1) * th)
            x0, x1 = int(c * tw), int((c + 1) * tw)
            if mask[y0:y1, x0:x1].size and float(mask[y0:y1, x0:x1].mean()) >= min_frac:
                out.add(r * rows + c)
    return out


def main():
    manifest = json.load(open("gt_exact/manifest.json", encoding="utf-8"))
    for m in manifest:
        m["image"] = str(m["image"]).replace("\\", "/")
    grids = [m for m in manifest if m["rows"] == 4][:N]
    W = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    det = fasterrcnn_mobilenet_v3_large_320_fpn(weights=W).eval().to("cuda")
    cats = W.meta["categories"]
    sam = sam_model_registry["vit_b"](checkpoint=CKPT).to("cuda")
    pred = SamPredictor(sam)
    exact = {"sam_best": 0, "sam_union": 0, "det_boxes": 0}
    for m in grids:
        want = NORM.get(m["label"], m["label"])
        img = cv2.cvtColor(cv2.imread(m["image"]), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float().to("cuda") / 255.0
        with torch.no_grad():
            out = det([t])[0]
        boxes = [(float(s), [float(v) for v in b]) for s, l, b in
                 zip(out["scores"].tolist(), out["labels"].tolist(), out["boxes"].tolist())
                 if cats[l] == want and float(s) >= 0.30]
        exp = set(m["picks"])
        if not boxes:
            print(f"{m['image']} want={want} NO DETECTION", flush=True)
            continue
        # detector-box tiles (baseline)
        h, w = img.shape[:2]
        det_mask = np.zeros((h, w), bool)
        for _s, b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            det_mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
        if tiles_of_mask(det_mask) == exp:
            exact["det_boxes"] += 1
        # SAM refinement
        pred.set_image(img)
        best_union = np.zeros((h, w), bool)
        best_single, best_score = None, -1.0
        for _s, b in boxes:
            masks, scores, _ = pred.predict(box=np.array(b), multimask_output=True)
            k = int(np.argmax(scores))
            best_union |= masks[k]
            if float(scores[k]) > best_score:
                best_score, best_single = float(scores[k]), masks[k]
        if best_single is not None and tiles_of_mask(best_single) == exp:
            exact["sam_best"] += 1
        if tiles_of_mask(best_union) == exp:
            exact["sam_union"] += 1
        print(f"{m['image']} want={want} boxes={len(boxes)} exp={sorted(exp)} "
              f"sam_best={sorted(tiles_of_mask(best_single)) if best_single is not None else None}", flush=True)
    print("exact:", exact, "of", len(grids))


if __name__ == "__main__":
    main()