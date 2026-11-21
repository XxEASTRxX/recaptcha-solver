"""4x4 multicaptcha: mask-overlay whole-grid question.

Idea not yet measured: give the local VLM the full 450x450 photo with (a) the 4x4 tile grid drawn
and numbered, and (b) the COCO instance masks that Mask R-CNN found for the target class, drawn as
translucent overlays.  The mask is a partial-recall HINT (74% tile recall, 83% precision), and the
model's job is to correct it using the photo - which is exactly the "half of what the per-tile VLM
misses are partials" failure.  Scored exact against the 101 exact 4x4 grids.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import httpx
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange")
sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange/src")
os.environ.setdefault("HC_BRAIN_BASE", "http://127.0.0.1:1234")
from service.solver import _brain_json  # noqa: E402

MAN = os.path.join(ROOT, "gt_exact", "manifest.json")
COCO = {2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus", 10: "traffic light", 11: "fire hydrant"}
NORM = {"Bike": "bicycle", "bicycle": "bicycle", "bus": "bus", "Bus": "bus",
        "motorcycle": "motorcycle", "car": "car", "Car": "car",
        "traffic light": "traffic light", "Traffic light": "traffic light",
        "fire hydrant": "fire hydrant", "Fire hydrant": "fire hydrant"}
PROMPT = """This is a reCAPTCHA grid photo (tiles numbered 1-16, left to right, top to bottom) with
translucent RED overlays marking an object detector's guess. The detector misses partially cut-off
objects, so its overlay is only a hint - trust the photo.
TASK: Select all images with {label}. A tile counts if ANY part of a {label} is visible in it, even
a small fragment cut by the tile edge.
Answer STRICT JSON only: {{"tiles": [<numbers 1-16>], "why": "<short>"}}"""


def render(im, masks, rows=4, cols=4):
    arr = np.asarray(im).copy()
    h, w = arr.shape[:2]
    over = arr.copy()
    for m in masks:
        mm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        over[mm > 0] = (0.45 * over[mm > 0] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    arr = cv2.addWeighted(over, 0.55, arr, 0.45, 0)
    for r in range(1, rows):
        cv2.line(arr, (0, r * h // rows), (w, r * h // rows), (255, 255, 255), 2)
    for c in range(1, cols):
        cv2.line(arr, (c * w // cols, 0), (c * w // cols, h), (255, 255, 255), 2)
    for r in range(rows):
        for c in range(cols):
            n = r * cols + c + 1
            cv2.putText(arr, str(n), (c * w // cols + 4, r * h // rows + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(arr, str(n), (c * w // cols + 4, r * h // rows + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    return arr


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    man = [x for x in json.load(open(MAN, encoding="utf-8")) if x["rows"] == 4][:n]
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT").eval()
    brain = httpx.Client(timeout=120.0)
    tf = transforms.functional.pil_to_tensor
    hit = 0
    out = []
    for i, g in enumerate(man):
        im = Image.open(os.path.join(ROOT, g["image"])).convert("RGB")
        x = tf(im).float() / 255
        with torch.no_grad():
            o = model([x])[0]
        want = NORM.get(g["label"], g["label"].lower())
        masks = [m[0].numpy() for lab, sc, m in zip(o["labels"].tolist(), o["scores"].tolist(),
                                                    o["masks"]) if COCO.get(lab) == want and sc >= 0.4]
        vis = render(im, masks)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
        ans = _brain_json(brain, PROMPT.format(label=want), buf.tobytes(), max_tokens=400) or {}
        tiles = sorted({int(t) - 1 for t in (ans.get("tiles") or [])
                        if str(t).isdigit() and 1 <= int(t) <= 16})
        gt = set(g["picks"])
        good = set(tiles) == gt
        hit += good
        out.append({"image": g["image"], "label": want, "gt": sorted(gt), "pred": tiles, "ok": good})
        print(f"{i+1}/{len(man)} {want} gt={sorted(gt)} pred={tiles} "
              f"{'HIT' if good else 'miss'}", flush=True)
    print(f"mask-overlay whole-grid: {hit}/{len(man)} = {hit/max(1,len(man)):.0%}")
    json.dump(out, open(os.path.join(ROOT, "probe_maskoverlay_4x4.json"), "w"), indent=1)


if __name__ == "__main__":
    main()