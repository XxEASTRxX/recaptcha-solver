"""4x4 multicaptcha: per-tile VLM with NEIGHBOUR PADDING.

The per-tile VLM misses 32% of true partial tiles (recall 0.68).  The tile crop cuts the object at
the tile edge with no visual continuation, so a fragment (a wheel, a bumper) looks like noise.  This
arm feeds each tile with a 22% margin taken from the surrounding photo, with the tile boundary
drawn in yellow, and asks the same 4x4 partial-tile question.  Evaluated exact against the 101 exact
4x4 grids; compare with the shipped tight-tile prompt.
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import httpx
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange")
sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange/src")
os.environ.setdefault("HC_BRAIN_BASE", "http://127.0.0.1:1234")
from service.solver import _brain_json  # noqa: E402

MAN = os.path.join(ROOT, "gt_exact", "manifest.json")
NORM = {"Bike": "bicycle", "bicycle": "bicycle", "bus": "bus", "Bus": "bus",
        "motorcycle": "motorcycle", "car": "car", "Car": "car",
        "traffic light": "traffic light", "Traffic light": "traffic light",
        "fire hydrant": "fire hydrant", "Fire hydrant": "fire hydrant"}
PROMPT = """This is ONE tile (the area inside the YELLOW rectangle) cut from a larger photo in a
reCAPTCHA 4x4 grid, shown with a margin of the surrounding photo so you can see what continues past
the tile edge. TASK: "Select all squares with {label}".
A tile counts if ANY part of a {label} is visible INSIDE the yellow rectangle, even a small fragment
cut by the edge. Answer no if the yellow rectangle shows only sky, road surface, trees, buildings,
people, or a painted word/sign that is not the object.
Reply STRICT JSON only: {{"yes": true/false, "what": "brief"}}"""


def padded(im, r, c, rows=4, cols=4, margin=0.22):
    w, h = im.size
    tw, th = w / cols, h / rows
    x1, y1 = int(c * tw), int(r * th)
    x2, y2 = int((c + 1) * tw), int((r + 1) * th)
    mx, my = int(tw * margin), int(th * margin)
    ax1, ay1 = max(0, x1 - mx), max(0, y1 - my)
    ax2, ay2 = min(w, x2 + mx), min(h, y2 + my)
    crop = np.array(im.crop((ax1, ay1, ax2, ay2)))[:, :, ::-1].copy()
    cv2.rectangle(crop, (x1 - ax1, y1 - ay1), (x2 - ax1 - 1, y2 - ay1 - 1), (0, 255, 255), 2)
    return crop


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    man = [x for x in json.load(open(MAN, encoding="utf-8")) if x["rows"] == 4][:n]
    brain = httpx.Client(timeout=120.0)
    hit = 0
    out = []
    for i, g in enumerate(man):
        im = cv2.imread(os.path.join(ROOT, g["image"]))
        from PIL import Image
        pim = Image.open(os.path.join(ROOT, g["image"])).convert("RGB")
        want = NORM.get(g["label"], g["label"].lower())
        sel = []
        for t in range(16):
            r, c = divmod(t, 4)
            tile = padded(pim, r, c)
            ok, buf = cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
            ans = _brain_json(brain, PROMPT.format(label=want), buf.tobytes(), max_tokens=60) or {}
            if bool(ans.get("yes")):
                sel.append(t)
        gt = set(g["picks"])
        good = set(sel) == gt
        hit += good
        out.append({"image": g["image"], "label": want, "gt": sorted(gt), "pred": sel, "ok": good})
        print(f"{i+1}/{len(man)} {want} gt={sorted(gt)} pred={sel} {'HIT' if good else 'miss'}",
              flush=True)
    print(f"padded per-tile: {hit}/{len(man)} = {hit/max(1,len(man)):.0%}")
    json.dump(out, open(os.path.join(ROOT, "probe_padded_tiles.json"), "w"), indent=1)


if __name__ == "__main__":
    main()