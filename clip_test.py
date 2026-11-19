"""Zero-shot tile classification with CLIP (isolated venv).

Why: five hand-crafted approaches failed on reCAPTCHA classes COCO does not have
(crosswalk, stairs, chimneys). CLIP can score a tile against free-text labels, which
covers every reCAPTCHA class at once and needs no training.

Measured GT for `capture/1794677201938_3x3.jpg` ("crosswalks", 3x3) is tiles {2, 8}.

Run (isolated venv):
  ./.venv/Scripts/python.exe clip_test.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "capture")

# every reCAPTCHA class we have seen, as CLIP text labels
LABELS = {
    "crosswalk": ["a photo of a zebra crossing on a road",
                  "a photo of a pedestrian crosswalk with white stripes",
                  "a photo of a road with painted white stripes"],
    "stairs": ["a photo of a staircase", "a photo of steps"],
    "traffic light": ["a photo of a traffic light"],
    "car": ["a photo of a car"],
    "bus": ["a photo of a bus"],
    "motorcycle": ["a photo of a motorcycle"],
    "bicycle": ["a photo of a bicycle"],
    "fire hydrant": ["a photo of a fire hydrant"],
    "chimney": ["a photo of a chimney"],
    "palm tree": ["a photo of a palm tree"],
}
NEGATIVE = ["a photo of a road", "a photo of a building", "a photo of a person",
            "a photo of a sky", "a photo of a wall", "a photo of grass"]


def load_clip():
    model, _pre, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tok


def tile_prompts(task: str) -> tuple[str, list[str]]:
    t = task.lower()
    key = next((k for k in LABELS if k in t), None)
    if key is None:
        return "", []
    pos = LABELS[key]
    return key, pos + NEGATIVE


def score_grid(model, preprocess, tok, path: str, task: str):
    rows = cols = 3
    meta = os.path.join(CAP, os.path.basename(path)[:-4] + ".json")
    if os.path.exists(meta):
        d = json.load(open(meta, encoding="utf-8"))
        rows, cols = int(d["rows"]), int(d["cols"])
    key, prompts = tile_prompts(task)
    if not prompts:
        return key, [], []
    im = cv2.imread(path)
    h, w = im.shape[:2]
    th, tw = h // rows, w // cols
    tiles = []
    for r in range(rows):
        for c in range(cols):
            t = im[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            tiles.append(preprocess(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))).unsqueeze(0))
    batch = torch.cat(tiles)
    with torch.no_grad():
        ie = model.encode_image(batch)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = model.encode_text(tok(prompts))
        te = te / te.norm(dim=-1, keepdim=True)
        sims = (ie @ te.T).numpy()
    n_pos = len(LABELS[key])
    pos_score = sims[:, :n_pos].max(1)
    neg_score = sims[:, n_pos:].max(1)
    picks = [int(i) for i in np.where(pos_score > neg_score)[0]]
    return key, picks, [(round(float(a), 3), round(float(b), 3)) for a, b in zip(pos_score, neg_score)]


def main() -> int:
    model, preprocess, tok = load_clip()
    path = os.path.join(CAP, "1794677201938_3x3.jpg")
    key, picks, scores = score_grid(model, preprocess, tok, path, "Select all images with crosswalks")
    print(f"class={key} picks={picks} (GT [2, 8])")
    for i, (p, n) in enumerate(scores):
        print(f"  tile {i}: pos={p} neg={n} {'PICK' if i in picks else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())