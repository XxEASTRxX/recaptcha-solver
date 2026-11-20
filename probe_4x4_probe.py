"""Fast 4x4 exact-GT evaluation: frozen 57k features + per-class logistic, grid-level holdout.

Why frozen: 52 grids is small; a full fine-tune memorises. The linear probe answers the question
"do the pretrained features separate the target object from the rest when trained on exact labels?"
If yes, a small head is enough; if no, no head will help.

Run with the repo venv:
  .venv/Scripts/python.exe probe_4x4_probe.py
"""
import json
import random
import sys
import types

import cv2
import numpy as np
import torch

h = types.ModuleType("httpx")
h.Client = lambda *a, **k: None
sys.modules["httpx"] = h
import browserless as B  # noqa: E402

NORM = B.MC_NORM
ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def main():
    manifest = json.load(open("gt_exact/manifest.json", encoding="utf-8"))
    grids = [m for m in manifest if m["rows"] == ROWS]
    print(f"{ROWS}x{ROWS} grids: {len(grids)}")
    net, classes, tf = B._classifier()
    feat = torch.nn.Sequential(*list(net.children())[:-1]).eval()

    def features(m):
        arr = cv2.imread(m["image"])
        hh, ww = arr.shape[:2]
        th, tw = hh // ROWS, ww // ROWS
        tiles = [arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw] for r in range(ROWS) for c in range(ROWS)]
        with torch.no_grad():
            x = torch.stack([tf(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in tiles])
            return feat(x).flatten(1)

    # stratify by class, hold out 30%
    random.seed(11)
    by = {}
    for m in grids:
        by.setdefault(NORM.get(m["label"], m["label"]), []).append(m)
    train, test = [], []
    for cls, ms in by.items():
        random.shuffle(ms)
        k = max(1, int(round(len(ms) * 0.3))) if len(ms) >= 3 else 0
        test += ms[:k]
        train += ms[k:]
    print(f"train {len(train)}, holdout {len(test)}; holdout classes "
          f"{sorted({NORM.get(m['label'], m['label']) for m in test})}")

    # per-class logistic on frozen features (train grids only)
    models = {}
    for cls in sorted({NORM.get(m["label"], m["label"]) for m in train}):
        Xs, ys = [], []
        for m in train:
            if NORM.get(m["label"], m["label"]) != cls:
                continue
            Xs.append(features(m))
            y = torch.zeros(ROWS * ROWS)
            y[[i for i in m["picks"] if 0 <= i < ROWS * ROWS]] = 1.0
            ys.append(y)
        X = torch.cat(Xs)
        y = torch.cat(ys)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=0.2, max_iter=300)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y) + 1e-3 * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        models[cls] = (w.detach(), b.detach())
        print(f"  {cls}: {int(y.sum())} pos / {len(y)} tiles")

    for thr in (0.3, 0.4, 0.5, 0.6):
        exact = eval_n = 0
        for m in test:
            cls = NORM.get(m["label"], m["label"])
            if cls not in models:
                continue
            w, b = models[cls]
            p = torch.sigmoid(features(m) @ w + b).numpy()
            got = set(np.where(p >= thr)[0].tolist())
            if got == set(m["picks"]):
                exact += 1
            eval_n += 1
        print(f"holdout thr={thr}: exact {exact}/{eval_n}")


if __name__ == "__main__":
    main()