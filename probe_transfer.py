"""Transfer probe: 57k-backbone features + per-class logistic probes trained on the 97 EXACT 3x3
grids, evaluated on the 13 EXACT 4x4 grids.

Acceptance is exact (all-16 failed 0/18), so a passing grid's picks ARE the true tile set. The 3x3
dynamic grids therefore give clean per-tile labels for their class; a 4x4 grid is only used for
evaluation. The earlier in-domain fine-tune (11/40 live) trained on weak passing-round labels; this
probe uses exact labels and a linear head, so it is a different experiment.

Run with the repo venv (torch/torchvision CPU):
  .venv/Scripts/python.exe probe_transfer.py
"""
import json
import os
import sys
import types

import cv2
import numpy as np
import torch

# browserless imports httpx (absent in this venv); only _classifier is needed.
h = types.ModuleType("httpx")
h.Client = lambda *a, **k: None
sys.modules["httpx"] = h
import browserless as B  # noqa: E402

NORM = {"Bike": "bicycle", "bicycle": "bicycle", "Bus": "bus", "bus": "bus",
        "Fire hydrant": "fire hydrant", "fire hydrant": "fire hydrant",
        "Pedestrian crossing": "pedestrian crossing", "car": "car", "Car": "car",
        "motorcycle": "motorcycle", "traffic light": "traffic light"}


def load_feature_net():
    net, classes, tf = B._classifier()
    feat = torch.nn.Sequential(*list(net.children())[:-1]).eval()
    return feat, tf


def grid_tiles(path, rows):
    arr = cv2.imread(path)
    h, w = arr.shape[:2]
    th, tw = h // rows, w // rows
    out = []
    for r in range(rows):
        for c in range(rows):
            out.append(arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw])
    return out


def feats(feat, tf, tiles):
    # Same channel order as the live path: cv2 decodes BGR, the classifier expects RGB.
    with torch.no_grad():
        x = torch.stack([tf(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in tiles])
        f = feat(x).flatten(1)
    return f


def train_logistic(X, y, steps=400, lr=0.1, l2=1e-3):
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=steps)
    Xt, yt = X, y

    def closure():
        opt.zero_grad()
        z = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def main():
    manifest = json.load(open("gt_exact/manifest.json", encoding="utf-8"))
    feat, tf = load_feature_net()
    train = [m for m in manifest if m["rows"] == 3]
    test = [m for m in manifest if m["rows"] == 4]
    Xs, ys, cs = [], [], []
    for m in train:
        cls = NORM.get(m["label"], m["label"])
        tiles = grid_tiles(m["image"], 3)
        f = feats(feat, tf, tiles)
        picks = set(m["picks"])
        for i in range(9):
            Xs.append(f[i])
            ys.append(1.0 if i in picks else 0.0)
            cs.append(cls)
    X = torch.stack(Xs)
    y = torch.tensor(ys)
    c = np.array(cs)
    print(f"train tiles {X.shape[0]}, positives {int(y.sum())}")
    models = {}
    for cls in sorted(set(cs)):
        mask = c == cls
        w, b = train_logistic(X[mask], y[mask])
        models[cls] = (w, b)
        print(f"  {cls}: {int(mask.sum())} tiles, {int(y[mask].sum())} pos")
    # evaluate on 4x4 grids
    print("\n4x4 evaluation (exact GT):")
    for thr in (0.3, 0.4, 0.5, 0.6):
        exact = 0
        for m in test:
            cls = NORM.get(m["label"], m["label"])
            if cls not in models:
                continue
            w, b = models[cls]
            f = feats(feat, tf, grid_tiles(m["image"], 4))
            p = torch.sigmoid(f @ w + b).numpy()
            got = set(np.where(p >= thr)[0].tolist())
            exp = set(m["picks"])
            ok = got == exp
            exact += ok
        print(f"  thr={thr}: exact {exact}/{len(test)}")


if __name__ == "__main__":
    main()
