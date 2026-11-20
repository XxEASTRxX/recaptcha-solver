"""4x4 tile classification with NEIGHBOUR CONTEXT.

A multicaptcha 4x4 is ONE photo split into 16 tiles, so a tile's identity depends on its neighbours
(a wheel is only a wheel next to the bike). The frozen linear probe (own features only) reaches 4/15
exact. This probe adds the mean feature of the 8 neighbours and a small MLP, trained with the same
exact-GT masked loss and the same 70/30 grid split as probe_4x4_probe.py.

  .venv/Scripts/python.exe probe_4x4_context.py
"""
import json
import random
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn as nn

h = types.ModuleType("httpx")
h.Client = lambda *a, **k: None
sys.modules["httpx"] = h
import browserless as B  # noqa: E402

NORM = B.MC_NORM
NEIGH = {}


def build_neigh():
    for r in range(4):
        for c in range(4):
            nb = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 4 and 0 <= cc < 4:
                        nb.append(rr * 4 + cc)
            NEIGH[r * 4 + c] = nb


def main():
    build_neigh()
    manifest = json.load(open("gt_exact/manifest.json", encoding="utf-8"))
    grids = [m for m in manifest if m["rows"] == 4]
    net, classes57, tf = B._classifier()
    backbone = torch.nn.Sequential(*list(net.children())[:-1]).eval()
    Xg = {}
    with torch.no_grad():
        for m in grids:
            arr = cv2.imread(m["image"])
            hh, ww = arr.shape[:2]
            th, tw = hh // 4, ww // 4
            tiles = [arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
                     for r in range(4) for c in range(4)]
            x = torch.stack([tf(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in tiles])
            Xg[m["image"]] = backbone(x).flatten(1)
    classes = sorted({NORM.get(m["label"], m["label"]) for m in grids})
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

    def feats(m):
        F = Xg[m["image"]]
        ctx = torch.stack([F[NEIGH[i]].mean(0) for i in range(16)])
        return torch.cat([F, ctx], 1)

    X = torch.cat([feats(m) for m in train])
    Y = torch.zeros(len(train) * 16, len(classes))
    C = torch.tensor([classes.index(NORM.get(m["label"], m["label"])) for m in train])
    for gi, mi in enumerate(train):
        ci = C[gi]
        Y[gi * 16 + torch.tensor([i for i in mi["picks"] if 0 <= i < 16]), ci] = 1.0
    Ct = C.repeat_interleave(16)
    rows = torch.arange(Y.shape[0])

    for hidden in (0, 64):
        model = (nn.Linear(X.shape[1], len(classes)) if hidden == 0
                 else nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                                    nn.Linear(hidden, len(classes))))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(400):
            opt.zero_grad()
            logits = model(X)
            loss = nn.functional.binary_cross_entropy_with_logits(logits[rows, Ct], Y[rows, Ct])
            loss.backward()
            opt.step()
        model.eval()
        print(f"hidden={hidden} final loss {float(loss):.4f}")
        for thr in (0.3, 0.4, 0.5, 0.6):
            exact = 0
            for m in test:
                ci = classes.index(NORM.get(m["label"], m["label"]))
                with torch.no_grad():
                    p = torch.sigmoid(model(feats(m))[:, ci]).numpy()
                got = set(np.where(p >= thr)[0].tolist())
                exact += got == set(m["picks"])
            print(f"  holdout thr={thr}: exact {exact}/{len(test)}")


if __name__ == "__main__":
    main()