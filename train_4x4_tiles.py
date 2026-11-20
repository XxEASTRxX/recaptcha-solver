"""Train a 4x4 tile classifier on EXACT labels harvested from passing rounds.

Acceptance is exact (all-16 failed 0/18), so a passing multicaptcha round's picks are the true tile
set. Each grid supervises only its own class head (a non-picked tile is a true negative for that
class; other classes are unknown and masked out).

Inputs : gt_exact/manifest.json (rows==4 entries)
Output : tile_cls_{ROWS}x{ROWS}.pt  {state, classes}
Report : grid-level holdout exact rate, and a leave-one-grid-out estimate when the set is small.

Run with the repo venv:
  .venv/Scripts/python.exe train_4x4_tiles.py
"""
import json
import random
import sys
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

NORM = {"Bike": "bicycle", "bicycle": "bicycle", "Bus": "bus", "bus": "bus",
        "Fire hydrant": "fire hydrant", "fire hydrant": "fire hydrant",
        "Pedestrian crossing": "pedestrian crossing", "car": "car", "Car": "car",
        "motorcycle": "motorcycle", "traffic light": "traffic light"}
TF = T.Compose([T.ToTensor(), T.Resize((160, 160)),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def tiles_of(path, rows=ROWS):
    arr = cv2.imread(path)
    h, w = arr.shape[:2]
    th, tw = h // rows, w // rows
    out = []
    for r in range(rows):
        for c in range(rows):
            t = arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            out.append(tf_rgb(t))
    return out


def tf_rgb(t):
    return TF(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))


def build_model(n_classes, ckpt="tile_cls57k.pt"):
    net = resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    try:
        old = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in old["state"].items() if not k.startswith("fc.")}
        missing = net.load_state_dict(sd, strict=False)
        print(f"  backbone init from {ckpt}: missing={len(missing.missing_keys)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  backbone init skipped: {exc}")
    return net


def train(grids, classes, epochs=8, lr=3e-4):
    net = build_model(len(classes))
    opt = torch.optim.Adam([
        {"params": [p for n, p in net.named_parameters() if not n.startswith("fc.")], "lr": lr / 5},
        {"params": net.fc.parameters(), "lr": lr},
    ])
    lossf = nn.BCEWithLogitsLoss()
    net.train()
    for ep in range(epochs):
        random.shuffle(grids)
        total = 0.0
        for m in grids:
            cls = NORM.get(m["label"], m["label"])
            if cls not in classes:
                continue
            ci = classes.index(cls)
            x = torch.stack(tiles_of(m["image"]))
            y = torch.zeros(ROWS * ROWS, len(classes))
            picks = [i for i in m["picks"] if 0 <= i < ROWS * ROWS]
            y[picks, ci] = 1.0
            logits = net(x)
            loss = lossf(logits[:, ci], y[:, ci])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        if ep % 6 == 5 or ep == epochs - 1:
            print(f"  epoch {ep + 1}: loss {total / max(1, len(grids)):.3f}")
    net.eval()
    return net


@torch.no_grad()
def predict(net, m, classes, thr=0.5):
    cls = NORM.get(m["label"], m["label"])
    if cls not in classes:
        return None
    ci = classes.index(cls)
    p = torch.sigmoid(net(torch.stack(tiles_of(m["image"])))[:, ci]).numpy()
    return set(np.where(p >= thr)[0].tolist())


def main():
    manifest = json.load(open("gt_exact/manifest.json", encoding="utf-8"))
    grids = [m for m in manifest if m["rows"] == ROWS]
    print(f"4x4 grids: {len(grids)}")
    if len(grids) < 8:
        print("not enough grids")
        return
    classes = sorted({NORM.get(m["label"], m["label"]) for m in grids})
    print("classes:", classes)
    # grid-level holdout: last 30% by label-stratified order
    random.seed(7)
    by = {}
    for m in grids:
        by.setdefault(NORM.get(m["label"], m["label"]), []).append(m)
    train_g, test_g = [], []
    for cls, ms in by.items():
        random.shuffle(ms)
        k = max(1, int(round(len(ms) * 0.3)))
        test_g += ms[:k]
        train_g += ms[k:]
    print(f"train grids {len(train_g)}, holdout {len(test_g)}")
    net = train(train_g, classes)
    for thr in (0.4, 0.5, 0.6):
        exact = 0
        for m in test_g:
            got = predict(net, m, classes, thr)
            if got is not None and got == set(m["picks"]):
                exact += 1
        print(f"holdout thr={thr}: exact {exact}/{len(test_g)}")
    # save a model trained on all grids
    net = train(grids, classes, epochs=8)
    torch.save({"state": net.state_dict(), "classes": classes}, f"tile_cls_{ROWS}x{ROWS}.pt")
    print(f"saved tile_cls_{ROWS}x{ROWS}.pt")


if __name__ == "__main__":
    main()