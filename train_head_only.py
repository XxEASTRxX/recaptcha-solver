"""Train only the classifier head on frozen 57k features, using EXACT harvested GT.

The frozen-feature probe reached 71/80 = 89% exact on held-out 3x3 grids, so the head is the only
thing that needs fitting. Precompute the features once, fit the linear head with masked BCE (each
grid supervises only its own class), then save a full resnet18 checkpoint that
browserless.picks_by_tile_net can load.

  .venv/Scripts/python.exe train_head_only.py 3
  .venv/Scripts/python.exe train_head_only.py 4
"""
import json
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

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    manifest = json.load(open("gt_exact/manifest.json", encoding="utf-8"))
    for _m in manifest:
        _m["image"] = str(_m["image"]).replace("\\", "/")
    grids = [m for m in manifest if m["rows"] == ROWS]
    classes = sorted({B.MC_NORM.get(m["label"], m["label"]) for m in grids})
    print(f"{ROWS}x{ROWS} grids {len(grids)}, classes {classes}")
    net, _old_classes, tf = B._classifier()
    backbone = torch.nn.Sequential(*list(net.children())[:-1]).to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    head = nn.Linear(512, len(classes)).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()

    Xs, Ys, Cs = [], [], []
    with torch.no_grad():
        for m in grids:
            arr = cv2.imread(m["image"])
            hh, ww = arr.shape[:2]
            th, tw = hh // ROWS, ww // ROWS
            tiles = [arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
                     for r in range(ROWS) for c in range(ROWS)]
            x = torch.stack([tf(cv2.cvtColor(t, cv2.COLOR_BGR2RGB)) for t in tiles]).to(DEVICE)
            Xs.append(backbone(x).flatten(1))
            y = torch.zeros(ROWS * ROWS, len(classes))
            ci = classes.index(B.MC_NORM.get(m["label"], m["label"]))
            y[[i for i in m["picks"] if 0 <= i < ROWS * ROWS], ci] = 1.0
            Ys.append(y)
            Cs.append(ci)
    X = torch.cat(Xs)
    Y = torch.cat(Ys).to(DEVICE)
    C = torch.tensor(Cs).to(DEVICE)
    Ct = C.repeat_interleave(ROWS * ROWS)
    print(f"tiles {X.shape[0]}, positives {int(Y.sum())}")
    for ep in range(400):
        opt.zero_grad()
        logits = head(X)
        rows_ix = torch.arange(Y.shape[0])
        loss = lossf(logits[rows_ix, Ct], Y[rows_ix, Ct])
        loss.backward()
        opt.step()
        if ep % 100 == 99:
            print(f"  ep {ep + 1}: loss {float(loss):.4f}")
    net.fc = head
    torch.save({"state": net.state_dict(), "classes": classes},
               f"tile_cls_{ROWS}x{ROWS}.pt")
    print(f"saved tile_cls_{ROWS}x{ROWS}.pt")


if __name__ == "__main__":
    main()