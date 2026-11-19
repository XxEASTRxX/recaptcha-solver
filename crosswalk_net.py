"""Crosswalk tile verifier: frozen ImageNet features + logistic head, LOO-CV by grid.

Why: the brain's whole-grid answer had 85% recall but only 63% precision on the 17
hand-labelled grids (it adds roughly one extra tile per grid). A verifier must keep the
recall and cut the extras. 40 positives is far too few to fine-tune, so the backbone stays
frozen and only a linear head is fitted; the metric that matters is leave-one-GRID-out,
because tiles from one grid share a scene and would leak otherwise.

    C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange/.venv/Scripts/python.exe crosswalk_net.py
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "crosswalk_gt.json")
CAP = os.path.join(HERE, "capture")


def load_tiles():
    gt = json.load(open(GT, encoding="utf-8"))
    xs, ys, gids, tids = [], [], [], []
    for gi, (name, g) in enumerate(sorted(gt.items())):
        im = cv2.imread(os.path.join(CAP, name + ".jpg"))
        if im is None:
            continue
        h, w = im.shape[:2]
        n = 3
        th, tw = h // n, w // n
        amb = set(g.get("amb") or [])
        pos = set(g["pos"])
        for r in range(n):
            for c in range(n):
                i = r * n + c
                if i in amb:
                    continue
                t = im[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
                xs.append(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
                ys.append(1 if i in pos else 0)
                gids.append(name)
                tids.append(i)
    return xs, np.array(ys), np.array(gids), np.array(tids)


def embed(model, xs, views=6):
    """Mean feature over light augmentations - cheap regularisation for 150 samples."""
    pre = ResNet18_Weights.DEFAULT.transforms()
    feats = []
    for img in xs:
        vs = []
        for k in range(views):
            v = img
            if k & 1:
                v = v[:, ::-1]
            if k & 2:
                v = np.clip(v.astype(np.int16) + (k - 3) * 8, 0, 255).astype(np.uint8)
            vs.append(pre(torch.from_numpy(np.ascontiguousarray(v)).permute(2, 0, 1)))
        with torch.no_grad():
            f = model(torch.stack(vs)).mean(0).numpy()
        feats.append(f)
    return np.stack(feats)


def main() -> int:
    xs, ys, gids, tids = load_tiles()
    print(f"tiles: {len(xs)}  positives: {int(ys.sum())}  grids: {len(set(gids))}")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    X = embed(model, xs)
    print("features:", X.shape)

    def fit_predict(Xtr, ytr, Xte):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        a = torch.from_numpy((Xtr - mu) / sd).float()
        b = torch.from_numpy((Xte - mu) / sd).float()
        t = torch.from_numpy(ytr).float().unsqueeze(1)
        w = torch.zeros(a.shape[1], 1, requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        pos_w = torch.tensor([(len(ytr) - ytr.sum()) / max(1, ytr.sum())])
        opt = torch.optim.LBFGS([w, bias], max_iter=400, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                a @ w + bias, t, pos_weight=pos_w)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            return (torch.sigmoid(b @ w + bias).squeeze(1) >= 0.5).numpy().astype(int)

    pred = np.zeros(len(ys), dtype=int)
    for g in sorted(set(gids)):
        tr, te = gids != g, gids == g
        pred[te] = fit_predict(X[tr], ys[tr], X[te])

    tp = int(((pred == 1) & (ys == 1)).sum())
    fp = int(((pred == 1) & (ys == 0)).sum())
    fn = int(((pred == 0) & (ys == 1)).sum())
    rec = tp / max(1, tp + fn)
    prec = tp / max(1, tp + fp)
    print(f"\nLOO-GRID-CV: recall {tp}/{tp+fn} = {rec:.1%}   precision {tp}/{tp+fp} = {prec:.1%}")
    print(f"  false positives {fp}, false negatives {fn}")
    for g in sorted(set(gids)):
        m = gids == g
        if ((pred[m] == 1) & (ys[m] == 0)).any() or ((pred[m] == 0) & (ys[m] == 1)).any():
            print(f"  {g}: GT={sorted(tids[m][ys[m]==1].tolist())} "
                  f"pred={sorted(tids[m][pred[m]==1].tolist())}")
    ok = rec >= 0.80 and prec >= 0.95
    print("\nmeets objective (>=80% recall, >=95% precision):", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
