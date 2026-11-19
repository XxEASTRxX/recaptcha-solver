"""Per-(tile, label) classifier for reCAPTCHA grids, trained on dumped passing rounds.

Passing rounds are free ground truth: the tiles we clicked were accepted by the server, so they are
positives and the rest are negatives. The point is to replace the STAGE-ONE candidate generator,
which the 100-round analysis showed is the bottleneck (verification cannot recover a tile that was
never proposed).

Leakage rule: tiles from one challenge share a scene, so the split is by GRID, never by tile.

    python tile_net.py --data bl_bulk --holdout 0.25
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rows(dirs):
    rows = []
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                meta = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            if not meta.get("pass") or not meta.get("picks"):
                continue
            img = cv2.imread(f[:-5] + ".jpg")
            if img is None:
                continue
            n = 4 if img.shape[0] > 300 else 3
            rows.append((f[:-5], img, n, meta["label"], set(meta["picks"])))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["bl_bulk", "bl_dump", "bl_dump2"])
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = load_rows([os.path.join(HERE, d) if not os.path.isabs(d) else d for d in a.data])
    labels = sorted({r[3] for r in rows})
    lab_ix = {l: i for i, l in enumerate(labels)}
    print(f"rounds {len(rows)}  labels {len(labels)}: {labels}")

    xs, ys, gids = [], [], []
    for base, img, n, lab, picks in rows:
        h, w = img.shape[:2]
        th, tw = h // n, w // n
        for i in range(n * n):
            r, c = divmod(i, n)
            t = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            if t.size == 0:
                continue
            xs.append(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))
            ys.append(1 if i in picks else 0)
            gids.append(base)
    ys = np.array(ys); gids = np.array(gids)
    print(f"tiles {len(xs)}  positives {int(ys.sum())}  negatives {int((ys == 0).sum())}")

    net = resnet18(weights=ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval()
    pre = ResNet18_Weights.DEFAULT.transforms()
    feats = []
    with torch.no_grad():
        for i in range(0, len(xs), 64):
            batch = torch.stack([pre(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1))
                                 for x in xs[i:i + 64]])
            feats.append(net(batch).numpy())
    X = np.concatenate(feats)
    print("features:", X.shape)

    # label one-hot appended, so one head serves every class
    lab_ids = np.array([lab_ix[r[3]] for r in rows for _ in range((4 if r[2] == 4 else 3) ** 2)][:len(X)])
    onehot = np.zeros((len(X), len(labels)), dtype=np.float32)
    onehot[np.arange(len(X)), lab_ids] = 1.0
    X = np.concatenate([X, onehot], axis=1)

    grids = sorted(set(gids))
    rng = np.random.default_rng(a.seed)
    rng.shuffle(grids)
    n_hold = max(1, int(len(grids) * a.holdout))
    hold = set(grids[:n_hold])
    tr = ~np.isin(gids, list(hold))
    te = np.isin(gids, list(hold))
    print(f"split by grid: train {len(set(gids[tr]))} grids / {int(tr.sum())} tiles, "
          f"holdout {len(hold)} grids / {int(te.sum())} tiles")

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    A = torch.from_numpy(((X[tr] - mu) / sd)).float()
    B = torch.from_numpy(((X[te] - mu) / sd)).float()
    t = torch.from_numpy(ys[tr]).float().unsqueeze(1)
    w = torch.zeros(A.shape[1], 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    pos_w = torch.tensor([(len(ys[tr]) - ys[tr].sum()) / max(1, ys[tr].sum())])
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, t, pos_weight=pos_w)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        prob = torch.sigmoid(B @ w + b).squeeze(1).numpy()

    yte = ys[te]
    gte = gids[te]
    lab_te = [lab_ids[i] for i in np.where(te)[0]]
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
        pred = prob >= thr
        tp = int((pred & (yte == 1)).sum()); fp = int((pred & (yte == 0)).sum())
        fn = int((~pred & (yte == 1)).sum())
        # scene-level: a grid passes only if the predicted set equals the GT set
        scene_ok = scene_tot = 0
        for g in set(gte):
            m = gte == g
            scene_tot += 1
            scene_ok += int(np.array_equal(pred[m], yte[m] == 1))
        print(f"  thr {thr:.1f}: tile recall {tp}/{tp+fn} = {tp/max(1,tp+fn):.1%}  "
              f"precision {tp/max(1,tp+fp):.1%}  scene-exact {scene_ok}/{scene_tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
