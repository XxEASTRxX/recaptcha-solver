"""Export dumped rounds as whole-GRID samples for a per-grid mask model.

The per-tile formulation failed three times (3%, 17%, 14% scene-exact) because it throws away the
context that matters: a 4x4 multicaptcha is one photo, so which tiles are positive depends on what
the whole scene contains. This exports (grid image, label, tile mask) so the model sees everything.

    python export_grids.py --out grids_dataset.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 160
MAXN = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["bl_bulk", "bl_dump", "bl_dump2"])
    ap.add_argument("--out", default="grids_dataset.npz")
    a = ap.parse_args()

    xs, ys, labs, ns, grids = [], [], [], [], []
    label_ix: dict[str, int] = {}
    n_pass = 0
    for d in a.dirs:
        for f in glob.glob(os.path.join(HERE, d, "*.json")):
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
            lab = meta["label"]
            if lab not in label_ix:
                label_ix[lab] = len(label_ix)
            mask = np.zeros(MAXN * MAXN, np.float32)
            for i in meta["picks"]:
                if 0 <= i < n * n:
                    mask[i] = 1.0
            xs.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
            ys.append(mask)
            labs.append(label_ix[lab])
            ns.append(n)
            grids.append(os.path.basename(f)[:-5])
            n_pass += 1

    X = np.stack(xs).astype(np.uint8)
    np.savez_compressed(a.out, X=X, Y=np.stack(ys), lab=np.array(labs, np.int64),
                        n=np.array(ns, np.int64), grid=np.array(grids),
                        labels=np.array(list(label_ix)))
    print(f"passing rounds {n_pass}, grids {X.shape}, labels {len(label_ix)}")
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
