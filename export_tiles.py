"""Export dumped reCAPTCHA rounds into a training set for the homeserver GPU.

Passing rounds are free ground truth (the clicks were accepted). Each tile becomes one sample with
its grid id, so the training split can be made by GRID and never leak a scene across train/val.

    python export_tiles.py --out tiles_dataset.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 96


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["bl_bulk", "bl_dump", "bl_dump2"])
    ap.add_argument("--out", default="tiles_dataset.npz")
    a = ap.parse_args()

    xs, ys, labs, grids = [], [], [], []
    label_ix: dict[str, int] = {}
    n_rounds = n_pass = 0
    for d in a.dirs:
        for f in glob.glob(os.path.join(HERE, d, "*.json")):
            n_rounds += 1
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
            h, w = img.shape[:2]
            th, tw = h // n, w // n
            lab = meta["label"]
            if lab not in label_ix:
                label_ix[lab] = len(label_ix)
            picks = set(meta["picks"])
            for i in range(n * n):
                r, c = divmod(i, n)
                t = img[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
                if t.size == 0:
                    continue
                xs.append(cv2.resize(t, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
                ys.append(1 if i in picks else 0)
                labs.append(label_ix[lab])
                grids.append(os.path.basename(f)[:-5])
            n_pass += 1

    X = np.stack(xs).astype(np.uint8)
    np.savez_compressed(a.out, X=X, y=np.array(ys, np.int64), lab=np.array(labs, np.int64),
                        grid=np.array(grids), labels=np.array(list(label_ix)))
    print(f"rounds scanned {n_rounds}, passing {n_pass}")
    print(f"tiles {X.shape}, positives {int(np.sum(ys))}, labels {len(label_ix)}")
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
