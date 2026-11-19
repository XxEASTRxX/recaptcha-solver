"""Score the crosswalk decision path against the hand-labelled grids.

`crosswalk_gt.json` holds 5 unique 300x300 grids labelled by zooming each ambiguous tile
(12 positives / 45 tiles). Re-run this after any prompt or filter change; the shipped
configuration measured 11/12 = 91.7% recall, 2 false positives, 84.6% precision.

    C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange/.venv/Scripts/python.exe eval_crosswalk.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("HC_BRAIN_BASE", "http://127.0.0.1:1234")

from solve_demo import ask_grid_brain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    gt = json.load(open(os.path.join(HERE, "crosswalk_gt.json"), encoding="utf-8"))
    hit = tot = fp = 0
    for name, g in gt.items():
        png = open(os.path.join(HERE, "capture", name + ".jpg"), "rb").read()
        meta = json.load(open(os.path.join(HERE, "capture", name + ".json"), encoding="utf-8"))
        picks = sorted(ask_grid_brain(png, 3, 3, meta["task"][:200]))
        pos = set(g["pos"])
        amb = set(g.get("amb") or [])
        # an ambiguous tile is excluded from BOTH sides: counting it as a false positive
        # would punish the solver for a label I could not decide
        picks = [i for i in picks if i not in amb]
        hit += len(pos & set(picks))
        tot += len(pos)
        fp += len(set(picks) - pos)
        print(f"{name}: picks={picks} GT={sorted(pos)} amb={g.get('amb') or []}")
    print(f"\nrecall {hit}/{tot} = {hit / tot:.1%}   false positives {fp}   "
          f"precision {hit / max(1, hit + fp):.1%}")
    return 0 if hit == tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
