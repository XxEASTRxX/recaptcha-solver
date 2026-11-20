"""Whole-grid VLM with EXPLICIT partial-tile semantics, measured against the 101 exact 4x4 grids.

The exact GT rule is: a tile is correct iff >=15% of the object's pixels fall in it - i.e. a tile
with ANY visible part counts. The serving prompt ("Which tiles satisfy it?") never states this, so
the VLM likely drops edge tiles. This probe adds the rule and reports exact-set accuracy.

  .venv/Scripts/python.exe grid_partial_probe.py [n]
"""
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange")
from service.solver import _brain_json  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 101

PROMPT = ('This image is a reCAPTCHA 4x4 grid; tiles are numbered row-major 0..15 (0 = top-left). '
          'The instruction is: "Select all squares with {cls}". A square MUST be included if ANY '
          'part of a {cls} appears in it - even a small sliver at the tile edge. A square is '
          'excluded only if no part of a {cls} is visible in it. '
          'Reply STRICT JSON only: {{"tiles": [indices]}}')


def main():
    man = json.load(open("gt_exact/manifest.json", encoding="utf-8"))
    for m in man:
        m["image"] = str(m["image"]).replace("\\", "/")
    grids = [m for m in man if m["rows"] == 4][:N]
    brain = httpx.Client(timeout=120.0)
    exact = 0
    tp = fp = fn = 0
    for m in grids:
        try:
            ib = Path(m["image"]).read_bytes()
        except OSError:
            continue
        r = _brain_json(brain, PROMPT.format(cls=m["label"]), ib, max_tokens=120) or {}
        got = set()
        for v in (r.get("tiles") or []):
            try:
                i = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= i < 16:
                got.add(i)
        exp = set(m["picks"])
        if got == exp:
            exact += 1
        tp += len(got & exp)
        fp += len(got - exp)
        fn += len(exp - got)
        print(f"{m['image']} {m['label']} exp={sorted(exp)} got={sorted(got)}", flush=True)
    n = len(grids)
    print(f"exact {exact}/{n} = {exact / max(1, n):.0%}; tile tp={tp} fp={fp} fn={fn}")
    brain.close()


if __name__ == "__main__":
    main()