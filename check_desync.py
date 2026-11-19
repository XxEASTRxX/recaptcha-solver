"""Regression test: no image may be paired with a task its pixels cannot support.

The solver used to read the instruction text BEFORE waiting for the next /payload image,
so a challenge arriving in between was answered against the previous question. That was
measured across the capture history: 22 of 137 unique images carried two or more tasks,
covering 69 of 361 captures, with at least 27 definitely-wrong pairings.

IMPORTANT refinement (measured): a multi-task image is NOT automatically a desync. reCAPTCHA
reuses one stock photo across challenges, so `fc4d1b7cf3` legitimately carries 'bicycles' AND
'motorcycles' (detector: bicycle 0.99 per-tile, motorcycle 1.00) and `8b9c5de2b5` carries
'bicycles' AND 'stairs' (both visible in the photo). The test therefore fails only when a
claimed class is ABSENT from the image on BOTH detection paths - positive evidence of a
wrong pairing. Non-COCO classes (stairs, crosswalk) cannot be verified automatically and are
reported as unverified rather than failures.

    python check_desync.py                 # whole history (expect the pre-fix count)
    python check_desync.py --since-fix     # only captures newer than solve_demo.py
    python check_desync.py --since 1794684775

Exit code 0 when no offending image is found, 1 otherwise.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(HERE, "capture")


def scan(since_mtime: float | None):
    by_img: dict[str, list[tuple[str, float, str]]] = collections.defaultdict(list)
    for meta in glob.glob(os.path.join(CAP, "*.json")):
        img = meta[:-5] + ".jpg"
        if not os.path.exists(img):
            continue
        mt = os.path.getmtime(img)
        if since_mtime is not None and mt <= since_mtime:
            continue
        try:
            task = json.load(open(meta, encoding="utf-8"))["task"]
        except Exception:
            continue
        label = task.split("\n")[1].strip() if "\n" in task else task.strip()
        arr = cv2.imread(img)
        if arr is None:
            continue
        h = hashlib.md5(arr.tobytes()).hexdigest()[:10]
        by_img[h].append((label, mt, os.path.basename(img)))
    return by_img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-fix", action="store_true",
                    help="only captures newer than solve_demo.py (the shipped fix)")
    ap.add_argument("--since", type=float, default=None,
                    help="only captures with mtime greater than this epoch value")
    a = ap.parse_args()

    cut = a.since
    if a.since_fix:
        cut = os.path.getmtime(os.path.join(HERE, "solve_demo.py"))

    by_img = scan(cut)
    total = sum(len(v) for v in by_img.values())
    bad = {h: v for h, v in by_img.items() if len({t for t, _, _ in v}) > 1}
    print(f"captures scanned: {total}   unique images: {len(by_img)}")
    print(f"images carrying MORE THAN ONE task: {len(bad)}")
    for h, v in list(bad.items())[:10]:
        print(f"  {h}: {sorted({t for t, _, _ in v})}")
    affected = sum(len(v) for v in bad.values())
    print(f"captures involved: {affected}"
          + (f" ({affected / total:.1%})" if total else ""))
    if not bad:
        print("\nOK: every image maps to exactly one task")
        return 0

    import cv2 as _cv2
    from solve_demo import detect, load_detector, target_class
    net, cats = load_detector()
    desync, unverified = [], []
    for h, v in bad.items():
        img = os.path.join(CAP, v[0][2])
        arr, dets = detect(net, cats, open(img, "rb").read())
        if arr is None:
            continue
        hh, ww = arr.shape[:2]
        n = 3 if ww == 300 else 4
        th, tw = hh // n, ww // n
        absent, unknown = [], []
        for t in sorted({t for t, _, _ in v}):
            want = target_class(t)
            if want is None:
                unknown.append(t)
                continue
            wg = max([s for s, nm, _ in dets if nm == want], default=0.0)
            pt = 0.0
            for r in range(n):
                for c in range(n):
                    tile = _cv2.resize(arr[r*th:(r+1)*th, c*tw:(c+1)*tw], (320, 320))
                    _a, dd = detect(net, cats, _cv2.imencode(".jpg", tile)[1].tobytes(), 0.15)
                    pt = max([pt] + [s for s, nm, _ in dd if nm == want])
            if max(wg, pt) < 0.15:
                absent.append(t)
        if absent:
            desync.append((h, sorted({t for t, _, _ in v}), absent))
        elif unknown:
            unverified.append((h, sorted(unknown)))
    print(f"\nimages with a claimed class ABSENT (real desync): {len(desync)}")
    for h, tasks, absent in desync:
        print(f"  {h}: tasks={tasks} ABSENT={absent}")
    if unverified:
        print(f"images whose only multi-task explanation is a non-COCO class: {len(unverified)}")
        for h, unk in unverified:
            print(f"  {h}: unverifiable={unk}")
    if desync:
        print("\nFAIL: task/payload desync present")
        return 1
    print("\nOK: no image carries a task its pixels cannot support")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
