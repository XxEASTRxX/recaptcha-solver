"""Reference-anchored 4x4 tile comparison probe.

Mechanism: the per-tile yes/no prompt confuses object TYPES (a tree trunk answered "bicycle",
a white car answered "motorcycle", empty sky answered "car"). Ask instead whether a tile shows the
same kind of object as a KNOWN example tile from the same photo. One reference + 16 comparisons.

Run with the hCaptcha venv (has httpx):
  HC_BRAIN_BASE=http://127.0.0.1:1234 <captchachalange>/.venv/Scripts/python.exe ref_probe.py
"""
import io
import os
import sys

import cv2
import httpx
from PIL import Image

sys.path.insert(0, r"C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange")
from service.solver import _brain_json  # noqa: E402

CASES = [
    # (image, label, reference tile, expected true set)
    ("capture_gate60/1795122684912_motorcycle.jpg", "motorcycle", 6, {1, 2, 5, 6, 9, 10}),
    ("capture_gate60/1795122695979_bicycle.jpg", "bicycle", 9, {4, 5, 8, 9, 12, 13}),
    ("capture_gate60/1795122647935_bicycle.jpg", "bicycle", 10, {5, 6, 10, 11}),
    ("capture_gate60/1795122610683_bus.jpg", "bus", 5, {4, 5, 8, 9}),
]


def tile_jpg(arr, rows, cols, i, scale=2):
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    r, c = divmod(i, cols)
    t = arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
    t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.imencode(".jpg", t)[1].tobytes()


def main():
    brain = httpx.Client(timeout=90.0)
    try:
        for path, label, ref, expected in CASES:
            arr = cv2.imread(path)
            if arr is None:
                print(f"MISSING {path}")
                continue
            rows = cols = 4
            ref_jpg = tile_jpg(arr, rows, cols, ref)
            got = set()
            for i in range(rows * cols):
                prompt = (
                    f'The FIRST image is a known example of a "{label}" from a reCAPTCHA photo. '
                    f'The SECOND image is one tile from the SAME photo. Does the second image show '
                    f'any part of the SAME KIND of object as the first image? Ignore people, road, '
                    f'sky, trees, walls and other object types. '
                    'Reply STRICT JSON only: {"same": true/false}')
                ans = _brain_json(brain, prompt, [ref_jpg, tile_jpg(arr, rows, cols, i)],
                                  max_tokens=40, text_first=True) or {}
                if bool(ans.get("same")):
                    got.add(i)
            print(f"{os.path.basename(path)} label={label} ref={ref}")
            print(f"  expected {sorted(expected)}")
            print(f"  got      {sorted(got)}  exact={got == expected}")
    finally:
        brain.close()


if __name__ == "__main__":
    main()
