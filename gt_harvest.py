"""Harvest EXACT 4x4/3x3 ground truth from PASSING rounds.

Acceptance was measured EXACT (all-16 failed 0/18 multicaptcha rounds), so a passing round's
submitted pick set IS the true set. Each passing dump therefore yields one exact labelled grid.
Failing grids are NOT usable: their set is a mixture of true and false tiles with no per-tile label.

Output: gt_exact/manifest.json  [{image, type, label, rows, cols, picks, source}]
        gt_exact/<name>.jpg    (copied payload)
The six hand-checked grids are included with source="hand" and their visually verified sets.
"""
import glob
import json
import os
import shutil

OUT = "gt_exact"

# Visually verified from the rendered grids (2026-11-19). Only tiles that contain any part of the
# object count; see REPORT.md.
HAND = {
    "1795122684912_motorcycle.jpg": ("motorcycle", 4, 4, [1, 2, 5, 6, 9, 10]),
    "1795122695979_bicycle.jpg": ("bicycle", 4, 4, [4, 5, 8, 9, 12, 13]),
    "1795122647935_bicycle.jpg": ("bicycle", 4, 4, [5, 6, 10, 11]),
    "1795122610683_bus.jpg": ("bus", 4, 4, [4, 5, 8, 9]),
    "1795122632424_bus.jpg": ("bus", 4, 4, list(range(4, 12))),
    "1795122046507_car.jpg": ("car", 4, 4, [5, 6, 7, 8, 9]),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    seen = set()
    for jf in glob.glob("capture*/**/*.json", recursive=True):
        try:
            d = json.load(open(jf, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or not d.get("pass"):
            continue
        img = jf.replace(".json", ".jpg")
        if not os.path.exists(img):
            continue
        picks = d.get("picks") or []
        rows = 4 if d.get("type") == "multicaptcha" else 3
        key = os.path.basename(img)
        if key in seen:
            continue
        seen.add(key)
        dst = os.path.join(OUT, key)
        shutil.copyfile(img, dst)
        manifest.append({"image": dst, "type": d.get("type"), "label": d.get("label"),
                         "rows": rows, "cols": rows, "picks": picks, "source": "pass"})
    for name, (label, rows, cols, picks) in HAND.items():
        src = glob.glob(f"capture*/**/{name}", recursive=True)
        if not src:
            print("hand grid missing:", name)
            continue
        dst = os.path.join(OUT, name)
        shutil.copyfile(src[0], dst)
        manifest.append({"image": dst, "type": "multicaptcha", "label": label,
                         "rows": rows, "cols": cols, "picks": picks, "source": "hand"})
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    n4 = sum(1 for m in manifest if m["rows"] == 4)
    print(f"harvested {len(manifest)} exact grids: {n4} 4x4, {len(manifest) - n4} 3x3")


if __name__ == "__main__":
    main()
