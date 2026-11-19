"""reCAPTCHA v2 MVP v2: browser transport + LOCAL COCO detector for tile selection.

Why this shape (all measured):
  * raw HTTP cannot obtain a challenge (`/reload` -> ["rresp",null,...]);
  * headless Chromium can (real rresp, bframe, /payload);
  * the challenge is ONE square JPEG, tiles are CSS crops of it;
  * a pretrained COCO detector finds the requested objects in that JPEG in ~2 s and its
    boxes map straight onto tile indices - far better than asking a VLM yes/no per tile
    (which scored 0/2 at 50-68 s per round).

reCAPTCHA's vocabulary overlaps COCO almost exactly (car, bus, truck, motorcycle,
bicycle, train, boat, fire hydrant, traffic light), so the detector is the primary
decision path; the local brain is only a fallback for classes COCO does not have.

Run:
  C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange/.venv/Scripts/python.exe solve_demo.py --rounds 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from playwright.sync_api import sync_playwright
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_320_fpn)

DEMO = "https://www.google.com/recaptcha/api2/demo"
SCORE_THR = 0.30
CROSSWALK_MIN_WHITE = 0.10   # see _white_frac: keeps every true tile, drops 2 of 4 FPs
TILE_SCORE_THR = 0.15        # measured on coco_gt.json: 0.15 -> recall 16/19, 0 FPs;
                             # 0.30 -> 15/19, 0.60 -> 13/19, 0.10 -> 16/19 but 1 FP

# instruction keyword -> COCO category (singular, as torchvision names them)
COCO_MAP = {
    "car": "car", "cars": "car", "taxi": "car", "vehicle": "car",
    "bus": "bus", "buses": "bus",
    "truck": "truck", "trucks": "truck",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle",
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle", "bikes": "bicycle",
    "train": "train", "trains": "train",
    "boat": "boat", "boats": "boat",
    "fire hydrant": "fire hydrant", "fire hydrants": "fire hydrant", "hydrant": "fire hydrant",
    "traffic light": "traffic light", "traffic lights": "traffic light",
    "person": "person", "people": "person", "pedestrian": "person",
}


def load_detector():
    w = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    net = fasterrcnn_mobilenet_v3_large_320_fpn(weights=w).eval()
    return net, w.meta["categories"]


def detect(net, cats, png: bytes, thr: float = SCORE_THR):
    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None, []
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = net([t])[0]
    dets = []
    for s, l, box in zip(out["scores"].tolist(), out["labels"].tolist(), out["boxes"].tolist()):
        if s < thr:
            break
        dets.append((s, cats[l], box))
    return arr, dets


def ask_grid_brain(png: bytes, rows: int, cols: int, task: str) -> list[int]:
    """Fallback for classes COCO does not have (crosswalk, stairs, chimney...).

    ONE call on the whole grid, not nine per-tile yes/no calls. Measured on the two
    hand-labelled crosswalk grids (GT {2,8} and {4,5}): the whole-grid call found every
    true tile (2/2 recall on both) while the per-tile loop missed a true tile on the
    first grid. Precision is the weak side either way, so the caller varies the answer
    between attempts rather than trusting a single set.
    """
    import httpx
    sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange")
    os.environ.setdefault("HC_BRAIN_BASE", "http://127.0.0.1:1234")
    from service.solver import _brain_json
    prompt = ('This image is a reCAPTCHA {r}x{c} grid; tiles are numbered row-major 0..{n} '
              '(0 = top-left, last = bottom-right). The instruction is: "{t}". Which tiles '
              'satisfy it? Reply STRICT JSON only: {{"tiles": [indices]}}'
              ).format(r=rows, c=cols, n=rows * cols - 1, t=task[:200])
    brain = httpx.Client(timeout=120.0)
    try:
        out = _brain_json(brain, prompt, png, max_tokens=120) or {}
    finally:
        brain.close()
    n = rows * cols
    picks = []
    for v in (out.get("tiles") or []):
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n and i not in picks:
            picks.append(i)
    # "crosswalk" is the reCAPTCHA demo wording; the KG label served by the browserless
    # chain is "Pedestrian crossing". Matching only the first silently skipped the white
    # filter and the per-pick verification on every crossing round (4 of 7 dynamic failures).
    _t = task.lower()
    if picks and ("crosswalk" in _t or "pedestrian crossing" in _t):
        picks = [i for i in picks if _white_frac(png, rows, cols, i) >= CROSSWALK_MIN_WHITE]
        picks = [i for i in picks if _verify_pick(png, rows, cols, i)]
    return picks


VERIFY_PROMPT = ('This is ONE tile cropped from a reCAPTCHA grid. The task was: "Select all '
                 'squares with crosswalks". Answer yes ONLY if white zebra-crossing stripes are '
                 'clearly painted on the road in this tile. Answer no for lane markings, stop '
                 'lines, plain road, buildings, walls, vegetation or vehicles. '
                 'Reply STRICT JSON only: {"yes": true} or {"yes": false}')


def _verify_pick(png: bytes, rows: int, cols: int, i: int) -> bool:
    """Second opinion on ONE already-picked tile.

    Measured on the 17 hand-labelled grids: whole-grid picks give 87.8% recall but 89.6%
    precision (five errors: lane dashes, a house, a street, a hillside, a white van). Asking
    about ONLY the picked tiles (about three calls per grid, not nine) gives **83.7% recall
    at 100% precision** - it meets the >=80% / >=95% bar, which no cheap image statistic did
    (white floor, asphalt fraction and bar counting all failed to separate).
    """
    import httpx
    sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange")
    os.environ.setdefault("HC_BRAIN_BASE", "http://127.0.0.1:1234")
    from service.solver import _brain_json
    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return True
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    tile = arr[(i // cols) * th:(i // cols + 1) * th, (i % cols) * tw:(i % cols + 1) * tw]
    tile = cv2.resize(tile, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    brain = httpx.Client(timeout=120.0)
    try:
        out = _brain_json(brain, VERIFY_PROMPT,
                          cv2.imencode(".jpg", tile)[1].tobytes(), max_tokens=40) or {}
    except Exception:
        return True          # never lose a pick to a transport hiccup
    finally:
        brain.close()
    return bool(out.get("yes"))


def _white_frac(png: bytes, rows: int, cols: int, i: int) -> float:
    """Fraction of bright, low-saturation pixels ("paint") in one tile.

    Measured on the 5 hand-labelled grids: among the brain's picks every true crosswalk
    tile scores >= 0.123 while the two clearly-wrong picks score 0.013 and 0.072, so a
    0.10 floor removes those two and keeps every true tile. It is not a detector (one
    false positive scores 0.63 because a sunlit wall is also white); it only prunes."""
    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return 1.0
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    t = arr[(i // cols) * th:(i // cols + 1) * th, (i % cols) * tw:(i % cols + 1) * tw]
    if t.size == 0:
        return 1.0
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    return float(((hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 45)).mean())


def vary_picks(picks: list[int], attempt: int, n: int) -> list[int]:
    """The same challenge is re-served after a wrong answer, so resubmitting the identical
    set is a guaranteed second failure. Alternate between the set, the set minus its
    weakest member, and the set plus the first unpicked tile."""
    if attempt <= 0 or not picks:
        return picks
    if attempt % 3 == 1:
        return picks[:-1] if len(picks) > 1 else picks
    if attempt % 3 == 2:
        missing = [i for i in range(n) if i not in picks]
        return picks + missing[:1] if missing else picks[:-1]
    return picks


def target_class(task: str) -> str | None:
    t = task.lower()
    for k in sorted(COCO_MAP, key=len, reverse=True):
        if k in t:
            return COCO_MAP[k]
    return None


def tiles_for(arr, dets, want: str, rows: int, cols: int) -> list[int]:
    """Whole-grid detections mapped to tiles. Kept for reference only: measured on a real
    grid it MISSED a clear fire hydrant (0.29 'bicycle' over the whole image), because the
    detector downsamples a 450 px grid to 320 px. Per-tile detection finds it at 0.99."""
    h, w = arr.shape[:2]
    th, tw = h / rows, w / cols
    hits: dict[int, float] = {}
    for s, name, box in dets:
        if name != want:
            continue
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        r = min(rows - 1, int(cy // th))
        c = min(cols - 1, int(cx // tw))
        i = r * cols + c
        hits[i] = max(hits.get(i, 0.0), s)
    return sorted(hits)


def picks_quadrants(net, cats, arr, want: str, rows: int, cols: int) -> list[int]:
    """Overlapping 2x2 windows, mapped back to tiles.

    Measured on coco_gt.json: per-tile + whole-grid gives 84.2% recall, adding these windows
    gives **94.7% at 100% precision** - it recovers exactly the objects split across a tile
    border (a car in tile 5 and a bicycle in tile 2 that neither single path found)."""
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    out: set[int] = set()
    for r in range(rows - 1):
        for c in range(cols - 1):
            win = arr[r * th:(r + 2) * th, c * tw:(c + 2) * tw]
            if win.size == 0:
                continue
            win = cv2.resize(win, (320, 320))
            _a, dd = detect(net, cats, cv2.imencode(".jpg", win)[1].tobytes(), TILE_SCORE_THR)
            for _s, nm, b in dd:
                if nm != want:
                    continue
                cx = (b[0] + b[2]) / 2.0 / 320.0 * (2 * tw) + c * tw
                cy = (b[1] + b[3]) / 2.0 / 320.0 * (2 * th) + r * th
                out.add(min(rows - 1, int(cy // th)) * cols + min(cols - 1, int(cx // tw)))
    return sorted(out)


def picks_per_tile(net, cats, arr, want: str, rows: int, cols: int) -> list[int]:
    """Run the detector on each tile at model resolution (320). Measured: 0.6 s for 9
    tiles and it finds what the whole-grid pass loses."""
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    picks = []
    for r in range(rows):
        for c in range(cols):
            tile = cv2.resize(arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw], (320, 320))
            _arr, dets = detect(net, cats, cv2.imencode(".jpg", tile)[1].tobytes(), TILE_SCORE_THR)
            if any(name == want for _s, name, _b in dets):
                picks.append(r * cols + c)
    return picks


CAPDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capture")


def save_grid(png: bytes, task: str, rows: int, cols: int, picks: list[int]) -> str:
    """Persist every grid so tile accuracy can be measured offline (this project's
    equivalent of the hCaptcha corpus)."""
    os.makedirs(CAPDIR, exist_ok=True)
    stamp = f"{int(time.time()*1000)}_{rows}x{cols}"
    path = os.path.join(CAPDIR, stamp + ".jpg")
    open(path, "wb").write(png)
    json.dump({"task": task, "rows": rows, "cols": cols, "picks": picks},
              open(os.path.join(CAPDIR, stamp + ".json"), "w", encoding="utf-8"))
    return path


def parse_proxy(url: str) -> dict | None:
    """'http://user:pass@host:port' -> Playwright proxy dict."""
    from urllib.parse import urlparse
    u = urlparse(url)
    if not u.hostname:
        return None
    d = {"server": f"http://{u.hostname}:{u.port or 80}"}
    if u.username:
        d["username"] = u.username
    if u.password:
        d["password"] = u.password
    return d


def read_token(page) -> str:
    """The token lives in the hidden field for checkbox v2, but v2 INVISIBLE submits the form
    immediately and the token appears only in the POST result page, as
    `[g-recaptcha-response] => 0cAFcWeA...`. Measured on recaptcha-v2-invisible.php."""
    try:
        tok = page.evaluate("() => document.querySelector('#g-recaptcha-response')?.value || ''")
    except Exception:
        tok = ""
    if tok:
        return tok
    try:
        body = page.inner_text("body")
    except Exception:
        return ""
    marker = "g-recaptcha-response] =>"
    if marker in body:
        rest = body.split(marker, 1)[1].strip()
        return rest.split()[0] if rest else ""
    return ""


def run_round(net, cats, round_no: int, proxy: str | None = None,
              headed: bool = False, url: str = DEMO,
              invisible: bool = False) -> tuple[bool, str, float]:
    t0 = time.time()
    payload = {"png": None, "url": None}
    with sync_playwright() as pw:
        kw = {"headless": not headed,
              "args": ["--disable-blink-features=AutomationControlled"]}
        pd = parse_proxy(proxy) if proxy else None
        if pd:
            kw["proxy"] = pd
        browser = pw.chromium.launch(**kw)
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"),
            locale="en-US", viewport={"width": 1280, "height": 900})
        page = ctx.new_page()

        def on_response(resp):
            # the page fetches several payload resources; the GRID is the largest one
            if "/recaptcha/api2/payload" in resp.url:
                try:
                    b = resp.body()
                except Exception:
                    return
                if payload["png"] is None or len(b) > len(payload["png"]):
                    payload["png"] = b
                    payload["url"] = resp.url

        page.on("response", on_response)
        page.goto(url, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(1800)
        if invisible:
            # v2 INVISIBLE has no anchor checkbox: the challenge (if any) is triggered by the
            # form's g-recaptcha button, which calls grecaptcha.execute via data-callback.
            try:
                page.click("button.g-recaptcha", timeout=8000)
            except Exception:
                try:
                    page.click("form button", timeout=5000)
                except Exception:
                    pass
        else:
            for fr in page.frames:
                if "anchor" in (fr.url or ""):
                    try:
                        fr.click("#recaptcha-anchor", timeout=8000)
                    except Exception:
                        pass
                    break
        page.wait_for_timeout(3500)
        attempts: dict[tuple, int] = {}
        rejects = 0

        for step in range(4):
            tok = read_token(page)
            if tok:
                browser.close()
                return True, f"token after {step} grids: {tok[:20]}...", time.time() - t0
            bf = next((f for f in page.frames if "/bframe" in (f.url or "")), None)
            if bf is None:
                page.wait_for_timeout(1500)
                continue
            # A new grid is served as a NEW /payload request: wait for it by URL FIRST.
            prev_url = payload["url"]
            for _ in range(12):
                if payload["png"] is not None and payload["url"] != prev_url:
                    break
                page.wait_for_timeout(250)
            if payload["png"] is None:
                browser.close()
                return False, "no payload image", time.time() - t0
            # Read the task AFTER the new payload landed. Reading it first paired the
            # PREVIOUS challenge's wording with the NEW image: three captures of one photo
            # carried the tasks motorcycle / bus / bicycle, and only one could be right.
            url_used = payload["url"]
            # The payload can land BEFORE the DOM repaints the new instruction, so any
            # time-based wait is a guess (a 220 ms settle still left 2 of 18 images carrying
            # two tasks, and in those the second class was absent from the image on BOTH
            # detection paths). Sync on the rendered tile instead: its background-image URL
            # IS the payload URL, so matching them proves the DOM is showing this image.
            want_tok = payload["url"].split("payload")[-1]
            synced = False
            for _ in range(12):
                bg = bf.evaluate("""() => {
                    const img = document.querySelector('.rc-imageselect-tile img');
                    return img ? img.src : '';
                }""")
                if want_tok and want_tok in bg:
                    synced = True
                    break
                page.wait_for_timeout(200)
            if not synced:
                # No payload matched the DOM within 2.4 s. Answering here pairs the OLD image
                # with whatever task is currently displayed - measured as the last remaining
                # desync (one image carried 'motorcycles' while its pixels contain only
                # traffic lights). Ask for a fresh challenge instead of answering blind.
                refreshed = bf.evaluate(
                    "() => { const r = document.querySelector('.rc-imageselect-refresh');"
                    " if (r) { r.click(); return true; } return false; }")
                print(f"    grid {step+1}: unsynced -> {'refreshed' if refreshed else 'skipped'}",
                      flush=True)
                page.wait_for_timeout(1500 if refreshed else 600)
                continue
            task = ""
            info: dict = {}
            for _ in range(4 if synced else 2):
                info = json.loads(bf.evaluate("""() => {
                    const hdr = document.querySelector('.rc-imageselect-desc, .rc-imageselect-desc-no-canonical');
                    const tbl = document.querySelector('table');
                    return JSON.stringify({task: hdr ? hdr.innerText : '',
                                           rows: tbl ? tbl.rows.length : 0,
                                           cols: tbl ? tbl.rows[0].cells.length : 0});
                }"""))
                t = str(info.get("task") or "")
                if t and t == task:
                    break
                task = t
                page.wait_for_timeout(220)
            rows = int(info.get("rows") or 3)
            cols = int(info.get("cols") or 3)
            if payload["url"] != url_used:
                # a newer payload landed while the DOM was being read; redo this step so the
                # task text and the image are always from the same challenge
                continue
            arr, dets = detect(net, cats, payload["png"])
            want = target_class(task)
            key = (payload["url"], len(payload["png"]))
            attempt = attempts.get(key, 0)
            attempts[key] = attempt + 1
            if attempt >= 1:
                # The same grid is being re-served after a wrong answer. Answering it again
                # (even with a varied set) burns the rest of the round: measured, 24 of 25
                # rounds died on exactly this. Ask for a fresh challenge instead.
                refreshed = bf.evaluate(
                    "() => { const r = document.querySelector('.rc-imageselect-refresh');"
                    " if (r) { r.click(); return true; } return false; }")
                if refreshed:
                    print(f"    grid {step+1}: repeat image -> refreshed", flush=True)
                    page.wait_for_timeout(1800)
                    continue
            if want is None:
                picks = ask_grid_brain(payload["png"], rows, cols, task[:200])
            else:
                # Union of both detectors. Measured on real rounds: per-tile finds small
                # objects the whole-grid pass downsamples away (hydrant 0.99 vs 0.29), while
                # whole-grid finds objects that SPAN tile boundaries (round 17: 0.88 'car'
                # whole-grid, zero per-tile picks). Either alone loses a class of tile.
                picks = sorted(set(picks_per_tile(net, cats, arr, want, rows, cols))
                               | set(tiles_for(arr, dets, want, rows, cols))
                               | set(picks_quadrants(net, cats, arr, want, rows, cols)))
            picks = vary_picks(picks, attempt, rows * cols)
            print(f"    grid {step+1}: {rows}x{cols} sync={int(synced)} want={want} picks={picks} "
                  f"dets={[(round(s,2), n) for s, n, _ in dets[:6]]}", flush=True)
            save_grid(payload["png"], task, rows, cols, picks)
            if picks:
                bf.evaluate("""(picks) => {
                    const t = [...document.querySelectorAll('td[role=button], .rc-imageselect-tile')];
                    for (const i of picks) if (t[i]) t[i].click();
                }""", picks)
            bf.click("#recaptcha-verify-button", timeout=8000)
            page.wait_for_timeout(2500)
            err = bf.evaluate("""() => {
                const e = document.querySelector('.rc-imageselect-incorrect-response');
                return e && e.offsetParent ? e.innerText : '';
            }""")
            if err:
                rejects += 1
                if rejects > 1:
                    browser.close()
                    return False, f"rejected: {err[:40]}", time.time() - t0
                page.wait_for_timeout(1200)
                continue

        tok = read_token(page)
        browser.close()
        return bool(tok), ("token" if tok else "no token"), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--url", default=DEMO,
                    help="challenge page (default the v2 checkbox demo)")
    ap.add_argument("--invisible", action="store_true",
                    help="v2 invisible: trigger via the form's g-recaptcha button")
    ap.add_argument("--headed", action="store_true",
                    help="run a real browser window: headless is a strong reCAPTCHA signal")
    ap.add_argument("--proxies-file", default=None,
                    help="rotate one proxy per round; the demo site rate-limits per IP "
                         "(measured: 0/5 direct with a hard 429, 2/3 through a pool)")
    args = ap.parse_args()
    pool = []
    if args.proxies_file:
        pool = [ln.strip() for ln in open(args.proxies_file, encoding="utf-8")
                if ln.strip().startswith("http")]
        print(f"proxy pool: {len(pool)} exits", flush=True)
    net, cats = load_detector()
    ok = 0
    for i in range(args.rounds):
        px = pool[i % len(pool)] if pool else None
        try:
            good, why, dt = run_round(net, cats, i + 1, px, args.headed,
                                      args.url, args.invisible)
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "TargetClosedError":
                # v2 INVISIBLE submits the form on success and the renderer can die mid-flow;
                # that is a page-lifecycle race, not a solve failure, so retry once.
                try:
                    good, why, dt = run_round(net, cats, i + 1, px, args.headed,
                                              args.url, args.invisible)
                except Exception as exc2:  # noqa: BLE001
                    good, why, dt = False, f"{type(exc2).__name__}: {str(exc2)[:70]}", 0.0
            else:
                good, why, dt = False, f"{type(exc).__name__}: {str(exc)[:70]}", 0.0
        ok += int(good)
        print(f"round {i+1}: {'PASS' if good else 'FAIL'} ({why}) {dt:.1f}s", flush=True)
    print(f"\n{ok}/{args.rounds} tokens")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())