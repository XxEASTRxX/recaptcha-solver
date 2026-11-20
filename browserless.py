"""Browserless reCAPTCHA v2 solver: anchor -> reload -> payload -> solve -> userverify.

No browser anywhere. Chrome TLS impersonation via curl_cffi; the challenge is a 300x300
montage that the existing detector solves.

Established this session, every step measured:
  * anchor GET with the right `v` returns the page, sets `_GRECAPTCHA`, and carries both the
    `03AF...` anchor token and the `0dAF...` bframe token. Without `v` it is a JS shell.
  * reload is a protobuf POST; field 1 = `v`, field 2 = the anchor token, the rest can be a
    captured template. Response is JSON after `)]}'`.
  * payload needs BOTH `p=<rresp[9]>` AND `k=<sitekey>`; `p` alone returns 400.
  * userverify takes form fields v/c/response(b64 JSON)/t/ct/ch/oc/pm. ch/oc/pm are
    VM-derived; the static literals below are what the reviewed shim ships, and they are
    accepted (HTTP 200 `["uvresp",...]`) - a null token means the ANSWER was wrong, not the
    request.

    python browserless.py --rounds 10
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time

import httpx

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36")
DEFAULT_V = "4i3aN5wAweL7i3ADI7BSX4B-"
DEMO_SITEKEY = "6Le-wvkSAAAAAPBMRTvw0Q4Muexq9bi0DJwx_mJ-"

# Knowledge-Graph id -> instruction class (e2e.py:25, cross-checked against captured rresps)
MID_LABEL = {
    "/m/015qff": "traffic light", "/m/04_sv": "motorcycle", "/m/01bjv": "bus",
    "/m/0199g": "bicycle", "/m/01lynh": "fire hydrant", "/m/014xcs": "car",
    "/m/02kx1l": "bridge", "/m/0k4j": "chimney", "/m/02sgy": "palm tree",
    "/m/04h7h": "stairs", "/m/09g5v": "crosswalk", "/m/0k2k": "boat",
    "/m/015258": "car", "/m/01pns0": "fire hydrant",
}
X_BROWSER_VALIDATION = "rvacWodMMC8CsnYVvG35HCYGtZY="
CH = "2801408"
OC = "0Il3sfQw"
PM = ("51bGwsbnVsbCxbbnVsbCxudWxsLG51bGwsWzcsNS41NDI4NTcxNDQ1NjAxMzI1LDAuMDA3MzYxMTcxNzE4NzAwMTE1LDE2XSxbMCxudWxsLDBdLDEsMCwxXV0")


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def walk(buf: bytes):
    i, out = 0, []
    while i < len(buf):
        tag = shift = 0
        while True:
            b = buf[i]; i += 1
            tag |= (b & 0x7F) << shift; shift += 7
            if not b & 0x80:
                break
        fn, wt = tag >> 3, tag & 7
        if wt == 2:
            ln = shift = 0
            while True:
                b = buf[i]; i += 1
                ln |= (b & 0x7F) << shift; shift += 7
                if not b & 0x80:
                    break
            out.append((fn, wt, buf[i:i + ln])); i += ln
        elif wt == 0:
            val = shift = 0
            while True:
                b = buf[i]; i += 1
                val |= (b & 0x7F) << shift; shift += 7
                if not b & 0x80:
                    break
            out.append((fn, wt, val))
        else:
            raise ValueError(f"wire type {wt}")
    return out


def build(fields) -> bytes:
    out = bytearray()
    for fn, wt, val in fields:
        out += varint((fn << 3) | wt)
        if wt == 2:
            out += varint(len(val)) + val
        else:
            out += varint(val)
    return bytes(out)


def acquire(sess, sitekey: str, v: str, template: bytes, origin="https://www.google.com"):
    """anchor -> reload -> payload. Returns (image_bytes, meta) or (None, meta)."""
    co = base64.b64encode(origin.encode()).decode().rstrip("=")
    r = sess.get(f"https://www.google.com/recaptcha/api2/anchor?ar=1&k={sitekey}"
                 f"&co={co}&hl=en&v={v}&size=normal", timeout=30)
    tok = re.search(r'id="recaptcha-token" value="([^"]+)"', r.text)
    bfts = re.findall(r'0dAF[A-Za-z0-9_\-]{20,}', r.text)
    if not tok:
        return None, {"error": "no anchor token (v may have rotated)"}
    anchor_token, bft = tok.group(1), (bfts[0] if bfts else "")

    fields = walk(template)
    new = [(fn, wt, (v.encode() if fn == 1 else anchor_token.encode() if fn == 2 else val))
           for fn, wt, val in fields]
    t0 = time.time()
    rr = sess.post(f"https://www.google.com/recaptcha/api2/reload?k={sitekey}",
                   data=build(new),
                   headers={"content-type": "application/x-protobuffer"}, timeout=30)
    text = rr.text.split("\n", 1)[1] if rr.text.startswith(")]}'") else rr.text
    try:
        d = json.loads(text)
    except Exception:
        return None, {"error": "reload parse", "head": text[:120]}

    def find_mid(x):
        """First /m/ id anywhere in the pmeta tree. `dynamic` puts it at pmeta[1][i],
        `multicaptcha` nests it at pmeta[5][0][i][0] (measured), so search recursively."""
        if isinstance(x, str):
            return x if x.startswith("/m/") else None
        if isinstance(x, list):
            for y in x:
                r = find_mid(y)
                if r:
                    return r
        return None

    mid = find_mid(d[4]) if len(d) > 4 else None
    # pmeta[1] is ["/m/...", null, 3, 3, 3, null, "Fire hydrant"] - the ENGLISH LABEL is
    # index 6. That is authoritative; the KG map is only a fallback. (A recursive /m/ search
    # picked the wrong id from unrelated candidate lists and solved the wrong task.)
    label = None
    try:
        p1 = d[4][1]
        if isinstance(p1, list) and len(p1) > 6 and isinstance(p1[6], str):
            label = p1[6]
    except Exception:
        pass
    kind = d[5] if len(d) > 5 else None
    p_tok = d[9] if len(d) > 9 else None
    meta = {"type": kind, "mid": mid,
            "label": label or MID_LABEL.get(mid or "", None),
            "c": d[1] if len(d) > 1 else "", "bft": bft, "t0": t0}
    if not p_tok:
        return None, {**meta, "error": "no payload token"}
    # The browser loads the bframe page before userverify, so its session carries that state.
    # We never did; loading it is the cheapest remaining difference in the submission path.
    if bft:
        try:
            sess.get(f"https://www.google.com/recaptcha/api2/bframe?hl=en&v={v}&k={sitekey}"
                     f"&bft={bft}&ca=true", timeout=20)
        except Exception:
            pass
    pr = sess.get(f"https://www.google.com/recaptcha/api2/payload?p={p_tok}&k={sitekey}",
                  timeout=30)
    if pr.status_code != 200 or not pr.content:
        return None, {**meta, "error": f"payload HTTP {pr.status_code}"}
    return pr.content, meta


def submit(sess, sitekey: str, v: str, meta: dict, indices: list[int]):
    inner = {"response": [int(i) for i in indices], "s": os.environ.get("HC_S_TOKEN", "4cd7"), "e": os.environ.get("HC_E_TOKEN", "bXVs")}
    resp = base64.b64encode(json.dumps(inner, separators=(",", ":")).encode()).decode()
    # 4-15 s clamp. Reporting the TRUE elapsed (up to 60 s) was measured over 100 rounds at 48%
    # vs the 52% baseline, i.e. no gain, so the clamp stays.
    t = int(max(4000, min(15000, (time.time() - meta.get("t0", time.time())) * 1000)))
    data = {"v": v, "c": meta.get("c") or "", "response": resp, "t": t, "ct": t}
    # ch/oc/pm are captured constants from an OLD session. If the server ties them to the current
    # one they would cause exactly the "correct answer rejected" pattern, so they can be omitted.
    if os.environ.get("HC_NO_PM") != "1":
        data.update({"ch": CH, "oc": OC, "pm": PM})
    ref = f"https://www.google.com/recaptcha/api2/bframe?hl=en&v={v}&k={sitekey}"
    if meta.get("bft"):
        ref += f"&bft={meta['bft']}"
    r = sess.post(f"https://www.google.com/recaptcha/api2/userverify?k={sitekey}",
                  data=data,
                  headers={"accept": "*/*",
                           "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
                           "origin": "https://www.google.com",
                           "x-browser-validation": X_BROWSER_VALIDATION,
                           "referer": ref},
                  timeout=30)
    text = r.text.split("\n", 1)[1] if r.text.startswith(")]}'") else r.text
    try:
        d = json.loads(text)
    except Exception:
        return None, f"parse: {text[:80]}"
    token = d[1] if isinstance(d, list) and len(d) > 1 else None
    if os.environ.get("HC_DEBUG_SUBMIT") == "1":
        # Evidence capture: a multicaptcha serves SUCCESSIVE grids, so a correct first answer also
        # returns no token. Without the raw body we cannot tell "wrong answer" from "next grid".
        # Print PASSING bodies too: the shape of d[2:] is what distinguishes a token from a
        # continuation, and we had never recorded a passing multicaptcha body.
        print(f"[userverify] type={meta.get('type')} ok={bool(token)} len={len(text)} "
              f"body={text[:600]}", flush=True)
    return token, text[:160]



def tiles_for_overlap(arr, dets, want: str, rows: int, cols: int, min_frac: float = 0.15):
    """Every tile a detection BOX actually covers, not just the one under its centre.

    A 4x4 `multicaptcha` is ONE photo split into 16 tiles, so an object routinely spans
    several of them; centre-mapping assigns it to one tile and under-selects, which is why
    those rounds failed while 3x3 montages passed."""
    h, w = arr.shape[:2]
    th, tw = h / rows, w / cols
    tile_area = max(1.0, th * tw)
    out: set[int] = set()
    for _s, name, box in dets:
        if name != want:
            continue
        x1, y1, x2, y2 = box
        for r in range(rows):
            for c in range(cols):
                tx1, ty1 = c * tw, r * th
                ix = max(0.0, min(x2, tx1 + tw) - max(x1, tx1))
                iy = max(0.0, min(y2, ty1 + th) - max(y1, ty1))
                if (ix * iy) / tile_area >= min_frac:
                    out.add(r * cols + c)
    return sorted(out)



# Labels where per-tile VLM beats the detector as a CANDIDATE GENERATOR, measured on free GT from
# passing rounds (12 grids): tile recall 82.9% and 6/12 scene-exact, versus the pipeline's
# bus 28.6% / motorcycle 23.5% live. The detector under-selects these (multi-instance, spanning).
# Measured on free GT from passing rounds: per-tile VLM tile recall 82.9% (weak labels) and 89.1%
# (strong labels, 10/14 scene-exact) - it beats the detector on BOTH groups, so every label routes
# through it. Cost is 9-16 calls per grid (~9-18 s) versus ~1.6 s for the detector.
VLM_TILE_LABELS = {"bus", "Bus", "motorcycle", "Pedestrian crossing", "traffic light", "Traffic light"}
# NOT all labels: routing bicycle/Bike/Fire hydrant through the VLM measured WORSE live (strong-label
# rounds rejected where the detector passes them at 80-89%), even though the offline probe read
# 10/14. Keep the detector for the classes it is good at.


def picks_per_tile_vlm(png: bytes, rows: int, cols: int, label: str) -> list[int]:
    """Ask the local VLM about every tile. ~9-16 calls, so only for the labels where it wins."""
    import httpx
    import cv2 as _cv2
    import numpy as _np
    sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange")
    os.environ.setdefault("HC_BRAIN_BASE", "http://127.0.0.1:1234")
    from service.solver import _brain_json
    arr = _cv2.imdecode(_np.frombuffer(png, _np.uint8), _cv2.IMREAD_COLOR)
    if arr is None:
        return []
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    # A 4x4 `multicaptcha` is ONE photo split into 16 tiles, so most tiles hold only a FRAGMENT of
    # the object. Without saying so, the VLM answers "no" to every tile (measured: `no picks
    # (label=bus)` on a 4x4 grid), which reads as a solver failure when it is a prompt failure.
    # 4x4 REFINEMENT (measured 2026-11-19 on two fresh live grids): the old prompt answered yes to
    # an empty sky tile ("car") and to a painted "BUS" road marking ("bus"). Adding the exclusion
    # clause fixed both - bus [4,5,8,9,10] exact (was [4,5,8,9,10,12,14]) and car [5,6,7,9,10]
    # (was [2,4,5,6,7]; 2 was sky, 8 a real car still missed). The detector union added nothing on
    # either grid, so the prompt is the whole defect. HC_MC_PROMPT=0 restores the old wording.
    if rows == 4 and os.environ.get("HC_MC_PROMPT", "1") == "1":
        prompt = (f'This is ONE tile cut from a larger photo in a reCAPTCHA 4x4 grid. Task: "Select '
                  f'all squares with {label}". Answer yes only if a {label} itself is visible in this '
                  f'tile, even partially cut off by the edge. Answer no if the tile shows only sky, '
                  f'road surface, trees, buildings, people, or a painted word/sign that is not the '
                  'object. Reply STRICT JSON only: {"yes": true/false, "what": "brief"}')
    else:
        prompt = (f'This is ONE tile cut from a larger photo in a reCAPTCHA grid. Task: "Select all '
                  f'squares with {label}". The object is often cut off by the tile edge - answer yes if '
                  f'ANY part of a {label} is visible in this tile, even partially. '
                  'Reply STRICT JSON only: {"yes": true} or {"yes": false}')
    brain = httpx.Client(timeout=90.0)
    out = []
    try:
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            t = arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
            t = _cv2.resize(t, None, fx=2, fy=2, interpolation=_cv2.INTER_CUBIC)
            o = _brain_json(brain, prompt, _cv2.imencode(".jpg", t)[1].tobytes(), max_tokens=30) or {}
            if bool(o.get("yes")):
                out.append(i)
    finally:
        brain.close()
    return out



# ---- 14-class tile classifier (57k public dataset, 91.1% val accuracy) ------------------------
# Trained on DannyLuna/recaptcha-57k-images-dataset (MIT): 56,186 tiles, classes that map onto
# reCAPTCHA's vocabulary. This replaces both the COCO detector's limited vocabulary and the
# 9-16-call per-tile VLM: one batched forward pass per grid.
LABEL_TO_CLASS = {
    "bus": "Bus", "Bus": "Bus", "motorcycle": "Motorcycle", "Motorcycle": "Motorcycle",
    "Pedestrian crossing": "Crosswalk", "crosswalk": "Crosswalk", "Crosswalk": "Crosswalk",
    "Fire hydrant": "Hydrant", "fire hydrant": "Hydrant", "Hydrant": "Hydrant",
    "Bike": "Bicycle", "bicycle": "Bicycle", "Bicycle": "Bicycle",
    "car": "Car", "Car": "Car", "taxis": "Car", "Taxi": "Car",
    "traffic light": "Traffic Light", "Traffic light": "Traffic Light",
    "mountain": "Mountain", "stairs": "Stair", "Stair": "Stair", "chimney": "Chimney",
    "palm tree": "Palm", "bridge": "Bridge", "tractor": "Tractor",
}
_CLS = {"model": None, "classes": None, "transform": None}


def _classifier():
    if _CLS["model"] is None:
        import torch
        import torchvision.transforms as T
        from torchvision.models import resnet18
        here = os.path.dirname(os.path.abspath(__file__))
        # HC_CLS_CKPT picks the checkpoint so an in-domain fine-tune can be A/B'd against the
        # public-dataset model instead of silently replacing it.
        ck_name = os.environ.get("HC_CLS_CKPT", "tile_cls57k.pt")
        ck = torch.load(os.path.join(here, ck_name), map_location="cpu", weights_only=False)
        net = resnet18(weights=None)
        net.fc = torch.nn.Linear(net.fc.in_features, len(ck["classes"]))
        net.load_state_dict(ck["state"])
        net.eval()
        _CLS["model"] = net
        _CLS["classes"] = ck["classes"]
        # ToTensor FIRST: it accepts a numpy HWC array, while Resize needs PIL.
        _CLS["transform"] = T.Compose([
            T.ToTensor(), T.Resize((160, 160)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return _CLS["model"], _CLS["classes"], _CLS["transform"]


def picks_by_classifier(png: bytes, rows: int, cols: int, label: str) -> list[int]:
    """Classify every tile and select those whose predicted class is the requested one."""
    import torch
    import cv2 as _cv2
    import numpy as _np
    # Case-insensitive: the KG label arrives as "Bus"/"bus"/"Traffic light" and one missing
    # exact key silently skipped the classifier, falling through to the weak detector.
    target = LABEL_TO_CLASS.get(label)
    if target is None and label:
        for _k, _v in LABEL_TO_CLASS.items():
            if _k.lower() == label.lower():
                target = _v
                break
    if target is None:
        return []
    net, classes, tf = _classifier()
    if target not in classes:
        return []
    arr = _cv2.imdecode(_np.frombuffer(png, _np.uint8), _cv2.IMREAD_COLOR)
    if arr is None:
        return []
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    tiles = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        t = arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
        if t.size == 0:
            t = _np.zeros((th, tw, 3), _np.uint8)
        tiles.append(tf(_cv2.cvtColor(t, _cv2.COLOR_BGR2RGB)))
    with torch.no_grad():
        logits = net(torch.stack(tiles))
        idx = logits.argmax(1).tolist()
    want_ix = classes.index(target)
    return [i for i, k in enumerate(idx) if k == want_ix]



# ---- tile net trained on EXACT ground truth (a passing round's picks ARE the true set) ---------
MC_NORM = {"Bike": "bicycle", "bicycle": "bicycle", "Bus": "bus", "bus": "bus",
           "Fire hydrant": "fire hydrant", "fire hydrant": "fire hydrant",
           "Pedestrian crossing": "pedestrian crossing", "car": "car", "Car": "car",
           "motorcycle": "motorcycle", "traffic light": "traffic light"}
_TILE_NETS: dict = {}


def _tile_net(rows: int):
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet18
    if rows in _TILE_NETS:
        return _TILE_NETS[rows]
    here = os.path.dirname(os.path.abspath(__file__))
    ck = torch.load(os.path.join(here, f"tile_cls_{rows}x{rows}.pt"),
                    map_location="cpu", weights_only=False)
    net = resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(ck["classes"]))
    net.load_state_dict(ck["state"])
    net.eval()
    tf = T.Compose([T.ToTensor(), T.Resize((160, 160)),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    _TILE_NETS[rows] = (net, ck["classes"], tf)
    return _TILE_NETS[rows]


def picks_by_tile_net(png: bytes, rows: int, cols: int, label: str) -> list[int]:
    """Select tiles whose head (trained on exact GT) scores the target >= HC_MC_NET_THR.

    Exact acceptance was proven with HC_MC_ALL (all-16 failed 0/18), so a passing round is a clean
    label source. Trained per class on the harvested 4x4/3x3 grids; this is the serving candidate
    for multicaptcha, not the 57k public model (which is exact on 2/6 hand grids and blind to
    motorcycles).
    """
    import torch
    import cv2 as _cv2
    import numpy as _np
    target = MC_NORM.get(label)
    if target is None:
        return []
    net, classes, tf = _tile_net(rows)
    if target not in classes:
        return []
    ci = classes.index(target)
    arr = _cv2.imdecode(_np.frombuffer(png, _np.uint8), _cv2.IMREAD_COLOR)
    if arr is None:
        return []
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    tiles = []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        tiles.append(tf(_cv2.cvtColor(arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw],
                                       _cv2.COLOR_BGR2RGB)))
    thr = float(os.environ.get("HC_MC_NET_THR", "0.5"))
    with torch.no_grad():
        p = torch.sigmoid(net(torch.stack(tiles))[:, ci]).numpy()
    return [i for i in range(rows * cols) if float(p[i]) >= thr]


def picks_cascade(png: bytes, rows: int, cols: int, label: str,
                  conf_thr: float = 0.80) -> tuple[list[int], int]:
    """Fast classifier first; VLM only for the tiles the classifier is unsure about.

    Returns (picks, n_vlm_calls). The classifier is ~1 s for a whole grid; the per-tile VLM is
    ~1 s PER TILE. Most grids resolve with zero VLM calls, so this is both cheap and accurate -
    the two options measured in isolation were 41% at 1 s (classifier) and 54.5% at 9-18 s (VLM).
    """
    import torch
    import cv2 as _cv2
    import numpy as _np
    target = LABEL_TO_CLASS.get(label)
    if target is None:
        return [], 0
    net, classes, tf = _classifier()
    if target not in classes:
        return [], 0
    arr = _cv2.imdecode(_np.frombuffer(png, _np.uint8), _cv2.IMREAD_COLOR)
    if arr is None:
        return [], 0
    h, w = arr.shape[:2]
    th, tw = h // rows, w // cols
    tiles, raws = [], []
    for i in range(rows * cols):
        r, c = divmod(i, cols)
        t = arr[r * th:(r + 1) * th, c * tw:(c + 1) * tw]
        if t.size == 0:
            t = _np.zeros((th, tw, 3), _np.uint8)
        raws.append(t)
        tiles.append(tf(_cv2.cvtColor(t, _cv2.COLOR_BGR2RGB)))
    with torch.no_grad():
        probs = torch.softmax(net(torch.stack(tiles)), dim=1)
        conf, idx = probs.max(1)
    want_ix = classes.index(target)
    picks = {i for i, k in enumerate(idx.tolist()) if k == want_ix and float(conf[i]) >= conf_thr}
    unsure = [i for i in range(rows * cols) if float(conf[i]) < conf_thr]
    if not unsure:
        return sorted(picks), 0
    # Ask the VLM only about the unsure tiles.
    import httpx
    sys.path.insert(0, "C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange")
    os.environ.setdefault("HC_BRAIN_BASE", "http://127.0.0.1:1234")
    from service.solver import _brain_json
    prompt = (f'This is ONE tile cut from a larger photo in a reCAPTCHA grid. Task: "Select all '
              f'squares with {label}". The object is often cut off by the tile edge - answer yes if '
              f'ANY part of a {label} is visible in this tile. '
              'Reply STRICT JSON only: {"yes": true} or {"yes": false}')
    brain = httpx.Client(timeout=90.0)
    try:
        for i in unsure:
            t = _cv2.resize(raws[i], None, fx=2, fy=2, interpolation=_cv2.INTER_CUBIC)
            o = _brain_json(brain, prompt, _cv2.imencode(".jpg", t)[1].tobytes(), max_tokens=30) or {}
            if bool(o.get("yes")):
                picks.add(i)
    finally:
        brain.close()
    return sorted(picks), len(unsure)


def solve_one(sess, net, cats, sitekey: str, v: str, template: bytes):
    from solve_demo import (detect, picks_per_tile, picks_quadrants, tiles_for,
                            target_class, ask_grid_brain)
    # `doscaptcha` ("you're a robot") has NO image payload - the GET returns 400. It is a trust
    # verdict on the anchor, not a challenge we can fail, so re-acquiring is free and may draw a
    # solvable type. Measured: ~2/20 rounds were lost this way; retrying converts some of them.
    img, meta = None, {}
    for _try in range(int(os.environ.get('HC_ACQ_RETRY', '1'))):
        img, meta = acquire(sess, sitekey, v, template)
        if img is not None:
            break
        _err = str(meta.get("error") or "")
        if meta.get("type") != "doscaptcha" and "payload HTTP" not in _err:
            break
        time.sleep(0.4)
    if img is not None:
        meta["img_bytes"] = img
    if img is None:
        return False, f"acquire: {meta.get('error')}", meta
    label = meta.get("label")
    # 450x450 payloads are class `rc-image-tile-44`: a 4x4 GRID (16 tiles), not a point-click.
    # Measured in the browser (naturalWidth 450, className rc-image-tile-44). 300x300 is 3x3.
    try:
        import cv2 as _cv2
        import numpy as _np
        _arr = _cv2.imdecode(_np.frombuffer(img, _np.uint8), _cv2.IMREAD_COLOR)
        _h = _arr.shape[0] if _arr is not None else 300
    except Exception:
        _h = 300
    rows = cols = 4 if _h > 300 else 3
    want = target_class(label) if label else None
    # 1) per-tile VLM for the weak labels (measured best: 54.5% over 200 rounds). The 57k
    #    classifier was FASTER (1 s vs 9-18 s) but measured 41% live - its validation set does not
    #    match live challenge images, so it is only a fallback here.
    # Per-tile VLM for the weak labels: measured 54.5% over 200 rounds. The 57k classifier (41%)
    # and its in-domain fine-tune (34%) both lost - in-domain tiles come from PASSING rounds, so
    # training on them reinforces the bias that made us right in the first place.
    # HC_NO_VLM=1 disables every local-VLM call (the GPU is in use): the CPU classifier then
    # handles everything. Production stays local either way - no cloud in the serving path.
    _no_vlm = os.environ.get("HC_NO_VLM") == "1"
    # WHOLE-GRID first: one call on the full grid measured 87.8% recall / 89.6% precision on the
    # 17 hand-labelled grids, vs a per-tile yes/no loop that missed a true tile on the first grid.
    # Off by default so the 53% baseline stays measurable; HC_GRID_VLM=1 turns it on.
    picks = []
    # EXACT-GT tile net first when enabled. It is trained only on passing-round labels (exact by
    # the all-16 proof), so it carries no public-dataset bias. Falls through to the existing VLM
    # path when the checkpoint is missing or the label is unknown. HC_MC_NET=0 leaves it off.
    if (os.environ.get("HC_MC_NET") == "1" and label
            and (rows == 3 or os.environ.get("HC_MC_NET_4X4") == "1")):
        try:
            picks = picks_by_tile_net(img, rows, cols, label)
        except Exception as exc:  # noqa: BLE001 - a missing checkpoint must not fail the round
            print(f"[mc-net] {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            picks = []
    if not picks and not _no_vlm and os.environ.get("HC_GRID_VLM") == "1" and label:
        try:
            # Pass the FULL instruction, not the bare KG label: the measured run over-selected
            # badly on bare labels (8 of 9 tiles for "car"), which is the signature of the model
            # treating a one-word prompt as "which tiles could plausibly relate to cars".
            _instr = ("Select all images with " + label) if label else ""
            picks = ask_grid_brain(img, rows, cols, _instr)
            # A grid where more than 2/3 of tiles are picked is almost always a hallucinated
            # answer set, not a real dense grid; drop it rather than submit a guaranteed miss.
            if len(picks) > (rows * cols) * 2 // 3:
                picks = []
        except Exception:
            picks = []
    if not picks and not _no_vlm and os.environ.get("HC_CASCADE") == "1":
        # Classifier first, VLM only for tiles the classifier is unsure about (~1 s whole grid +
        # a handful of VLM calls) instead of 9-16 blind VLM calls. Designed for a strong VL model.
        try:
            picks, _nv = picks_cascade(img, rows, cols, label or "")
        except Exception:
            picks = []
    if not picks and not _no_vlm:
        # HC_VLM_ALL=1 routes EVERY label through the per-tile VLM. The old narrow
        # VLM_TILE_LABELS set was chosen against the weak coder model; with a dedicated VL model
        # that routing rule is probably obsolete and must be re-measured rather than inherited.
        _all = os.environ.get("HC_VLM_ALL") == "1"
        picks = (
            picks_per_tile_vlm(img, rows, cols, label)
            if (_all or VLM_TILE_LABELS is None or label in VLM_TILE_LABELS) else [])
    if not picks:
        picks = picks_by_classifier(img, rows, cols, label)
    # 3) COCO detector as a last resort (limited vocabulary, but never worse than nothing).
    # `multicaptcha` is a 4x4 split of ONE photo, so a per-tile classifier sees fragments of an
    # object and usually misses it (measured 8/22 = 36% vs 65% for `dynamic`). A whole-image
    # detection pass mapped to every tile the box OVERLAPS recovers the spanning objects.
    # VLM-ONLY option for 4x4 (measured 2026-11-19). Dumped a multicaptcha grid and read all 16 tiles:
    # the per-tile VLM answered [1,2,3,5,6,7,9,10,11] - every tile that genuinely shows a motorcycle,
    # including the background scooter at 2 - while the failing round had submitted that set PLUS 13,
    # an empty-road tile. 13 came from the detector UNION, not from the VLM. So the union can add a
    # false positive that the VLM never claimed.
    if rows == 4 and want is not None and os.environ.get("HC_MC_VLM_ONLY") == "1":
        pass
    elif rows == 4 and want is not None:
        try:
            _arr4, _dets4 = detect(net, cats, img)
            _ov = set(tiles_for_overlap(_arr4, _dets4, want, rows, cols))
            # NEGATIVE RESULT (measured 2026-11-18): making the detector's box set AUTHORITATIVE for
            # 4x4 (replacing the per-tile picks instead of unioning) measured 12/20 overall with
            # multicaptcha 5/12, vs 14/20 with multicaptcha 2/6 for the union. Small samples, no
            # clear win, and the detector UNDER-selects (a motorcycle round returned [6,9,10] where
            # the known-correct set was 7 tiles). Reverted to the union.
            picks = sorted(set(picks) | _ov)
        except Exception:
            pass
    # ACCEPTANCE-RULE PROBE (2026-11-19): a 4x4 hydrant round passed with 12 of 16 tiles picked
    # where only 3 tiles visibly contain the hydrant. That is only possible if the server accepts a
    # SUPERSET of the true set (under-selection fails, over-selection is tolerated). HC_MC_ALL=1
    # submits every tile to test that directly; it is a probe, never a serving default.
    if rows == 4 and want is not None and os.environ.get("HC_MC_ALL") == "1":
        picks = list(range(rows * cols))
    if not picks and want is not None:
        arr, dets = detect(net, cats, img)
        picks = sorted(set(picks_per_tile(net, cats, arr, want, rows, cols))
                       | set(tiles_for(arr, dets, want, rows, cols))
                       | set(tiles_for_overlap(arr, dets, want, rows, cols))
                       | set(picks_quadrants(net, cats, arr, want, rows, cols)))
    # 4) whole-grid VLM question if everything else declined - but NEVER when the GPU is not
    #    ours: this call is what silently JIT-loaded Qwen during HC_NO_VLM runs.
    if not picks and not _no_vlm:
        picks = ask_grid_brain(img, rows, cols, f"Select all squares with {label}")
    if not picks:
        return False, f"no picks (label={label})", meta
    token, info = submit(sess, sitekey, v, meta, picks)
    return bool(token), (f"token {token[:18]}..." if token else f"rejected {info}"), {**meta, "picks": picks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--template", default="reload_template.bin")
    ap.add_argument("--sitekey", default=DEMO_SITEKEY)
    ap.add_argument("--v", default=DEFAULT_V)
    ap.add_argument("--dump", default=None,
                    help="directory: save every challenge image + its picks/outcome as JSON")
    a = ap.parse_args()

    from curl_cffi import requests as creq
    from solve_demo import load_detector
    net, cats = load_detector()
    template = open(a.template, "rb").read()

    ok = 0
    for i in range(a.rounds):
        sess = creq.Session(impersonate="chrome")
        t = time.time()
        try:
            good, why, meta = solve_one(sess, net, cats, a.sitekey, a.v, template)
        except Exception as exc:  # noqa: BLE001
            good, why, meta = False, f"{type(exc).__name__}: {str(exc)[:70]}", {}
        ok += int(good)
        if a.dump and meta.get("img_bytes"):
            import os as _os
            _os.makedirs(a.dump, exist_ok=True)
            _ts = int(time.time() * 1000)
            _base = _os.path.join(a.dump, f"{_ts}_{(meta.get('label') or 'none').replace(' ', '_')}")
            open(_base + ".jpg", "wb").write(meta["img_bytes"])
            json.dump({"label": meta.get("label"), "type": meta.get("type"),
                       "picks": meta.get("picks"), "pass": bool(good), "why": why},
                      open(_base + ".json", "w"))
        print(f"round {i+1}: {'PASS' if good else 'FAIL'} ({why}) {time.time()-t:.1f}s "
              f"type={meta.get('type')} label={meta.get('label')} picks={meta.get('picks')}",
              flush=True)
    print(f"\n{ok}/{a.rounds} tokens (browserless)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
