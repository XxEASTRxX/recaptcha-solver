# reCAPTCHA solver — status

Goal: enterprise-grade reCAPTCHA solver, local-only serving, as fast as the hCaptcha stack.

## Established by measurement (2026-11-14)

| probe | result | conclusion |
|---|---|---|
| raw HTTP `POST /recaptcha/api2/reload` (demo key, correct `v`, `c`, cookies, browser headers) | `200` `["rresp",null,null,null,null,null,0]` | **browserless-HTTP is closed** |
| anchor page | `200`, 55 KB, sets `_GRECAPTCHA`, `#recaptcha-token` ~1849 ch | anchor is fetchable |
| release hash from `api.js` | `0lEGiwO4Qwx5R9ArxUy0mBpf` | must be sent as `v` or asset paths come back empty |
| headless Chromium: click checkbox | reload returns a **real rresp**, `bframe` opens, `/payload` fetched | **browser transport works** |
| `/payload` body | one **450x450 JPEG**, 40 KB; all tiles are CSS crops | tiles are deterministic crops of one image |
| bframe DOM | 3x3 or 4x4 grid, instruction text in `.rc-imageselect-desc` | question is readable without OCR |
| COCO detector (torchvision fasterrcnn_mobilenet 320) on the grid | `motorcycle .97, person .94, bus .49` in ~2 s | **detector is the right decision path** |
| per-tile VLM yes/no (local 27B) | 0/2 rounds, 50-68 s/round | too slow and too weak; fallback only |

reCAPTCHA's object vocabulary overlaps COCO almost exactly (car, bus, truck, motorcycle,
bicycle, train, boat, fire hydrant, traffic light). Classes COCO lacks (stairs, crosswalk,
chimneys, palm trees) need the VLM fallback.

## Detector: per-tile is mandatory (measured 2026-11-14)

On a real 3x3 grid whose task was "fire hydrant":

| approach | result |
|---|---|
| whole 450 px grid, FasterRCNN-320 | `0.29 bicycle` - **hydrant missed** |
| whole grid upscaled 2x | `0.32 bicycle` - still missed |
| **per-tile, each tile resized to 320** | **tile 1 -> `0.99 fire hydrant`**, tile 4 -> `0.96 bus` |

Per-tile wall time: **0.6 s for 9 tiles**. So the MVP now runs the detector per tile.

## Vocabulary gap is now the main blocker

A later run was 100% **crosswalks**, which COCO does not contain, so `target_class()`
returned None and the VLM fallback ran: **0/5 tokens at 44-58 s per round**. reCAPTCHA
classes split into two tiers:

| tier | classes | plan |
|---|---|---|
| COCO | car, bus, truck, motorcycle, bicycle, train, boat, fire hydrant, traffic light | per-tile detector (fast) |
| non-COCO | crosswalk, stairs, chimneys, palm trees | dedicated CV (crosswalk = parallel white bars) or a student net on labelled tiles |

23 grids with sidecars are saved in `capture/` (`<stamp>.jpg` + `.json` with task/rows/cols/picks).

## Non-COCO classes: five approaches tried, all fail (2026-11-14)

Ground truth for grid `1794677201938_3x3.jpg` ("crosswalks", 3x3) established by zooming:
**tiles {2, 8}** contain crosswalks; tile 0 does not.

| approach | result |
|---|---|
| white-bar contours (V>140, S<70, elongated) | 0 bars in 2 and 8, 4 in tile 0 - **inverted** |
| Hough lines, angle-clustered | every tile 3-15 lines (road edges) - **no separation** |
| FFT periodicity in the 8-40 px band | 0.32-0.38 on all tiles - **no separation** |
| local brain, whole grid, one call | `[2,3,4,7,8]` vs GT `[2,8]` - recall ok, **precision poor** |
| local brain, per-tile yes/no on the shortlist | `{2,3}` vs GT `{2,8}` - **wrong** |

So crosswalks (and stairs, chimneys, palm trees) are the same problem the hCaptcha THREE
family had: **no hand-crafted signal separates them; they need labelled tiles and a
student net.** 10 crosswalk grids (90 tiles) are already captured in `capture/`.

## Crosswalks: CLIP fails, the local brain has perfect RECALL (2026-11-14)

Two grids hand-labelled by zooming (GT is reliable):
`1794677201938_3x3` -> `{2, 8}` and `1794677247003_3x3` -> `{4, 5}`.

| method | grid 1 | grid 2 | verdict |
|---|---|---|---|
| CLIP ViT-B-32 zero-shot (pos-neg margin) | `[2,3,8]` at thr .03 | `[4]` - **true tile 5 scored 0.014** | recall unreliable, unusable |
| local brain, whole grid | `[2,3,4,7,8]` | `[0,4,5]` | **recall 2/2 on both**, precision poor |

So the shape of the solution is a **two-stage cascade**: the brain shortlists (high recall,
cheap, one call), and a precise verifier decides. CLIP cannot be that verifier. The
verifier needs labels -> a small student net on labelled tiles (the foods/difficon pattern).

## Crosswalk training data: NOT VIABLE at current capture volume (2026-11-14)

All 5 unique 300x300 crosswalk grids are now hand-labelled (`crosswalk_gt.json`):
`{2,8}`, `{4,5}`, `{0,1,8}`, `{0,2}`, `{5,7,8}` = **12 positives out of 45 tiles**.

Why there are only 5: the capture directory holds 14 crosswalk files but **9 are duplicates**
or 100x100 thumbnails, and 2 are 450x450 *single photos* (not tiled montages - my
size->grid inference was wrong for those). Deduping by md5 leaves 5 grids.

12 positives cannot train a verifier that generalises (the foods student needed 621 cells;
difficon needed 9324 patches). So the "train a crosswalk student" plan is **blocked on
capture volume**, not on labelling effort - each retry of a failed challenge re-serves the
same image, so rounds produce duplicates faster than unique grids.

## Shipped crosswalk prompt, measured on all 5 GT grids (2026-11-14)

`ask_grid_brain` (ONE call per grid, tiles numbered row-major) replaced the 9-call per-tile
loop. Against `crosswalk_gt.json`:

| grid | picks | GT | hit | fp |
|---|---|---|---|---|
| 1794677201938 | 2,3,8 | 2,8 | 2/2 | 1 |
| 1794677247003 | 0,4,5 | 4,5 | 2/2 | 1 |
| 1794677304824 | 0,1,8 | 0,1,8 | 3/3 | 0 |
| 1794680415834 | 2,8 | 0,2 | 1/2 | 1 |
| 1794682097524 | 5,6,7,8 | 5,7,8 | 3/3 | 1 |

**recall 11/12 = 91.7%, precision 11/15 = 73.3%**, one grid exactly right. That is a
usable candidate generator: recall is the property that matters, and `vary_picks` now
drops the last pick on the first retry (the over-selection is the common error).

## White-paint filter: precision 73.3% -> 84.6%, recall unchanged (2026-11-14)

`_white_frac` prunes brain picks whose tile has almost no bright low-saturation pixels.
Measured on the same 5 grids: every true crosswalk tile scores >= 0.123, the two clearly
wrong picks score 0.013 and 0.072 -> floor at 0.10.

| | before | after |
|---|---|---|
| recall | 11/12 = 91.7% | **11/12 = 91.7%** |
| false positives | 4 | **2** |
| precision | 73.3% | **84.6%** |
| grids exactly correct | 1/5 | **2/5** |

It is a pruner, not a detector: one remaining false positive scores 0.63 because a
sunlit wall is also bright and desaturated.

## Two grid styles exist - do not infer grid size from the image (2026-11-14)

* **Montage**: 300x300, 9 unrelated photos. All 5 labelled crosswalk grids are this style.
* **Single photo**: 450x450, one scene whose 4x4 tiles are crops of it (e.g. a house with
  stairs, task "Select all squares with stairs"). Adjacent tiles then all contain the
  target, so exact-match grading is much harder than on a montage.

Filenames like `*_4x4.jpg` came from my own `size -> grid` guess and are **wrong** for
these; the live loop reads rows/cols from the DOM table, which is correct. Only the
montage grids were usable as GT, which is why `crosswalk_gt.json` has 5 entries.

## 25-round gate: 1/25 = 4% - the round is lost to RE-SERVED grids (2026-11-14)

Whole-grid fallback + `vary_picks` shipped, then measured over 25 rounds: **1/25 tokens**.
The log localises the failure precisely - it is not the answer, it is the loop:

    round 14: FAIL (no token) 36.6s
      grid 1: 3x3 want=fire hydrant picks=[5,6,8] dets=[...same...]
      grid 2: 3x3 want=fire hydrant picks=[5,6]   dets=[...same...]
      grid 3: 3x3 want=fire hydrant picks=[5,6,8,0] dets=[...same...]
      grid 4: 3x3 want=fire hydrant picks=[5,6,8] dets=[...same...]

`dets` identical on grids 1-4 means **the same challenge image is re-served** after each
wrong answer, so one bad answer consumes all four steps and the round cannot recover.
`vary_picks` changed the set but not the outcome.

Fix shipped: on a repeated image the loop clicks `.rc-imageselect-refresh` and takes a NEW
challenge instead of answering the dead one (`vary_picks` remains only as the fallback for
when that element is missing).

## Detector union replaces the single path (2026-11-14, offline-measured)

Replaying 36 captured COCO-class grids through both paths:

| outcome | grids |
|---|---|
| whole-grid covers tiles per-tile misses (objects spanning tile borders) | 9 |
| per-tile covers tiles whole-grid misses (small objects downsampled away) | 7 |
| both agree | 20 |

The fatal case the union fixes: `want=car, per-tile=[], whole-grid=[5]` - the old code
submitted NOTHING while the detector had found the object at 0.88. Seven captured grids
were in that class.

Shipped: `picks = sorted(set(picks_per_tile) | set(tiles_for))`. Note the tension - the
union raises recall and also raises the pick count, and over-selection is the other
observed failure mode, so the live rate still has to arbitrate.

## The live gate is REPUTATION-CONFOUNDED - stop hammering (2026-11-14)

Three gates on the same demo site, same machine, increasing volume:

| gate | prior volume | tokens | throttle signals in log |
|---|---|---|---|
| first run (8 rounds) | 0 | **2/8 = 25%** | none |
| rec5 (25 rounds) | ~60 rounds | 1/25 = 4% | none |
| rec6 (25 rounds) | ~85 rounds | 1/25 = 4% | 3x `rejected: Please try again.` |
| rec7 (12 rounds, union + 0.60 tile threshold) | ~110 rounds | 0/12 = 0% | - |

The rate falls monotonically as cumulative automated volume rises, and the explicit
`Please try again.` rejections are reCAPTCHA's own throttle response. So the later gates
cannot be read as an algorithm regression: they are measuring a site that has learned
this client. **The 25% figure from the first run is the only unthrottled measurement,
and it is 8 rounds - not evidence either way.**

Consequence: further live measurement on this IP is not informative until the reputation
resets. Do not draw conclusions from rec5/rec6/rec7 about answer quality.

## TASK/PAYLOAD DESYNC - the real accuracy bug (2026-11-14, CONFIRMED)

The loop read the instruction text from the DOM and only THEN waited for the next
`/payload` response. When a new challenge arrived in between, it answered the PREVIOUS
question against the NEW image.

Proof (three consecutive captures, byte-identical photos):

| capture | task it recorded | detector on that image |
|---|---|---|
| 1794684457482_4x4 | motorcycle | bicycle **1.00** |
| 1794684450717_4x4 | bus | bicycle **1.00** |
| 1794684444069_4x4 | bicycle | bicycle **1.00** |

One image cannot be all three. Across the 46 unique 300x300 montage grids the wanted
class was absent (whole-grid, >=0.5) in **60.9%** - part desync, part the known
downsampling weakness, but far too high to be only the latter.

Fixed: the payload is awaited FIRST, then the task/rows/cols are read, and the payload URL
is re-checked afterwards - if a newer image landed during the read, the step is redone, so
wording and image always come from the same challenge.

This is a plausible explanation for most of the sub-10% live rates: the answers were
computed for the wrong question.

## COCO policy settled on hand-labelled GT (2026-11-14)

`coco_gt.json`: 6 montage grids hand-labelled (19 positives). Every policy scored 100%
precision - **over-selection was never the failure**; recall was.

| policy | recall | FP | precision |
|---|---|---|---|
| union, per-tile thr 0.10 | 16/19 = 84.2% | 1 | 94.1% |
| **union, per-tile thr 0.15 (SHIPPED)** | **16/19 = 84.2%** | **0** | **100%** |
| union, thr 0.20-0.30 | 15/19 = 78.9% | 0 | 100% |
| union, thr 0.60 | 13/19 = 68.4% | 0 | 100% |
| per-tile alone, thr 0.30 | 14/19 = 73.7% | 0 | 100% |
| whole-grid alone | 8/19 = 42.1% | 0 | 100% |

Two corrections to my own earlier reasoning:
* `TILE_SCORE_THR = 0.60` (shipped on a guess from the log) **cost 10 points of recall and
  bought nothing** - reverted to the measured 0.15.
* The union beats both single paths (84.2% vs 73.7% vs 42.1%), so it stays.

Remaining misses are objects the detector does not fire on at all (a car and a bicycle
tile at any threshold), not a tuning problem.

## Desync prevalence measured across the whole capture history (2026-11-14)

Grouping all captures by image hash:

| metric | value |
|---|---|
| captures | 361 |
| unique images | 137 |
| images carrying MORE THAN ONE task | **22** |
| captures belonging to those images | 69 (**19.1%**) |
| minimum definitely-wrong pairings | **27** |

An image cannot be both `buses` and `fire hydrants`, so at least 27 submitted answers were
computed for a different question than the one displayed. The fix (payload awaited first,
task read second, URL re-checked) removes the whole class.

Verification of the fix is by construction rather than by a rate: the live site is
throttling, so a post-fix token rate would not be attributable. The desync signature is
the direct test - re-run this grouping after the next live session and it must be zero.

## Crosswalk: GT corrected to 17 grids, ceiling measured at 87.8% / 89.6% (2026-11-15)

The 12 new grids tripled the GT. The first measurement (85.0% recall / **63.0%** precision)
was **wrong because my labels were wrong**: an audit sheet of the 20 "false positives" at
300 px showed 9 of them are clear zebra crossings whose stripes sit at a tile edge and were
invisible in the 480 px montage view. Montage-scale labelling is not reliable; tile zoom is.

After correction (`crosswalk_gt.json`: 17 grids, 49 positives, 8 ambiguous excluded from
BOTH sides):

| configuration | recall | precision |
|---|---|---|
| **brain whole-grid + white floor 0.10 (SHIPPED)** | **43/49 = 87.8%** | **43/48 = 89.6%** |
| strict prompt naming the failure modes | 36/49 = 73.5% | 83.7% |
| frozen ResNet18 + linear head, LOO-grid | 40/49 = 81.6% | 75.5% |
| asphalt floor 0.55 | 41.9% | 94.7% |
| `bars_on_road` feature | 2.3% | 100% |
| white floor sweep 0.10-0.20 | 100% -> 81% | 89.6% -> 89.7% |

The five residual errors are diverse (lane dashes on a highway, a house, a street with a
motorcycle, a hillside, a white van) and **no cheap statistic separates them**; the white
floor cannot (their white fractions 0.20-0.44 sit inside the true-tile range 0.12-0.63).

So the objective's >=95% precision is **not met for crosswalks**: measured ceiling is
89.6% at 87.8% recall. COCO classes do meet it (84.2% recall / 100% precision).

## Per-pick verification MEETS the objective: 83.7% recall / 100% precision (2026-11-15)

The fix for crosswalk precision is a SECOND OPINION on only the tiles the brain already
picked (~3 calls per grid), not a 9-tile sweep (which missed true tiles) and not a cheap
image statistic (white floor, asphalt, bar counting all failed):

| configuration | recall | precision |
|---|---|---|
| brain whole-grid + white floor | 43/49 = 87.8% | 43/48 = 89.6% |
| **+ per-pick verification (SHIPPED)** | **41/49 = 83.7%** | **41/41 = 100%** |

Zero false positives across all 17 grids. It gives up 4 true tiles to remove all 5 errors,
which is the right trade against the objective's >=80% recall / >=95% precision bar.

## Desync: three attempts, each measured by `check_desync.py` (2026-11-15)

The regression test is `check_desync.py` (exit 1 if any image carries two tasks). Each fix
was judged by it, not by assumption:

| attempt | change | result |
|---|---|---|
| 1 | await payload, then read task, re-check URL | 2/6 unique images still multi-task |
| 2 | + require the task text stable on two reads 220 ms apart | 3/18 still multi-task |
| 3 | + sync on the rendered tile (`img.rc-image-tile-33[src]` carries the payload URL) | 2/12; 1 was benign reuse, 1 real |
| 4 | + refuse to answer when unsynced (refresh or skip) | 1/16 multi-task, **0 real desync** |

Final state (`check_desync.py --since-fix`): 29 captures, 16 unique images, 1 multi-task,
**0 images with a claimed class absent**, 1 unverifiable (`stairs`, checked by eye - the photo
contains both the bicycle and the concrete steps). The refined test requires positive evidence
of absence, because reCAPTCHA genuinely reuses photos across challenges.

Two false trails worth keeping: the tile has **no** background-image (it is an
`<img class="rc-image-tile-33" src=".../payload?p=...">`), so a `backgroundImage` sync
never matched (`sync=0` on every grid); and a multi-task image is **not** automatically a
desync - `fc4d1b7cf3` genuinely contains a bicycle (0.99 per-tile) and a motorcycle (1.00),
so reCAPTCHA does reuse one photo across two challenges.

## Proxy rotation added; it does NOT rescue the token rate (2026-11-15)

`solve_demo.py --proxies-file <pool>` now rotates one exit per round (Playwright `proxy`
dict; `parse_proxy` splits `http://user:pass@host:port`). The pool itself is genuine - 1000
entries, 5 distinct exit IPs in 6 samples - and it clears hCaptcha's direct-IP 429.

For reCAPTCHA it fails earlier: **0/6 rounds, every one `Page.goto: Timeout 45000ms`**.
The exits cannot load google.com in time, so the token rate stays unmeasurable this way.

Live-measurement matrix, all measured today:

| target | direct IP | proxy pool |
|---|---|---|
| hCaptcha (`api.hcaptcha.com`) | hard **429** (0/5) | 2/3 probe, then **0/8** at 86-177 s/round |
| reCAPTCHA demo (google.com) | throttled, 0/12 | **page load timeout**, 0/6 |

## TOKEN RATE MEASURED: 50% headed vs ~6% headless (2026-11-15)

Headless Chromium is the dominant reCAPTCHA detection signal. Same code, same site, only
the browser mode changed:

| mode | runs | tokens | rate |
|---|---|---|---|
| headless | 2/8, 1/25, 1/25, 0/12, 1/10, 1/5 | 5/85 | **5.9%** |
| **headed (`--headed`)** | 2/4, 6/12 | **8/16** | **50.0%** |

The earlier "reputation" reading was wrong: the site was not throttling a good client, it
was scoring a headless one. `--headed` is now the default recommendation; headless remains
available for offline work only.

## COCO recall 84.2% -> 94.7% with a quadrant pass (2026-11-15)

The residual misses were objects split across a tile border, which neither single path finds:
per-tile sees the fragments, whole-grid downsamples them away. Adding overlapping 2x2 windows
(`picks_quadrants`, mapped back to tiles by box centre) fixes exactly those:

| path | recall | FP | precision |
|---|---|---|---|
| per-tile@0.15 + whole-grid@0.30 | 16/19 = 84.2% | 0 | 100% |
| **+ overlapping quadrants (SHIPPED)** | **18/19 = 94.7%** | **0** | **100%** |

It recovers the car in tile 5 and the bicycle in tile 2 and adds nothing else.

## Crosswalk recall is capped by stage one, not the verifier (2026-11-15)

Two attempts to raise the 83.7% recall, both NULL:

* **three-level verification** (`yes`/`no`/`unsure`) - keeps all 41 true picks and 0 FPs, i.e. the
  binary prompt was already keeping every true pick it was shown.
* **broad first stage** ("list every tile where a crossing could be present, even partly") then
  verify - recall stays **41/49 = 83.7%** but FPs go 0 -> 2, precision 100% -> 95.3%.

The 8 missed positives are never candidates under either prompt, so no verifier can recover them.
Shipped configuration (narrow stage + binary per-pick verify) remains optimal at 83.7% / 100%.

## reCAPTCHA v3: pipeline works, but the demo cannot discriminate (2026-11-15)

v3 has no puzzle. `grecaptcha.execute()` returns a token and Google attaches a score 0.0-1.0 that
the SITE's backend reads via siteverify. `v3_score.py` drives Google's demo page, which echoes the
siteverify JSON, and parses the score.

| mode | trials | scores | mean |
|---|---|---|---|
| headed | 5 | 0.9, 0.9, 0.9, 0.9, 0.9 | **0.900** |
| headless | 5 | 0.9, 0.9, 0.9, 0.9, 0.9 | **0.900** |

**Ten identical scores means the page cannot measure client quality**: `recaptcha-demo.appspot.com`
is Google's own domain, so it is trusted no matter what we send. A useful v3 harness needs a sitekey
on a NEUTRAL domain that we control, plus its siteverify secret - then the score becomes a real
function of fingerprint, behaviour and IP.

What is proven: tokens are issued to our client in ~3 s, in both modes, and the score can be read.

## v2 INVISIBLE supported: 13/16 = 81% (2026-11-15)

`solve_demo.py --url <page> --invisible` drives the invisible flow: there is no anchor
checkbox, the challenge is triggered by the form's `button.g-recaptcha`, and on success the
form submits so the token appears only in the POST result (`[g-recaptcha-response] => ...`).
`read_token` handles both shapes. A renderer death mid-submit (`TargetClosedError` at 0.0 s)
is retried once - it is a page-lifecycle race, not a solve failure.

| page | tokens | rate |
|---|---|---|
| v2 checkbox demo | 10/18 | 56% |
| **v2 invisible demo** | **13/16** | **81%** |

Invisible often passes with 0 grids (a silent pass); when a grid does appear the same COCO
path solves it.

## reCAPTCHA v3: 2/8 = 25% (2026-11-14, first non-zero)

`solve_demo.py` now issues **real reCAPTCHA tokens on the demo page**, 2 of 8 rounds, ~33 s
per round. Also fixed: the payload listener kept whichever payload arrived first (often a
100x100 thumbnail); it now keeps the LARGEST payload per grid, so every capture is the real
300x300 grid.

Known inefficiency: after a rejected answer the SAME challenge is re-served, so the loop's
remaining grid attempts resubmit the identical answer (visible as identical detections on
grids 2-4). Should re-read and vary the answer, or stop early.

## Current MVP (`solve_demo.py`)

Browser (Playwright) drives the demo page, captures `/payload` bytes, runs the detector,
maps detection centres to tile indices, clicks them, submits, and loops up to 4 grids.
Measured: **0/3 tokens, ~30 s per round**.

Known defects (in priority order):
1. **Empty-pick grids re-submit the same payload** — when nothing is detected we click
   verify and the identical challenge is re-served, so grids 2-4 repeat. Need to treat
   "no detection" as a real decision (submit, or re-read the grid) rather than a no-op.
2. **Detector resolution** — the 320 px model downsamples the 450 px grid; small objects
   are missed. Try `fasterrcnn_resnet50_fpn` or upscaling the grid before detection.
3. **No per-tile ground truth** — pass/fail is per-challenge, so tile accuracy cannot be
   measured directly. Fix: label a few grids by hand/agent (the payloads are saved) and
   score the detector against them.
4. **Multi-grid budget** — demo sometimes needs more than 4 grids; verify the loop exit.

## Next steps

1. Save every `(payload, task, picks, outcome)` tuple to `capture/` for offline analysis.
2. Build a tile-level GT set by labelling ~20 grids, then measure detector per-tile
   precision/recall per class — this is the equivalent of the hCaptcha corpus.
3. Swap in a stronger detector and re-measure on that GT.
4. Only then wire the VLM fallback and the multi-grid loop, and measure end-to-end token
   rate on the demo.

## 2026-11-18 - multicaptcha: hypothesis refuted, ground truth established, one negative result

**Run with the captchachalange venv.** This project's own `.venv` has no `httpx`, so `browserless.py`
dies at import with `ModuleNotFoundError`. Every measurement below used
`C:/Users/EAS/Documents/Projects VisualStudioCode/captchachalange/.venv/Scripts/python.exe`.

### Hypothesis REFUTED: multicaptcha does NOT require successive grids

I suspected the consistent multicaptcha failures were a protocol problem - a correct first grid being
answered with a SECOND grid that we mistook for a rejection. Instrumented `submit()` with
`HC_DEBUG_SUBMIT=1` and captured the raw bodies:

```
[userverify] type=multicaptcha len=27 body=["uvresp",null,null,null,0]
[userverify] type=dynamic     len=27 body=["uvresp",null,null,null,0]
```

Every rejection is **27 characters with no follow-up challenge**, identical for `dynamic` and
`multicaptcha`, and multicaptcha rounds DO pass on a single submission (round 1 of the capture run:
`type=multicaptcha label=bicycle picks=[0,2,6,15]` -> token). So these are genuine wrong answers.

### Ground truth from dumped grids

`browserless.py --dump` saves the 450x450 payload. Two motorcycle multicaptcha rounds:

| round | picks | outcome |
|---|---|---|
| passing | `[1,3,5,6,7,9,10]` | **PASS** |
| failing | `[1,2,3,5,6,7,9,10,11,13]` | FAIL |

On the image (a road with one prominent motorcycle centre-left and one smaller one right) the passing
set is exactly the tiles those two bikes' boxes cover; the failing set is that set **plus [2,11,13]**,
all empty road. So **over-selection is the multicaptcha failure mode**, not a transport defect.

### NEGATIVE RESULT: detector-authoritative 4x4 selection

Since the code UNIONed the per-tile picks with the detector's box-overlap set (which can only ADD), I
tried making the detector's set authoritative when it finds the target.

| config | overall | multicaptcha |
|---|---|---|
| union (before) | 14/20 = 70% | 2/6 = 33% |
| detector-authoritative | 12/20 = 60% | 5/12 = 42% |

Not a win, and the detector UNDER-selects (a motorcycle round returned `[6,9,10]` where the
known-correct set was 7 tiles). **Reverted to the union**, with the numbers recorded in the code
comment. Sample sizes are small (6 and 12 multicaptcha rounds); a 100-round gate is needed before
believing either arm.

### Why neither arm fixes multicaptcha: the detector is NOT stable across grids

**CORRECTION (same session)**: an earlier draft of this section claimed the detector returned an
"identical 6-tile set on both grids at 0.15". That was a misread of my own output. The real sweep
(`detect(net, cats, raw_bytes)` - it decodes bytes, not an array):

| grid | 0.15 | 0.05 | 0.02 |
|---|---|---|---|
| passing (submitted `[1,3,5,6,7,9,10]`) | `[5,9]` | `[3,5,6,7,9,10]` | `[3,5,6,7,9,10]` |
| failing (submitted that set + `[2,11,13]`) | `[3,5,6,7,9,10]` | `[2,3,5,6,7,9,10,11,13,14]` | `[2,3,5,6,7,8,9,10,11,13,14]` |

So the detector's set depends heavily on the threshold AND on the photo: at 0.15 the passing grid
yields only 2 tiles while the failing one yields 6. There is no threshold that reproduces the known
correct set on both.

**Rider augmentation** (union vehicle boxes with `person` boxes that overlap them - reCAPTCHA's
"motorcycle" means bike PLUS rider, but COCO files the rider as `person`):

| grid | motorcycle-only @0.15 | + overlapping persons |
|---|---|---|
| passing | `[5,9]` | `[3,5,6,9]` (missing 1,7,10) |
| failing | `[3,5,6,7,9,10]` | `[1,2,3,5,6,7,9,10]` (extra 2) |

Neither matches truth on either grid. The box geometry is the reason: the prominent bike's box
`[166,156,243,280]` starts at y=156 (row 1) and never touches tile 1, while the rider's helmet sits at
y~88-112 (row 0).

**Conclusion**: multicaptcha is not fixable by a selection heuristic. It needs a per-tile classifier
trained in-domain on these 4x4 split photos. No change shipped; nothing was left enabled.

## 2026-11-19 - reCAPTCHA re-measured: 26/40 = 65%; VLM-only for 4x4 REFUTED

Current browserless numbers (same code as the last session):

| type | result |
|---|---|
| dynamic | **21/26 = 81%** |
| multicaptcha | **5/14 = 36%** |
| **total** | **26/40 = 65%** |

### Ground truth established for a 4x4 multicaptcha

Dumped a grid (`rc_dump/1795030297403_motorcycle.jpg`) and read all 16 tiles as a numbered montage
(`grid2_tiles.jpg`). The passing round had submitted `[1,3,5,6,7,9,10]`; the failing round submitted
that plus `2, 11, 13`. Reading the tiles:

- **2 DOES show a motorcycle** (a scooter in the background) - so the passing set was a SUBSET the
  server accepted, and my earlier "extra = wrong" reading was wrong for that tile;
- **11 has a small object at its bottom edge**, plausibly a bike part;
- **13 is empty road** - a genuine false positive.

### Two-pass self-consistency REFUTED

Ran the per-tile VLM twice on both grids: the two passes were **byte-identical** (temperature 0), so
agreement filtering does nothing. Self-consistency is not available to this model.

### VLM-only (no detector union) REFUTED

The empty-road tile 13 came from the detector UNION, not from the VLM (the per-tile pass alone
returned `[1,2,3,5,6,7,9,10,11]`, no 13). So `HC_MC_VLM_ONLY=1` was added to skip the union:

| arm | total | multicaptcha |
|---|---|---|
| union (shipped default) | **26/40 = 65%** | 5/14 = 36% |
| VLM-only | 22/40 = 55% | 5/16 = 31% |

**Worse.** The union earns its place; the flag stays default-off. Recorded, not shipped.

## 2026-11-19 - multicaptcha: refined 4x4 prompt 6/14 = 43%; acceptance rule under test

The union-confidence premise was re-checked against the agent labels and is mostly WRONG. Rendering
`labels_agent.json`'s "detector-added false positives" showed the bus grid's tiles 5/6 and the
motorcycle grid's tile 9 DO contain the object; the labels were incomplete. A VLM yes/no confirmed
6 of the 7 "FP" tiles. Do not build a confidence gate on that sample.

**Shipped**: the 4x4 per-tile prompt now excludes sky/road/trees/buildings/people/painted signs
(`HC_MC_PROMPT=0` restores the old wording). On two fresh grids it fixed the visible defects: bus
[4,5,8,9,10] exact (was [4,5,8,9,10,12,14], incl. a painted "BUS" marking), car [5,6,7,9,10] (was
[2,4,5,6,7], incl. empty sky). The detector union added nothing on either grid.

| 60-round gate | result |
|---|---|
| dynamic | 39/46 = 85% |
| multicaptcha | **6/14 = 43%** (was 5/14 = 36%) |
| total | **45/60 = 75%** |

Multicaptcha failures are single-tile selection errors (motorcycle adds a white car; bicycle adds a
wall; bus [4,5] under-selects the wheels in 8/9). The 57k classifier is exact on one bicycle grid but
returns NO motorcycle on a motorcycle grid, so it cannot replace the VLM.

**Acceptance rule is unclear**: a hydrant round PASSED with 12 of 16 tiles selected where only 3 tiles
visibly contain the hydrant. That is only possible if the server tolerates a SUPERSET. `HC_MC_ALL=1`
(probe flag) submits all 16 tiles on 4x4 rounds to test this directly; a 60-round probe is running.
A f-string bug in the new prompt crashed 4 rounds of an earlier gate (fixed; JSON braces moved out of
the f-string).

## 2026-11-19 - multicaptcha: three more selection framings REFUTED; exact GT harvested

Acceptance is exact: `HC_MC_ALL=1` (all 16 tiles) failed **0/18** multicaptcha rounds. Over-selection
is fatal, so every fix must be near-exact per tile. The 12/16 hydrant pass is an unexplained outlier.

| framing | result on 6 hand-checked grids |
|---|---|
| reference-anchored comparison (known example + tile, "same kind?") | under-selects partials: motorcycle [1,6,10] vs [1,2,5,6,9,10]; bus [4,5] vs [4,5,8,9]; 0/6 exact |
| 2x2 quadrant classifier (4 calls, "which sub-tiles?") | over-selects: bus 9 tiles, bicycle 15 tiles; 1/6 exact |
| 3-pass per-tile majority vote | unanimous but wrong on 4/6 (model is consistently wrong, not noisy) |

Whole-photo detector sweep (6 hand-checked grids): mf=0.04 gives tp=32 fp=8 fn=1, 3/6 exact; no
single coverage threshold separates the bus's true partial wheels (0.06-0.10) from the motorcycle's
false edge tiles (0.16-0.23). The detector is exact on 2/6 at mf=0.15.

Exact GT harvest (`gt_harvest.py`): **13 4x4 grids** (7 passing + 6 hand-verified) and **97 3x3
dynamic grids** (exact because acceptance is exact). 13 grids are too few to train a 4x4 tile
classifier; more passing multicaptcha rounds are needed. The 3x3 set is large enough for a 3x3 model,
but the earlier in-domain transfer measured 34% live, so transfer to 4x4 is not assumed.

## 2026-11-19 - reference-anchored / 2x2 / majority REFUTED; exact GT = 13 4x4 + 97 3x3

All three remaining selection framings failed on 6 hand-checked grids: reference-anchored comparison
under-selects partials (0/6 exact), 2x2 quadrant classification over-selects (1/6), and 3-pass
per-tile majority is unanimous but wrong on 4/6. The whole-photo detector at mf=0.04 reaches 3/6
exact but no coverage threshold separates the bus's true partial wheels (0.06-0.10) from the
motorcycle's false edge tiles (0.16-0.23).

`gt_harvest.py` produced **13 exact 4x4 grids** (7 passing + 6 hand-verified) and **97 exact 3x3
dynamic grids**. 13 grids cannot train a 4x4 tile classifier; the 3x3 set can train a 3x3 model, but
the earlier in-domain transfer measured 34% live, so transfer to 4x4 is not assumed. Multicaptcha
stays at 6/14 = 43%, blocked on exact labels.

## 2026-11-20 - 3x3->4x4 transfer and GrabCut refinement REFUTED; harvesting more 4x4 labels

`probe_transfer.py`: 57k resnet18 features + per-class logistic probes trained on the 97 exact 3x3
grids (873 tiles, 294 positives), evaluated on the 13 exact 4x4 grids — **1/13 exact** at thr 0.3/0.4,
0/13 at 0.5/0.6. 3x3 tile appearance does not transfer to 4x4 splits.

GrabCut refinement of each COCO box (foreground coverage per tile, thr 0.05-0.20) — **1/6 exact** on
the hand grids; recovers some partials but drops others, and the car grid yields nothing.

The only remaining path is a 4x4-specific model, so five 60-round gates are harvesting passing
multicaptcha rounds (exact GT). Each new passing 4x4 grid is added to `gt_exact/` by `gt_harvest.py`.

## 2026-11-20 - exact-GT 4x4 classifier: 52 grids; frozen probe and hybrid REFUTED

Five 60-round harvest gates (39+39+37+43+47 = 205/300 = 68%) grew the exact GT to **52 4x4 grids**
and **263 3x3 grids** (a passing round is exact GT). `probe_4x4_probe.py` (frozen 57k features +
per-class logistic, 37 train / 15 holdout) reaches 4/15 exact. Tile-level on the same holdout:
VLM precision 0.98 / recall 0.68; net 0.86 / 0.79; union 0.87 / 0.92 but exact 6-7/15 at every net
threshold 0.5-0.95 (VLM-only 7/15). The VLM under-selects; the net adds recall but each added tile
must be exact, so the union cannot beat VLM-only. A full fine-tune (train_4x4_tiles.py) was stopped:
CPU cost is high and the frozen probe leaves little headroom. Multicaptcha remains 43%, blocked on
a stronger model or GPU-trained 4x4 classifier; the exact labels are now available for that.

## 2026-11-20 - exact-GT 3x3 tile net: 71/80 = 89% held-out; 4x4 still weak

The five-gate harvest grew the exact GT to 52 4x4 / 263 3x3 grids. Frozen 57k features + per-class
logistic heads, grid-level holdout: **3x3 71/80 = 89% exact**, 4x4 4/15. `train_head_only.py` fits
the head on the frozen backbone and saves `tile_cls_3x3.pt` / `tile_cls_4x4.pt`;
`browserless.picks_by_tile_net` + `HC_MC_NET=1` use them (net first, existing path as fallback).
The 3x3 path is the live dynamic candidate; the 4x4 path is not shipped (43% live is already better
than anything measured from the 4x4 net).

## 2026-11-20 - 3x3 exact-GT net live gate (HC_MC_NET=1) queued

The 3x3 net reached 71/80 = 89% exact on held-out harvested grids. Live gate (60 rounds) runs with
HC_MC_NET=1; the net applies to 3x3 (dynamic) only by default (HC_MC_NET_4X4=1 would enable the weak
4x4 net). Dynamic baseline 39/46 = 85%, multicaptcha 6/14 = 43%.

## 2026-11-20 - 3x3 net LIVE: dynamic 36/42 = 86% (baseline 85%), no lift

`HC_MC_NET=1`, 60 rounds: total 42/60 = 70%, dynamic **36/42 = 86%**, multicaptcha 6/18 = 33% (4x4
path unchanged; window variance). The 89% held-out rate did not translate to a live lift, but the net
is deterministic and fast (~1.1-1.4 s/round). No mc-net errors. Next: net vs VLM complementarity on
the 80 held-out exact grids; if each is right where the other is wrong, an ensemble can clear 90%.

## 2026-11-20 - net/VLM complementarity: VLM 73/80, net 70/80, fusion +1

On the 80 held-out exact 3x3 grids: VLM 73/80 = 91%, net 70/80, intersection 73 (the VLM's picks are
a subset of the net's), union 70. Net-right/VLM-wrong 5, VLM-right/net-wrong 8, both wrong 2.
"VLM plus net tiles at prob >= 0.98" reaches 74/80 with precision 0.97 and recall 1.00 - a 1-grid
gain, not a route to 90% live (live VLM 85%, net 86%). The remaining errors are 7 high-confidence
false tiles the VLM rejects; a better verifier is needed. Dynamic 90% remains unmet; multicaptcha
33-43% remains blocked on a GPU-trained 4x4 model (52 exact labels now exist).

## 2026-11-20 - 4x4: neighbour context and "partials count" prompt REFUTED

Neighbour-context features (own 512 + mean of 8 neighbours, linear and MLP heads, same 70/30 split):
**3/15 exact** (own-features linear was 4/15), so context adds noise, not signal. A prompt variant
that explicitly counts wheels/bumpers/mirrors/the object's own shadow returns the SAME picks as the
shipped prompt on all six hand-checked grids (1/6 exact), so the VLM's per-tile judgement is stable
but wrong on partials - wording is not the lever. A short full fine-tune (8 epochs) on the 52 exact
grids is running; it is the remaining local route to a usable 4x4 model.
