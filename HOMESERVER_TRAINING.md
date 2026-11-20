# Homeserver training bundle — in-domain 4x4 tile classifier

Purpose: train the reCAPTCHA 4x4 (multicaptcha) tile classifier on the **exact GT** (a passing
round's picks ARE the true tile set; acceptance was proven exact — all-16 failed 0/18) using the
RTX 3060 Ti instead of the workstation CPU, where the fine-tune measured 2/16 (overfit) and the
frozen head 4/25 (feature-limited).

## Contents

- `gt_exact/` — 101 exact 4x4 grids + 545 exact 3x3 grids (md5-deduped) + `manifest.json`
  (`{image, type, label, rows, cols, picks, source}`).
- `train_head_only.py` — frozen 57k backbone + fitted linear head (fast, ~1 min GPU). Run:
  `python train_head_only.py 4` and `python train_head_only.py 3`.
- `train_4x4_tiles.py` — full fine-tune with a grid-level holdout; prints the holdout exact rate.
- `browserless.py` — provides `_classifier()` / `MC_NORM`; the scripts import it.
- `tile_cls57k.pt` — the public-dataset backbone used for initialisation.

## Commands

On the workstation (this repo):

```bash
scp -r homeserver_training <user>@10.10.10.236:~/captcharesearch/
```

On the homeserver:

```bash
cd ~/captcharesearch/homeserver_training
~/brain_env/bin/python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
~/brain_env/bin/python train_head_only.py 4      # frozen head, 4x4
~/brain_env/bin/python train_head_only.py 3      # frozen head, 3x3
~/brain_env/bin/python train_4x4_tiles.py       # fine-tune + holdout exact rate
```

If torch/torchvision are missing in `brain_env`: `pip install torch torchvision` (CUDA build).

Then copy the checkpoints back:

```bash
scp <user>@10.10.10.236:~/captcharesearch/homeserver_training/tile_cls_4x4.pt .
scp <user>@10.10.10.236:~/captcharesearch/homeserver_training/tile_cls_3x3.pt .
```

## What we need back

- `tile_cls_4x4.pt` and `tile_cls_3x3.pt` (the trained checkpoints).
- The holdout exact rates printed by the scripts (grid-level, 30% by class).

Integration on return: `browserless.picks_by_tile_net` already loads `tile_cls_{rows}x{rows}.pt`;
the 4x4 path is enabled with `HC_MC_NET_4X4=1` (currently off because the CPU models are weak). The
3x3 path is the dynamic candidate.

## Why this matters

The 4x4 multicaptcha is the last reCAPTCHA blocker (33–43% live vs the 80% goal). The CPU
measurements show the limit is the model/features, not the data volume (86 grids → 4/25 frozen);
a GPU fine-tune with a proper schedule is the untested step. The 3x3 net is also a candidate for
the dynamic family (86% live, 90% goal).
