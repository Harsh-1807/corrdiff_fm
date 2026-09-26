# -*- coding: utf-8 -*-
r"""
Train.py -- SRDRN training: a single, purely supervised, deterministic network.
================================================================================
Structurally the closest analogue in this package to corrdiff_fm/Regressor.py:
a k-fold loop, distributed support, EMA, a checkpoint carrying an `"arch"`
sub-dict for exact reconstruction, and validation on RMSE/MAE in mm/day
through the same tiled inference path used at deployment. The training-loop
skeleton below is deliberately close to Regressor.py's -- not copied wholesale
(SRDRN has no coarse-intensity augmentation import from a diffusion-adjacent
module, no Stage-2 counterpart to defer to, and a three-way loss switch instead
of a fixed L2) but built from the same template, because that template is a
serious, battle-tested piece of engineering and there is no reason to redesign
a k-fold-training-with-EMA-and-tiled-validation loop from scratch when a good
one already exists one directory over.

THERE IS NO STAGE 1 / STAGE 2 SPLIT HERE
------------------------------------------
Unlike corrdiff_fm, SRDRN is single-stage: one network maps LR conditioning
straight to the HR conditional mean, with no residual-diffusion corrector on
top. So there is exactly one training script, one checkpoint directory
(Config.CKPT_DIR), and one set of checkpoints per fold -- there is no analogue
of "train Stage 1 once, point every arm's Stage 2 at it".

WHY THREE LOSS VARIANTS, AND WHY WMAE BY DEFAULT
--------------------------------------------------
The paper trains three separate models, SRDRN-MSE / SRDRN-MAE / SRDRN-WMAE,
and reports WMAE as the best performer for reproducing precipitation extremes.
`Config.LOSS_KIND` selects among them (see Network.srdrn_loss); the default is
"wmae" for that reason. Note the important caveat, restated here because it
matters for how you read validation curves: `srdrn_loss("wmae", ...)` is THIS
IMPLEMENTATION'S RECONSTRUCTION of the paper's stated intent, not a verified
reproduction of the authors' exact weighting formula (see
Network.wmae_pixel_weight's docstring for the full argument). Whatever
LOSS_KIND you train with, `validate()` below always additionally reports
plain, UNWEIGHTED MAE and RMSE in mm/day -- this is the "WMAE vs
plain-MAE-computed-for-comparison-only" sanity check the README asks you to
watch: if training loss (say, WMAE) keeps improving while the always-reported
plain MAE stagnates or worsens, the weighting is trading off mean accuracy for
tail accuracy more aggressively than intended, and `Config.WMAE_ALPHA` is a
candidate to reduce.

HYPERPARAMETERS -- NOT THE PAPER'S BATCH=64/EPOCHS=500
---------------------------------------------------------
The paper's Adam lr=1e-4 is kept EXACTLY (it is a paper-stated hyperparameter
with no dependency on dataset size). Batch size and epoch count are NOT copied
from the paper: those were tuned for the paper's own India-region 0.8->0.1
degree dataset and offer no particular reason to transfer to this package's
k-fold protocol over a different domain and resolution. Instead this package
adopts corrdiff_fm/Regressor.py's own defaults (BATCH=16, EPOCHS=600, the same
warmup+cosine LR schedule, patience and EMA decay) since Regressor.py is the
closest architecture-and-protocol analogue available (deterministic,
single-stage, MSE-family loss, same data pipeline) and its hyperparameters
were already tuned against exactly this dataset and fold protocol. The paper's
lr value survives; its batch/epoch counts do not, and this paragraph is the
documentation the task explicitly asked for so nobody mistakes that for
copying the paper blindly.

Run:
  single GPU : python Train.py
  multi-GPU  : torchrun --nproc_per_node=2 Train.py
"""

import os
import copy
import math
import random
import traceback
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from Config import (HR_FILES, ORO_8KM, CKPT_DIR, DS_FACTOR, PRECIP_CH,
                    IN_CH_LR, PATCH, KFOLD_K, VAL_RATIO, SEED,
                    NUM_RES_BLOCKS, BASE_CH, N_UP_BLOCKS, LOSS_KIND,
                    WMAE_ALPHA, ensure_dirs)
from Dataset import ClimateDataset, get_climate_kfolds
from Network import SRDRN, expand_topo, denorm_precip_mmday, srdrn_loss, TOPO_CHANNELS
from Tiling import regress_tiled

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------
# HYPERPARAMETERS -- see the module docstring for what is paper-faithful (LR)
# vs. adapted-from-corrdiff_fm (everything else).
# ------------------------------------------------------------------------------
BATCH        = 16
ACCUM_STEPS  = 1
LR           = 1e-4        # paper-stated value, kept exactly
BETAS        = (0.9, 0.99)
MIN_LR       = 1e-6
EPOCHS       = 600
WARMUP_FRAC  = 0.02
PATIENCE     = 60
VAL_EVERY    = 2

WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9995

# Coarse-input augmentation. At training LR = avg_pool(HR); at deployment LR
# is a genuine coarser-resolution RCM field, which retains slightly more
# sub-grid intensity than a pure area mean. A small blend toward the block
# maximum makes the network less brittle to that shift when this package's
# cascade is later run 8km->2km the same way corrdiff_fm's is (see README).
# Set AUG_P = 0.0 to reproduce Dataset.__getitem__'s LR definition exactly.
AUG_P        = 0.15
AUG_MAX      = 0.20

PATCH_PER_SAMPLE = True


# ------------------------------------------------------------------------------
# DISTRIBUTED / UTILS
# ------------------------------------------------------------------------------

def setup():
    rank = int(os.environ.get("RANK", 0))
    ws = int(os.environ.get("WORLD_SIZE", 1))
    loc = int(os.environ.get("LOCAL_RANK", 0))
    if ws > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(loc)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return rank, ws, loc, torch.device(f"cuda:{loc}" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def pick_amp(dev):
    if dev.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    if dev.type == "cuda":
        return torch.float16, True
    return torch.float32, False


def build_sched(opt, total, warm, base_lr, min_lr):
    def fn(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
        cos = 0.5 * (1.0 + math.cos(math.pi * p))
        return (min_lr + (base_lr - min_lr) * cos) / base_lr
    return LambdaLR(opt, fn)


class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay, self.step = decay, 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        msd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        for k, v in self.shadow.state_dict().items():
            src = msd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src, alpha=1 - d)
            else:
                v.copy_(src)

    def state_dict(self):
        return self.shadow.state_dict()


# ------------------------------------------------------------------------------
# CROPPING / COARSENING -- same pattern as corrdiff_fm's Regressor.py.
# ------------------------------------------------------------------------------

def random_crop_hr(hr, topo, size, ds, per_sample=True):
    """Aligned random crop of HR and topo.

    Offsets are multiples of `ds` so the LR grid stays exactly nested. topo is
    cropped from the FULL-DOMAIN tensor rather than recomputed per patch --
    expand_topo standardizes elevation and lays down y/x coordinate channels
    over the whole domain, so recomputing it per patch would give every patch
    a different, mutually incomparable encoding.
    """
    H, W = hr.shape[-2:]
    if size is None or (size >= H and size >= W):
        return hr, topo
    ph, pw = min(size, H), min(size, W)
    if not per_sample:
        i = random.randrange(0, (H - ph) // ds + 1) * ds
        j = random.randrange(0, (W - pw) // ds + 1) * ds
        return hr[..., i:i + ph, j:j + pw], topo[..., i:i + ph, j:j + pw]

    hrs, tps = [], []
    for b in range(hr.shape[0]):
        i = random.randrange(0, (H - ph) // ds + 1) * ds
        j = random.randrange(0, (W - pw) // ds + 1) * ds
        hrs.append(hr[b:b + 1, ..., i:i + ph, j:j + pw])
        tb = topo[b:b + 1] if topo.shape[0] == hr.shape[0] else topo[:1]
        tps.append(tb[..., i:i + ph, j:j + pw])
    return torch.cat(hrs, 0), torch.cat(tps, 0)


def coarsen(hr, ds=DS_FACTOR, aug_p=0.0, aug_max=0.0):
    """LR = area mean of HR, optionally nudged toward the sub-grid maximum.

    Only the precip channel is perturbed -- blending temperature or pressure
    toward a block maximum is not physically meaningful.
    """
    avg = F.avg_pool2d(hr, kernel_size=ds, stride=ds)
    if aug_p <= 0.0 or aug_max <= 0.0 or torch.rand(()).item() > aug_p:
        return avg
    mx = F.max_pool2d(hr[:, PRECIP_CH:PRECIP_CH + 1], kernel_size=ds, stride=ds)
    alpha = torch.rand(hr.shape[0], 1, 1, 1, device=hr.device) * aug_max
    out = avg.clone()
    out[:, PRECIP_CH:PRECIP_CH + 1] = (
        (1 - alpha) * avg[:, PRECIP_CH:PRECIP_CH + 1] + alpha * mx)
    return out


# ------------------------------------------------------------------------------
# CHECKPOINT RECONSTRUCTION -- shared by Evaluate.py and Inference.py, exactly
# the role corrdiff_fm's TrainStage2.py plays for its own load_frozen_regressor.
# ------------------------------------------------------------------------------

def build_srdrn(arch, dev):
    """Rebuild an SRDRN from a checkpoint's own recorded `"arch"` sub-dict,
    never from whatever Config.py constants happen to be sitting around when
    Evaluate.py/Inference.py is run. The same "reconstruct from the
    checkpoint, not from the currently-imported module" discipline
    corrdiff_fm's TrainStage2.load_frozen_regressor and Inference.py's
    load_frozen_diffusion both enforce, and for the identical reason: editing
    NUM_RES_BLOCKS or BASE_CH in Config.py after training a fold must not
    silently produce a shape-mismatched (or worse, silently WRONG-shaped but
    loadable) reconstruction of an old checkpoint.
    """
    return SRDRN(
        in_channels=arch["in_channels"], out_channels=arch["out_channels"],
        base_channels=arch["base_channels"], num_res_blocks=arch["num_res_blocks"],
        n_up_blocks=arch["n_up_blocks"], ds_factor=arch["ds_factor"],
        topo_channels=arch["topo_channels"],
    ).to(dev)


def assert_precip_transform_compatible(ckpt, meta, path, rank, label="checkpoint"):
    """Refuse to mix precip scales between training and eval/inference.

    A checkpoint trained under a different PRECIP_SCALE or normalization
    convention will load without complaint and produce numbers that are wrong
    by whatever factor separates the two conventions -- exactly the failure
    mode corrdiff_fm's identically-named function guards against.
    """
    old = ckpt.get("precip_transform")
    if old is None:
        raise RuntimeError(
            f"[PRECIP SAFETY] {path} ({label}) has no 'precip_transform' -- refusing to use it.")
    bad = []
    for k in ("precip_scale", "norm_mean", "norm_std"):
        a, b = old.get(k), meta.get(k)
        if a is None or b is None or abs(a - b) > 1e-6:
            bad.append(f"{k}: {label}={a}  current={b}")
    if bad:
        raise RuntimeError(
            f"[PRECIP SAFETY] {path} ({label}) precip_transform mismatch:\n  "
            + "\n  ".join(bad) + "\nDo not mix precip scales between training and eval.")
    if rank == 0:
        print(f"[PRECIP SAFETY] {path} ({label}): OK.")


def load_frozen_srdrn(path, dev, meta, rank=0):
    """Load a trained SRDRN fold checkpoint for eval/inference: EMA weights
    preferred over raw, eval() mode, gradients disabled, precip-scale checked.
    """
    ck = torch.load(path, map_location=dev, weights_only=False)
    assert_precip_transform_compatible(ck, meta, path, rank, "SRDRN")
    arch = ck.get("arch")
    if arch is None:
        raise RuntimeError(
            f"{path} predates architecture-carrying checkpoints. Retrain with the "
            "current Train.py.")
    net = build_srdrn(arch, dev)
    net.load_state_dict(ck.get("ema_state_dict") or ck["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if rank == 0:
        print(f"[SRDRN] {path} ({'EMA' if ck.get('ema_state_dict') else 'raw'} weights, "
              f"loss={ck.get('loss_kind')}, val_rmse={ck.get('val_rmse_mmday')})")
    return net, ck


# ------------------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, topo_full, dev, pt, amp_dtype, tile_hr):
    """RMSE and MAE in mm/day (always UNWEIGHTED, regardless of LOSS_KIND --
    see module docstring on why this is the sanity-check metric to watch),
    plus the normalized-space loss actually being optimized (whichever
    LOSS_KIND is configured).

    Runs through the SAME tiled path used at inference, so the number selected
    on is the number you will get.
    """
    model.eval()
    tot = {"se": 0.0, "ae": 0.0, "n": 0.0, "loss_norm": 0.0, "nb": 0}
    for b in loader:
        hr = b["hr"].to(dev, non_blocking=True)
        B = hr.shape[0]
        topo = topo_full.expand(B, -1, -1, -1)
        lr = coarsen(hr, DS_FACTOR)
        target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()
        mu = regress_tiled(model, lr, topo, DS_FACTOR, tile_hr=tile_hr, amp_dtype=amp_dtype)

        tot["loss_norm"] += float(srdrn_loss(mu, target, LOSS_KIND, pt, WMAE_ALPHA))
        tot["nb"] += 1

        p = denorm_precip_mmday(mu, pt)
        o = denorm_precip_mmday(target, pt)
        tot["se"] += float(((p - o) ** 2).sum())
        tot["ae"] += float((p - o).abs().sum())
        tot["n"] += p.numel()

    n = max(tot["n"], 1.0)
    return (math.sqrt(tot["se"] / n), tot["ae"] / n,
            tot["loss_norm"] / max(tot["nb"], 1))


# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------

def train():
    rank, ws, loc, dev = setup()
    set_seed(SEED + rank)
    if rank == 0:
        ensure_dirs(CKPT_DIR)
    amp_dtype, need_scaler = pick_amp(dev)

    ds = ClimateDataset(HR_FILES, ORO_8KM, rank=rank)
    pt = ds.get_precip_transform_meta()

    oro = torch.as_tensor(ds.oro, dtype=torch.float32, device=dev)
    while oro.dim() < 4:
        oro = oro.unsqueeze(0)
    topo_full = expand_topo(oro)      # computed ONCE over the full domain

    if rank == 0:
        print(f"[SETUP] SRDRN | loss={LOSS_KIND} | patch={PATCH} | "
              f"domain={ds.H}x{ds.W} | ds_factor={DS_FACTOR} | "
              f"num_res_blocks={NUM_RES_BLOCKS} | base_ch={BASE_CH}")
        print(f"[SETUP] precip transform: {pt}")
        if LOSS_KIND == "wmae":
            print(f"[SETUP] WMAE alpha={WMAE_ALPHA} -- reconstructed weighting, "
                  "see Network.wmae_pixel_weight docstring for the honesty caveat.")
        if PATCH is None:
            print("[SETUP] [WARNING] PATCH=None ties this model to the 8 km domain size "
                  "and forfeits the 8km->2km cascade.")

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)

    for fi, fold in enumerate(folds):
        name = fold["name"]
        if rank == 0:
            print(f"\n{'='*64}\nSRDRN -- Fold {fi+1}/{KFOLD_K}  [{name}]")
            print(f"  train {len(fold['train_idx'])} | val {len(fold['val_idx'])} "
                  f"| test {len(fold['test_idx'])} (held out entirely)\n{'='*64}")

        tr, va = Subset(ds, fold["train_idx"]), Subset(ds, fold["val_idx"])
        tr_s = torch.utils.data.distributed.DistributedSampler(
            tr, num_replicas=ws, rank=rank, shuffle=True) if ws > 1 else None
        va_s = torch.utils.data.distributed.DistributedSampler(
            va, num_replicas=ws, rank=rank, shuffle=False) if ws > 1 else None
        tl = DataLoader(tr, batch_size=BATCH, sampler=tr_s, shuffle=(tr_s is None),
                        num_workers=4, pin_memory=True, drop_last=True)
        vl = DataLoader(va, batch_size=BATCH, sampler=va_s, shuffle=False,
                        num_workers=4, pin_memory=True)

        model = SRDRN(
            in_channels=IN_CH_LR, out_channels=1, base_channels=BASE_CH,
            num_res_blocks=NUM_RES_BLOCKS, n_up_blocks=N_UP_BLOCKS,
            ds_factor=DS_FACTOR, topo_channels=TOPO_CHANNELS,
        ).to(dev)
        if rank == 0:
            n_par = sum(p.numel() for p in model.parameters())
            print(f"  parameters: {n_par/1e6:.2f} M")
        if ws > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[loc],
                                                        output_device=loc)

        ema = EMA(model.module if hasattr(model, "module") else model, EMA_DECAY)
        opt = AdamW(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        total = (EPOCHS * len(tl)) // ACCUM_STEPS
        sched = build_sched(opt, total, int(total * WARMUP_FRAC), LR, MIN_LR)
        scaler = torch.amp.GradScaler("cuda") if need_scaler else None

        best, no_improve = float("inf"), 0

        for ep in range(EPOCHS):
            if ws > 1:
                tr_s.set_epoch(ep)
            model.train()
            opt.zero_grad(set_to_none=True)
            run = 0.0

            for i, b in enumerate(tl):
                hr = b["hr"].to(dev, non_blocking=True)
                topo_b = topo_full.expand(hr.shape[0], -1, -1, -1)
                hr, topo_b = random_crop_hr(hr, topo_b, PATCH, DS_FACTOR, PATCH_PER_SAMPLE)

                lr_x = coarsen(hr, DS_FACTOR, AUG_P, AUG_MAX)
                target = hr[:, PRECIP_CH:PRECIP_CH + 1]

                with autocast(device_type=dev.type, dtype=amp_dtype,
                              enabled=amp_dtype != torch.float32):
                    mu = model(lr_x, topo_b)
                    loss = srdrn_loss(mu.float(), target.float(), LOSS_KIND,
                                      pt, WMAE_ALPHA) / ACCUM_STEPS

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(tl):
                    if scaler is not None:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    if scaler is not None:
                        scaler.step(opt); scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    ema.update(model)
                    sched.step()

                run += loss.item() * ACCUM_STEPS

            avg = torch.tensor(run / max(1, len(tl)), device=dev)
            if ws > 1:
                dist.all_reduce(avg, op=dist.ReduceOp.SUM); avg /= ws
            if rank == 0:
                print(f"Epoch {ep+1:04d}/{EPOCHS} | {LOSS_KIND} {avg.item():.5f} | "
                      f"LR {sched.get_last_lr()[0]:.2e}")

            if (ep + 1) % VAL_EVERY == 0 or ep == EPOCHS - 1:
                rmse, mae, loss_n = validate(ema.shadow, vl, topo_full, dev, pt,
                                             amp_dtype, PATCH)
                m = torch.tensor([rmse, mae, loss_n], device=dev)
                if ws > 1:
                    dist.all_reduce(m, op=dist.ReduceOp.SUM); m /= ws
                rmse, mae, loss_n = m.tolist()
                if rank == 0:
                    print(f"  -> val RMSE {rmse:.4f} mm/day | MAE {mae:.4f} mm/day "
                          f"(unweighted, comparison metric) | "
                          f"normalized {LOSS_KIND} {loss_n:.5f}")

                if rmse < best:
                    best, no_improve = rmse, 0
                    if rank == 0:
                        torch.save({
                            "model_state_dict": (model.module.state_dict() if ws > 1
                                                 else model.state_dict()),
                            "ema_state_dict": ema.state_dict(),
                            # Everything needed to rebuild this exactly, mirroring
                            # corrdiff_fm's checkpoint convention.
                            "arch": {
                                "in_channels": IN_CH_LR,
                                "out_channels": 1,
                                "base_channels": BASE_CH,
                                "num_res_blocks": NUM_RES_BLOCKS,
                                "n_up_blocks": N_UP_BLOCKS,
                                "ds_factor": DS_FACTOR,
                                "topo_channels": TOPO_CHANNELS,
                            },
                            "precip_lr_ch": PRECIP_CH,
                            "patch": PATCH,
                            "loss_kind": LOSS_KIND,
                            "wmae_alpha": WMAE_ALPHA if LOSS_KIND == "wmae" else None,
                            "precip_transform": pt,
                            "val_rmse_mmday": rmse,
                            "val_mae_mmday": mae,
                            "val_loss_norm": loss_n,
                            "epoch": ep + 1,
                        }, os.path.join(CKPT_DIR, f"SRDRN_{name}_best.pth"))
                        print("  [*] saved new best")
                else:
                    no_improve += 1
                    if rank == 0:
                        print(f"  [!] no improvement for {no_improve * VAL_EVERY} epochs")

            if no_improve >= (PATIENCE // VAL_EVERY):
                if rank == 0:
                    print(f"Early stopping fold {name} (best RMSE {best:.4f} mm/day)")
                break

        del model, ema, opt, sched, scaler, tl, vl
        torch.cuda.empty_cache()

    if rank == 0:
        print(f"\nSRDRN training complete. Checkpoints in {CKPT_DIR}")
    if ws > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"CRITICAL ERROR: {e}")
            traceback.print_exc()
        if int(os.environ.get("WORLD_SIZE", 1)) > 1 and dist.is_initialized():
            dist.destroy_process_group()
        raise
