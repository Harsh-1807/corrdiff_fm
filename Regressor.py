# -*- coding: utf-8 -*-
r"""
Regressor.py -- STAGE 1: the conditional mean  mu = E[precip_HR | LR, topo]
===========================================================================
Trained with a plain L2 (mean-squared-error) loss, per the paper: "The model is
optimized using a Mean-Squared-Error (MSE) loss during training" (Sec. 5.2.1).

THIS FILE IS IDENTICAL IN THE EDM AND FLOW-MATCHING PACKAGES. Train it ONCE and
point both Stage-2 arms at the same checkpoints. If the two arms sat on
different Stage-1 models, every difference in their CRPS would be confounded by
the difference in the mean they are correcting, and the comparison would be
worthless.

WHY L2 IS THE RIGHT LOSS HERE, SPECIFICALLY
-------------------------------------------
Not a stylistic choice. The entire CorrDiff decomposition

    x = E[x|y] + (x - E[x|y])
        \_ Stage 1 _/  \_ Stage 2 _/

rests on Stage 1 actually estimating the conditional MEAN, because that is what
makes the residual zero-mean, and a zero-mean residual is what gives the
variance reduction in Eq. (2) that makes the diffusion problem easier than
modelling p(x) directly.

The squared-error minimizer IS the conditional mean. An L1 loss would converge
to the conditional MEDIAN instead -- which for precipitation, with its huge
point mass at zero and a long right tail, is dramatically lower than the mean
and often exactly zero. The residual would then be strongly positively biased,
Stage 2 would have to spend capacity correcting a systematic offset, and the
zero-mean assumption underpinning the whole method would simply be false.

So: L2, and the temptation to "robustify" it with Huber should be resisted for
this stage. (Stage 2 is different -- there a robust loss is defensible, which is
why LOSS_KIND exists there and not here.)

Note the loss is computed in the normalized log1p space the model works in, not
in mm/day. Squared error on raw precipitation would be dominated almost
entirely by a handful of extreme wet grid points.

Run:
  single GPU : python Regressor.py
  multi-GPU  : torchrun --nproc_per_node=2 Regressor.py
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

from Config import (HR_FILES, ORO_8KM, REGRESSOR_CKPT_DIR, DS_FACTOR, PRECIP_CH,
                    IN_CH_LR, PATCH, KFOLD_K, VAL_RATIO, SEED, REG_BASE_CH,
                    REG_CHANNEL_MULT, REG_NUM_BLOCKS, ensure_dirs)
from Dataset import ClimateDataset, get_climate_kfolds
from Network import (CorrDiffRegressor, expand_topo, denorm_precip_mmday,
                     augment_coarse_intensity, TOPO_CHANNELS)
from Tiling import regress_tiled

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------
# HYPERPARAMETERS
# ------------------------------------------------------------------------------
BATCH        = 16          # Stage 1 is much cheaper than Stage 2
ACCUM_STEPS  = 1
LR           = 2e-4        # paper Sec. 5.3.2
BETAS        = (0.9, 0.99)
MIN_LR       = 1e-6
EPOCHS       = 600
WARMUP_FRAC  = 0.02
PATIENCE     = 60
VAL_EVERY    = 2

DROPOUT      = 0.10
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9995

# Coarse-input augmentation. At training LR = avg_pool(HR); at 2 km deployment
# LR is a real 8 km RCM field, which retains slightly more sub-grid intensity
# than a pure area mean. A small blend toward the block maximum makes Stage 1
# less brittle to that shift. Set AUG_P = 0.0 to reproduce Dataset.__getitem__
# exactly.
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
# CROPPING
# ------------------------------------------------------------------------------

def random_crop_hr(hr, topo, size, ds, per_sample=True):
    """Aligned random crop of HR and topo.

    Offsets are multiples of `ds` so the LR grid stays exactly nested. topo is
    cropped from the FULL-DOMAIN tensor rather than recomputed per patch --
    expand_topo standardizes elevation and lays down y/x coordinate channels
    over the whole domain, so recomputing per patch would give every patch a
    different, mutually incomparable encoding.
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
# VALIDATION
# ------------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, topo_full, dev, pt, amp_dtype, tile_hr):
    """RMSE and MAE in mm/day, plus the normalized-space L2 that is actually
    being optimized.

    Runs through the SAME tiled path used at inference, so the number selected
    on is the number you will get. Reported in mm/day because that is the unit
    a decision gets made in -- normalized MSE is not interpretable and hides
    whether the model is useful.
    """
    model.eval()
    tot = {"se": 0.0, "ae": 0.0, "n": 0.0, "mse_norm": 0.0, "nb": 0}
    for b in loader:
        hr = b["hr"].to(dev, non_blocking=True)
        B = hr.shape[0]
        topo = topo_full.expand(B, -1, -1, -1)
        lr = coarsen(hr, DS_FACTOR)
        target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()
        mu = regress_tiled(model, lr, topo, DS_FACTOR, tile_hr=tile_hr, amp_dtype=amp_dtype)

        tot["mse_norm"] += float(((mu - target) ** 2).mean())
        tot["nb"] += 1

        p = denorm_precip_mmday(mu, pt)
        o = denorm_precip_mmday(target, pt)
        tot["se"] += float(((p - o) ** 2).sum())
        tot["ae"] += float((p - o).abs().sum())
        tot["n"] += p.numel()

    n = max(tot["n"], 1.0)
    return (math.sqrt(tot["se"] / n), tot["ae"] / n,
            tot["mse_norm"] / max(tot["nb"], 1))


# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------

def train():
    rank, ws, loc, dev = setup()
    set_seed(SEED + rank)
    if rank == 0:
        ensure_dirs(REGRESSOR_CKPT_DIR)
    amp_dtype, need_scaler = pick_amp(dev)

    ds = ClimateDataset(HR_FILES, ORO_8KM, rank=rank)
    pt = ds.get_precip_transform_meta()

    oro = torch.as_tensor(ds.oro, dtype=torch.float32, device=dev)
    while oro.dim() < 4:
        oro = oro.unsqueeze(0)
    topo_full = expand_topo(oro)      # computed ONCE over the full domain

    if rank == 0:
        print(f"[SETUP] Stage 1 regressor | loss=L2 (MSE) | patch={PATCH} | "
              f"domain={ds.H}x{ds.W} | ds_factor={DS_FACTOR}")
        print(f"[SETUP] precip transform: {pt}")
        if PATCH is None:
            print("[SETUP] [WARNING] PATCH=None ties this model to the 8 km domain size "
                  "and forfeits the 8km->2km cascade. Stage 2 must match whatever is set here.")

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)

    for fi, fold in enumerate(folds):
        name = fold["name"]
        if rank == 0:
            print(f"\n{'='*64}\nStage 1 -- Fold {fi+1}/{KFOLD_K}  [{name}]")
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

        model = CorrDiffRegressor(
            in_channels=IN_CH_LR, out_channels=1, base_channels=REG_BASE_CH,
            channel_mult=REG_CHANNEL_MULT, num_blocks=REG_NUM_BLOCKS,
            topo_channels=TOPO_CHANNELS, dropout=DROPOUT, ds_factor=DS_FACTOR,
            precip_lr_ch=PRECIP_CH,
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
                    # L2 / MSE. The squared-error minimizer is the conditional
                    # mean, which is exactly what the residual decomposition
                    # requires -- see the module docstring.
                    loss = F.mse_loss(mu.float(), target.float()) / ACCUM_STEPS

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
                print(f"Epoch {ep+1:04d}/{EPOCHS} | L2 {avg.item():.5f} | "
                      f"LR {sched.get_last_lr()[0]:.2e}")

            if (ep + 1) % VAL_EVERY == 0 or ep == EPOCHS - 1:
                rmse, mae, mse_n = validate(ema.shadow, vl, topo_full, dev, pt,
                                            amp_dtype, PATCH)
                m = torch.tensor([rmse, mae, mse_n], device=dev)
                if ws > 1:
                    dist.all_reduce(m, op=dist.ReduceOp.SUM); m /= ws
                rmse, mae, mse_n = m.tolist()
                if rank == 0:
                    print(f"  -> val RMSE {rmse:.4f} mm/day | MAE {mae:.4f} mm/day "
                          f"| normalized MSE {mse_n:.5f}")

                if rmse < best:
                    best, no_improve = rmse, 0
                    if rank == 0:
                        torch.save({
                            "model_state_dict": (model.module.state_dict() if ws > 1
                                                 else model.state_dict()),
                            "ema_state_dict": ema.state_dict(),
                            # Everything Stage 2 needs to rebuild this exactly.
                            "in_channels": IN_CH_LR,
                            "out_channels": 1,
                            "base_ch": REG_BASE_CH,
                            "channel_mult": list(REG_CHANNEL_MULT),
                            "num_blocks": REG_NUM_BLOCKS,
                            "topo_channels": TOPO_CHANNELS,
                            "ds_factor": DS_FACTOR,
                            "precip_lr_ch": PRECIP_CH,
                            "patch": PATCH,
                            "loss": "l2_mse",
                            "precip_transform": pt,
                            "val_rmse_mmday": rmse,
                            "val_mae_mmday": mae,
                            "epoch": ep + 1,
                        }, os.path.join(REGRESSOR_CKPT_DIR, f"Regressor_{name}_best.pth"))
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
        print("\nStage 1 complete. Both Stage-2 arms must now point at "
              f"{REGRESSOR_CKPT_DIR} -- do not retrain it per arm.")
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
