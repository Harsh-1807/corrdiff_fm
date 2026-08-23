# -*- coding: utf-8 -*-
"""
TrainStage2.py -- STAGE 2: the residual corrector.
===================================================
IDENTICAL IN THE EDM AND FLOW-MATCHING PACKAGES. The parameterization enters
through `Param.py` and nowhere else -- two function calls, `build_train_batch`
and `sample_residual`. Same data, same crops, same frozen Stage 1, same residual
normalization, same conditioning, same EMA, same optimizer, same CRPS protocol,
same seeds.

Trains on the normalized residual

    r = (precip_HR - mu_hat - res_mean) / res_std        mu_hat = frozen Stage 1

Both moments are used. Training subtracts res_mean and inference adds it back;
in the version this replaces, res_mean appeared only on the reconstruction side,
injecting a constant precipitation bias into every downscaled field.

MODEL SELECTION IS ON ENSEMBLE CRPS, NOT TRAINING LOSS
-------------------------------------------------------
For a probabilistic downscaler the denoising loss is only a surrogate, and the
two genuinely diverge: the loss keeps falling while the ensemble spread quietly
collapses. CRPS is a proper scoring rule on the quantity that matters, in
mm/day. Spread-skill is logged alongside so calibration is visible rather than
inferred.

Validation runs through the SAME tiled sampling path as deployment, so the CRPS
you early-stop on is the CRPS you get.

Run:
  single GPU : python TrainStage2.py
  multi-GPU  : torchrun --nproc_per_node=2 TrainStage2.py
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
                    IN_CH_LR, PATCH, KFOLD_K, VAL_RATIO, SEED, EVAL_SEED,
                    UNET_IN_CH, BASE_CH, CHANNEL_MULT, NUM_RES_BLOCKS,
                    ATTN_LEVELS, USE_SPECTRAL, DROPOUT, ensure_dirs)
from Dataset import ClimateDataset, get_climate_kfolds
from Network import (UNet, CorrDiffRegressor, expand_topo, denorm_precip_mmday,
                     TOPO_CHANNELS)
from Tiling import regress_tiled
from Param import (PARAM, CFG, CFG_KEY, CKPT_DIR, SAMPLE_STEPS,
                   build_train_batch, sample_residual, nfe_count)

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------
# HYPERPARAMETERS -- identical for both arms
# ------------------------------------------------------------------------------
LOSS_KIND    = "mse"       # "mse" or "pseudo_huber"
HUBER_C_SC   = 0.00054

BATCH        = 8
ACCUM_STEPS  = 2
LR           = 2e-4        # paper Sec. 5.3.2
BETAS        = (0.9, 0.99)
MIN_LR       = 1e-6
EPOCHS       = 1500
WARMUP_FRAC  = 0.02
PATIENCE     = 100
VAL_EVERY    = 5

WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9999

PATCH_PER_SAMPLE = True

# Conditioning dropout. Only useful if you intend to sample with guidance != 1.
# At 0.10 with guidance 1.0 you would train a tenth of every batch to ignore its
# conditioning and then never use the unconditional branch.
CFG_DROP_P   = 0.0
GUIDANCE     = 1.0

# Coarse-input augmentation, for cascade robustness -- see Regressor.coarsen.
LR_AUG_P     = 0.15
LR_AUG_MAX   = 0.20

CRPS_MEMBERS = 8
CRPS_BATCHES = 3


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


def arch_dict():
    """Everything needed to rebuild this network. Written into the checkpoint so
    Inference.py never reconstructs it from constants sitting in this file --
    which is correct exactly until someone edits a hyperparameter, after which it
    either raises (lucky) or loads a mismatched model (unlucky)."""
    return {
        "in_channels": UNET_IN_CH, "out_channels": 1, "base_channels": BASE_CH,
        "channel_mult": list(CHANNEL_MULT), "num_res_blocks": NUM_RES_BLOCKS,
        "attn_levels": list(ATTN_LEVELS), "use_spectral": USE_SPECTRAL,
        "topo_channels": TOPO_CHANNELS,
    }


def build_unet(arch, dev, dropout=0.0):
    return UNet(
        in_channels=arch["in_channels"], out_channels=arch["out_channels"],
        base_channels=arch["base_channels"], channel_mult=tuple(arch["channel_mult"]),
        num_res_blocks=arch["num_res_blocks"], dropout=dropout,
        attn_levels=tuple(arch["attn_levels"]), use_spectral=arch["use_spectral"],
        topo_channels=arch["topo_channels"],
    ).to(dev)


# ------------------------------------------------------------------------------
# STAGE-1 LOADING AND SAFETY
# ------------------------------------------------------------------------------

def assert_precip_transform_compatible(ckpt, meta, path, rank, label="checkpoint"):
    """Refuse to mix precip scales between stages.

    A checkpoint trained under the old PRECIP_SCALE=24.0 convention will load
    without complaint and produce numbers that are wrong by a factor of 24. This
    is the guard against that.
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
            + "\n  ".join(bad) + "\nDo not mix precip scales between Stage 1 and Stage 2.")
    if rank == 0:
        print(f"[PRECIP SAFETY] {path} ({label}): OK.")


def load_frozen_regressor(path, dev, meta, rank):
    ck = torch.load(path, map_location=dev, weights_only=False)
    assert_precip_transform_compatible(ck, meta, path, rank, "Stage-1 regressor")
    reg = CorrDiffRegressor(
        in_channels=ck.get("in_channels", IN_CH_LR),
        out_channels=ck.get("out_channels", 1),
        base_channels=ck["base_ch"],
        channel_mult=tuple(ck["channel_mult"]),
        num_blocks=ck.get("num_blocks", 2),
        topo_channels=ck.get("topo_channels", TOPO_CHANNELS),
        dropout=0.0,                     # deterministic conditioning mean
        ds_factor=ck.get("ds_factor", DS_FACTOR),
        precip_lr_ch=ck.get("precip_lr_ch", PRECIP_CH),
    ).to(dev)
    reg.load_state_dict(ck.get("ema_state_dict") or ck["model_state_dict"])
    reg.eval()
    for p in reg.parameters():
        p.requires_grad_(False)
    if rank == 0:
        print(f"[STAGE-1] {path} ({'EMA' if ck.get('ema_state_dict') else 'raw'} weights, "
              f"loss={ck.get('loss')}, val_rmse={ck.get('val_rmse_mmday')})")
        if ck.get("patch") not in (None, PATCH):
            print(f"[STAGE-1] [WARNING] Stage 1 trained at patch={ck.get('patch')} but "
                  f"Stage 2 is set to PATCH={PATCH}. They must match, or the tiled "
                  "cascade feeds each stage a grid size it never saw.")
    return reg


# ------------------------------------------------------------------------------
# LOSS
# ------------------------------------------------------------------------------

def per_sample_loss(pred, target, kind=LOSS_KIND):
    d = pred.float() - target.float()
    if kind == "pseudo_huber":
        c = HUBER_C_SC * math.sqrt(target[0].numel())
        return (torch.sqrt(d * d + c * c) - c).mean(dim=[1, 2, 3])
    return (d * d).mean(dim=[1, 2, 3])


# ------------------------------------------------------------------------------
# CROPPING AND CONDITIONING
# ------------------------------------------------------------------------------

def random_crop_hr(hr, topo, size, ds, per_sample=True):
    """Aligned random crop of HR and topo; offsets are multiples of `ds` so the
    LR grid stays exactly nested. topo is cropped from the FULL-DOMAIN tensor,
    never recomputed per patch -- expand_topo standardizes elevation and lays
    down y/x coordinate channels domain-wide."""
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
    avg = F.avg_pool2d(hr, kernel_size=ds, stride=ds)
    if aug_p <= 0.0 or aug_max <= 0.0 or torch.rand(()).item() > aug_p:
        return avg
    mx = F.max_pool2d(hr[:, PRECIP_CH:PRECIP_CH + 1], kernel_size=ds, stride=ds)
    alpha = torch.rand(hr.shape[0], 1, 1, 1, device=hr.device) * aug_max
    out = avg.clone()
    out[:, PRECIP_CH:PRECIP_CH + 1] = (
        (1 - alpha) * avg[:, PRECIP_CH:PRECIP_CH + 1] + alpha * mx)
    return out


@torch.no_grad()
def build_conditioning(hr, topo, regressor, ds=DS_FACTOR, amp_dtype=torch.float32,
                       tile_hr=None, aug_p=0.0, aug_max=0.0):
    """-> (target, mu_hat, cond_stack, topo), all on the HR grid."""
    lr_x = coarsen(hr, ds, aug_p, aug_max)
    target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()
    mu = regress_tiled(regressor, lr_x, topo, ds, tile_hr=tile_hr, amp_dtype=amp_dtype)
    lr_up = F.interpolate(lr_x, size=hr.shape[-2:], mode="bilinear", align_corners=False)
    return target, mu, torch.cat([mu, lr_up.float()], 1), topo


# ------------------------------------------------------------------------------
# RESIDUAL SCALE
# ------------------------------------------------------------------------------

@torch.no_grad()
def estimate_residual_stats(regressor, loader, topo_full, dev, amp_dtype,
                            tile_hr, max_batches=40):
    """Stage 2 must see a target with zero mean and unit variance. For EDM that
    is what sigma_data=1.0 asserts; for flow matching it is what makes x1
    comparable in scale to the x0 ~ N(0,I) it is paired against. Either way the
    noise schedule is calibrated to it."""
    n = s1 = s2 = 0.0
    for k, b in enumerate(loader):
        if k >= max_batches:
            break
        hr = b["hr"].to(dev, non_blocking=True)
        topo = topo_full.expand(hr.shape[0], -1, -1, -1)
        target, mu, _, _ = build_conditioning(hr, topo, regressor, amp_dtype=amp_dtype,
                                              tile_hr=tile_hr)
        r = (target - mu).float()
        n += r.numel(); s1 += r.sum().item(); s2 += (r * r).sum().item()
    mean = s1 / max(n, 1)
    var = max(s2 / max(n, 1) - mean * mean, 1e-12)
    return mean, math.sqrt(var)


# ------------------------------------------------------------------------------
# ENSEMBLE VALIDATION
# ------------------------------------------------------------------------------

def crps_ensemble(ens, obs):
    """Fair (unbiased) ensemble CRPS. ens [M,B,1,H,W], obs [B,1,H,W].

    The second term is the pairwise spread correction that removes the small-M
    bias -- without it, an 8-member ensemble is systematically penalized
    relative to a 32-member one and the metric is not comparable across
    ensemble sizes.
    """
    M = ens.shape[0]
    mae = (ens - obs.unsqueeze(0)).abs().mean(0)
    if M > 1:
        pair = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum(dim=(0, 1))
        spread = pair / (2.0 * M * (M - 1))
    else:
        spread = torch.zeros_like(mae)
    return (mae - spread).mean(), spread.mean()


@torch.no_grad()
def ensemble_validate(net, loader, regressor, topo_full, dev, pt, res_mean, res_std,
                      amp_dtype, tile_hr, members=CRPS_MEMBERS, steps=SAMPLE_STEPS,
                      n_batches=CRPS_BATCHES):
    net.eval()
    gen = torch.Generator(device=dev).manual_seed(EVAL_SEED)
    tot = {"crps": 0.0, "rmse": 0.0, "spread": 0.0, "n": 0}

    for k, b in enumerate(loader):
        if k >= n_batches:
            break
        hr = b["hr"].to(dev, non_blocking=True)
        B = hr.shape[0]
        topo = topo_full.expand(B, -1, -1, -1)
        target, mu, cond, topo = build_conditioning(hr, topo, regressor,
                                                    amp_dtype=amp_dtype, tile_hr=tile_hr)
        shape = (B, 1, *target.shape[-2:])

        members_mm = []
        for _ in range(members):
            r = sample_residual(net, cond, topo, shape, dev, CFG, n_steps=steps,
                                generator=gen, tile=tile_hr, amp_dtype=amp_dtype,
                                guidance=GUIDANCE)
            members_mm.append(denorm_precip_mmday(mu + res_mean + res_std * r, pt))

        ens = torch.stack(members_mm, 0)
        obs = denorm_precip_mmday(target, pt)
        crps, sprd = crps_ensemble(ens, obs)
        rmse = ((ens.mean(0) - obs) ** 2).mean().sqrt()
        tot["crps"] += crps.item(); tot["rmse"] += rmse.item()
        tot["spread"] += sprd.item(); tot["n"] += 1

    n = max(tot["n"], 1)
    crps, rmse, spread = tot["crps"] / n, tot["rmse"] / n, tot["spread"] / n
    return crps, rmse, spread / max(rmse, 1e-6)


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
    topo_full = expand_topo(oro)

    if rank == 0:
        print(f"[SETUP] Stage 2 | param={PARAM} | patch={PATCH} | domain={ds.H}x{ds.W}")
        print(f"[SETUP] {CFG}")
        print(f"[SETUP] sampling: {SAMPLE_STEPS} steps = {nfe_count(SAMPLE_STEPS)} NFE/member "
              "-- identical in both arms; quote this in any comparison")
        print(f"[SETUP] checkpoints -> {CKPT_DIR}")
        if PATCH is None:
            print("[SETUP] [WARNING] PATCH=None forfeits the 8km->2km cascade.")

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)

    for fi, fold in enumerate(folds):
        name = fold["name"]
        if rank == 0:
            print(f"\n{'='*64}\nStage 2 ({PARAM.upper()}) -- Fold {fi+1}/{KFOLD_K}  [{name}]\n{'='*64}")

        tr, va = Subset(ds, fold["train_idx"]), Subset(ds, fold["val_idx"])
        tr_s = torch.utils.data.distributed.DistributedSampler(
            tr, num_replicas=ws, rank=rank, shuffle=True) if ws > 1 else None
        va_s = torch.utils.data.distributed.DistributedSampler(
            va, num_replicas=ws, rank=rank, shuffle=False) if ws > 1 else None
        tl = DataLoader(tr, batch_size=BATCH, sampler=tr_s, shuffle=(tr_s is None),
                        num_workers=4, pin_memory=True, drop_last=True)
        vl = DataLoader(va, batch_size=BATCH, sampler=va_s, shuffle=False,
                        num_workers=4, pin_memory=True)

        reg_path = os.path.join(REGRESSOR_CKPT_DIR, f"Regressor_{name}_best.pth")
        if not os.path.exists(reg_path):
            raise FileNotFoundError(
                f"Missing Stage-1 checkpoint for fold {name}: {reg_path}\n"
                "Run Regressor.py first. Do NOT substitute another fold's checkpoint -- "
                "it was trained on this fold's held-out years and would leak them.")
        regressor = load_frozen_regressor(reg_path, dev, pt, rank)

        if rank == 0:
            print("Estimating residual statistics...")
        res_mean, res_std = estimate_residual_stats(regressor, tl, topo_full, dev,
                                                    amp_dtype, tile_hr=PATCH)
        if ws > 1:
            st = torch.tensor([res_mean, res_std], device=dev)
            dist.broadcast(st, 0)
            res_mean, res_std = st[0].item(), st[1].item()
        if rank == 0:
            print(f"Residual stats -> mean={res_mean:.5f}  std={res_std:.5f} "
                  "(normalized target: mean 0, std 1)")

        arch = arch_dict()
        net = build_unet(arch, dev, dropout=DROPOUT)
        if rank == 0:
            print(f"  parameters: {sum(p.numel() for p in net.parameters())/1e6:.2f} M")
        if ws > 1:
            net = nn.parallel.DistributedDataParallel(net, device_ids=[loc], output_device=loc)

        ema = EMA(net.module if hasattr(net, "module") else net, EMA_DECAY)
        opt = AdamW(net.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        total = (EPOCHS * len(tl)) // ACCUM_STEPS
        sched = build_sched(opt, total, int(total * WARMUP_FRAC), LR, MIN_LR)
        scaler = torch.amp.GradScaler("cuda") if need_scaler else None

        best, no_improve = float("inf"), 0

        for ep in range(EPOCHS):
            if ws > 1:
                tr_s.set_epoch(ep)
            net.train()
            opt.zero_grad(set_to_none=True)
            run = 0.0

            for i, b in enumerate(tl):
                hr = b["hr"].to(dev, non_blocking=True)
                topo_b = topo_full.expand(hr.shape[0], -1, -1, -1)
                hr, topo_b = random_crop_hr(hr, topo_b, PATCH, DS_FACTOR, PATCH_PER_SAMPLE)

                target, mu, cond, topo_b = build_conditioning(
                    hr, topo_b, regressor, amp_dtype=amp_dtype, tile_hr=None,
                    aug_p=LR_AUG_P, aug_max=LR_AUG_MAX)

                r = (target - mu - res_mean) / res_std
                Bc = r.shape[0]

                # THE ONLY PARAMETERIZATION-DEPENDENT LINE IN TRAINING.
                state, time_input, loss_target = build_train_batch(r, CFG, dev)
                inp = torch.cat([state, cond], 1)

                cfg_drop = (torch.rand(Bc, device=dev) < CFG_DROP_P) if CFG_DROP_P > 0 else None

                with autocast(device_type=dev.type, dtype=amp_dtype,
                              enabled=amp_dtype != torch.float32):
                    pred = net(inp, time_input, topo_b, cfg_drop=cfg_drop)
                    # UNIFORM weight in both arms. For EDM the c_out division in
                    # the target IS Karras' lambda(sigma); for flow matching,
                    # logit-normal timestep sampling already makes uniform
                    # weighting near-optimal. Adding a weighting function to
                    # either applies it twice.
                    loss = per_sample_loss(pred, loss_target, LOSS_KIND).mean() / ACCUM_STEPS

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(tl):
                    if scaler is not None:
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                    if scaler is not None:
                        scaler.step(opt); scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    ema.update(net)
                    sched.step()

                run += loss.item() * ACCUM_STEPS

            avg = torch.tensor(run / max(1, len(tl)), device=dev)
            if ws > 1:
                dist.all_reduce(avg, op=dist.ReduceOp.SUM); avg /= ws
            if rank == 0:
                print(f"Epoch {ep+1:04d}/{EPOCHS} | Loss {avg.item():.4f} | "
                      f"LR {sched.get_last_lr()[0]:.2e}")

            if (ep + 1) % VAL_EVERY == 0 or ep == EPOCHS - 1:
                crps, rmse, ssr = ensemble_validate(
                    ema.shadow, vl, regressor, topo_full, dev, pt,
                    res_mean, res_std, amp_dtype, tile_hr=PATCH)
                m = torch.tensor([crps, rmse, ssr], device=dev)
                if ws > 1:
                    dist.all_reduce(m, op=dist.ReduceOp.SUM); m /= ws
                crps, rmse, ssr = m.tolist()
                if rank == 0:
                    tag = ("[under-dispersive]" if ssr < 0.8 else
                           "[over-dispersive]" if ssr > 1.25 else "[calibrated]")
                    print(f"  -> CRPS {crps:.4f} mm/day | RMSE {rmse:.4f} | "
                          f"spread/skill {ssr:.3f}   {tag}")

                if crps < best:
                    best, no_improve = crps, 0
                    if rank == 0:
                        torch.save({
                            "model_state_dict": (net.module.state_dict() if ws > 1
                                                 else net.state_dict()),
                            "ema_state_dict": ema.state_dict(),
                            "arch": arch,
                            "param": PARAM,
                            CFG_KEY: CFG.to_dict(),
                            "patch": PATCH,
                            "ds_factor": DS_FACTOR,
                            "res_mean": res_mean, "res_std": res_std,
                            "precip_transform": pt,
                            "sample_steps": SAMPLE_STEPS,
                            "nfe": nfe_count(SAMPLE_STEPS),
                            "crps": crps, "rmse": rmse, "spread_skill": ssr,
                            "epoch": ep + 1,
                        }, os.path.join(CKPT_DIR, f"Stage2_{name}_best.pth"))
                        print("  [*] saved new best")
                else:
                    no_improve += 1
                    if rank == 0:
                        print(f"  [!] no improvement for {no_improve * VAL_EVERY} epochs")

            if no_improve >= (PATIENCE // VAL_EVERY):
                if rank == 0:
                    print(f"Early stopping fold {name} (best CRPS {best:.4f} mm/day)")
                break

        del net, ema, opt, sched, scaler, regressor, tl, vl
        torch.cuda.empty_cache()

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
