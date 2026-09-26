# -*- coding: utf-8 -*-
r"""
Train.py -- two-phase SRGAN training: MSE pretraining, then adversarial fine-tuning.
=====================================================================================
Implements Ledig et al. 2017 CVPR Sec. 3.2's training protocol exactly: first
train the generator alone as a plain super-resolution regressor ("SRResNet"),
then switch on the discriminator and fine-tune the SAME generator weights with
a combined content + adversarial loss. Skipping straight to adversarial
training from a randomly-initialized generator is not a shortcut -- it is a
different (and, per the paper's own ablation, worse) recipe: early in
training G produces close to noise, D trivially tells it apart from real
fields, and the resulting gradient through G's adversarial term carries
almost no useful signal about what a plausible field looks like. Pretraining
first gives D something non-trivial to discriminate from the very first
adversarial step.

THIS IS A SINGLE-STAGE ARCHITECTURE -- NOT corrdiff_fm's TWO STAGES
----------------------------------------------------------------------
There is no frozen Stage-1 regressor here, no residual target, no
`res_mean`/`res_std` normalization step. One generator, `Network.Generator`,
maps (LR climate stack, HR topo) directly to HR precip. The "two phases" in
this file's name are two TRAINING REGIMES for that one network (first
pixel-loss only, then + adversarial), not two different networks being
composed at inference the way corrdiff_fm's mean+residual decomposition works.
Only ONE checkpoint format matters for downstream use (Evaluate.py,
Inference.py): the Phase-2 (GAN fine-tuned) generator. The Phase-1 checkpoint
is kept only for ablation/diagnostics (see README's caveats section).

WHY THE 1e-3 ADVERSARIAL WEIGHT, SPECIFICALLY (Ledig et al. Eq. 3)
----------------------------------------------------------------------
    l^SR = l^SR_X (content loss) + 1e-3 * l^SR_Gen (adversarial loss)

This is not an arbitrary or lightly-tunable knob. `l^SR_Gen = -log(D(G(x)))`
(equivalently `BCE(D(G(x)), 1)` in the non-saturating form used here) has a
gradient magnitude set by how confidently D currently rejects G's output,
which is a property of THE DISCRIMINATOR'S CURRENT STATE, not of how far G's
output currently is from the target in any content sense. Early in Phase 2,
right after Phase-1 pretraining, D has not seen any of G's (now much better)
outputs yet and can be very confidently correct that they are fake -- which
means -log(D(G(x))) starts LARGE and its gradient can dominate the content
loss by orders of magnitude for pixels/patches where D is confident. Without
the 1e-3 down-weighting, that adversarial gradient would overwhelm the content
term precisely in the epochs right after switching phases, undoing much of
what Phase 1 achieved before the discriminator has learned anything
sufficiently well-calibrated to be a useful training signal. 1e-3 keeps the
content loss as the dominant term throughout, letting the adversarial term
act as a texture/realism regularizer on top of it rather than a competing
objective.

TWO DELIBERATE, DOCUMENTED CONTRASTS WITH THE `wgan_gp` SIBLING PACKAGE
--------------------------------------------------------------------------
  * Adam betas=(0.9, 0.999) here (paper's own setting) vs. wgan_gp's
    (0.0, 0.9). WGAN-GP's critic loss includes a gradient-penalty term whose
    interaction with Adam's first-moment (heavy-ball) momentum is known to
    destabilize training unless beta1 is reduced toward 0 (Gulrajani et al.
    2017, "Improved Training of Wasserstein GANs", Sec 4). This package's
    discriminator has no gradient penalty -- it is a plain sigmoid+BCE
    classifier -- so there is no such interaction, and the paper's own,
    higher-momentum Adam setting is correct here.
  * Standard (non-Wasserstein) adversarial loss: BCE with a sigmoid output,
    real label 1, fake label 0. No critic, no Lipschitz constraint, no
    gradient penalty. See Network.Discriminator's docstring for the full
    contrast.

MODEL SELECTION -- WHY NOT JUST VALIDATION RMSE/MAE
--------------------------------------------------------
Since this architecture produces one deterministic field per input (see
Network.py's "WHY THE GENERATOR HAS NO NOISE INPUT"), there is no ensemble to
compute a proper CRPS over -- it collapses exactly to MAE (see Evaluate.py).
That degeneracy has a sharp consequence for MODEL SELECTION DURING PHASE 2
specifically: RMSE/MAE are minimized by the CONDITIONAL MEAN, and the entire
point of turning on the adversarial loss is to push the generator's output
away from a blurry conditional-mean-like solution toward one with realistic
high-frequency statistics -- a strictly harder-to-satisfy criterion that a
plain RMSE-minimizer will actively resist. Selecting purely on RMSE during
Phase 2 would systematically prefer checkpoints that partially UNDID the GAN
fine-tuning and drifted back toward the safe, blurry Phase-1 solution -- the
opposite of what Phase 2 exists to buy. So Phase-2 selection here uses a
composite score,

    score = rmse * (1 + SPECTRUM_SELECTION_WEIGHT * |spectrum_logratio|)

(both computed exactly as Evaluate.py defines them: RMSE of precip in mm/day,
spectrum_logratio = mean |log10(power ratio)| over the top two octaves versus
the target spectrum). This keeps RMSE as the primary term -- a generator that
hallucinates unrelated high-frequency texture only to win the spectrum term
is not what's wanted either -- while still penalizing systematic blur, which
is exactly the failure mode a GAN is supposed to fix and RMSE alone cannot
see. Phase 1 selection is plain validation RMSE (there is no adversarial
term yet to fight against, so it is simply the right criterion, for the same
reason corrdiff_fm's Stage-1 regressor selects on RMSE).

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
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from Config import (HR_FILES, ORO_8KM, GENERATOR_CKPT_DIR, DISCRIMINATOR_CKPT_DIR,
                    DS_FACTOR, PRECIP_CH, IN_CH_LR, PATCH, KFOLD_K, VAL_RATIO, SEED,
                    EVAL_SEED, GEN_BASE_CH, NUM_RES_BLOCKS, GEN_TOPO_CHANNELS,
                    DISC_BASE_CH, DISC_IN_CH, DISC_PATCH, PRETRAIN_EPOCHS,
                    PRETRAIN_PATIENCE, GAN_EPOCHS, GAN_PATIENCE, VAL_EVERY,
                    ADV_WEIGHT, USE_VGG_PERCEPTUAL, VGG_LAYER_IDX,
                    VGG_PIXEL_ANCHOR_WEIGHT, SOBEL_LOSS_WEIGHT,
                    SPECTRUM_SELECTION_WEIGHT, LR, BETAS, WEIGHT_DECAY, GRAD_CLIP,
                    MIN_LR, WARMUP_FRAC, EMA_DECAY, BATCH, ACCUM_STEPS, LR_AUG_P,
                    LR_AUG_MAX, ensure_dirs)
from Dataset import ClimateDataset, get_climate_kfolds
from Network import (Generator, Discriminator, ContentLoss, expand_topo,
                     denorm_precip_mmday, TOPO_CHANNELS)
from Tiling import TiledGenerator

warnings.filterwarnings("ignore", category=UserWarning)

BCE_EPS = 1e-7   # clamp on D's sigmoid output before BCE, avoids log(0)


# ------------------------------------------------------------------------------
# DISTRIBUTED / UTILS -- same conventions as corrdiff_fm's training scripts
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
    """Exponential moving average of the generator's weights. NOT part of the
    paper -- corrdiff_fm's convention, kept here for the same reason: it gives
    a lower-variance checkpoint to evaluate/deploy than the raw training
    weights, which is particularly valuable once the adversarial loss is
    active and the raw weights are noisier step to step."""

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
# ARCHITECTURE DICTS -- written into checkpoints so Inference.py/Evaluate.py
# never reconstruct the network from whatever constants happen to be sitting
# in Config.py at the time they are run (see corrdiff_fm's TrainStage2.py for
# why that footgun matters: it is correct exactly until someone edits a
# hyperparameter, after which it either raises or silently loads a mismatched
# model).
# ------------------------------------------------------------------------------

def gen_arch_dict():
    return {
        "in_channels": IN_CH_LR, "out_channels": 1, "base_channels": GEN_BASE_CH,
        "num_res_blocks": NUM_RES_BLOCKS, "ds_factor": DS_FACTOR,
        "topo_channels": GEN_TOPO_CHANNELS,
    }


def build_generator(arch, dev):
    return Generator(
        in_channels=arch["in_channels"], out_channels=arch["out_channels"],
        base_channels=arch["base_channels"], num_res_blocks=arch["num_res_blocks"],
        ds_factor=arch["ds_factor"], topo_channels=arch["topo_channels"],
    ).to(dev)


def disc_arch_dict():
    return {
        "in_channels": DISC_IN_CH, "base_channels": DISC_BASE_CH, "patch": DISC_PATCH,
    }


def build_discriminator(arch, dev):
    return Discriminator(
        in_channels=arch["in_channels"], base_channels=arch["base_channels"],
        patch=arch["patch"],
    ).to(dev)


# ------------------------------------------------------------------------------
# PRECIP-TRANSFORM SAFETY CHECK -- same convention as corrdiff_fm
# ------------------------------------------------------------------------------

def assert_precip_transform_compatible(ckpt, meta, path, rank, label="checkpoint"):
    """Refuse to evaluate/deploy a checkpoint whose recorded precip transform
    does not match what the CURRENT Dataset.py produces. There is no separate
    Stage-1/Stage-2 pair to desynchronize here (this is a single-stage
    architecture), but the same failure mode this guards against in
    corrdiff_fm is still possible: retraining under a different
    PRECIP_SCALE/normalization convention and then silently reusing an old
    checkpoint would load without complaint and denormalize every downstream
    number wrong by whatever factor separates the two conventions."""
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
            + "\n  ".join(bad) + "\nRetrain rather than reuse this checkpoint.")
    if rank == 0:
        print(f"[PRECIP SAFETY] {path} ({label}): OK.")


# ------------------------------------------------------------------------------
# CROPPING AND CONDITIONING
# ------------------------------------------------------------------------------

def random_crop_hr(hr, topo, size, ds, per_sample=True):
    """Aligned random crop of HR and topo; offsets are multiples of `ds` so the
    LR grid stays exactly nested. topo is cropped from the FULL-DOMAIN tensor,
    never recomputed per patch -- expand_topo standardizes elevation and lays
    down y/x coordinate channels domain-wide, so recomputing per patch would
    give every patch a different, mutually incomparable encoding. Identical to
    corrdiff_fm's Regressor.py/TrainStage2.py (duplicated rather than shared
    across packages, following this codebase's own established convention --
    see e.g. how `setup`/`set_seed`/`EMA` are duplicated between corrdiff_fm's
    own Regressor.py and TrainStage2.py)."""
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

    At training time LR = avg_pool(HR); at 2 km deployment LR is a genuine
    8 km RCM field whose sub-grid intensity distribution differs. Only the
    precip channel is perturbed -- blending temperature or pressure toward a
    block maximum is not physically meaningful. Identical rationale and
    values to corrdiff_fm's `coarsen`."""
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
# VALIDATION METRICS (RMSE/MAE in mm/day + radial-spectrum log-ratio)
# ------------------------------------------------------------------------------

def radial_spectrum(x):
    """Isotropic power spectrum of a [B,1,H,W] field, averaged over the batch.
    Identical to Evaluate.py's version (duplicated here so Train.py does not
    have to import from Evaluate.py, which itself imports build_generator etc.
    from THIS file -- see Evaluate.py's module docstring)."""
    x = x.squeeze(1).float()
    B, H, W = x.shape
    f = torch.fft.fftshift(torch.fft.fft2(x - x.mean(dim=(1, 2), keepdim=True)), dim=(-2, -1))
    p = (f.real ** 2 + f.imag ** 2).mean(0)
    cy, cx = H // 2, W // 2
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device) - cy,
                            torch.arange(W, device=x.device) - cx, indexing="ij")
    r = torch.sqrt((yy.float()) ** 2 + (xx.float()) ** 2).round().long()
    nb = int(min(cy, cx))
    out = torch.zeros(nb, device=x.device)
    for k in range(1, nb):
        m = r == k
        if m.any():
            out[k] = p[m].mean()
    return out


def spectrum_score(pred_spec, targ_spec):
    """Mean |log10(P_pred / P_target)| over the top two octaves. 0 is perfect."""
    n = len(targ_spec)
    lo = max(1, n // 4)
    a = pred_spec[lo:n].clamp_min(1e-12)
    b = targ_spec[lo:n].clamp_min(1e-12)
    return float((torch.log10(a / b)).abs().mean())


@torch.no_grad()
def validate(generator, loader, topo_full, dev, pt, amp_dtype, tile_hr):
    """RMSE, MAE (mm/day) and spectrum log-ratio, run through the SAME tiled
    path used at deployment (Tiling.TiledGenerator) -- so the number selected
    on is the number you will get, exactly the discipline corrdiff_fm's own
    validate() follows."""
    generator.eval()
    se = ae = n = 0.0
    spec_p = spec_t = None
    n_spec = 0
    for b in loader:
        hr = b["hr"].to(dev, non_blocking=True)
        B = hr.shape[0]
        topo = topo_full.expand(B, -1, -1, -1)
        lr = coarsen(hr, DS_FACTOR)
        target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()

        tg = TiledGenerator(generator, topo, DS_FACTOR, tile_hr=tile_hr, amp_dtype=amp_dtype)
        pred = tg(lr)

        p = denorm_precip_mmday(pred, pt)
        o = denorm_precip_mmday(target, pt)
        se += float(((p - o) ** 2).sum())
        ae += float((p - o).abs().sum())
        n += p.numel()

        sp, st = radial_spectrum(p), radial_spectrum(o)
        spec_p = sp if spec_p is None else spec_p + sp
        spec_t = st if spec_t is None else spec_t + st
        n_spec += 1

    n = max(n, 1.0)
    rmse = math.sqrt(se / n)
    mae = ae / n
    spec_lr = spectrum_score(spec_p / n_spec, spec_t / n_spec) if n_spec else 0.0
    return rmse, mae, spec_lr


# ------------------------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------------------------

def train():
    rank, ws, loc, dev = setup()
    set_seed(SEED + rank)
    if rank == 0:
        ensure_dirs(GENERATOR_CKPT_DIR, DISCRIMINATOR_CKPT_DIR)
    amp_dtype, need_scaler = pick_amp(dev)

    ds = ClimateDataset(HR_FILES, ORO_8KM, rank=rank)
    pt = ds.get_precip_transform_meta()

    oro = torch.as_tensor(ds.oro, dtype=torch.float32, device=dev)
    while oro.dim() < 4:
        oro = oro.unsqueeze(0)
    topo_full = expand_topo(oro)      # computed ONCE over the full domain

    if rank == 0:
        print(f"[SETUP] SRGAN | patch={PATCH} | domain={ds.H}x{ds.W} | ds_factor={DS_FACTOR}")
        print(f"[SETUP] precip transform: {pt}")
        print(f"[SETUP] content loss: {'VGG54 perceptual + L1 anchor' if USE_VGG_PERCEPTUAL else 'L1 + Sobel-gradient'}")
        print(f"[SETUP] adversarial weight: {ADV_WEIGHT} (Ledig et al. Eq. 3)")
        if PATCH is None:
            print("[SETUP] [WARNING] PATCH=None ties this model to the 8 km domain size "
                  "and forfeits the 8km->2km cascade.")

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)

    for fi, fold in enumerate(folds):
        name = fold["name"]
        if rank == 0:
            print(f"\n{'='*64}\nSRGAN -- Fold {fi+1}/{KFOLD_K}  [{name}]")
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

        gen_arch = gen_arch_dict()
        G = build_generator(gen_arch, dev)
        if rank == 0:
            print(f"  Generator parameters: {sum(p.numel() for p in G.parameters())/1e6:.2f} M")
        if ws > 1:
            G = nn.parallel.DistributedDataParallel(G, device_ids=[loc], output_device=loc)

        ema = EMA(G.module if hasattr(G, "module") else G, EMA_DECAY)

        # ==========================================================================
        # PHASE 1 -- pretrain G alone on pixel (MSE) loss. See module docstring.
        # ==========================================================================
        opt_g = Adam(G.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        total1 = (PRETRAIN_EPOCHS * len(tl)) // ACCUM_STEPS
        sched_g = build_sched(opt_g, total1, int(total1 * WARMUP_FRAC), LR, MIN_LR)
        scaler_g = torch.amp.GradScaler("cuda") if need_scaler else None

        best_rmse, no_improve = float("inf"), 0
        pretrain_ckpt_path = os.path.join(GENERATOR_CKPT_DIR, f"Generator_{name}_pretrain.pth")

        if rank == 0:
            print(f"\n--- Phase 1: MSE pretraining ({PRETRAIN_EPOCHS} epochs budget) ---")

        for ep in range(PRETRAIN_EPOCHS):
            if ws > 1:
                tr_s.set_epoch(ep)
            G.train()
            opt_g.zero_grad(set_to_none=True)
            run = 0.0

            for i, b in enumerate(tl):
                hr = b["hr"].to(dev, non_blocking=True)
                topo_b = topo_full.expand(hr.shape[0], -1, -1, -1)
                hr, topo_b = random_crop_hr(hr, topo_b, PATCH, DS_FACTOR)

                lr_x = coarsen(hr, DS_FACTOR, LR_AUG_P, LR_AUG_MAX)
                target = hr[:, PRECIP_CH:PRECIP_CH + 1]

                with autocast(device_type=dev.type, dtype=amp_dtype,
                              enabled=amp_dtype != torch.float32):
                    pred = G(lr_x, topo_b)
                    # Plain MSE, per Ledig et al. Sec 3.2's SRResNet baseline
                    # ("optimized for MSE"). The squared-error minimizer is the
                    # conditional MEAN -- exactly the target this phase needs,
                    # since it is what Phase 2 then fine-tunes away from.
                    loss = F.mse_loss(pred.float(), target.float()) / ACCUM_STEPS

                if scaler_g is not None:
                    scaler_g.scale(loss).backward()
                else:
                    loss.backward()

                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(tl):
                    if scaler_g is not None:
                        scaler_g.unscale_(opt_g)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
                    if scaler_g is not None:
                        scaler_g.step(opt_g); scaler_g.update()
                    else:
                        opt_g.step()
                    opt_g.zero_grad(set_to_none=True)
                    ema.update(G)
                    sched_g.step()

                run += loss.item() * ACCUM_STEPS

            avg = torch.tensor(run / max(1, len(tl)), device=dev)
            if ws > 1:
                dist.all_reduce(avg, op=dist.ReduceOp.SUM); avg /= ws
            if rank == 0:
                print(f"[Phase1] Epoch {ep+1:04d}/{PRETRAIN_EPOCHS} | MSE {avg.item():.5f} | "
                      f"LR {sched_g.get_last_lr()[0]:.2e}")

            if (ep + 1) % VAL_EVERY == 0 or ep == PRETRAIN_EPOCHS - 1:
                rmse, mae, spec_lr = validate(ema.shadow, vl, topo_full, dev, pt, amp_dtype, PATCH)
                m = torch.tensor([rmse, mae, spec_lr], device=dev)
                if ws > 1:
                    dist.all_reduce(m, op=dist.ReduceOp.SUM); m /= ws
                rmse, mae, spec_lr = m.tolist()
                if rank == 0:
                    print(f"  -> val RMSE {rmse:.4f} mm/day | MAE {mae:.4f} mm/day "
                          f"| spectrum log-ratio {spec_lr:.4f}")

                if rmse < best_rmse:
                    best_rmse, no_improve = rmse, 0
                    if rank == 0:
                        torch.save({
                            "model_state_dict": (G.module.state_dict() if ws > 1
                                                 else G.state_dict()),
                            "ema_state_dict": ema.state_dict(),
                            "arch": gen_arch,
                            "phase": "pretrain",
                            "patch": PATCH,
                            "ds_factor": DS_FACTOR,
                            "precip_transform": pt,
                            "val_rmse_mmday": rmse,
                            "val_mae_mmday": mae,
                            "val_spectrum_logratio": spec_lr,
                            "epoch": ep + 1,
                        }, pretrain_ckpt_path)
                        print("  [*] saved new best (pretrain)")
                else:
                    no_improve += 1
                    if rank == 0:
                        print(f"  [!] no improvement for {no_improve * VAL_EVERY} epochs")

            if no_improve >= (PRETRAIN_PATIENCE // VAL_EVERY):
                if rank == 0:
                    print(f"Early stopping Phase 1, fold {name} (best RMSE {best_rmse:.4f} mm/day)")
                break

        # Reload the best Phase-1 weights (raw, not EMA -- Phase 2 continues
        # optimizing the raw weights; EMA is re-initialized fresh from them so
        # its own averaging window does not straddle the phase transition).
        if rank == 0:
            print(f"Loading best Phase-1 checkpoint for Phase 2: {pretrain_ckpt_path}")
        best_ck = torch.load(pretrain_ckpt_path, map_location=dev, weights_only=False)
        (G.module if ws > 1 else G).load_state_dict(best_ck["model_state_dict"])
        ema = EMA(G.module if hasattr(G, "module") else G, EMA_DECAY)

        del opt_g, sched_g, scaler_g
        torch.cuda.empty_cache()

        # ==========================================================================
        # PHASE 2 -- adversarial fine-tuning. G continues from Phase 1; D is new.
        # ==========================================================================
        disc_arch = disc_arch_dict()
        D = build_discriminator(disc_arch, dev)
        if rank == 0:
            print(f"\n--- Phase 2: adversarial fine-tuning ({GAN_EPOCHS} epochs budget) ---")
            print(f"  Discriminator parameters: {sum(p.numel() for p in D.parameters())/1e6:.2f} M")
        if ws > 1:
            D = nn.parallel.DistributedDataParallel(D, device_ids=[loc], output_device=loc)

        content_loss_fn = ContentLoss(
            use_vgg=USE_VGG_PERCEPTUAL, vgg_layer_idx=VGG_LAYER_IDX,
            pixel_anchor_weight=VGG_PIXEL_ANCHOR_WEIGHT, sobel_weight=SOBEL_LOSS_WEIGHT,
        ).to(dev)

        opt_g = Adam(G.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        opt_d = Adam(D.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        total2 = (GAN_EPOCHS * len(tl)) // ACCUM_STEPS
        sched_g = build_sched(opt_g, total2, int(total2 * WARMUP_FRAC), LR, MIN_LR)
        sched_d = build_sched(opt_d, total2, int(total2 * WARMUP_FRAC), LR, MIN_LR)
        scaler_g = torch.amp.GradScaler("cuda") if need_scaler else None
        scaler_d = torch.amp.GradScaler("cuda") if need_scaler else None

        best_score, no_improve = float("inf"), 0

        for ep in range(GAN_EPOCHS):
            if ws > 1:
                tr_s.set_epoch(ep + PRETRAIN_EPOCHS)
            G.train(); D.train()
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            run_g = run_d = run_adv = 0.0
            d_real_acc = d_fake_acc = 0.0
            n_batches = 0

            for i, b in enumerate(tl):
                hr = b["hr"].to(dev, non_blocking=True)
                topo_b = topo_full.expand(hr.shape[0], -1, -1, -1)
                hr, topo_b = random_crop_hr(hr, topo_b, PATCH, DS_FACTOR)

                lr_x = coarsen(hr, DS_FACTOR, LR_AUG_P, LR_AUG_MAX)
                target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()
                lr_up = F.interpolate(lr_x, size=target.shape[-2:],
                                      mode="bilinear", align_corners=False).float()
                Bc = target.shape[0]
                real_lbl = torch.ones(Bc, 1, device=dev)
                fake_lbl = torch.zeros(Bc, 1, device=dev)

                # ---------------- Train D ----------------
                with autocast(device_type=dev.type, dtype=amp_dtype,
                              enabled=amp_dtype != torch.float32):
                    with torch.no_grad():
                        fake = G(lr_x, topo_b)
                    d_real = D(target, lr_up, topo_b).clamp(BCE_EPS, 1 - BCE_EPS)
                    d_fake = D(fake.detach(), lr_up, topo_b).clamp(BCE_EPS, 1 - BCE_EPS)
                    d_loss = (F.binary_cross_entropy(d_real, real_lbl)
                             + F.binary_cross_entropy(d_fake, fake_lbl)) / ACCUM_STEPS

                if scaler_d is not None:
                    scaler_d.scale(d_loss).backward()
                else:
                    d_loss.backward()

                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(tl):
                    if scaler_d is not None:
                        scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(D.parameters(), GRAD_CLIP)
                    if scaler_d is not None:
                        scaler_d.step(opt_d); scaler_d.update()
                    else:
                        opt_d.step()
                    opt_d.zero_grad(set_to_none=True)
                    sched_d.step()

                # ---------------- Train G ----------------
                with autocast(device_type=dev.type, dtype=amp_dtype,
                              enabled=amp_dtype != torch.float32):
                    fake = G(lr_x, topo_b)
                    d_fake_for_g = D(fake, lr_up, topo_b).clamp(BCE_EPS, 1 - BCE_EPS)
                    # Non-saturating generator loss: maximize log(D(fake)),
                    # i.e. minimize BCE(D(fake), 1). Ledig et al. Eq. 3's
                    # l^SR_Gen (the paper's Eq. 4 defines the saturating
                    # -log(1-D(fake)) form; the non-saturating form used here
                    # is the now-standard fix for its well-known vanishing-
                    # gradient problem early in adversarial training, when D
                    # easily rejects G's output -- exactly Phase 2's starting
                    # condition here).
                    adv_loss = F.binary_cross_entropy(d_fake_for_g, real_lbl)
                    content = content_loss_fn(fake.float(), target)
                    g_loss = (content + ADV_WEIGHT * adv_loss) / ACCUM_STEPS

                if scaler_g is not None:
                    scaler_g.scale(g_loss).backward()
                else:
                    g_loss.backward()

                if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(tl):
                    if scaler_g is not None:
                        scaler_g.unscale_(opt_g)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
                    if scaler_g is not None:
                        scaler_g.step(opt_g); scaler_g.update()
                    else:
                        opt_g.step()
                    opt_g.zero_grad(set_to_none=True)
                    ema.update(G)
                    sched_g.step()

                run_g += content.item(); run_d += d_loss.item() * ACCUM_STEPS
                run_adv += adv_loss.item()
                d_real_acc += float((d_real > 0.5).float().mean())
                d_fake_acc += float((d_fake < 0.5).float().mean())
                n_batches += 1

            n_batches = max(n_batches, 1)
            stats = torch.tensor([run_g / n_batches, run_d / n_batches, run_adv / n_batches,
                                  d_real_acc / n_batches, d_fake_acc / n_batches], device=dev)
            if ws > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM); stats /= ws
            content_avg, d_avg, adv_avg, real_acc, fake_acc = stats.tolist()
            disc_acc = 0.5 * (real_acc + fake_acc)
            if rank == 0:
                # Sanity check worth watching (see README): disc_acc should
                # hover near 0.5 (D cannot reliably tell real from fake -- the
                # adversarial signal is informative but not overwhelming). A
                # sustained ~1.0 means D has "won" and G's adversarial
                # gradient has likely collapsed to near-zero; a sustained
                # ~0.0 means G has "won" (or D has stopped learning), and the
                # adversarial term is no longer doing anything useful either.
                tag = ("[D dominating]" if disc_acc > 0.85 else
                       "[G dominating]" if disc_acc < 0.15 else "[balanced]")
                print(f"[Phase2] Epoch {ep+1:04d}/{GAN_EPOCHS} | content {content_avg:.5f} | "
                      f"D loss {d_avg:.4f} | adv {adv_avg:.4f} | D acc {disc_acc:.3f} {tag} | "
                      f"LR {sched_g.get_last_lr()[0]:.2e}")

            if (ep + 1) % VAL_EVERY == 0 or ep == GAN_EPOCHS - 1:
                rmse, mae, spec_lr = validate(ema.shadow, vl, topo_full, dev, pt, amp_dtype, PATCH)
                m = torch.tensor([rmse, mae, spec_lr], device=dev)
                if ws > 1:
                    dist.all_reduce(m, op=dist.ReduceOp.SUM); m /= ws
                rmse, mae, spec_lr = m.tolist()
                # Composite selection score -- see module docstring for why
                # plain RMSE is the wrong criterion once the adversarial term
                # is active.
                score = rmse * (1.0 + SPECTRUM_SELECTION_WEIGHT * abs(spec_lr))
                if rank == 0:
                    print(f"  -> val RMSE {rmse:.4f} mm/day | MAE {mae:.4f} mm/day | "
                          f"spectrum log-ratio {spec_lr:.4f} | selection score {score:.4f}")

                if score < best_score:
                    best_score, no_improve = score, 0
                    if rank == 0:
                        torch.save({
                            "model_state_dict": (G.module.state_dict() if ws > 1
                                                 else G.state_dict()),
                            "ema_state_dict": ema.state_dict(),
                            "arch": gen_arch,
                            "phase": "gan",
                            "patch": PATCH,
                            "ds_factor": DS_FACTOR,
                            "precip_transform": pt,
                            "val_rmse_mmday": rmse,
                            "val_mae_mmday": mae,
                            "val_spectrum_logratio": spec_lr,
                            "selection_score": score,
                            "adv_weight": ADV_WEIGHT,
                            "use_vgg_perceptual": USE_VGG_PERCEPTUAL,
                            "epoch": ep + 1,
                            "pretrain_epoch": best_ck.get("epoch"),
                        }, os.path.join(GENERATOR_CKPT_DIR, f"Generator_{name}_best.pth"))
                        torch.save({
                            "model_state_dict": (D.module.state_dict() if ws > 1
                                                 else D.state_dict()),
                            "arch": disc_arch,
                            "paired_generator_epoch": ep + 1,
                            "disc_acc": disc_acc,
                        }, os.path.join(DISCRIMINATOR_CKPT_DIR, f"Discriminator_{name}_best.pth"))
                        print("  [*] saved new best (gan)")
                else:
                    no_improve += 1
                    if rank == 0:
                        print(f"  [!] no improvement for {no_improve * VAL_EVERY} epochs")

            if no_improve >= (GAN_PATIENCE // VAL_EVERY):
                if rank == 0:
                    print(f"Early stopping Phase 2, fold {name} (best score {best_score:.4f})")
                break

        del G, D, ema, opt_g, opt_d, sched_g, sched_d, scaler_g, scaler_d, tl, vl
        torch.cuda.empty_cache()

    if rank == 0:
        print("\nSRGAN training complete for all folds.")
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
