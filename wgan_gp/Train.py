# -*- coding: utf-8 -*-
r"""
Train.py -- WGAN-GP training: the single stage that replaces corrdiff_fm's
Regressor.py + TrainStage2.py split.
=============================================================================
Implements Gulrajani et al. 2017, "Improved Training of Wasserstein GANs"
(arXiv:1704.00028), Algorithm 1, faithfully, for a CONDITIONAL image-to-image
task (LR + topo -> HR precip) rather than the paper's original unconditional
generate-from-noise setting. Builds on Arjovsky et al. 2017's WGAN (the
Wasserstein-distance objective and weight-clipped critic); Gulrajani's
contribution, and the one this package implements, is replacing weight
clipping with a gradient penalty on random real/fake interpolates.

WHY NO STAGE-1/STAGE-2 SPLIT
-----------------------------
corrdiff_fm decomposes  x = E[x|y] + (x - E[x|y])  because that decomposition
is what makes its DIFFUSION problem tractable: Stage 1 (L2-trained) estimates
the conditional mean, which makes the Stage-2 residual zero-mean, which is
what gives the variance reduction (corrdiff paper Eq. 2) that makes modelling
the residual easier than modelling p(x) directly with a denoiser.

None of that machinery is needed here. An adversarial framework does not need
a variance-reduced target to make its training problem easier -- the
Generator is trained end-to-end to output the FULL field directly, and the
Wasserstein critic supplies exactly the "this doesn't look like a real
downscaled field" pressure that corrdiff_fm gets from its residual denoiser's
score-matching objective. Introducing a frozen Stage-1 mean here would buy
nothing (WGAN-GP has no analogous variance-reduction argument) while adding
exactly the kind of two-checkpoint bookkeeping complexity Regressor.py exists
to manage in the other package. One network, one training script.

WHY THE CONTENT (L1) LOSS TERM EXISTS AT ALL
-----------------------------------------------
The original WGAN-GP is unconditional: G maps pure noise z to a sample, and
the adversarial term alone is the entire training signal because there is no
"correct" specific target for a given z to hit -- only the marginal output
distribution has to look real.

This task is conditional (LR+topo -> a SPECIFIC paired HR field), and that
changes what the adversarial term can and cannot enforce. E[D(G(lr,topo,z))]
only pressures the MARGINAL distribution of G's outputs (over all lr,topo,z)
to look like the marginal distribution of real HR fields. Nothing in that
objective ties a PARTICULAR G(lr) to the PARTICULAR truth paired with that lr
-- a Generator could satisfy the adversarial objective perfectly while
producing plausible-looking but arbitrary fields decorrelated from the actual
paired target (e.g. correct climatology and texture statistics, storms in the
wrong place). Conditioning the Critic on (lr, topo) (see Network.py) recovers
part of this correspondence, but only through the Critic's own capacity and
training dynamics, which is a much weaker and noisier constraint early in
training than a direct pixel-space anchor.

The L1 content term is that anchor: `lambda_content * ||G(lr,topo,z) -
target||_1` (in normalized log1p space) directly ties each specific G(lr) to
its specific paired truth, the same way corrdiff_fm's Stage-1 L2 loss ties
its mean estimate to the truth. L1, not L2, for the same reason
Regressor.py's docstring gives for NOT using L2 there: were this stage's
target the CONDITIONAL MEAN, L2 would be correct, but here we want the
Generator to reproduce plausible individual REALIZATIONS (the adversarial
term is what pushes towards realizations rather than an averaged mean), and
minimizing L2 pixel-wise pulls every realization back towards the
conditional mean -- exactly the blurring failure mode a GAN is meant to
avoid. L1 anchors less aggressively (it targets the conditional MEDIAN, not
mean) and is the standard, empirically more robust choice for anchoring
GAN-family generators against heavy-tailed image content (SRGAN, pix2pix,
and the wider literature all use L1/perceptual anchors rather than L2 for
exactly this reason).

WGAN-GP ALGORITHM, AS IMPLEMENTED HERE
-----------------------------------------
Per generator step:
  for t in 1..N_CRITIC:
      sample a real (lr, target, topo) triple
      fake = G(lr, topo, z)                    [z fresh, detached from G]
      eps ~ U(0,1) per-sample
      x_hat = eps*target + (1-eps)*fake
      GP = E[(||grad_{x_hat} D(x_hat, lr_up, topo)||_2 - 1)^2]
      L_D = E[D(fake)] - E[D(target)] + LAMBDA_GP * GP
      update Critic (Adam, betas=(0.0, 0.9), lr=1e-4)
  sample one more real (lr, target, topo) triple
  fake = G(lr, topo, z)                        [z fresh, NOT detached]
  L_G = -E[D(fake, lr_up, topo)] + LAMBDA_CONTENT * L1(fake, target)
  update Generator (Adam, betas=(0.0, 0.9), lr=1e-4)

One departure from the paper's literal Algorithm 1 worth flagging explicitly:
the ORIGINAL unconditional generator step consumes only z (no real data at
all -- there is nothing conditional to pair it with). This conditional
setting's generator step also needs a paired (lr, target, topo) triple,
because the content-loss anchor term requires a specific target to compare
against; the adversarial term by itself would still only need (lr, topo, z).

ADAM BETAS = (0.0, 0.9), NOT ADAM'S DEFAULT, NOT WGAN'S RMSPROP
-------------------------------------------------------------------
Gulrajani et al., Sec 4: with Adam's own default beta1=0.9, the paper reports
training instability once the gradient penalty is added -- the penalty term
differentiates the Critic a second time (autograd.grad with create_graph=True
below), and Adam's default first-moment momentum interacts badly with how
quickly that second-order signal needs to be able to change direction between
optimizer steps. beta1=0.0 removes that stale momentum. beta2 is lowered from
0.999 to 0.9 for the same reason: less smoothing of the second-moment
estimate that scales the update. This is NOT what the original (Arjovsky
et al. 2017) WGAN paper's weight-clipped critic uses (RMSProp, no momentum
concept at all) -- the two papers solve different critic-stabilization
problems (weight clipping vs. a differentiable penalty) with different
optimizers, and neither's answer transfers automatically to the other.

WHY THE WGAN-GP STEPS ARE NOT RUN UNDER AUTOCAST/AMP
---------------------------------------------------------
The gradient penalty requires a clean SECOND derivative: `torch.autograd.grad`
with `create_graph=True` builds the graph for d(||grad_xhat D||)/d(critic
params), which the subsequent `d_loss.backward()` then differentiates again.
Combining that with float16 autocast + GradScaler (which corrdiff_fm's stages
use for their SINGLE-backward denoising loss) is a well-known source of NaNs
and silently-wrong scaled gradients in double-backward settings, and gets
materially more failure-prone still if AMP is stacked under DistributedDataParallel's
gradient bucketing. All Critic/Generator adversarial-training forward/backward
passes below therefore run in plain float32. AMP is still used at inference
time (TiledGenerator's `amp_dtype`), where there is only ever a single forward
pass with no gradient penalty involved.

MODEL SELECTION: CRPS ACROSS THE NOISE ENSEMBLE, NOT THE ADVERSARIAL LOSS
------------------------------------------------------------------------
Mirrors TrainStage2.py's "CRPS, not the training loss" philosophy, and for an
even sharper reason here: L_D and L_G are BOTH moving targets in a two-player
game -- L_G can fall because the Generator improved, or because the Critic
got WORSE at telling real from fake, and there is no way to distinguish the
two from the loss curve alone. Neither loss is a proper scoring rule against
ground truth; CRPS (computed by holding lr/topo fixed and resampling z to
build an ensemble, exactly mirroring how the diffusion siblings resample their
sampler's initial noise) is, and is what early stopping and checkpoint
selection use here.

Run:
  single GPU : python Train.py
  multi-GPU  : torchrun --nproc_per_node=2 Train.py
"""

import os
import copy
import random
import traceback
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from Config import (HR_FILES, ORO_8KM, GENERATOR_CKPT_DIR, CRITIC_CKPT_DIR,
                    DS_FACTOR, PRECIP_CH, IN_CH_LR, PATCH, KFOLD_K, VAL_RATIO,
                    SEED, EVAL_SEED, Z_CH, GEN_BASE_CH, GEN_CHANNEL_MULT,
                    GEN_NUM_BLOCKS, GEN_DROPOUT, CRITIC_BASE_CH,
                    CRITIC_CHANNEL_MULT, CRITIC_DROPOUT, N_CRITIC, LAMBDA_GP,
                    LAMBDA_CONTENT, CONTENT_LOSS, BETAS, LR, WEIGHT_DECAY,
                    EMA_DECAY, BATCH, EPOCHS, WARMUP_FRAC, PATIENCE, VAL_EVERY,
                    CRPS_MEMBERS, CRPS_BATCHES, ensure_dirs)
from Dataset import ClimateDataset, get_climate_kfolds
from Network import Generator, Critic, expand_topo, denorm_precip_mmday, TOPO_CHANNELS
from Tiling import TiledGenerator

warnings.filterwarnings("ignore", category=UserWarning)

# Coarse-input augmentation, for cascade robustness (identical rationale and
# values to corrdiff_fm's Regressor.py / TrainStage2.py `coarsen`): at
# training LR = avg_pool(HR); at 2 km deployment LR is a real 8 km RCM field
# whose sub-grid intensity distribution differs, and a small blend toward the
# block maximum makes the Generator less brittle to that shift.
LR_AUG_P = 0.15
LR_AUG_MAX = 0.20

PATCH_PER_SAMPLE = True


# ------------------------------------------------------------------------------
# DISTRIBUTED / UTILS -- same pattern as corrdiff_fm's Regressor.py / TrainStage2.py
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


def build_sched(opt, warm):
    """Linear warmup only, then FLAT (no decay).

    WGAN-GP's Algorithm 1 specifies a constant learning rate throughout --
    unlike corrdiff_fm's cosine-decayed schedule, there is no paper basis for
    decaying this one. The short linear warmup is kept anyway, purely for
    early-training numerical stability: the gradient penalty's double-backward
    term is more sensitive to an oversized first few optimizer steps than a
    typical single-backward loss, and a few hundred warmup steps cost nothing
    against a run of this length.
    """
    def fn(step):
        return min(1.0, (step + 1) / max(1, warm))
    return LambdaLR(opt, fn)


class EMA:
    """Exponential moving average of the Generator's weights ONLY.

    Not part of the WGAN-GP paper -- see Config.py's EMA_DECAY docstring for
    why it is added anyway (matches the sibling packages' checkpoint/eval
    convention) and why it is deliberately NOT applied to the Critic (the
    Critic is discarded after training; only the Generator's smoothed output
    quality is ever consumed downstream).
    """

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


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------
# ARCHITECTURE <-> CHECKPOINT (mirrors TrainStage2.py's arch_dict()/build_unet())
# ------------------------------------------------------------------------------

def generator_arch_dict():
    """Everything needed to rebuild the Generator. Written into the checkpoint
    so Inference.py / Evaluate.py never reconstruct it from whatever constants
    happen to be sitting in Config.py at the time they are run -- which is
    correct only until someone edits a hyperparameter, after which it either
    raises (lucky) or silently loads a mismatched model (unlucky)."""
    return {
        "lr_channels": IN_CH_LR, "z_channels": Z_CH, "out_channels": 1,
        "base_channels": GEN_BASE_CH, "channel_mult": list(GEN_CHANNEL_MULT),
        "num_blocks": GEN_NUM_BLOCKS, "topo_channels": TOPO_CHANNELS,
        "ds_factor": DS_FACTOR, "precip_lr_ch": PRECIP_CH,
    }


def build_generator(arch, dev, dropout=0.0):
    return Generator(
        lr_channels=arch["lr_channels"], z_channels=arch["z_channels"],
        out_channels=arch["out_channels"], base_channels=arch["base_channels"],
        channel_mult=tuple(arch["channel_mult"]), num_blocks=arch["num_blocks"],
        topo_channels=arch["topo_channels"], dropout=dropout,
        ds_factor=arch["ds_factor"], precip_lr_ch=arch["precip_lr_ch"],
    ).to(dev)


def critic_arch_dict():
    return {
        "precip_channels": 1, "cond_channels": IN_CH_LR, "topo_channels": TOPO_CHANNELS,
        "base_channels": CRITIC_BASE_CH, "channel_mult": list(CRITIC_CHANNEL_MULT),
    }


def build_critic(arch, dev, dropout=0.0):
    return Critic(
        precip_channels=arch["precip_channels"], cond_channels=arch["cond_channels"],
        topo_channels=arch["topo_channels"], base_channels=arch["base_channels"],
        channel_mult=tuple(arch["channel_mult"]), dropout=dropout,
    ).to(dev)


# ------------------------------------------------------------------------------
# PRECIP-TRANSFORM SAFETY (same convention as corrdiff_fm's TrainStage2.py)
# ------------------------------------------------------------------------------

def assert_precip_transform_compatible(ckpt, meta, path, rank, label="checkpoint"):
    """Refuse to evaluate/infer with a checkpoint whose precip transform does
    not match the CURRENT dataset's. A checkpoint trained under a different
    precip-unit convention (e.g. an old PRECIP_SCALE=24.0 file, or a re-run of
    PrepareData.py against different source data) will load without complaint
    and silently produce numbers that are wrong by whatever factor changed.
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
            + "\n  ".join(bad) + "\nDo not mix precip scales between training runs.")
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
    give every patch a different, mutually incomparable encoding."""
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
# GRADIENT PENALTY (Gulrajani et al. 2017, Algorithm 1)
# ------------------------------------------------------------------------------

def gradient_penalty(critic, real, fake, lr_up, topo, dev):
    """GP = E[(||grad_{x_hat} D(x_hat, lr_up, topo)||_2 - 1)^2]

    x_hat = eps*real + (1-eps)*fake, eps ~ U(0,1) PER SAMPLE (not per batch --
    a single shared eps for the whole batch would make every interpolate lie
    along the same real/fake mixing ratio, sampling a much thinner slice of
    the space between the two distributions than the paper intends).

    The interpolation is only ever taken along the PRECIP field being judged;
    the conditioning (lr_up, topo) is held fixed and shared between the real
    and fake side of each pair, matching how the Critic is conditioned
    everywhere else in this file -- interpolating the conditioning too would
    be asking the Critic a question ("is this LR/topo combination real?")
    that has nothing to do with what the penalty is supposed to regularize
    (the Critic's Lipschitz constant with respect to the precip field itself).
    """
    B = real.shape[0]
    eps = torch.rand(B, 1, 1, 1, device=dev, dtype=real.dtype)
    x_hat = (eps * real + (1.0 - eps) * fake).detach().requires_grad_(True)
    d_hat = critic(x_hat, lr_up, topo)
    grad = torch.autograd.grad(
        outputs=d_hat.sum(), inputs=x_hat, create_graph=True, retain_graph=True,
    )[0]
    grad_norm = grad.reshape(B, -1).norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()


# ------------------------------------------------------------------------------
# CRPS-ENSEMBLE VALIDATION (model selection metric -- see module docstring)
# ------------------------------------------------------------------------------

def crps_ensemble(ens, obs):
    """Fair (unbiased) ensemble CRPS, identical formula to corrdiff_fm's
    TrainStage2.py. ens [M,B,1,H,W], obs [B,1,H,W].

    The second term is the pairwise spread correction that removes the
    small-M bias -- without it an 8-member ensemble is systematically
    penalized relative to a 32-member one and the metric is not comparable
    across ensemble sizes (this matters here specifically because M is
    "however many times we resampled z", a free choice, not a property of
    the model the way step count is for a diffusion sampler).
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
def ensemble_validate(ema_gen, loader, topo_full, dev, pt, amp_dtype, tile_hr,
                      members=CRPS_MEMBERS, n_batches=CRPS_BATCHES):
    """Runs the SAME tiled generation path used at inference (TiledGenerator),
    so the CRPS selected on is the CRPS you actually get downstream. The
    ensemble is built by holding (lr, topo) fixed per validation day and
    resampling z `members` times -- the direct Generator analogue of how
    corrdiff_fm's ensemble_validate resamples its sampler's initial noise.
    """
    ema_gen.eval()
    gen_rng = torch.Generator(device=dev).manual_seed(EVAL_SEED)
    tot = {"crps": 0.0, "rmse": 0.0, "spread": 0.0, "n": 0}

    for k, b in enumerate(loader):
        if k >= n_batches:
            break
        hr = b["hr"].to(dev, non_blocking=True)
        B = hr.shape[0]
        topo = topo_full.expand(B, -1, -1, -1)
        lr = coarsen(hr, DS_FACTOR)
        target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()

        tiled = TiledGenerator(ema_gen, topo, DS_FACTOR, tile_hr=tile_hr, amp_dtype=amp_dtype)
        members_mm = []
        for _ in range(members):
            fake = tiled(lr, generator=gen_rng)
            members_mm.append(denorm_precip_mmday(fake, pt))
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
        ensure_dirs(GENERATOR_CKPT_DIR, CRITIC_CKPT_DIR)

    ds = ClimateDataset(HR_FILES, ORO_8KM, rank=rank)
    pt = ds.get_precip_transform_meta()

    oro = torch.as_tensor(ds.oro, dtype=torch.float32, device=dev)
    while oro.dim() < 4:
        oro = oro.unsqueeze(0)
    topo_full = expand_topo(oro)

    if rank == 0:
        print(f"[SETUP] WGAN-GP | patch={PATCH} | domain={ds.H}x{ds.W} | "
              f"n_critic={N_CRITIC} | lambda_gp={LAMBDA_GP} | lambda_content={LAMBDA_CONTENT}")
        print(f"[SETUP] precip transform: {pt}")
        print(f"[SETUP] Adam betas={BETAS} (NOT (0.9,0.999) -- see Train.py docstring), lr={LR}")
        if PATCH is None:
            print("[SETUP] [WARNING] PATCH=None forfeits the 8km->2km tiled cascade.")

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)

    for fi, fold in enumerate(folds):
        name = fold["name"]
        if rank == 0:
            print(f"\n{'='*64}\nWGAN-GP -- Fold {fi+1}/{KFOLD_K}  [{name}]")
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

        gen_arch = generator_arch_dict()
        crit_arch = critic_arch_dict()
        generator = build_generator(gen_arch, dev, dropout=GEN_DROPOUT)
        critic = build_critic(crit_arch, dev, dropout=CRITIC_DROPOUT)
        if rank == 0:
            ng = sum(p.numel() for p in generator.parameters())
            nc = sum(p.numel() for p in critic.parameters())
            print(f"  Generator: {ng/1e6:.2f} M params | Critic: {nc/1e6:.2f} M params")

        if ws > 1:
            # find_unused_parameters=True: the gradient-penalty term backpropagates
            # through the Critic via torch.autograd.grad (not via .backward()) before
            # the Critic's own .backward() call runs, which DDP's autograd hooks do
            # not see in the usual single-pass way. This setting is the safe,
            # documented way to keep DDP correct in that situation, at a small
            # extra-bookkeeping cost.
            generator = nn.parallel.DistributedDataParallel(
                generator, device_ids=[loc], output_device=loc)
            critic = nn.parallel.DistributedDataParallel(
                critic, device_ids=[loc], output_device=loc, find_unused_parameters=True)

        gen_bare = generator.module if hasattr(generator, "module") else generator
        ema = EMA(gen_bare, EMA_DECAY)

        opt_g = Adam(generator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        opt_d = Adam(critic.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

        steps_per_epoch = max(1, len(tl) // (N_CRITIC + 1))
        total_gsteps = EPOCHS * steps_per_epoch
        warm = max(1, int(total_gsteps * WARMUP_FRAC))
        sched_g = build_sched(opt_g, warm)
        sched_d = build_sched(opt_d, warm)

        best, no_improve = float("inf"), 0

        for ep in range(EPOCHS):
            if ws > 1:
                tr_s.set_epoch(ep)
            generator.train(); critic.train()
            it = iter(tl)

            def next_batch():
                """Pulls the next training batch, transparently starting a new
                pass over `tl` if the current one runs out mid-epoch.

                `steps_per_epoch = len(tl) // (N_CRITIC+1)` guarantees enough
                batches for a full epoch whenever `len(tl) >= N_CRITIC+1`, which
                holds for any dataset of realistic size (thousands of days). This
                guard only ever engages for a pathologically small fold (e.g. a
                toy/smoke-test run with far fewer than `(N_CRITIC+1)*BATCH`
                training samples), where wrapping mid-epoch is preferable to
                crashing with StopIteration.
                """
                nonlocal it
                try:
                    return next(it)
                except StopIteration:
                    it = iter(tl)
                    return next(it)

            d_run = g_run = gp_run = wd_run = adv_run = content_run = 0.0

            for _gstep in range(steps_per_epoch):
                # ---------------- CRITIC UPDATES ----------------
                set_requires_grad(critic, True)
                d_acc = gp_acc = wd_acc = 0.0
                for _ct in range(N_CRITIC):
                    b = next_batch()
                    hr = b["hr"].to(dev, non_blocking=True)
                    topo_b = topo_full.expand(hr.shape[0], -1, -1, -1)
                    hr, topo_b = random_crop_hr(hr, topo_b, PATCH, DS_FACTOR, PATCH_PER_SAMPLE)

                    lr_x = coarsen(hr, DS_FACTOR, LR_AUG_P, LR_AUG_MAX)
                    target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()
                    lr_up = F.interpolate(lr_x, size=hr.shape[-2:], mode="bilinear",
                                          align_corners=False).float()

                    with torch.no_grad():
                        fake = generator(lr_x, topo_b).float()

                    real_score = critic(target, lr_up, topo_b)
                    fake_score = critic(fake, lr_up, topo_b)
                    gp = gradient_penalty(critic, target, fake, lr_up, topo_b, dev)
                    d_loss = fake_score.mean() - real_score.mean() + LAMBDA_GP * gp

                    opt_d.zero_grad(set_to_none=True)
                    d_loss.backward()
                    opt_d.step()
                    sched_d.step()

                    d_acc += d_loss.item()
                    gp_acc += gp.item()
                    wd_acc += (real_score.mean() - fake_score.mean()).item()

                # ---------------- GENERATOR UPDATE ----------------
                set_requires_grad(critic, False)
                b = next_batch()
                hr = b["hr"].to(dev, non_blocking=True)
                topo_b = topo_full.expand(hr.shape[0], -1, -1, -1)
                hr, topo_b = random_crop_hr(hr, topo_b, PATCH, DS_FACTOR, PATCH_PER_SAMPLE)

                lr_x = coarsen(hr, DS_FACTOR, LR_AUG_P, LR_AUG_MAX)
                target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()
                lr_up = F.interpolate(lr_x, size=hr.shape[-2:], mode="bilinear",
                                      align_corners=False).float()

                fake = generator(lr_x, topo_b).float()
                fake_score = critic(fake, lr_up, topo_b)
                adv_loss = -fake_score.mean()
                if CONTENT_LOSS == "l1":
                    content_loss = F.l1_loss(fake, target)
                else:
                    content_loss = F.mse_loss(fake, target)
                g_loss = adv_loss + LAMBDA_CONTENT * content_loss

                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                opt_g.step()
                sched_g.step()
                ema.update(generator)

                d_run += d_acc / N_CRITIC
                gp_run += gp_acc / N_CRITIC
                wd_run += wd_acc / N_CRITIC
                g_run += g_loss.item()
                adv_run += adv_loss.item()
                content_run += content_loss.item()

            n = max(steps_per_epoch, 1)
            stats = torch.tensor([d_run, g_run, gp_run, wd_run, adv_run, content_run], device=dev) / n
            if ws > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM); stats /= ws
            d_avg, g_avg, gp_avg, wd_avg, adv_avg, content_avg = stats.tolist()

            if rank == 0:
                print(f"Epoch {ep+1:04d}/{EPOCHS} | D {d_avg:+.4f} | G {g_avg:+.4f} "
                      f"(adv {adv_avg:+.4f} + {LAMBDA_CONTENT:g}*L1 {content_avg:.4f}) | "
                      f"GP {gp_avg:.4f} | W-est {wd_avg:+.4f} | "
                      f"LR {sched_g.get_last_lr()[0]:.2e}")

            if (ep + 1) % VAL_EVERY == 0 or ep == EPOCHS - 1:
                crps, rmse, ssr = ensemble_validate(
                    ema.shadow, vl, topo_full, dev, pt, torch.float32, tile_hr=PATCH)
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
                            "model_state_dict": (generator.module.state_dict() if ws > 1
                                                 else generator.state_dict()),
                            "ema_state_dict": ema.state_dict(),
                            "arch": gen_arch,
                            "algo": "wgan_gp",
                            "n_critic": N_CRITIC, "lambda_gp": LAMBDA_GP,
                            "lambda_content": LAMBDA_CONTENT, "content_loss": CONTENT_LOSS,
                            "patch": PATCH,
                            "ds_factor": DS_FACTOR,
                            "precip_transform": pt,
                            "crps": crps, "rmse": rmse, "spread_skill": ssr,
                            "epoch": ep + 1,
                        }, os.path.join(GENERATOR_CKPT_DIR, f"Generator_{name}_best.pth"))
                        torch.save({
                            "model_state_dict": (critic.module.state_dict() if ws > 1
                                                 else critic.state_dict()),
                            "arch": crit_arch,
                            "paired_generator_crps": crps,
                            "epoch": ep + 1,
                        }, os.path.join(CRITIC_CKPT_DIR, f"Critic_{name}_latest.pth"))
                        print("  [*] saved new best (selected on CRPS, not adversarial loss)")
                else:
                    no_improve += 1
                    if rank == 0:
                        print(f"  [!] no improvement for {no_improve * VAL_EVERY} epochs")

            if no_improve >= (PATIENCE // VAL_EVERY):
                if rank == 0:
                    print(f"Early stopping fold {name} (best CRPS {best:.4f} mm/day)")
                break

        del generator, critic, ema, opt_g, opt_d, sched_g, sched_d, tl, vl
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
