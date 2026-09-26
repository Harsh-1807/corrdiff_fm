# -*- coding: utf-8 -*-
"""
Config.py -- every path and every cross-file constant, in one place.
======================================================================
Same discipline as corrdiff_fm's Config.py, and for the same reason: the single
most common way a multi-file pipeline breaks is two scripts disagreeing about a
constant that was copy-pasted into both. Train.py, Inference.py and Evaluate.py
all import from here so there is exactly one place to change and nothing to
keep in sync.

THIS PACKAGE SHARES corrdiff_fm's DATA, TASK AND PROTOCOL
----------------------------------------------------------
`srgan` is a sibling package to `corrdiff_fm`, not an independent project: same
NetCDF sources, same 8 km training grid / 2 km inference cascade, same k-fold
split, same PATCH/tiling discipline, same evaluation seed. That is deliberate --
a comparison between architecture FAMILIES (adversarial single-stage SRGAN vs.
two-stage regressor+diffusion) is only informative if everything except the
architecture itself is held fixed. So every constant in the "SHARED" section
below has the SAME VALUE as corrdiff_fm/Config.py, copied rather than imported
(this package must be runnable standalone, see the README) -- if you change one
of them here, change it in corrdiff_fm too, or the two are no longer comparable.

PATCH is still the one constant to understand before touching anything
------------------------------------------------------------------------
It is the grid size the generator (and, during training, the discriminator)
actually see, and it is the tile size used at 8 km -> 2 km inference.

Unlike corrdiff_fm's Stage-2 U-Net, this package's generator has NO blocks that
tie learned weights to absolute grid coordinates (no SpectralConv2d, no
attention-token-budget switch) -- see Tiling.py's module docstring for the
detailed comparison. So the resolution-transfer risk here is genuinely smaller.
It is not zero: BatchNorm layers throughout the residual blocks carry running
statistics estimated on PATCH-sized crops, and a much larger inference canvas
does not change what those frozen statistics do to the affine transform (in
eval mode BatchNorm uses the stored running mean/var, not the current batch's),
so the network you'd get from an un-tiled full-domain forward pass is *not* a
fundamentally different operator, only a mildly out-of-distribution one at the
seams from finite receptive field + convolution padding. Tiling still matters
for that reason. Read Tiling.py's docstring before deciding to skip it.
"""

import os

# ------------------------------------------------------------------------------
# PATHS -- identical to corrdiff_fm/Config.py. Edit both together.
# ------------------------------------------------------------------------------
ROOT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh"

_HR_DIR = f"{ROOT}/SRGAN_pipeline/HR_DATA_8"
_STATIC = f"{ROOT}/Singapore_Data/STATIC_Data"

# The uncorrected source file (mm/hr). PrepareData.py reads this.
RAW_PRECIP_FILE = f"{_HR_DIR}/precip_rcm_8km_daily_data_1995-2014.nc"

# What everything downstream actually trains on. The precip entry is the
# CORRECTED file written by PrepareData.py -- see the PRECIP_SCALE note in
# Dataset.py for why this matters. This is the SAME corrected file corrdiff_fm
# uses; PrepareData.py only ever needs to be run ONCE across every package in
# /home/ylale/extras/h/, not once per package (see README.md).
HR_FILES = [
    f"{_HR_DIR}/huss_rcm_8km_daily_data_1995-2014.nc",
    f"{_HR_DIR}/mslp_rcm_8km_daily_data_1995-2014.nc",
    f"{_HR_DIR}/tas_rcm_8km_daily_data_1995-2014.nc",
    f"{_HR_DIR}/precip_rcm_8km_daily_data_1995-2014_mmday_corrected.nc",
]

ORO_8KM = f"{_STATIC}/SG.orog.8km.nc"
ORO_2KM = f"{_STATIC}/SG.orog.2km.nc"   # optional but strongly recommended

VAR_MAP = {"huss": "huss", "mslp": "psl", "tas": "tas", "precip": "pr", "oro": "orog"}

# ------------------------------------------------------------------------------
# CHECKPOINTS / OUTPUTS -- this package's OWN directories (never shared with
# corrdiff_fm's checkpoints/regressor or checkpoints/stage2_*).
# ------------------------------------------------------------------------------
GENERATOR_CKPT_DIR = "checkpoints/generator_srgan/"
DISCRIMINATOR_CKPT_DIR = "checkpoints/discriminator_srgan/"

OUTPUT_DIR = "outputs/downscaled_2km_srgan/"

# ------------------------------------------------------------------------------
# SHARED CONSTANTS -- must agree with corrdiff_fm/Config.py (see module
# docstring). Copied, not imported, so this package stands alone.
# ------------------------------------------------------------------------------
DS_FACTOR = 4          # mirrors Dataset.DS_FACTOR; 32->8 km in training, 8->2 km at inference
PRECIP_CH = 3          # channel index of precip in (huss, mslp, tas, precip)
IN_CH_LR = 4           # coarse input channels

PATCH = 128            # see the module docstring. Training crop AND inference tile size.
PATCH_OVERLAP = 32     # tile overlap at inference (PATCH // 4)

KFOLD_K = 5
VAL_RATIO = 0.15
SEED = 1234
EVAL_SEED = 20260823   # shared with corrdiff_fm so held-out days line up 1:1 across packages

# ------------------------------------------------------------------------------
# GENERATOR ARCHITECTURE (SRResNet backbone, Ledig et al. 2017 Fig. 4)
# ------------------------------------------------------------------------------
GEN_BASE_CH = 64        # paper's "64 feature maps"
NUM_RES_BLOCKS = 16      # paper's "B residual blocks", B=16 -- do not shrink this casually;
                         # it is the one architectural number the paper reports an ablation on
GEN_TOPO_CHANNELS = 8    # = Network.TOPO_CHANNELS; kept here too so arch dicts are self-describing

# ------------------------------------------------------------------------------
# DISCRIMINATOR ARCHITECTURE (Ledig et al. 2017 Fig. 4, VGG-style)
# ------------------------------------------------------------------------------
DISC_BASE_CH = 64
# D is conditioned on upsampled-LR + topo (a deliberate deviation from the
# literal paper -- see Network.py's Discriminator docstring for why). Its
# input channel count is therefore precip(1) + LR(IN_CH_LR) + topo.
DISC_IN_CH = 1 + IN_CH_LR + GEN_TOPO_CHANNELS
# D's dense head size depends on PATCH (see Network.Discriminator) -- it is
# NEVER run at any other spatial size, unlike G, so this fixed-size literal
# paper head is safe. If you change PATCH you must retrain D from scratch.
DISC_PATCH = PATCH

# ------------------------------------------------------------------------------
# TRAINING SCHEDULE -- Ledig et al. Sec. 3.2's two-phase protocol
# ------------------------------------------------------------------------------
# Phase 1: pretrain G alone on pixel (MSE) loss. The paper trains its "SRResNet"
# baseline for a large number of updates before ever touching the adversarial
# loss, specifically to avoid the GAN signal dominating gradients from a
# randomly-initialized generator (an adversarial loss computed against a
# discriminator that has learned nothing yet, applied to a generator that has
# also learned nothing yet, is close to pure noise). Budget mirrors
# corrdiff_fm's Stage-1 regressor (also a pure-pixel-loss conditional-mean
# fit) since it is solving an analogous problem here.
PRETRAIN_EPOCHS = 600
PRETRAIN_PATIENCE = 60

# Phase 2: adversarial fine-tuning of G (+ training of D from scratch). Shorter
# than Phase 1 -- the paper fine-tunes from the converged MSE solution rather
# than from scratch, so far fewer updates are needed to reshape the *texture*
# statistics of an already-accurate mean-field predictor.
GAN_EPOCHS = 400
GAN_PATIENCE = 80

VAL_EVERY = 2

# Eq. 3 of the paper: l^SR = l^SR_X + 1e-3 * l^SR_Gen. This is not a knob to
# tune casually -- see Train.py's module docstring for why 1e-3 specifically.
ADV_WEIGHT = 1e-3

# Content-loss choice (Train.py's ContentLoss / Network.ContentLoss). Default
# False: pixel L1 + Sobel-gradient-magnitude loss, physically motivated for a
# single-channel geophysical field. True: literal paper fidelity via a frozen
# ImageNet VGG19 (see Network.VGGFeatureExtractor for the domain-mismatch
# caveat this drags in).
USE_VGG_PERCEPTUAL = False
VGG_LAYER_IDX = 35             # torchvision vgg19().features[:35] = conv5_4, PRE-activation
                               # (paper found pre-activation features avoid the sparse-
                               # activation / inconsistent-magnitude issues of post-ReLU features)
VGG_PIXEL_ANCHOR_WEIGHT = 0.1  # small L1 anchor added on top of the VGG term (common practice
                               # variant: pure VGG-feature MSE has no term at all forcing the
                               # network to also match ABSOLUTE precipitation intensity, since
                               # VGG features are scale/contrast-invariant by design)
SOBEL_LOSS_WEIGHT = 0.5        # weight of the Sobel term relative to the L1 pixel term
                               # (non-VGG default path)

# Secondary model-selection signal during Phase 2 (see Train.py's module
# docstring for why RMSE/MAE alone is the wrong criterion once the adversarial
# term is active): weight applied to the validation spectrum log-ratio in the
# composite selection score, score = rmse * (1 + w * |spectrum_logratio|).
SPECTRUM_SELECTION_WEIGHT = 0.5

# ------------------------------------------------------------------------------
# OPTIMIZATION -- Ledig et al. Sec. 3.2's exact settings
# ------------------------------------------------------------------------------
LR = 1e-4                 # paper: "Adam optimizer with beta1 = 0.9" and "learning rate of 1e-4"
BETAS = (0.9, 0.999)      # paper's plain Adam. Contrast with the wgan_gp sibling package's
                          # (0.0, 0.9): THAT package's critic loss includes a gradient-penalty
                          # term whose gradient interacts badly with Adam's first-moment
                          # (heavy-ball) momentum -- beta1=0 is the standard WGAN-GP fix
                          # (Gulrajani et al. 2017). This package's discriminator has no
                          # gradient penalty (it is a plain sigmoid+BCE classifier), so there
                          # is no such interaction and the paper's own, more standard,
                          # momentum-bearing Adam is the right choice.
WEIGHT_DECAY = 0.0        # paper specifies none; SRResNet/SRGAN's regularization comes from
                          # BatchNorm + the content loss itself, not weight decay
GRAD_CLIP = 1.0
MIN_LR = 1e-6
WARMUP_FRAC = 0.02

# EMA on G. NOT part of the paper -- corrdiff_fm's convention, kept here for
# the same reason: it gives a lower-variance checkpoint to evaluate/deploy
# than the raw (noisier, especially once the adversarial loss is active)
# training weights.
EMA_DECAY = 0.999

BATCH = 16
ACCUM_STEPS = 1

# Coarse-input augmentation (identical rationale and values to corrdiff_fm's
# Regressor.py: LR = avg_pool(HR) at training time, but LR is a genuine 8 km
# RCM field at 2 km deployment, which retains more sub-grid intensity than a
# pure area mean; blending a little max-pool in during training buys tolerance
# to that shift, exactly as it does for corrdiff_fm's cascade).
LR_AUG_P = 0.15
LR_AUG_MAX = 0.20


def ensure_dirs(*dirs):
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)
