# -*- coding: utf-8 -*-
"""
Config.py -- every path and every constant this package uses, in one place.
=============================================================================
This is the ``wgan_gp`` sibling of ``corrdiff_fm``: same task (8 km -> 2 km
precipitation downscaling via a 32 km -> 8 km training analogue), same data,
same k-fold protocol, same tiling convention -- but a single-stage adversarial
generator instead of a two-stage regressor + residual-diffusion cascade. See
the package README for the full architectural rationale.

PATHS, VAR_MAP, DS_FACTOR, PRECIP_CH, IN_CH_LR, PATCH, PATCH_OVERLAP, KFOLD_K,
VAL_RATIO, SEED and EVAL_SEED are copied byte-for-byte from corrdiff_fm's
Config.py. This is deliberate, not laziness: this package trains against the
SAME corrected precip file, reads the SAME orography, splits years into the
SAME folds with the SAME seeds, and tiles at the SAME patch size. Any
difference in results between corrdiff_fm and this package should be
attributable to the generative architecture (diffusion/flow-matching residual
correction vs. adversarial single-stage generation), not to a silently
different data split or patch size. If you change ROOT or PATCH here, you are
no longer running a comparable experiment -- change it in both places, or
better, don't.

PATCH is still the one constant to understand before touching anything, and
the reason it matters is now DIFFERENT from corrdiff_fm's reason (read
Network.py's module docstring and this package's README, "Does the PATCH /
tiling caveat still apply here?", before assuming it is the same argument).
The short version: this Generator has no absolute-FFT-mode or token-budget
block, so it is far closer to genuinely resolution-agnostic than the
diffusion siblings -- but PATCH still fixes the training crop size, which (a)
sets what the Critic ever sees during adversarial training (a Critic that only
ever judged 128x128 patches has no calibrated opinion about a 944x944 canvas),
and (b) GroupNorm statistics inside the Generator are computed over whatever
spatial extent is handed to it, so a full-domain forward pass computes
different normalization statistics than the training patches did. Both are
real but strictly weaker failure modes than corrdiff_fm's, and tiled inference
side-steps both by construction.
"""

import os

# ------------------------------------------------------------------------------
# PATHS -- identical to corrdiff_fm/Config.py. Edit ROOT for your cluster; the
# derived paths then match wherever PrepareData.py (run ONCE, in any one
# package under /home/ylale/extras/h/) wrote the corrected precip file.
# ------------------------------------------------------------------------------
ROOT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh"

_HR_DIR = f"{ROOT}/SRGAN_pipeline/HR_DATA_8"
_STATIC = f"{ROOT}/Singapore_Data/STATIC_Data"

# The uncorrected source file (mm/hr). PrepareData.py reads this.
RAW_PRECIP_FILE = f"{_HR_DIR}/precip_rcm_8km_daily_data_1995-2014.nc"

# What everything downstream actually trains on. The precip entry is the
# CORRECTED file written by PrepareData.py (see Dataset.py's PRECIP_SCALE
# docstring). This is the SAME file corrdiff_fm trains on -- PrepareData.py
# only needs to be run once across every package in /home/ylale/extras/h/,
# not once per package. See this package's README, "Run order", step 0.
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
# CHECKPOINTS -- this package's own directories (NOT shared with corrdiff_fm;
# there is no frozen Stage-1 to share, since WGAN-GP trains a single generator
# end-to-end). Kept as two separate directories, one per network, because only
# the Generator (plus its EMA shadow) is ever needed downstream by
# Inference.py / Evaluate.py -- the Critic checkpoint exists purely so
# training can be resumed or the adversarial dynamics inspected after the
# fact, and keeping it physically separate makes that asymmetry obvious rather
# than bundling a network nobody will reload into every eval-time load.
# ------------------------------------------------------------------------------
GENERATOR_CKPT_DIR = "checkpoints/generator_wgan_gp/"
CRITIC_CKPT_DIR = "checkpoints/critic_wgan_gp/"

OUTPUT_DIR = "outputs/downscaled_2km_wgan_gp/"

# ------------------------------------------------------------------------------
# SHARED DATA / TASK CONSTANTS -- identical values to corrdiff_fm/Config.py
# ------------------------------------------------------------------------------
DS_FACTOR = 4          # mirrors Dataset.DS_FACTOR; 32->8 km in training, 8->2 km at inference
PRECIP_CH = 3          # channel index of precip in (huss, mslp, tas, precip)
IN_CH_LR = 4           # coarse input channels (huss, mslp, tas, precip)

PATCH = 128            # training crop size AND inference tile size (see module docstring)
PATCH_OVERLAP = 32     # tile overlap at inference (PATCH // 4)

KFOLD_K = 5
VAL_RATIO = 0.15
SEED = 1234
EVAL_SEED = 20260823   # shared with corrdiff_fm so cross-package comparisons stay paired

# ------------------------------------------------------------------------------
# NOISE INPUT -- this arm's deliberate point of departure from a deterministic
# SR network (see the sibling SRGAN/SRDRN packages, which have none of this).
# ------------------------------------------------------------------------------
# z ~ N(0, I) is drawn as a [B, Z_CH, H/DS_FACTOR, W/DS_FACTOR] tensor -- i.e.
# at LR resolution, the same grid the coarse conditioning lives on -- and
# concatenated to the LR input channels before the Generator's stem conv. This
# is what lets Inference.py draw a calibrated-in-spirit ensemble (resample z
# with lr/topo held fixed) comparable to corrdiff_fm's diffusion ensemble,
# despite the Generator being a single feed-forward pass rather than an
# iterative sampler. Z_CH=4 is a reasonable default: enough independent
# spatial degrees of freedom for the Generator to translate into varied
# sub-grid placement, without letting the noise channel count dominate the
# real conditioning signal (4 noise channels next to 4 physical LR channels
# is a 1:1 information budget, not a noise-swamped one).
Z_CH = 4

# ------------------------------------------------------------------------------
# GENERATOR ARCHITECTURE
# ------------------------------------------------------------------------------
# Same "all heavy work at LR resolution, then two x2 PixelShuffleUp stages"
# shape as CorrDiffRegressor (len(channel_mult)-1 must satisfy 2**n == DS_FACTOR),
# but WITHOUT self-attention or spectral convolution -- see Network.py's module
# docstring for why those two specific blocks are the ones that break the
# resolution-agnosticity this Generator is trying to keep.
GEN_BASE_CH = 96
GEN_CHANNEL_MULT = (1, 2, 4)   # len-1 == 2 == log2(DS_FACTOR)
GEN_NUM_BLOCKS = 2
GEN_DROPOUT = 0.10

# ------------------------------------------------------------------------------
# CRITIC ARCHITECTURE
# ------------------------------------------------------------------------------
# Fully-convolutional PatchGAN-style critic: downsamples the HR precip patch
# (conditioned by concatenated bilinear-upsampled LR + topo) through strided
# convolutions, ending in a 1x1 conv to a single-channel score MAP that is then
# spatially averaged to one scalar per sample. Chosen (over global-pool-then-
# linear) because it keeps the critic fully convolutional -- no fixed-size
# linear layer to break if a tile shape ever differs -- and because averaging
# a *local* per-patch Wasserstein estimate is a slightly stronger training
# signal per gradient step than one global scalar (every spatial location
# contributes its own critic gradient rather than being pooled away first).
CRITIC_BASE_CH = 64
CRITIC_CHANNEL_MULT = (1, 2, 4, 8)   # 4 stride-2 downsamples: 128 -> 64 -> 32 -> 16 -> 8
CRITIC_DROPOUT = 0.0   # see Network.py: dropout interacts badly with the gradient penalty

# ------------------------------------------------------------------------------
# WGAN-GP ALGORITHM CONSTANTS (Gulrajani et al. 2017, arXiv:1704.00028)
# ------------------------------------------------------------------------------
N_CRITIC = 5          # critic updates per generator update (paper Algorithm 1)
LAMBDA_GP = 10.0      # gradient-penalty coefficient (paper Algorithm 1 / Sec 4)

# Content-loss anchor weight -- NOT part of the original (unconditional) WGAN-GP,
# see Train.py's module docstring for why a conditional image-to-image task
# needs one. 50.0 is a judgment call, justified against the expected relative
# magnitude of the two terms at initialization: the Critic is 1-Lipschitz by
# construction (that is what the gradient penalty enforces), so its output
# difference across a pair of inputs is bounded by the distance between them
# in the SAME normalized pixel space the L1 term is computed in -- the two
# losses live on comparable natural scales, not orders of magnitude apart.
# With an untrained (near-random) Critic, |E[D(fake)] - E[D(real)]| starts
# close to 0 while the L1 term starts at roughly the target field's own
# normalized-space dispersion (order 1, since precip is normalized to unit
# variance downstream of log1p). LAMBDA_CONTENT=50 therefore makes the pixel
# anchor dominate the Generator's gradient for the first many epochs, which is
# exactly the desired behaviour: early on the Critic has no useful opinion yet
# (it hasn't learned the data distribution), so the Generator should
# essentially do supervised regression-to-target; only once the Critic
# sharpens does the adversarial term contribute a comparable-magnitude
# distributional-realism correction. Sane range 10.0-100.0; do not set below
# ~5 (the adversarial term alone provides no correspondence to a SPECIFIC
# target field, only to the marginal output distribution -- see Train.py) or
# above ~200 (the adversarial term becomes vestigial and this degenerates to
# plain L1 regression, forfeiting the entire point of using WGAN-GP).
LAMBDA_CONTENT = 50.0
CONTENT_LOSS = "l1"   # L1, not L2 -- see Train.py's module docstring

# Adam betas -- a SPECIFIC, DELIBERATE WGAN-GP paper recommendation, distinct
# from BOTH the original (Arjovsky et al. 2017) WGAN's RMSProp AND from Adam's
# own default betas=(0.9, 0.999). Gulrajani et al. Sec 4 / Algorithm 1: with
# beta1=0.9 (Adam's default first-moment decay), the paper reports the
# gradient-penalty-regularized critic exhibiting training instability, and
# recommends beta1=0.0 (i.e. no momentum on the gradient penalty's own
# gradients, which are second-derivative terms through the interpolated-input
# gradient and are more sensitive to stale momentum than a typical loss term).
# beta2=0.9 is also lowered from Adam's default 0.999. Used for BOTH the
# Generator and the Critic optimizers -- do not "fix" this to the Adam
# default.
BETAS = (0.0, 0.9)
LR = 1e-4             # paper Algorithm 1 / Sec 4
WEIGHT_DECAY = 0.0    # the paper uses none; the gradient penalty is already the regularizer

# ------------------------------------------------------------------------------
# EMA -- OUR addition, not the paper's. The original WGAN-GP paper has no EMA
# at all. It is added here on the GENERATOR ONLY, for two reasons: (1) it
# matches corrdiff_fm's checkpoint/eval convention (Inference.py and
# Evaluate.py in both packages load `ema_state_dict` preferentially), which
# keeps this package a fair architectural comparison rather than also a
# training-recipe comparison; (2) only the Generator's OUTPUT QUALITY is ever
# consumed downstream -- the Critic is a training-time-only auxiliary network
# that is discarded at inference, so smoothing it buys nothing and only risks
# lagging its adaptation to a Generator that is itself changing every step.
EMA_DECAY = 0.999

# ------------------------------------------------------------------------------
# TRAINING SCHEDULE
# ------------------------------------------------------------------------------
BATCH = 16
EPOCHS = 1500
WARMUP_FRAC = 0.02
PATIENCE = 150         # in epochs; generator steps only (see Train.py)
VAL_EVERY = 5

# CRPS-ensemble validation (same protocol as corrdiff_fm's TrainStage2.py,
# reading crps_ensemble by resampling z instead of resampling a diffusion seed)
CRPS_MEMBERS = 8
CRPS_BATCHES = 3


def ensure_dirs(*dirs):
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)
