# -*- coding: utf-8 -*-
"""
Config.py -- every path and every cross-stage constant, in one place.
=====================================================================
Stage 1, Stage 2, inference and evaluation all import from here. That is
deliberate: the single most common way this kind of pipeline breaks is two
scripts disagreeing about a constant that was copy-pasted into both. In the
version this replaces, the inference script rebuilt the Stage-2 network from
whatever hyperparameters happened to be sitting in the training script at the
time it was run.

PATCH is the one to understand before you touch anything:

  * It is the grid size the networks actually see during training.
  * It is the tile size used at inference.
  * BOTH STAGES MUST USE THE SAME VALUE.

The 8 km -> 2 km cascade depends on this. `SpectralConv2d` ties learned weights
to absolute FFT mode indices, and `SelfAttn2d` switches to pooled attention once
H*W exceeds its token budget -- so handing a network a 4x-larger canvas than it
trained on is a genuinely different operator, not the same one applied more
times. Pinning the grid size is what makes the cascade legitimate.
"""

import os

# ------------------------------------------------------------------------------
# PATHS -- edit these
# ------------------------------------------------------------------------------
ROOT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh"

_HR_DIR = f"{ROOT}/SRGAN_pipeline/HR_DATA_8"
_STATIC = f"{ROOT}/Singapore_Data/STATIC_Data"

# The uncorrected source file (mm/hr). PrepareData.py reads this.
RAW_PRECIP_FILE = f"{_HR_DIR}/precip_rcm_8km_daily_data_1995-2014.nc"

# What everything downstream actually trains on. The precip entry is the
# CORRECTED file written by PrepareData.py -- see the PRECIP_SCALE note in
# Dataset.py for why this matters.
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
# CHECKPOINTS
# ------------------------------------------------------------------------------
# Stage 1 is shared by both parameterizations -- train it ONCE and point both
# arms at it. Any difference in Stage 1 between the two arms would confound the
# comparison entirely.
REGRESSOR_CKPT_DIR = "checkpoints/regressor/"

# Stage 2 is per-arm and is set by each package's own trainer.
DIFFUSION_CKPT_DIR = "checkpoints/stage2/"

OUTPUT_DIR = "outputs/downscaled_2km/"

# ------------------------------------------------------------------------------
# SHARED CONSTANTS -- must agree across both stages
# ------------------------------------------------------------------------------
DS_FACTOR = 4          # mirrors Dataset.DS_FACTOR; 32->8 km in training, 8->2 km at inference
PRECIP_CH = 3          # channel index of precip in (huss, mslp, tas, precip)
IN_CH_LR = 4           # coarse input channels

PATCH = 128            # see the module docstring. Same for Stage 1 and Stage 2.
PATCH_OVERLAP = 32     # tile overlap at inference (PATCH // 4)

KFOLD_K = 5
VAL_RATIO = 0.15
SEED = 1234
EVAL_SEED = 20260823   # shared by validation AND evaluation so runs are paired

# ------------------------------------------------------------------------------
# STAGE-1 ARCHITECTURE -- recorded in its checkpoint, read back by Stage 2
# ------------------------------------------------------------------------------
REG_BASE_CH = 96
REG_CHANNEL_MULT = (1, 2, 4)   # len-1 must satisfy 2**n == DS_FACTOR
REG_NUM_BLOCKS = 2

# ------------------------------------------------------------------------------
# STAGE-2 ARCHITECTURE -- identical for EDM and flow matching, so that the only
# difference between the two arms is the generative parameterization itself.
# ------------------------------------------------------------------------------
UNET_IN_CH = 1 + 1 + IN_CH_LR   # noisy state, Stage-1 mean, upsampled LR
BASE_CH = 96
CHANNEL_MULT = (1, 2, 2, 4)
NUM_RES_BLOCKS = 2
ATTN_LEVELS = (2, 3)
USE_SPECTRAL = True
DROPOUT = 0.13                  # paper Sec. 5.3.2


def ensure_dirs(*dirs):
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)
