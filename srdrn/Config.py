# -*- coding: utf-8 -*-
"""
Config.py -- every path and every cross-stage constant, in one place.
=====================================================================
Same convention as the sibling `corrdiff_fm` package: every script imports its
paths and shared constants from here, so there is exactly one place to change
and nothing to keep in sync between training, evaluation and inference.

THIS PACKAGE IS SINGLE-STAGE. Unlike `corrdiff_fm` (Stage 1 conditional mean +
Stage 2 residual diffusion), SRDRN is one deterministic network trained
end-to-end on the full LR -> HR map. There is therefore no
REGRESSOR_CKPT_DIR / DIFFUSION_CKPT_DIR split -- just CKPT_DIR.

PATHS ARE INTENTIONALLY IDENTICAL TO corrdiff_fm/Config.py
------------------------------------------------------------
ROOT, HR_FILES, ORO_8KM, ORO_2KM and VAR_MAP are copied verbatim from the
sibling package. This is deliberate, not laziness: both packages train on the
same corrected precip file, the same orography, and the same variables, so
that any difference measured between SRDRN and CorrDiff-EDM/FM is attributable
to the architecture family, not to a hundred incidental data-pipeline
differences. See PrepareData.py and this package's README for why step 0
(`PrepareData.py`) only needs to be run ONCE across every package that shares
this ROOT, not once per package.

DS_FACTOR=4, NOT THE PAPER'S 8x -- see the README's "Deviations from the
published SRDRN, and why" section for the full justification. Short version:
this package is built to be directly comparable (via Compare.py) with the
other packages under /home/ylale/extras/h/, which all target the 32km->8km
(and cascaded 8km->2km) task at 4x. Replicating the paper's literal 8x
India-region setup would make that comparison impossible.
"""

import os

# ------------------------------------------------------------------------------
# PATHS -- edit these (identical values to corrdiff_fm/Config.py; keep in sync
# if you repoint one, since the whole point of sharing them is comparability)
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
# CHECKPOINTS / OUTPUTS -- this package's OWN directories (not shared with
# corrdiff_fm's REGRESSOR_CKPT_DIR/DIFFUSION_CKPT_DIR; SRDRN is a fully separate
# architecture family and its weights are not interchangeable with either).
# ------------------------------------------------------------------------------
CKPT_DIR = "checkpoints/srdrn/"
OUTPUT_DIR = "outputs/downscaled_2km_srdrn/"

# ------------------------------------------------------------------------------
# SHARED CONSTANTS -- kept numerically identical to corrdiff_fm's Config.py so
# the two packages solve the literal same problem (same fold definitions, same
# patch size, same tile overlap, same eval seed => Compare.py's paired test is
# valid: both packages score the same held-out days in the same order).
# ------------------------------------------------------------------------------
DS_FACTOR = 4          # ADAPTED from the paper's 8x -- see README
PRECIP_CH = 3          # channel index of precip in (huss, mslp, tas, precip)
IN_CH_LR = 4           # coarse input channels -- ADAPTED from the paper's 3ch
                       # (LR precip, LR daily climatology, LR orography) input;
                       # see README. Topography is NOT part of IN_CH_LR here --
                       # it enters separately, at HR resolution, via expand_topo
                       # (see Network.py), matching corrdiff_fm's convention.

PATCH = 128            # training-crop / inference-tile size, in HR pixels.
PATCH_OVERLAP = 32     # tile overlap at inference (PATCH // 4)

KFOLD_K = 5
VAL_RATIO = 0.15
SEED = 1234
EVAL_SEED = 20260823   # shared across every package's Evaluate.py so paired
                       # comparisons via Compare.py are valid

# ------------------------------------------------------------------------------
# SRDRN ARCHITECTURE -- recorded in the checkpoint's "arch" sub-dict and read
# back verbatim by Evaluate.py / Inference.py, so a hyperparameter edited here
# after training cannot silently desynchronize the reconstructed network from
# the weights on disk (see Train.py's checkpoint format).
# ------------------------------------------------------------------------------
# 16 residual blocks, faithful to the paper (Sec. 3: "sixteen residual blocks").
NUM_RES_BLOCKS = 16

# The paper does not state its channel width. 64 is the value used by SRResNet
# / SRGAN / EDSR-style residual super-resolution networks generally (Ledig et
# al. 2017; Lim et al. 2017), which SRDRN's block design is visibly descended
# from, and is a reasonable, well-precedented default in the absence of a
# stated number. Judgment call -- documented here and in the README, not
# silently assumed.
BASE_CH = 64

# len(UP_FACTORS) upsampling blocks, each x2, must multiply out to DS_FACTOR.
# The published SRDRN uses 3 blocks (8x); this package uses 2 (4x) to match
# DS_FACTOR -- see README "Deviations from the published SRDRN, and why".
N_UP_BLOCKS = 2
assert 2 ** N_UP_BLOCKS == DS_FACTOR, (
    f"N_UP_BLOCKS={N_UP_BLOCKS} must satisfy 2**n == DS_FACTOR ({DS_FACTOR}), "
    "mirroring CorrDiffRegressor's own channel_mult assertion in corrdiff_fm.")

# Loss variant. The paper trains three separate models -- SRDRN-MSE, SRDRN-MAE,
# SRDRN-WMAE -- and reports WMAE as the best performer for reproducing
# precipitation extremes. Default to "wmae" for that reason; switch to "mse" or
# "mae" to reproduce the other two paper variants exactly.
LOSS_KIND = "wmae"      # one of "mse", "mae", "wmae"

# WMAE emphasis strength -- see the prominent caveat in Network.py's
# `wmae_pixel_weight` docstring: the paper's abstract does not give the exact
# weighting formula, so this is a documented reconstruction of its stated
# intent, not a verified reproduction. Treat as a hyperparameter to tune.
WMAE_ALPHA = 2.0


def ensure_dirs(*dirs):
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)
