# -*- coding: utf-8 -*-
"""
Dataset.py -- NetCDF loading, normalization, k-fold splits.
============================================================
Copied essentially verbatim from corrdiff_fm/Dataset.py. This file's job --
load the four HR NetCDF variables, log1p+normalize precip, per-channel
normalize the rest, build the LR grid by average-pooling the normalized HR
tensor, and split years into rolling k-fold train/val/test blocks -- has
nothing to do with the downstream architecture. A ClimateDataset instance
means the same thing whether the model reading it is a residual-diffusion
U-Net or, as here, a purely feed-forward residual network. Keeping the file
byte-for-byte identical across every package under /home/ylale/extras/h/ is
what makes cross-package comparisons (via each package's Compare.py) valid --
if this file drifted between packages, a difference in downscaling quality
could just as easily be a difference in what "the data" means.

See Config.py's module docstring and this package's README for why the paths
threaded through here (HR_FILES, ORO_8KM) are the SAME absolute paths as
corrdiff_fm's.
"""

import os
import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DS_FACTOR = 4

# ----------------------------------------------------------------------
# PRECIP_SCALE
# ------------
# Previously 24.0. That value was doing double duty: (1) putting precip
# into a numerically nice range for log1p before normalization, and
# (2) -- unintentionally -- compensating for the raw HR precip file
# (precip_rcm_8km_daily_data_1995-2014.nc) actually being stored in
# mm/hr rather than mm/day, which is why it needed *24 to line up with
# SEA-8 8km GT (confirmed via check2.py: RMSE/bias -> 0, slope -> 1.0
# only at factor 24).
#
# The training file has since been physically corrected at the source
# (see PrepareData.py) -- HR_FILES should now point at
#   precip_rcm_8km_daily_data_1995-2014_mmday_corrected.nc
# which already contains true mm/day values. Multiplying that by 24
# again here would double-apply the correction, so PRECIP_SCALE is now
# 1.0: we train directly on the raw (already-correct) precip values,
# only log1p + per-channel normalization are applied below.
#
# NOTE: any checkpoint trained under the old PRECIP_SCALE=24.0 + old
# (uncorrected) file combination is NOT compatible with this setting --
# its stored precip_transform (precip_scale=24.0, plus norm_mean/std
# fit on the old distribution) won't match what this pipeline now
# produces. Retrain from scratch against the corrected file rather than
# resuming/fine-tuning from an old checkpoint.
PRECIP_SCALE = 1.0

CROP_TO_16 = True

VAR_MAP = {"huss": "huss", "mslp": "psl", "tas": "tas", "precip": "pr"}

ORO_CANDIDATES = ["orog", "topology", "topo", "elevation", "elev", "z", "hgt", "surface_altitude", "unspecified"]

def load_oro(oro_path, target_hw):
    ds = xr.open_dataset(oro_path, decode_times=False)
    vname = next((k for k in ORO_CANDIDATES if k in ds.data_vars), None)
    if vname is None:
        cand = [v for v in ds.data_vars if ds[v].squeeze().ndim >= 2]
        if not cand:
            raise ValueError(f"No 2D var found in {oro_path}")
        vname = cand[0]
    arr = np.squeeze(ds[vname].values).astype(np.float32)
    ds.close()
    arr = np.where(arr >= 1e19, np.nan, arr)
    arr = np.nan_to_num(arr, nan=0.0)
    oro = torch.from_numpy(arr).unsqueeze(0)
    if tuple(oro.shape[-2:]) != tuple(target_hw):
        oro = F.interpolate(oro.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)
    return oro

class ClimateDataset(Dataset):
    PRECIP_CH = 3

    def __init__(self, hr_paths, oro_path, variables=("huss", "mslp", "tas", "precip"), rank=0):
        self.variables = list(variables)
        self.rank = rank

        def log(msg):
            if rank == 0:
                print(msg)

        log("Loading HR data...")
        arrays = self._load(hr_paths)

        self.years = self._extract_years(hr_paths[0])
        T0 = arrays[self.variables[0]].shape[0]
        m = min(len(self.years), T0)
        self.years = self.years[:m]
        for v in self.variables:
            arrays[v] = arrays[v][:m]

        H, W = arrays[self.variables[0]].shape[1:]
        if CROP_TO_16:
            H16, W16 = (H // 16) * 16, (W // 16) * 16
            if (H16, W16) != (H, W):
                log(f"Cropping {H}x{W} -> {H16}x{W16}")
                for v in self.variables:
                    arrays[v] = arrays[v][:, :H16, :W16]
                H, W = H16, W16
        self.H, self.W = H, W

        p_raw = arrays["precip"].astype(np.float32)
        log(f"Precip raw mean={p_raw.mean():.4f}, max={p_raw.max():.4f}")
        p = np.clip(p_raw * PRECIP_SCALE, 0.0, None)
        p = np.log1p(p).astype(np.float32)
        # Stored explicitly (not just used locally) so the full precip transform
        # -- raw mm -> *PRECIP_SCALE -> log1p -> (x - mean) / std -- can be
        # written into checkpoints and reproduced exactly at inference time.
        self.precip_scale = float(PRECIP_SCALE)
        self.precip_log_mean = float(np.mean(p))
        self.precip_log_std = float(np.std(p) + 1e-6)
        log(f"Precip log mean={self.precip_log_mean:.4f}, std={self.precip_log_std:.4f}")

        self.mean, self.std = {}, {}
        for v in self.variables:
            if v == "precip":
                arr = p
            else:
                arr = arrays[v].astype(np.float32)
            flat = arr.reshape(-1)
            flat = flat[np.isfinite(flat)]
            self.mean[v] = float(flat.mean())
            self.std[v] = float(flat.std() + 1e-6)
            arrays[v] = ((arr - self.mean[v]) / self.std[v]).astype(np.float32)

        self.hr = torch.from_numpy(np.stack([arrays[v] for v in self.variables], axis=1)).contiguous()
        self.length = self.hr.shape[0]
        self.oro = load_oro(oro_path, target_hw=(H, W))
        log(f"Orography grid: {tuple(self.oro.shape[-2:])} (target was {(H, W)})")

        log(f"Dataset ready: T={self.length}, HR={tuple(self.hr.shape[1:])}, LR={(H//DS_FACTOR, W//DS_FACTOR)}")

    def _load(self, paths):
        out = {}
        for p in paths:
            prefix = os.path.basename(p).split("_")[0]
            if prefix not in self.variables:
                continue
            nc_var = VAR_MAP[prefix]
            ds = xr.open_dataset(p, decode_times=False)
            if nc_var not in ds.data_vars:
                nc_var = list(ds.data_vars)[0]
            arr = ds[nc_var].values.astype(np.float32)
            ds.close()
            out[prefix] = np.nan_to_num(arr, nan=0.0)
        return {v: out[v] for v in self.variables}

    def _extract_years(self, hr_path):
        ds = xr.open_dataset(hr_path)
        tname = next((c for c in ["time", "Time", "date"] if c in ds.coords or c in ds.dims), None)
        if tname is None:
            ds.close()
            return np.zeros(1, dtype=int)
        times = ds[tname].values
        ds.close()
        try:
            return np.array([t.astype("datetime64[Y]").astype(int) + 1970 for t in times])
        except Exception:
            import pandas as pd
            return pd.to_datetime(times).year.values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hr = self.hr[idx]
        # LR is built by average-pooling the already per-channel-normalized HR
        # tensor (not raw physical units). This is intentional: the model is
        # trained end-to-end in normalized space, so LR and HR share the same
        # mean/std per channel and Train.py's own F.avg_pool2d(hr, ...) call
        # reproduces this exactly. If you ever want LR pooled from raw
        # (pre-normalization) values instead, pool `arrays[v]` in __init__
        # before the normalization step and store it separately.
        lr = F.avg_pool2d(hr.unsqueeze(0), kernel_size=DS_FACTOR, stride=DS_FACTOR).squeeze(0)
        return {"hr": hr, "lr": lr, "oro": self.oro, "idx": torch.tensor(idx, dtype=torch.long)}

    def denorm_precip_mmday(self, x_norm):
        x_log = x_norm * self.std["precip"] + self.mean["precip"]
        return torch.expm1(torch.clamp(x_log, min=0.0))

    def get_precip_transform_meta(self):
        """
        Everything needed to invert the precip pipeline at inference time:
        raw_mm -> *precip_scale -> log1p -> (x - precip_log_mean) / precip_log_std
        (the log1p mean/std below double as the standard per-channel norm
        stats, since precip is normalized post-log1p same as the other vars).

        precip_scale is now 1.0 (see the PRECIP_SCALE comment above) --
        this key is kept in the returned dict for backward compatibility
        with the existing checkpoint / eval-script format, which divides
        by pt["precip_scale"] when denormalizing.
        """
        return {
            "precip_scale": self.precip_scale,
            "precip_log_mean": self.precip_log_mean,
            "precip_log_std": self.precip_log_std,
            "norm_mean": self.mean["precip"],
            "norm_std": self.std["precip"],
            "ds_factor": DS_FACTOR,
        }


def get_climate_kfolds(ds, k=5, val_ratio=0.15):
    """
    Build k rolling-block folds over the years present in `ds.years`.

    Each fold holds out one contiguous block of years as the TEST set
    (never trained or validated on). Validation is a single contiguous,
    deterministic slice taken from the middle of the remaining years --
    simpler to reason about and debug than a scattered pick, at the cost
    of validation living in one stretch of time rather than sampling the
    full remaining climatology. Everything else remaining goes to train.

    With 20 years of data (1995-2014) and k=5, this produces exactly:
        1995-1998, 1999-2002, 2003-2006, 2007-2010, 2011-2014
    Fold names always use this "STARTYEAR-ENDYEAR" format -- it's what
    Train.py uses for checkpoint filenames and log headers, so keep it
    consistent if you touch this function.

    Returns
    -------
    list[dict] with keys:
        name        : str, e.g. "1995-1998" (used in checkpoint filenames)
        train_years : sorted list[int]
        val_years   : sorted list[int]
        test_years  : sorted list[int]
        train_idx   : list[int]  -- dataset indices for Subset()
        val_idx     : list[int]  -- dataset indices for Subset()
        test_idx    : list[int]  -- dataset indices for Subset(), for holdout eval
    """
    years = np.asarray(ds.years)
    uniq_years = np.unique(years)
    uniq_years.sort()

    if k > len(uniq_years):
        raise ValueError(f"k={k} folds requested but only {len(uniq_years)} unique years in dataset")

    test_blocks = np.array_split(uniq_years, k)

    folds = []
    for block in test_blocks:
        test_years = np.asarray(block)
        remaining = np.asarray([y for y in uniq_years if y not in test_years])

        n_val = max(1, int(round(len(remaining) * val_ratio)))
        start = max(0, (len(remaining) - n_val) // 2)
        val_years = remaining[start:start + n_val]
        train_years = np.asarray([y for y in remaining if y not in val_years])

        train_idx = np.where(np.isin(years, train_years))[0].tolist()
        val_idx = np.where(np.isin(years, val_years))[0].tolist()
        test_idx = np.where(np.isin(years, test_years))[0].tolist()

        name = f"{int(test_years.min())}-{int(test_years.max())}"

        folds.append({
            "name": name,
            "train_years": sorted(int(y) for y in train_years),
            "val_years": sorted(int(y) for y in val_years),
            "test_years": sorted(int(y) for y in test_years),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        })

    return folds
