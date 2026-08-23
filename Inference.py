# -*- coding: utf-8 -*-
"""
Inference.py -- CorrDiff cascade: apply the 32km->8km model to make 2 km fields
===============================================================================
Both stages were trained on

    LR (32 km, = 8 km target average-pooled by DS_FACTOR=4)  ->  HR (8 km)

This script reuses those same weights for the next rung of the ladder

    LR (8 km, a REAL RCM field)  ->  HR (2 km)

by feeding the true 8 km fields in where a pooled 32 km field used to go.

WHY THIS IS LEGITIMATE, AND WHERE IT ISN'T
-------------------------------------------
It works because the operator being learned is *local and scale-relative*:
"given an area-mean field and fine topography, put back the sub-grid structure
at 4x". Nothing in that statement mentions kilometres. What it does depend on:

  (a) The coarse input must look like an area mean of the target. An 8 km RCM
      field is close to, but not exactly, the area mean of a 2 km field.
      TrainDiffusion.py's LR_AUG jitter is there to buy tolerance to this.

  (b) The network must see the same PIXEL GRID SIZE it trained on. This is the
      part the previous version of this script got wrong: it handed the full
      2 km domain to models trained on 8 km patches. SpectralConv2d ties learned
      weights to absolute FFT modes, and SelfAttn2d switches to pooled attention
      past a token budget -- so a 4x-larger canvas is a genuinely different
      operator, not the same one applied more times. Both stages are now run
      TILED at exactly the training patch size (read from the checkpoint), with
      MultiDiffusion-style blending of the denoiser output at every sampler
      step so the result is one globally coherent realization, not a quilt of
      independently sampled squares.

  (c) Precipitation variance is NOT scale-invariant: the sub-grid variance the
      model learned to add going 32->8 km is not identical to what belongs at
      8->2 km. `--res-std-scale` exposes this as an explicit, documented knob
      rather than an unstated assumption. Leave it at 1.0 unless you have
      2 km validation data to tune it against, and say so when you report.

  (d) Normalization statistics are the 8 km ones. Fine fields have heavier
      tails than coarse ones, so expect the extreme tail to be conservative.

None of (a), (c), (d) is fixed by this script. They are the honest caveats of
the cascade; treat the 2 km output as a physically plausible refinement, not a
validated product, until you have held-out 2 km data to score it against.

Output
------
One NetCDF per calendar year, covering only that fold's held-out TEST years:
  precip_mean         ensemble-mean downscaled precip, mm/day  [time,lat,lon]
  precip_std          ensemble spread (uncertainty),   mm/day
  precip_stage1_mean  Stage-1-only deterministic baseline, mm/day
  precip_members      (with --save-members) individual members

Usage
-----
  python Inference.py --oro-2km /path/to/SG.orog.2km.nc
  python Inference.py --shard ${PBS_ARRAY_INDEX} --num-shards 8
  python Inference.py --years 2004 2005 2006
  python Inference.py --merge-only
"""

import os
import glob
import argparse

import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F

from Config import (HR_FILES, ORO_8KM, ORO_2KM, REGRESSOR_CKPT_DIR, OUTPUT_DIR,
                    KFOLD_K, VAL_RATIO, EVAL_SEED, PATCH_OVERLAP)
from Dataset import ClimateDataset, get_climate_kfolds, DS_FACTOR, load_oro, ORO_CANDIDATES
from Network import expand_topo, denorm_precip_mmday
from Tiling import regress_tiled
from Param import (PARAM, CKPT_DIR as DIFFUSION_CKPT_DIR, SAMPLE_STEPS as DEFAULT_STEPS,
                   sample_residual, config_from_dict, with_stochasticity,
                   with_sigma_max_sample, nfe_count)
from TrainStage2 import (load_frozen_regressor, assert_precip_transform_compatible,
                         pick_amp, build_unet)

# ------------------------------------------------------------------------------
# DEFAULTS (all overridable via CLI)
# ------------------------------------------------------------------------------
ENSEMBLE_MEMBERS = 16
SAMPLE_STEPS = DEFAULT_STEPS
INFER_BATCH = 4
SAVE_FULL_ENSEMBLE = False
ORO_2KM_DEFAULT = ORO_2KM


# ------------------------------------------------------------------------------
# CHECKPOINT LOADING
# ------------------------------------------------------------------------------

def _find_ckpt(ckpt_dir, exact_name):
    candidate = os.path.join(ckpt_dir, exact_name)
    if os.path.exists(candidate):
        return candidate
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith((".pt", ".pth"))]) \
        if os.path.isdir(ckpt_dir) else []
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {os.path.abspath(ckpt_dir)}")
    chosen = sorted([f for f in files if "best" in f.lower()] or files)[0]
    print(f"  [WARN] '{exact_name}' not found in {ckpt_dir}; falling back to {chosen}. "
          "If this is not the fold that held these years out, the result is contaminated.")
    return os.path.join(ckpt_dir, chosen)


def load_frozen_diffusion(path, dev):
    """Rebuild Stage 2 from the checkpoint's OWN recorded architecture.

    The previous version reconstructed the UNet from whatever constants happened
    to be sitting in TrainDiffusion.py at the time the script was run, which is
    correct exactly until someone edits a hyperparameter -- after which it either
    raises a state_dict error (lucky) or silently loads a mismatched model
    (unlucky). Old checkpoints without an 'arch' key are rejected loudly rather
    than guessed at.
    """
    ck = torch.load(path, map_location=dev, weights_only=False)
    arch = ck.get("arch")
    if arch is None:
        raise RuntimeError(
            f"{path} predates architecture-carrying checkpoints. Retrain with the "
            "current TrainDiffusion.py -- the residual-mean and loss-weighting "
            "fixes make old weights incorrect anyway, not merely unloadable.")
    net = build_unet(arch, dev, dropout=0.0)
    net.load_state_dict(ck.get("ema_state_dict") or ck["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    param = ck.get("param", PARAM)
    if param != PARAM:
        raise RuntimeError(
            f"{path} was trained with param='{param}' but this package implements "
            f"'{PARAM}'. Use the other package's Inference.py -- the two are not "
            "interchangeable, and loading the weights would silently apply the wrong "
            "sampler to them.")
    cfg = config_from_dict(ck.get("edm_cfg") or ck.get("fm_cfg"))
    meta = {
        "param": param,
        "patch": ck.get("patch"),
        "res_mean": ck["res_mean"],
        "res_std": ck["res_std"],
        "cfg": cfg,
    }
    print(f"  [stage-2] param={meta['param']}  train_patch={meta['patch']}  "
          f"res_mean={meta['res_mean']:.5f}  res_std={meta['res_std']:.5f}")
    if meta["patch"] is None:
        print("  [stage-2] [WARNING] this model was trained full-domain, not on patches. "
              "Tiling it at 2 km is still the safer option, but the tile size will be "
              "guessed from the 8 km domain and transfer quality is not guaranteed.")
    return net, ck, meta


# ------------------------------------------------------------------------------
# OUTPUT COORDINATES
# ------------------------------------------------------------------------------

def _precip_path(hr_files):
    return next(p for p in hr_files if os.path.basename(p).startswith("precip"))


def _refine_1d(coord, factor):
    """Disaggregate a 1-D coordinate onto a `factor`x finer CELL-CENTRE grid.

    The old implementation used linspace(0, n-1, n*factor), which spans the same
    end POINTS as the coarse grid. That is wrong for cell centres: refining a
    cell of width d into 4 sub-cells puts their centres at -3d/8, -d/8, +d/8,
    +3d/8 relative to the parent centre, so the refined grid extends half a
    coarse cell BEYOND the original centres at each end. The old version instead
    squeezed the fine grid inside the coarse centres, shrinking the domain by
    one coarse cell and shifting every point -- a systematic half-cell
    geolocation error that then propagated into the orography regrid.
    """
    coord = np.asarray(coord, dtype=np.float64)
    n = len(coord)
    idx = np.arange(n, dtype=np.float64)
    fine_idx = (np.arange(n * factor, dtype=np.float64) + 0.5) / factor - 0.5
    d = np.gradient(coord)
    lo = coord[0] + (fine_idx[fine_idx < 0]) * d[0]
    hi = coord[-1] + (fine_idx[fine_idx > n - 1] - (n - 1)) * d[-1]
    mid = np.interp(fine_idx[(fine_idx >= 0) & (fine_idx <= n - 1)], idx, coord)
    return np.concatenate([lo, mid, hi]).astype(np.float32)


def _build_output_coords(hr_files, H, W, ds_length):
    precip_path = _precip_path(hr_files)
    dsx = xr.open_dataset(precip_path, decode_times=False)

    time_name = next((c for c in ("time", "Time", "date", "t")
                      if c in dsx.coords or c in dsx.dims), None)
    time_vals = np.asarray(dsx[time_name].values) if time_name is not None else np.arange(ds_length)
    time_vals = time_vals[:ds_length]
    time_attrs = dict(dsx[time_name].attrs) if time_name is not None else {}

    lat_name = next((c for c in ("rlat", "lat", "latitude", "y") if c in dsx.coords), None)
    lon_name = next((c for c in ("rlon", "lon", "longitude", "x") if c in dsx.coords), None)
    lat = np.asarray(dsx[lat_name].values) if lat_name is not None else np.arange(H, dtype=np.float32)
    lon = np.asarray(dsx[lon_name].values) if lon_name is not None else np.arange(W, dtype=np.float32)
    dsx.close()

    lat, lon = lat[:H], lon[:W]
    lat2, lon2 = _refine_1d(lat, DS_FACTOR), _refine_1d(lon, DS_FACTOR)
    print(f"  [coords] '{lat_name}' x '{lon_name}' from {precip_path}")
    print(f"  [coords] 8 km {lat.min():.4f}..{lat.max():.4f} / {lon.min():.4f}..{lon.max():.4f}")
    print(f"  [coords] 2 km {lat2.min():.4f}..{lat2.max():.4f} ({len(lat2)}) / "
          f"{lon2.min():.4f}..{lon2.max():.4f} ({len(lon2)})")
    return time_vals, time_attrs, lat2, lon2


def _get_rotated_pole_params(path):
    try:
        ds = xr.open_dataset(path, decode_times=False)
        for _, var in ds.variables.items():
            if var.attrs.get("grid_mapping_name") == "rotated_latitude_longitude":
                lon = float(var.attrs.get("grid_north_pole_longitude", np.nan))
                lat = float(var.attrs.get("grid_north_pole_latitude", np.nan))
                ds.close()
                return lon, lat
        ds.close()
    except Exception as e:
        print(f"  [WARN] could not read rotated_pole params from {path}: {e}")
    return None, None


def load_oro_2km_regridded(oro_2km_path, target_lat, target_lon):
    """Coordinate-aware regrid of a REAL 2 km orography onto the model's output grid.

    Interpolates on coordinate VALUES, not pixel indices, because the native
    2 km product is generally not a clean nested 4x refinement of the cropped
    8 km grid (a 939x939 native grid is not even divisible by 4).
    """
    ds = xr.open_dataset(oro_2km_path, decode_times=False)
    vname = next((k for k in ORO_CANDIDATES if k in ds.data_vars), None)
    if vname is None:
        cand = [v for v in ds.data_vars if ds[v].squeeze().ndim >= 2]
        if not cand:
            raise ValueError(f"No 2D var found in {oro_2km_path}")
        vname = cand[0]

    da = ds[vname].squeeze()
    dims = da.dims
    lat_dim = next((d for d in dims if d.lower() in ("rlat", "lat", "latitude", "y")), dims[0])
    lon_dim = next((d for d in dims if d.lower() in ("rlon", "lon", "longitude", "x")), dims[1])

    arr = np.where(da.values.astype(np.float32) >= 1e19, np.nan, da.values.astype(np.float32))
    native_lat = ds[lat_dim].values.astype(np.float64) if lat_dim in ds.coords else np.arange(arr.shape[0])
    native_lon = ds[lon_dim].values.astype(np.float64) if lon_dim in ds.coords else np.arange(arr.shape[1])
    ds.close()

    print(f"  [oro-2km] native {arr.shape}  {lat_dim} {native_lat.min():.4f}..{native_lat.max():.4f}  "
          f"{lon_dim} {native_lon.min():.4f}..{native_lon.max():.4f}")

    lat_oob = float(np.mean((target_lat < native_lat.min()) | (target_lat > native_lat.max())))
    lon_oob = float(np.mean((target_lon < native_lon.min()) | (target_lon > native_lon.max())))
    if lat_oob > 0 or lon_oob > 0:
        print(f"  [oro-2km] [WARNING] target grid runs outside the native file: "
              f"{lat_oob:.1%} of lat, {lon_oob:.1%} of lon will be EXTRAPOLATED. "
              "Large fractions mean the two domains do not overlap properly.")

    da_native = xr.DataArray(arr, dims=(lat_dim, lon_dim),
                             coords={lat_dim: native_lat, lon_dim: native_lon})
    out = da_native.interp({lat_dim: target_lat, lon_dim: target_lon},
                           method="linear", kwargs={"fill_value": None}).values.astype(np.float32)
    n_nan = int(np.isnan(out).sum())
    if n_nan:
        print(f"  [oro-2km] [WARNING] {n_nan}/{out.size} ({n_nan/out.size:.1%}) NaN after regrid "
              "(likely _FillValue-masked ocean) -- filling with 0.0.")
    return torch.from_numpy(np.nan_to_num(out, nan=0.0)).unsqueeze(0)


# ------------------------------------------------------------------------------
# NETCDF WRITING
# ------------------------------------------------------------------------------

def _write_year_nc(path, year, fold_name, time_vals, time_attrs, lat, lon,
                   mean_mm, std_mm, stage1_mm, members_mm, meta):
    data_vars = {
        "precip_mean": (("time", "lat", "lon"), mean_mm),
        "precip_std": (("time", "lat", "lon"), std_mm),
        "precip_stage1_mean": (("time", "lat", "lon"), stage1_mm),
    }
    coords = {"time": time_vals, "lat": lat, "lon": lon}
    if members_mm is not None:
        data_vars["precip_members"] = (("time", "member", "lat", "lon"), members_mm)
        coords["member"] = np.arange(members_mm.shape[1])

    out = xr.Dataset(data_vars, coords=coords)
    out["time"].attrs.update(time_attrs)
    out["precip_mean"].attrs.update(units="mm/day", long_name="Ensemble-mean downscaled precipitation")
    out["precip_std"].attrs.update(units="mm/day", long_name="Ensemble standard deviation")
    out["precip_stage1_mean"].attrs.update(units="mm/day", long_name="Stage-1 regression only (no diffusion)")
    out.attrs.update(
        description=("CorrDiff cascade: a 32km->8km-trained residual-corrective diffusion model "
                     "applied to real 8 km fields to synthesize 2 km structure. Tiled at the "
                     "training patch size with MultiDiffusion blending."),
        year=int(year), source_fold=fold_name, ds_factor=DS_FACTOR,
        ensemble_members=int(members_mm.shape[1]) if members_mm is not None else ENSEMBLE_MEMBERS,
        sample_steps=SAMPLE_STEPS, param=meta["param"], tile=int(meta["tile"]),
        res_std_scale=float(meta["res_std_scale"]),
        caveat=("Normalization statistics and residual variance are inherited from the 8 km "
                "training distribution; extremes at 2 km are expected to be conservative."),
    )
    enc = {v: {"zlib": True, "complevel": 4} for v in data_vars}
    out.to_netcdf(path, encoding=enc)
    print(f"    wrote {path}  ({out.nbytes/1e6:.1f} MB, {len(time_vals)} days)")


def merge_all_years(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "precip_2km_[0-9][0-9][0-9][0-9].nc")))
    if not files:
        print("No yearly files found to merge yet.")
        return
    print(f"Merging {len(files)} yearly file(s)")
    merged = xr.open_mfdataset(files, combine="by_coords").sortby("time")
    out_path = os.path.join(out_dir, "precip_2km_all_years.nc")
    merged.to_netcdf(out_path)
    print(f"wrote {out_path}  ({merged.sizes['time']} days)")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    global ENSEMBLE_MEMBERS, SAMPLE_STEPS, INFER_BATCH, SAVE_FULL_ENSEMBLE, OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None,
                    help="0-based shard index; splits YEARS (not folds) into --num-shards groups.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--fold", type=int, default=None, help="Run exactly this fold's test years.")
    ap.add_argument("--years", type=int, nargs="+", default=None, help="Explicit year list.")
    ap.add_argument("--members", type=int, default=ENSEMBLE_MEMBERS)
    ap.add_argument("--steps", type=int, default=SAMPLE_STEPS)
    ap.add_argument("--batch", type=int, default=INFER_BATCH)
    ap.add_argument("--save-members", action="store_true")
    ap.add_argument("--out-dir", type=str, default=OUTPUT_DIR)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--oro-2km", type=str, default=ORO_2KM_DEFAULT,
                    help="Real 2 km orography, coordinate-regridded onto the target grid. "
                         "Empty string falls back to a bilinear pixel-resize of the 8 km file.")
    ap.add_argument("--diff-dir", type=str, default=None,
                    help="Stage-2 checkpoint directory. Defaults to TrainDiffusion.CKPT_DIR, "
                         "which follows whatever PARAM is currently set there. With both an "
                         "EDM and a flow-matching arm trained you almost always want to name "
                         "the directory explicitly rather than rely on that default.")
    ap.add_argument("--reg-dir", type=str, default=None,
                    help="Stage-1 checkpoint directory (default TrainDiffusion.REGRESSOR_CKPT_DIR).")
    ap.add_argument("--tile", type=int, default=None,
                    help="Tile size in OUTPUT (2 km) pixels. Default: the training patch size "
                         "recorded in the checkpoint. Changing it breaks the guarantee that the "
                         "network sees the grid size it trained on.")
    ap.add_argument("--overlap", type=int, default=None,
                    help="Tile overlap in output pixels (default tile//4). More overlap = smoother "
                         "blending at linearly more compute.")
    ap.add_argument("--tiles-per-batch", type=int, default=8)
    ap.add_argument("--res-std-scale", type=float, default=1.0,
                    help="Multiplier on the residual std when reconstructing at 2 km. See caveat "
                         "(c) in the module docstring: sub-grid precipitation variance is not "
                         "scale-invariant, so the 32->8 km residual amplitude is not guaranteed "
                         "to be the right 8->2 km amplitude. 1.0 = assume it is. Tune only "
                         "against held-out 2 km data, and report the value you used.")
    ap.add_argument("--sigma-max-sample", type=float, default=None,
                    help="EDM package only (accepted and ignored by the flow-matching "
                         "package, whose time grid is always [0,1]). Sampling-time "
                         "sigma_max, overriding the checkpoint. sigma_max is a SAMPLER "
                         "choice -- it never appears in training -- so this changes nothing "
                         "about the trained model. The paper's 800 with only 18 steps "
                         "over-disperses by ~20%% from Heun discretization alone; 200 with "
                         "32 steps, or 800 with 64, is materially better calibrated.")
    ap.add_argument("--churn", type=float, default=None,
                    help="Override the sampler's stochasticity (EDM S_churn / FM churn). "
                         "Use this to isolate how much of the ensemble spread comes from "
                         "the sampler rather than the parameterization.")
    ap.add_argument("--deterministic", action="store_true",
                    help="Ablation: S_churn=0 (probability-flow ODE). Expect visible spread "
                         "collapse and over-smooth fields -- this is the control, not the product.")
    args = ap.parse_args()

    ENSEMBLE_MEMBERS = args.members
    SAMPLE_STEPS = args.steps
    INFER_BATCH = args.batch
    SAVE_FULL_ENSEMBLE = args.save_members
    OUTPUT_DIR = args.out_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.merge_only:
        merge_all_years(OUTPUT_DIR)
        return

    diff_dir = args.diff_dir or DIFFUSION_CKPT_DIR
    reg_dir = args.reg_dir or REGRESSOR_CKPT_DIR
    print(f"Stage-1 dir: {reg_dir}\nStage-2 dir: {diff_dir}")

    dev = torch.device(args.device)
    amp_dtype, _ = pick_amp(dev)

    print("Loading 8 km data (normalization stats AND the real LR input for this stage)...")
    ds = ClimateDataset(HR_FILES, ORO_8KM, rank=0)
    pt = ds.get_precip_transform_meta()
    H, W = ds.H, ds.W
    H2, W2 = H * DS_FACTOR, W * DS_FACTOR
    print(f"8 km grid: {H}x{W}  ->  2 km grid: {H2}x{W2}")

    time_full, time_attrs, lat2km, lon2km = _build_output_coords(HR_FILES, H, W, ds.length)

    if args.oro_2km:
        print(f"Regridding real 2 km orography from {args.oro_2km}...")
        p8, p2 = _get_rotated_pole_params(_precip_path(HR_FILES)), _get_rotated_pole_params(args.oro_2km)
        if None not in p8 and None not in p2:
            if abs(p8[0] - p2[0]) > 1e-3 or abs(p8[1] - p2[1]) > 1e-3:
                print(f"  [WARNING] rotated_pole differs: 8km={p8} vs 2km={p2}. Coordinate values "
                      "are not directly comparable without a proper rotation; the regridded "
                      "orography may be geographically misaligned.")
            else:
                print(f"  rotated_pole matches: {p2}")
        oro_2km = load_oro_2km_regridded(args.oro_2km, lat2km, lon2km).to(dev).unsqueeze(0)
    else:
        print("  [NOTE] no 2 km orography given -- bilinearly upsampling the 8 km file by pixel "
              "shape. This invents no new terrain detail, so the model loses the single most "
              "informative predictor it has at 2 km. Supply --oro-2km if you possibly can.")
        oro_2km = load_oro(ORO_8KM, target_hw=(H2, W2)).to(dev).unsqueeze(0)

    # Computed ONCE over the full 2 km domain, then cropped per tile -- expand_topo
    # standardizes elevation and lays down y/x coordinate channels domain-wide, so
    # recomputing it per tile would give every tile a different encoding.
    topo_2km = expand_topo(oro_2km)

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)
    year_to_fold = {y: f for f in folds for y in f["test_years"]}
    all_years = sorted(year_to_fold.keys())

    if args.years is not None:
        my_years = args.years
    elif args.shard is not None:
        my_years = [int(y) for y in np.array_split(np.array(all_years), args.num_shards)[args.shard]]
    elif args.fold is not None:
        my_years = folds[args.fold]["test_years"]
    else:
        my_years = all_years
    print(f"This job will process {len(my_years)} year(s): {my_years}")

    regressor = net = None
    current_fold = None
    s2 = None

    for yr in my_years:
        if yr not in year_to_fold:
            print(f"  [SKIP] year {yr} is in no fold's test set.")
            continue
        fold = year_to_fold[yr]
        fold_name = fold["name"]

        if fold_name != current_fold:
            if regressor is not None:
                del regressor, net
                torch.cuda.empty_cache()
            reg_path = _find_ckpt(reg_dir, f"Regressor_{fold_name}_best.pth")
            diff_path = _find_ckpt(diff_dir, f"Stage2_{fold_name}_best.pth")
            print(f"\n[fold {fold_name}] regressor: {reg_path}")
            print(f"[fold {fold_name}] diffusion: {diff_path}")

            regressor = load_frozen_regressor(reg_path, dev, pt, rank=0)
            net, diff_ck, s2 = load_frozen_diffusion(diff_path, dev)
            assert_precip_transform_compatible(diff_ck, pt, diff_path, 0, "Stage-2 diffusion")
            current_fold = fold_name

            tile = args.tile or s2["patch"] or min(H, W)
            cfg = s2["cfg"]
            if args.deterministic:
                cfg = with_stochasticity(cfg, 0.0)
                print("  [stage-2] stochasticity forced to 0 (deterministic ODE ablation)")
            if args.sigma_max_sample is not None:
                cfg = with_sigma_max_sample(cfg, args.sigma_max_sample)
                print(f"  [stage-2] sampling sigma_max -> {args.sigma_max_sample}")
            if args.churn is not None:
                cfg = with_stochasticity(cfg, args.churn)
                print(f"  [stage-2] sampler stochasticity -> {args.churn}")
            # Printed AFTER the overrides above, so the log shows the config
            # actually used for sampling rather than the one on disk.
            print(f"  [stage-2] effective config: {cfg}")
            print(f"  [stage-2] {SAMPLE_STEPS} steps = {nfe_count(SAMPLE_STEPS)} NFE/member")
            res_mean = s2["res_mean"]
            res_std = s2["res_std"] * args.res_std_scale
            if args.res_std_scale != 1.0:
                print(f"  [stage-2] res_std scaled by {args.res_std_scale} -> {res_std:.5f}")
            print(f"  [stage-2] tiling 2 km domain {H2}x{W2} into {tile}x{tile} tiles "
                  f"(overlap {args.overlap or max(8, tile//4)})")

        idx = np.where(np.asarray(ds.years) == yr)[0].tolist()
        print(f"=== Year {yr}  (fold {fold_name}, {len(idx)} days) ===")

        mean_out = np.empty((len(idx), H2, W2), dtype=np.float32)
        std_out = np.empty((len(idx), H2, W2), dtype=np.float32)
        stage1_out = np.empty((len(idx), H2, W2), dtype=np.float32)
        members_out = (np.empty((len(idx), ENSEMBLE_MEMBERS, H2, W2), dtype=np.float32)
                       if SAVE_FULL_ENSEMBLE else None)

        gen = torch.Generator(device=dev).manual_seed(EVAL_SEED + int(yr))

        for start in range(0, len(idx), INFER_BATCH):
            bidx = idx[start:start + INFER_BATCH]
            lr = ds.hr[bidx].to(dev)                     # real, normalized 8 km fields
            B = lr.shape[0]
            topo_b = topo_2km.expand(B, -1, -1, -1)

            with torch.no_grad():
                # Stage 1, tiled at the same scale ratio the regressor trained on.
                mu = regress_tiled(regressor, lr, topo_b, DS_FACTOR,
                                   tile_hr=tile, overlap=args.overlap, amp_dtype=amp_dtype)
                if tuple(mu.shape[-2:]) != (H2, W2):
                    raise RuntimeError(
                        f"Stage-1 output {tuple(mu.shape[-2:])} != expected {(H2, W2)}. "
                        f"The cascade assumes a fixed x{DS_FACTOR} upsample independent of "
                        "input size -- check CorrDiffRegressor.forward.")

                lr_up = F.interpolate(lr, size=(H2, W2), mode="bilinear", align_corners=False)
                cond = torch.cat([mu, lr_up.float()], dim=1)
                shape = (B, 1, H2, W2)

                members_mm = []
                for _ in range(ENSEMBLE_MEMBERS):
                    # ONE global latent + ONE global noise draw; tiles contribute
                    # only their denoiser (EDM) or velocity (FM) output, blended
                    # each step. Seamless either way.
                    r = sample_residual(net, cond, topo_b, shape, dev, cfg,
                                        n_steps=SAMPLE_STEPS, generator=gen, tile=tile,
                                        overlap=args.overlap, amp_dtype=amp_dtype,
                                        max_tiles_per_batch=args.tiles_per_batch)
                    members_mm.append(denorm_precip_mmday(mu + res_mean + res_std * r, pt))

                ens = torch.stack(members_mm, 0)                 # [M,B,1,H2,W2]
                stage1_mm = denorm_precip_mmday(mu + res_mean, pt)

            mean_out[start:start + B] = ens.mean(0).squeeze(1).cpu().numpy()
            std_out[start:start + B] = ens.std(0).squeeze(1).cpu().numpy()
            stage1_out[start:start + B] = stage1_mm.squeeze(1).cpu().numpy()
            if SAVE_FULL_ENSEMBLE:
                members_out[start:start + B] = ens.squeeze(2).permute(1, 0, 2, 3).cpu().numpy()
            print(f"  [{yr}] {min(start + B, len(idx))}/{len(idx)} days")

        _write_year_nc(
            os.path.join(OUTPUT_DIR, f"precip_2km_{yr}.nc"),
            yr, fold_name, time_full[idx], time_attrs, lat2km, lon2km,
            mean_out, std_out, stage1_out, members_out,
            {"param": s2["param"], "tile": tile, "res_std_scale": args.res_std_scale},
        )

    if regressor is not None:
        del regressor, net
        torch.cuda.empty_cache()

    expected = {f"precip_2km_{y}.nc" for y in all_years}
    present = {os.path.basename(p) for p in glob.glob(os.path.join(OUTPUT_DIR, "precip_2km_*.nc"))}
    missing = sorted(expected - present)
    if not missing:
        print(f"\nAll {len(expected)} yearly files present in {OUTPUT_DIR}.")
        print("Run `python Inference.py --merge-only` for a single combined file.")
    else:
        print(f"\n{len(missing)}/{len(expected)} outputs still missing: {missing}")


if __name__ == "__main__":
    main()
