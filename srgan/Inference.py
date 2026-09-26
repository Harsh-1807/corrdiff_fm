# -*- coding: utf-8 -*-
"""
Inference.py -- SRGAN cascade: apply the 32km->8km-trained generator to make
2 km fields from real 8 km data.
==============================================================================
Trained on

    LR (32 km, = 8 km target average-pooled by DS_FACTOR=4)  ->  HR (8 km)

This script reuses those same weights for the next rung of the ladder

    LR (8 km, a REAL RCM field)  ->  HR (2 km)

by feeding the true 8 km fields in where a pooled 32 km field used to go --
identical cascade logic to corrdiff_fm's Inference.py, and the same caveats
(a)/(c)/(d) below apply for the same reasons. Caveat (b) -- "the network must
see the same pixel grid size it trained on" -- applies in a WEAKER form here;
see Tiling.py's module docstring for exactly why, and do not conflate the two.

WHY THIS IS LEGITIMATE, AND WHERE IT ISN'T
-------------------------------------------
It works because the operator being learned is *local and scale-relative*:
"given an area-mean field and fine topography, put back the sub-grid structure
at 4x". Nothing in that statement mentions kilometres. What it does depend on:

  (a) The coarse input must look like an area mean of the target. An 8 km RCM
      field is close to, but not exactly, the area mean of a 2 km field.
      Train.py's LR_AUG jitter is there to buy tolerance to this.

  (b) The network should ideally see a similar pixel grid size to what it
      trained on. Unlike corrdiff_fm's Stage-2 U-Net, this package's Generator
      has no absolute-FFT-mode or attention-token-budget block (see Tiling.py's
      docstring for the detailed argument), so this is a much weaker
      requirement here -- but tiling is still used, both because finite-
      receptive-field boundary effects between independently-computed tiles
      are real, and because a full 2 km domain forward pass may simply not
      fit in memory.

  (c) Precipitation variance is NOT scale-invariant in general: whatever
      sub-grid structure the generator learned to synthesize going 32->8 km
      is not guaranteed identical in amplitude to what belongs at 8->2 km.
      Unlike corrdiff_fm, there is no explicit residual-std knob to expose
      here -- the generator's output IS the field, not a residual added to a
      separately-scaled baseline -- so this caveat cannot be mitigated by a
      post-hoc multiplier the way `--res-std-scale` does for corrdiff_fm.
      Treat any systematic amplitude bias at 2 km as evidence for this, not as
      a bug to patch with a knob that does not exist for this architecture.

  (d) Normalization statistics are the 8 km ones. Fine fields have heavier
      tails than coarse ones, so expect the extreme tail to be conservative.

None of (a), (c), (d) is fixed by this script. They are the honest caveats of
the cascade; treat the 2 km output as a physically plausible refinement, not a
validated product, until you have held-out 2 km data to score it against.

OUTPUT -- ONE DETERMINISTIC FIELD, NO ENSEMBLE
-------------------------------------------------
Unlike corrdiff_fm's Inference.py, this script writes ONLY `precip_mean` (the
generator's single deterministic output). There is deliberately no
`precip_std` and no `precip_members`: this package's generator has no noise
input (see Network.py's docstring) and therefore nothing to sample repeatedly
-- one forward pass per day is the entire computation, and "precip_mean" is a
slight abuse of the name kept only for output-schema parity with corrdiff_fm
(it is not a mean over anything; it is the one prediction). Every yearly
NetCDF's global attributes say this explicitly so a reader who only has the
file, not this script, cannot mistake the absence of `precip_std` for a bug.

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

from Config import (HR_FILES, ORO_8KM, ORO_2KM, GENERATOR_CKPT_DIR, OUTPUT_DIR,
                    KFOLD_K, VAL_RATIO, EVAL_SEED, PATCH_OVERLAP, DS_FACTOR)
from Dataset import ClimateDataset, get_climate_kfolds, load_oro, ORO_CANDIDATES
from Network import expand_topo, denorm_precip_mmday
from Tiling import TiledGenerator
from Train import assert_precip_transform_compatible, pick_amp
from Evaluate import _find_ckpt, load_generator

# ------------------------------------------------------------------------------
# DEFAULTS (all overridable via CLI)
# ------------------------------------------------------------------------------
INFER_BATCH = 4


# ------------------------------------------------------------------------------
# OUTPUT COORDINATES -- identical geometry/regridding logic to corrdiff_fm's
# Inference.py. Duplicated rather than imported (this package must stand
# alone -- see README) but not model-specific in any way: this is purely
# NetCDF coordinate bookkeeping.
# ------------------------------------------------------------------------------

def _precip_path(hr_files):
    return next(p for p in hr_files if os.path.basename(p).startswith("precip"))


def _refine_1d(coord, factor):
    """Disaggregate a 1-D coordinate onto a `factor`x finer CELL-CENTRE grid.

    Refining a cell of width d into 4 sub-cells puts their centres at
    -3d/8, -d/8, +d/8, +3d/8 relative to the parent centre, so the refined
    grid extends half a coarse cell BEYOND the original centres at each end
    -- naive linspace-to-the-same-endpoints refinement is off by half a cell.
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
    """Coordinate-aware regrid of a REAL 2 km orography onto the model's
    output grid (interpolated on coordinate VALUES, not pixel indices, since
    the native 2 km product is generally not a clean nested 4x refinement of
    the cropped 8 km grid)."""
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
              f"{lat_oob:.1%} of lat, {lon_oob:.1%} of lon will be EXTRAPOLATED.")

    da_native = xr.DataArray(arr, dims=(lat_dim, lon_dim),
                             coords={lat_dim: native_lat, lon_dim: native_lon})
    out = da_native.interp({lat_dim: target_lat, lon_dim: target_lon},
                           method="linear", kwargs={"fill_value": None}).values.astype(np.float32)
    n_nan = int(np.isnan(out).sum())
    if n_nan:
        print(f"  [oro-2km] [WARNING] {n_nan}/{out.size} ({n_nan/out.size:.1%}) NaN after regrid "
              "-- filling with 0.0.")
    return torch.from_numpy(np.nan_to_num(out, nan=0.0)).unsqueeze(0)


# ------------------------------------------------------------------------------
# NETCDF WRITING
# ------------------------------------------------------------------------------

def _write_year_nc(path, year, fold_name, time_vals, time_attrs, lat, lon, mean_mm, meta):
    data_vars = {
        "precip_mean": (("time", "lat", "lon"), mean_mm),
    }
    coords = {"time": time_vals, "lat": lat, "lon": lon}
    out = xr.Dataset(data_vars, coords=coords)
    out["time"].attrs.update(time_attrs)
    out["precip_mean"].attrs.update(
        units="mm/day",
        long_name="SRGAN downscaled precipitation (single deterministic forward pass)",
        note=("Named 'precip_mean' only for output-schema parity with corrdiff_fm; this "
              "is NOT an ensemble mean. SRGAN's generator is deterministic (no noise "
              "input) -- there is exactly one prediction per day, and no precip_std / "
              "precip_members variable exists in this file because there is nothing "
              "to average or spread over. See srgan/README.md."),
    )
    out.attrs.update(
        description=("SRGAN cascade: a 32km->8km-trained SRResNet-backbone generator, "
                     "adversarially fine-tuned, applied to real 8 km fields to synthesize "
                     "2 km structure. Tiled at the training patch size with raised-cosine "
                     "blending (see Tiling.py)."),
        architecture="srgan",
        deterministic=1,
        year=int(year), source_fold=fold_name, ds_factor=DS_FACTOR,
        tile=int(meta["tile"]),
        caveat=("Normalization statistics and any learned sub-grid amplitude are "
                "inherited from the 8 km training distribution; extremes at 2 km are "
                "expected to be conservative. See Inference.py's module docstring."),
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
    global INFER_BATCH, OUTPUT_DIR

    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None,
                    help="0-based shard index; splits YEARS (not folds) into --num-shards groups.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--fold", type=int, default=None, help="Run exactly this fold's test years.")
    ap.add_argument("--years", type=int, nargs="+", default=None, help="Explicit year list.")
    ap.add_argument("--batch", type=int, default=INFER_BATCH)
    ap.add_argument("--out-dir", type=str, default=OUTPUT_DIR)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--oro-2km", type=str, default=ORO_2KM,
                    help="Real 2 km orography, coordinate-regridded onto the target grid. "
                         "Empty string falls back to a bilinear pixel-resize of the 8 km file.")
    ap.add_argument("--gen-dir", type=str, default=None,
                    help="Generator checkpoint directory (default Config.GENERATOR_CKPT_DIR).")
    ap.add_argument("--tile", type=int, default=None,
                    help="Tile size in OUTPUT (2 km) pixels. Default: the training patch size "
                         "recorded in the checkpoint.")
    ap.add_argument("--overlap", type=int, default=None,
                    help="Tile overlap in output pixels (default tile//4).")
    args = ap.parse_args()

    INFER_BATCH = args.batch
    OUTPUT_DIR = args.out_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.merge_only:
        merge_all_years(OUTPUT_DIR)
        return

    gen_dir = args.gen_dir or GENERATOR_CKPT_DIR
    print(f"Generator dir: {gen_dir}")

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
                print(f"  [WARNING] rotated_pole differs: 8km={p8} vs 2km={p2}. The regridded "
                      "orography may be geographically misaligned.")
            else:
                print(f"  rotated_pole matches: {p2}")
        oro_2km = load_oro_2km_regridded(args.oro_2km, lat2km, lon2km).to(dev).unsqueeze(0)
    else:
        print("  [NOTE] no 2 km orography given -- bilinearly upsampling the 8 km file by pixel "
              "shape. This invents no new terrain detail, so the generator loses the single "
              "most informative predictor it has at 2 km. Supply --oro-2km if you possibly can.")
        oro_2km = load_oro(ORO_8KM, target_hw=(H2, W2)).to(dev).unsqueeze(0)

    # Computed ONCE over the full 2 km domain, then cropped per tile.
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

    net = None
    current_fold = None
    tile = overlap = None

    for yr in my_years:
        if yr not in year_to_fold:
            print(f"  [SKIP] year {yr} is in no fold's test set.")
            continue
        fold = year_to_fold[yr]
        fold_name = fold["name"]

        if fold_name != current_fold:
            if net is not None:
                del net
                torch.cuda.empty_cache()
            gen_path = _find_ckpt(gen_dir, f"Generator_{fold_name}_best.pth")
            print(f"\n[fold {fold_name}] generator: {gen_path}")

            net, ck = load_generator(gen_path, dev)
            assert_precip_transform_compatible(ck, pt, gen_path, 0, "SRGAN generator")
            current_fold = fold_name

            tile = args.tile or ck.get("patch") or min(H, W)
            overlap = args.overlap or max(8, tile // 4)
            print(f"  [srgan] train_patch={ck.get('patch')}  val_rmse={ck.get('val_rmse_mmday')}")
            print(f"  [srgan] tiling 2 km domain {H2}x{W2} into {tile}x{tile} tiles "
                  f"(overlap {overlap})")

        idx = np.where(np.asarray(ds.years) == yr)[0].tolist()
        print(f"=== Year {yr}  (fold {fold_name}, {len(idx)} days) ===")

        mean_out = np.empty((len(idx), H2, W2), dtype=np.float32)

        for start in range(0, len(idx), INFER_BATCH):
            bidx = idx[start:start + INFER_BATCH]
            lr = ds.hr[bidx].to(dev)                     # real, normalized 8 km fields
            B = lr.shape[0]
            topo_b = topo_2km.expand(B, -1, -1, -1)

            with torch.no_grad():
                tg = TiledGenerator(net, topo_b, DS_FACTOR, tile_hr=tile, overlap=overlap,
                                    amp_dtype=amp_dtype)
                pred = tg(lr)
                if tuple(pred.shape[-2:]) != (H2, W2):
                    raise RuntimeError(
                        f"Generator output {tuple(pred.shape[-2:])} != expected {(H2, W2)}. "
                        f"The cascade assumes a fixed x{DS_FACTOR} upsample independent of "
                        "input size -- check Network.Generator.forward.")
                mm = denorm_precip_mmday(pred, pt)

            mean_out[start:start + B] = mm.squeeze(1).cpu().numpy()
            print(f"  [{yr}] {min(start + B, len(idx))}/{len(idx)} days")

        _write_year_nc(
            os.path.join(OUTPUT_DIR, f"precip_2km_{yr}.nc"),
            yr, fold_name, time_full[idx], time_attrs, lat2km, lon2km,
            mean_out, {"tile": tile},
        )

    if net is not None:
        del net
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
