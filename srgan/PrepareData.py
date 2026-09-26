# -*- coding: utf-8 -*-
"""
PrepareData.py -- build and validate the 8 km training inputs. RUN THIS FIRST.
==============================================================================
SHARED INFRASTRUCTURE -- RUN ONCE FOR ALL OF /home/ylale/extras/h/, NOT ONCE
PER PACKAGE. This file is byte-identical across corrdiff_fm (both its `edm`
and `fm` arms), this `srgan` package, and the sibling `wgan_gp`/`srdrn`
packages. They all train on the exact same corrected precipitation file and
the exact same static orography, so there is exactly one unit correction to
apply and one set of pre-flight checks to pass -- running it again in each
package would at best waste time re-deriving the identical corrected file,
and at worst (if ever run with different flags in different packages) produce
per-package precip files that are silently different, which would confound
any comparison between architectures before a single model is trained. Run it
from any ONE package's directory; every package's Config.HR_FILES points at
the same corrected file on disk.

Two jobs:

  1. UNIT CORRECTION. The original precip file
     (precip_rcm_8km_daily_data_1995-2014.nc) is stored in mm/hr despite the
     "daily" name. That was previously compensated inside Dataset.py by a
     PRECIP_SCALE of 24.0 -- a fudge factor doing physics work, which is exactly
     the kind of thing that silently survives a refactor and poisons every
     downstream number. This script fixes it at the source, writing
     *_mmday_corrected.nc, so PRECIP_SCALE can stay at 1.0 and mean what it says.

     If your file is ALREADY in mm/day, pass --already-mmday and it will copy
     with corrected metadata rather than multiplying again. Read the diagnostic
     below before deciding -- the script tells you which case you are in.

  2. PRE-FLIGHT CHECKS. Grid sizes, time-axis alignment across variables, NaN
     and fill-value counts, orography coverage, and the crop-to-16 behaviour.
     Every one of these has a failure mode that produces a *trained model* rather
     than an error message, which is the worst kind.

Usage
-----
  python PrepareData.py --check-only          # inspect, write nothing
  python PrepareData.py                       # correct units + full validation
  python PrepareData.py --already-mmday       # skip the x24, still validate
"""

import os
import argparse
import shutil

import numpy as np
import xarray as xr

from Config import ROOT, HR_FILES, ORO_8KM, ORO_2KM, RAW_PRECIP_FILE, VAR_MAP

SEC_PER_DAY = 24.0


def _open(p):
    return xr.open_dataset(p, decode_times=False)


def _varname(ds, prefix):
    want = VAR_MAP.get(prefix)
    if want and want in ds.data_vars:
        return want
    cands = [v for v in ds.data_vars if ds[v].ndim >= 2]
    if not cands:
        raise ValueError(f"no data variable with >=2 dims found")
    return cands[0]


def diagnose_precip(path):
    """Decide, from the data itself, whether this file is mm/hr or mm/day.

    Southeast Asian domain-mean daily precipitation is roughly 5-12 mm/day.
    Divided by 24 that is 0.2-0.5, which is unmistakably different. We report
    the number rather than guessing silently, because getting this wrong by a
    factor of 24 changes nothing about whether training converges -- it just
    makes every millimetre you report wrong.
    """
    ds = _open(path)
    v = _varname(ds, "precip")
    arr = ds[v].values.astype(np.float32)
    ds.close()
    arr = np.where(arr >= 1e19, np.nan, arr)
    m = float(np.nanmean(arr))
    print(f"  variable '{v}', shape {arr.shape}")
    print(f"  domain mean {m:.4f}   max {float(np.nanmax(arr)):.2f}   "
          f"p99 {float(np.nanpercentile(arr, 99)):.2f}")
    if m < 1.5:
        print(f"  -> looks like mm/HR (a plausible daily mean would be ~{m*24:.1f} mm/day)")
        return "mmhr"
    if m < 30:
        print(f"  -> looks like mm/DAY already; do NOT multiply by 24")
        return "mmday"
    print(f"  -> mean {m:.1f} is high for either unit. Inspect manually before proceeding.")
    return "unknown"


def correct_precip(src, dst, already_mmday=False):
    print(f"\n[1] PRECIP UNIT CORRECTION\n    source: {src}")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    kind = diagnose_precip(src)

    if already_mmday or kind == "mmday":
        print(f"    copying without rescaling -> {dst}")
        shutil.copyfile(src, dst)
        ds = _open(dst)
        v = _varname(ds, "precip")
        ds[v].attrs["units"] = "mm day-1"
        ds.attrs["unit_correction"] = "none applied; source already mm/day"
        ds.load().to_netcdf(dst + ".tmp")
        ds.close()
        os.replace(dst + ".tmp", dst)
        return

    print(f"    multiplying by {SEC_PER_DAY:.0f} (mm/hr -> mm/day) -> {dst}")
    ds = _open(src)
    v = _varname(ds, "precip")
    ds[v] = ds[v] * SEC_PER_DAY
    ds[v].attrs["units"] = "mm day-1"
    ds[v].attrs["long_name"] = "daily precipitation"
    ds.attrs["unit_correction"] = f"multiplied by {SEC_PER_DAY:.0f}: source was mm/hr"
    enc = {v: {"zlib": True, "complevel": 4}}
    ds.load().to_netcdf(dst, encoding=enc)
    ds.close()
    print("    done. Verifying:")
    diagnose_precip(dst)


def validate():
    print("\n[2] PRE-FLIGHT VALIDATION")
    shapes, times, ok = {}, {}, True

    for p in HR_FILES:
        prefix = os.path.basename(p).split("_")[0]
        if not os.path.exists(p):
            print(f"  [FAIL] missing: {p}")
            ok = False
            continue
        ds = _open(p)
        v = _varname(ds, prefix)
        a = ds[v]
        shapes[prefix] = tuple(a.shape)
        tname = next((c for c in ("time", "Time", "date", "t")
                      if c in ds.coords or c in ds.dims), None)
        times[prefix] = int(ds.sizes[tname]) if tname else None
        raw = a.values.astype(np.float32)
        n_nan = int(np.isnan(raw).sum())
        n_fill = int((raw >= 1e19).sum())
        ds.close()
        flag = "" if (n_nan == 0 and n_fill == 0) else "  <- nan_to_num will zero these"
        print(f"  {prefix:7s} {v:6s} shape {shapes[prefix]}  "
              f"nan {n_nan}  fill {n_fill}{flag}")

    # Time alignment. Dataset.py truncates to the shortest, silently -- so a
    # variable that is short by a few days shifts NOTHING but quietly discards
    # the tail of every other variable.
    tv = [t for t in times.values() if t]
    if tv and len(set(tv)) > 1:
        print(f"  [WARN] time lengths differ across variables: {times}")
        print(f"         Dataset.py truncates all to {min(tv)} -- confirm that is intended.")
        ok = False
    elif tv:
        print(f"  time steps: {tv[0]} (consistent across variables)")

    sp = {s[1:] for s in shapes.values()}
    if len(sp) > 1:
        print(f"  [FAIL] spatial grids differ across variables: {shapes}")
        ok = False
    elif sp:
        H, W = sp.pop()
        H16, W16 = (H // 16) * 16, (W // 16) * 16
        print(f"  spatial grid {H}x{W} -> cropped to {H16}x{W16} (CROP_TO_16)")
        if (H16, W16) != (H, W):
            print(f"         discarding {H-H16} rows and {W-W16} cols from the far edge")
        print(f"  LR grid at DS_FACTOR=4: {H16//4}x{W16//4}")

    for label, path, expect in (("8 km", ORO_8KM, None), ("2 km", ORO_2KM, None)):
        if not os.path.exists(path):
            print(f"  [{'FAIL' if label=='8 km' else 'WARN'}] orography missing: {path}")
            if label == "8 km":
                ok = False
            continue
        ds = _open(path)
        v = _varname(ds, "oro")
        a = np.squeeze(ds[v].values).astype(np.float32)
        ds.close()
        a = np.where(a >= 1e19, np.nan, a)
        land = float(np.mean(np.nan_to_num(a, nan=0.0) > 0.5))
        print(f"  oro {label}: '{v}' shape {a.shape}  range "
              f"[{np.nanmin(a):.1f}, {np.nanmax(a):.1f}] m  land fraction {land:.1%}")
        if land < 0.01:
            print(f"         [WARN] almost no land -- topography will carry little signal")

    # Years / folds
    try:
        ds = xr.open_dataset(HR_FILES[0])
        tname = next((c for c in ("time", "Time", "date") if c in ds.coords or c in ds.dims), None)
        yrs = np.unique([np.datetime64(t, "Y").astype(int) + 1970 for t in ds[tname].values])
        ds.close()
        print(f"  years: {yrs.min()}-{yrs.max()} ({len(yrs)} unique)")
        blocks = np.array_split(yrs, 5)
        print(f"  5-fold test blocks: {[f'{b.min()}-{b.max()}' for b in blocks]}")
    except Exception as e:
        print(f"  [WARN] could not parse the time axis into years: {e}")

    print("\n" + ("  ALL CHECKS PASSED -- ready to train." if ok else
                  "  PROBLEMS FOUND ABOVE -- fix them before training."))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="Validate only; write no files.")
    ap.add_argument("--already-mmday", action="store_true",
                    help="Source precip is already mm/day; copy without rescaling.")
    ap.add_argument("--raw-precip", default=RAW_PRECIP_FILE)
    args = ap.parse_args()

    print("=" * 70)
    print("Shared precipitation-downscaling data preparation")
    print("(used by corrdiff_fm, srgan, and every other sibling package)")
    print("=" * 70)
    print(f"ROOT: {ROOT}")

    if not args.check_only:
        target = next(p for p in HR_FILES if os.path.basename(p).startswith("precip"))
        if os.path.exists(target):
            print(f"\n[1] corrected precip already exists: {target}")
            print("    delete it if you want to regenerate. Diagnosing it:")
            diagnose_precip(target)
        else:
            correct_precip(args.raw_precip, target, already_mmday=args.already_mmday)

    validate()


if __name__ == "__main__":
    main()
