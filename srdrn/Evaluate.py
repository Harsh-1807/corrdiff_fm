# -*- coding: utf-8 -*-
"""
Evaluate.py -- score SRDRN on the held-out test years.
========================================================
Writes a JSON of per-fold, per-day metrics using the SAME schema keys as
corrdiff_fm's Evaluate.py (crps, mae, rmse, spread_skill, rank_chi2,
spectrum_logratio, wet_frac_pred/target, p99_pred/target, n_days,
_crps_per_day), so this package's output is directly comparable via
Compare.py against corrdiff_fm's (and any sibling wgan_gp/srgan-style
package's) evaluation JSON.

THE CENTRAL STRUCTURAL DIFFERENCE FROM corrdiff_fm's Evaluate.py: SRDRN IS
DETERMINISTIC
----------------------------------------------------------------------------
corrdiff_fm draws `--members` independent samples per day from a stochastic
sampler and its metrics (CRPS, spread/skill, rank histogram) are built around
that ensemble. SRDRN has no sampler and no stochasticity: the same input
always produces exactly the same output. There is no meaningful ensemble here,
only a single point prediction per day, so effectively `members=1` always.
This degenerates several of the shared-schema metrics in ways that are
EXPECTED and MATHEMATICALLY CORRECT, not implementation bugs:

  * CRPS with a size-1 ensemble collapses exactly to MAE: the fair-CRPS
    formula subtracts a pairwise-spread correction term that is defined to be
    zero when there is only one member (see `crps_map` below -- this is the
    same formula corrdiff_fm's Evaluate.py uses, evaluated at M=1, not a
    special-cased shortcut). Reported anyway, under the same `"crps"` key, so
    Compare.py's paired bootstrap still works unmodified across a
    deterministic-vs-stochastic comparison.
  * Spread is identically zero (there is nothing to disagree with itself
    about), so spread/skill is identically zero too. This is guarded against
    division by zero explicitly below and is not a sign the model failed to
    disperse -- there is no ensemble to disperse.
  * The rank histogram degenerates to a single bin threshold (does the
    observation fall above or below the one prediction?) rather than a
    genuine calibration diagnostic. Computed anyway for schema compatibility;
    do not read anything into its chi-square value for this package.

A one-line note to this effect is also printed at the start of `main()` so it
is impossible to miss when reading the run's console output.

Usage
-----
  python Evaluate.py --tile 128 --batch 8
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from Config import (HR_FILES, ORO_8KM, CKPT_DIR, KFOLD_K, VAL_RATIO,
                    EVAL_SEED, PRECIP_CH, PATCH)
from Dataset import ClimateDataset, get_climate_kfolds, DS_FACTOR
from Network import expand_topo, denorm_precip_mmday
from Tiling import regress_tiled
from Train import load_frozen_srdrn, assert_precip_transform_compatible, pick_amp, coarsen

PARAM = "srdrn"


# ------------------------------------------------------------------------------
# METRICS -- identical formulas to corrdiff_fm/Evaluate.py, so that a
# deterministic (M=1) evaluation and a stochastic (M>1) one are computed by
# the literal same code path and differ only through M, never through a
# parallel "simplified" implementation that could quietly drift from the
# other package's.
# ------------------------------------------------------------------------------

def crps_map(ens, obs):
    """Fair (unbiased) ensemble CRPS, returned per grid point. ens [M,B,1,H,W],
    obs [B,1,H,W]. At M=1 the pairwise-spread term is defined as zero (there
    is no pair to form), so this reduces exactly to MAE -- see module
    docstring."""
    M = ens.shape[0]
    mae = (ens - obs.unsqueeze(0)).abs().mean(0)
    if M > 1:
        pair = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum(dim=(0, 1))
        spread = pair / (2.0 * M * (M - 1))
    else:
        spread = torch.zeros_like(mae)
    return mae - spread, spread


def rank_histogram(ens, obs, n_bins=None):
    """Where the observation falls within the sorted ensemble. At M=1 this is
    just a 2-bin "above/below the single prediction" split -- not a genuine
    calibration diagnostic, kept only for schema compatibility (see module
    docstring)."""
    M = ens.shape[0]
    n_bins = n_bins or (M + 1)
    ranks = (ens < obs.unsqueeze(0)).sum(0).flatten().cpu().numpy()
    return np.bincount(ranks, minlength=n_bins)[:n_bins].astype(np.float64)


def radial_spectrum(x):
    """Isotropic power spectrum of a [B,1,H,W] field, averaged over the batch."""
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
    """Mean |log10(P_pred / P_target)| over the top two octaves -- the scales
    a downscaler is supposed to be reconstructing. 0 is perfect. For a purely
    MSE/MAE-trained deterministic network, expect this to be noticeably worse
    than a generative arm's: minimizing pixelwise error pushes predictions
    toward the conditional mean, which is systematically SMOOTHER than any
    individual realization, and that smoothing shows up directly as missing
    high-frequency power here. This is the single most informative number to
    compare against corrdiff_fm/wgan_gp/srgan via Compare.py."""
    n = len(targ_spec)
    lo = max(1, n // 4)
    a = pred_spec[lo:n].clamp_min(1e-12)
    b = targ_spec[lo:n].clamp_min(1e-12)
    return float((torch.log10(a / b)).abs().mean())


# ------------------------------------------------------------------------------
# EVALUATION OF ONE FOLD
# ------------------------------------------------------------------------------

def _find_ckpt(ckpt_dir, exact_name):
    """Same fallback policy as corrdiff_fm's Evaluate.py/Inference.py: prefer
    the fold-matched checkpoint, but warn loudly rather than crash if it is
    absent, and say why it matters -- a mismatched fold leaks held-out years
    into the test score."""
    cand = os.path.join(ckpt_dir, exact_name)
    if os.path.exists(cand):
        return cand
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith((".pt", ".pth"))]) \
        if os.path.isdir(ckpt_dir) else []
    if not files:
        raise FileNotFoundError(f"No checkpoints in {os.path.abspath(ckpt_dir)}")
    chosen = sorted([f for f in files if "best" in f.lower()] or files)[0]
    print(f"  [WARN] '{exact_name}' not found; falling back to {chosen}. The absolute "
          "numbers for this fold are contaminated -- do not quote them as test-set scores.")
    return os.path.join(ckpt_dir, chosen)


@torch.no_grad()
def evaluate(net, ds, idx, topo_full, pt, dev, amp_dtype, batch, tile):
    per_day = {"crps": [], "mae": [], "mse": [], "spread": []}
    ranks = np.zeros(2)   # M=1 => M+1=2 rank bins
    spec_p = spec_t = None
    n_spec = 0
    wet_p = wet_t = 0.0
    all_p, all_t = [], []

    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        hr = ds.hr[b].to(dev)
        B = hr.shape[0]
        topo = topo_full.expand(B, -1, -1, -1)

        lr = coarsen(hr, DS_FACTOR)
        target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()
        mu = regress_tiled(net, lr, topo, DS_FACTOR, tile_hr=tile, amp_dtype=amp_dtype)

        pred_mm = denorm_precip_mmday(mu, pt)
        obs = denorm_precip_mmday(target, pt)
        ens = pred_mm.unsqueeze(0)   # [1,B,1,H,W] -- the trivial "ensemble"

        c, s = crps_map(ens, obs)
        for k in range(B):
            per_day["crps"].append(float(c[k].mean()))
            per_day["mae"].append(float((pred_mm[k] - obs[k]).abs().mean()))
            per_day["mse"].append(float(((pred_mm[k] - obs[k]) ** 2).mean()))
            per_day["spread"].append(float(s[k].mean()))

        ranks += rank_histogram(ens, obs, 2)

        sp, st = radial_spectrum(pred_mm), radial_spectrum(obs)
        spec_p = sp if spec_p is None else spec_p + sp
        spec_t = st if spec_t is None else spec_t + st
        n_spec += 1

        wet_p += float((pred_mm > 1.0).float().mean()) * B
        wet_t += float((obs > 1.0).float().mean()) * B
        all_p.append(pred_mm.flatten()[::37].cpu()); all_t.append(obs.flatten()[::37].cpu())

    n = len(per_day["crps"])
    rmse = float(np.sqrt(np.mean(per_day["mse"])))
    spread = float(np.mean(per_day["spread"]))          # == 0.0 by construction
    # spread/skill = spread / RMSE * sqrt(1 + 1/M), M=1 -> sqrt(2) multiplier
    # on a zero numerator. RMSE itself could in principle be exactly zero on a
    # degenerate (all-dry, perfectly-predicted) slice; guard both directions.
    ssr = spread / max(rmse, 1e-9) * np.sqrt(2.0)
    rh = ranks / max(ranks.sum(), 1)
    chi2 = float(((rh - 1.0 / len(rh)) ** 2).sum() * len(rh) * ranks.sum())

    p_all = torch.cat(all_p).numpy(); t_all = torch.cat(all_t).numpy()
    return {
        "n_days": n,
        "crps": float(np.mean(per_day["crps"])),   # == mae, by construction (M=1)
        "mae": float(np.mean(per_day["mae"])),
        "rmse": rmse,
        "spread_skill": float(ssr),                 # == 0.0, by construction (M=1)
        "rank_chi2": chi2,
        "spectrum_logratio": spectrum_score(spec_p / n_spec, spec_t / n_spec),
        "wet_frac_pred": wet_p / max(n, 1),
        "wet_frac_target": wet_t / max(n, 1),
        "p99_pred": float(np.percentile(p_all, 99)),
        "p99_target": float(np.percentile(t_all, 99)),
        "_crps_per_day": per_day["crps"],
    }


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--tile", type=int, default=None,
                    help="Default: the training patch recorded in the checkpoint.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"outputs/eval_{PARAM}.json"
    dev = torch.device(args.device)
    amp_dtype, _ = pick_amp(dev)

    print("[NOTE] SRDRN is deterministic (members=1 always): CRPS degenerates exactly "
          "to MAE and spread/spread-skill are identically 0 by construction -- this is "
          "expected, not a calibration failure. See this file's module docstring.")

    ds = ClimateDataset(HR_FILES, ORO_8KM, rank=0)
    pt = ds.get_precip_transform_meta()
    oro = torch.as_tensor(ds.oro, dtype=torch.float32, device=dev)
    while oro.dim() < 4:
        oro = oro.unsqueeze(0)
    topo_full = expand_topo(oro)

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)
    sel = args.folds if args.folds is not None else list(range(len(folds)))

    print(f"\nArm: {PARAM}")
    print(f"Ensemble size:  1 (deterministic)")
    print(f"Eval seed:      {EVAL_SEED} (unused by a deterministic model, kept in the "
          "JSON for schema parity with the stochastic packages' Evaluate.py)\n")

    results = {}
    for fi in sel:
        fold = folds[fi]
        name, idx = fold["name"], fold["test_idx"]
        print(f"{'='*78}\nFold {name}  --  {len(idx)} held-out test days\n{'='*78}")

        ck_path = os.path.join(args.ckpt_dir, f"SRDRN_{name}_best.pth")
        if not os.path.exists(ck_path):
            ck_path = _find_ckpt(args.ckpt_dir, f"SRDRN_{name}_best.pth")

        net, ck = load_frozen_srdrn(ck_path, dev, pt, rank=0)
        tile = args.tile or ck.get("patch") or PATCH or min(ds.H, ds.W)
        print(f"  [{PARAM}] {ck_path}  tile={tile}  val_rmse={ck.get('val_rmse_mmday'):.4f}")

        m = evaluate(net, ds, idx, topo_full, pt, dev, amp_dtype, args.batch, tile)
        print(f"  CRPS(=MAE) {m['crps']:8.4f} | MAE {m['mae']:8.4f} | "
              f"RMSE {m['rmse']:8.4f} | S/S {m['spread_skill']:5.3f} | "
              f"spec {m['spectrum_logratio']:5.3f} | "
              f"wet {m['wet_frac_pred']:.3f}/{m['wet_frac_target']:.3f} | "
              f"P99 {m['p99_pred']:7.2f}/{m['p99_target']:7.2f}")

        results[name] = {"default": m}
        del net
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"param": PARAM, "nfe": 1, "steps": 1, "members": 1,
                   "eval_seed": EVAL_SEED, "folds": results}, f, indent=2)
    print(f"\nWrote {out_path}")
    print("Compare against a stochastic arm with:  python Compare.py "
          f"outputs/eval_edm.json {out_path}")


if __name__ == "__main__":
    main()
