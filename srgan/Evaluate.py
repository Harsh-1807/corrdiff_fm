# -*- coding: utf-8 -*-
"""
Evaluate.py -- score the SRGAN generator on the held-out test years.
========================================================================
Writes a JSON with the SAME schema keys as corrdiff_fm's Evaluate.py (and the
`wgan_gp`/`srdrn` siblings' own Evaluate.py): `crps`, `mae`, `rmse`,
`spread_skill`, `rank_chi2`, `spectrum_logratio`, `wet_frac_pred`,
`wet_frac_target`, `p99_pred`, `p99_target`, `n_days`, `_crps_per_day`. That
schema agreement -- not shared code -- is what lets Compare.py pair this
package's JSON against any other architecture family's, without Compare.py
ever importing a single line of model code from either side.

WHY THE METRICS DEGENERATE HERE, AND WHY THAT IS NOT HIDDEN
----------------------------------------------------------------
This package's generator is deterministic (see Network.py's "WHY THE
GENERATOR HAS NO NOISE INPUT"): a given (lr, topo) input always produces the
same output field. There is therefore exactly ONE "ensemble member" per day,
always (`members=1`, not a CLI option here -- there is nothing to vary it
against). Two direct, mechanical consequences of M=1 in the CRPS/rank-
histogram formulas below (identical formulas to corrdiff_fm's Evaluate.py):

  * `crps_map`'s pairwise-spread correction term is exactly zero when M=1
    (the sum over M*(M-1) pairs of a single-element set is empty), so CRPS
    collapses EXACTLY to MAE. This is arithmetic, not an approximation --
    printed and flagged below rather than silently reported as if it were a
    meaningfully different, "proper scoring rule" number from MAE.
  * `spread` (ensemble std) is identically 0, so `spread_skill` = 0 /
    max(rmse, eps) = 0 always. A spread/skill ratio of 0 here does NOT mean
    "maximally under-dispersed relative to a calibrated probabilistic model"
    in the sense that comparison is usually read -- it means the question
    the metric is designed to answer (is the ensemble spread calibrated to
    the ensemble-mean error) does not apply to a model with no ensemble at
    all. Comparing this package's spread_skill=0 against corrdiff_fm's
    non-zero value is comparing "not applicable" to "applicable" -- read the
    printed note, not just the number.
  * `rank_chi2` still computes (it is well-defined for M=1: the "rank
    histogram" has 2 bins, "prediction under-shot" vs. "prediction over-
    shot"), and IS informative here -- it is really just a signed-bias
    check across the domain, not a calibration diagnostic.

Usage
-----
  python Evaluate.py --batch 4
  python Evaluate.py --folds 0 2 4 --out outputs/eval_srgan.json
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from Config import (HR_FILES, ORO_8KM, GENERATOR_CKPT_DIR, KFOLD_K, VAL_RATIO,
                    EVAL_SEED, PRECIP_CH, PATCH, DS_FACTOR)
from Dataset import ClimateDataset, get_climate_kfolds
from Network import expand_topo, denorm_precip_mmday
from Tiling import TiledGenerator
from Train import build_generator, assert_precip_transform_compatible, pick_amp


# ------------------------------------------------------------------------------
# METRICS -- identical formulas to corrdiff_fm's Evaluate.py (generic proper-
# scoring-rule / distributional-diagnostic code, not diffusion-specific).
# ------------------------------------------------------------------------------

def crps_map(ens, obs):
    """Fair (unbiased) ensemble CRPS, returned per grid point. ens [M,B,1,H,W],
    obs [B,1,H,W]. With M=1 (always true in this package) this is exactly MAE
    -- see module docstring."""
    M = ens.shape[0]
    mae = (ens - obs.unsqueeze(0)).abs().mean(0)
    if M > 1:
        pair = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum(dim=(0, 1))
        spread = pair / (2.0 * M * (M - 1))
    else:
        spread = torch.zeros_like(mae)
    return mae - spread, spread


def rank_histogram(ens, obs, n_bins=None):
    """Where the observation falls within the sorted ensemble. With M=1 this
    is just a 2-bin signed-bias indicator (under-shot / over-shot), not a
    calibration diagnostic in the usual ensemble-forecasting sense."""
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
    a GAN's texture-realism training is supposed to be recovering. 0 is perfect."""
    n = len(targ_spec)
    lo = max(1, n // 4)
    a = pred_spec[lo:n].clamp_min(1e-12)
    b = targ_spec[lo:n].clamp_min(1e-12)
    return float((torch.log10(a / b)).abs().mean())


# ------------------------------------------------------------------------------
# CHECKPOINT LOADING
# ------------------------------------------------------------------------------

def _find_ckpt(ckpt_dir, exact_name):
    """Same fallback policy as corrdiff_fm's Evaluate.py/Inference.py: prefer
    the fold-matched checkpoint, but warn loudly rather than crash if it is
    absent."""
    cand = os.path.join(ckpt_dir, exact_name)
    if os.path.exists(cand):
        return cand
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith((".pt", ".pth"))]) \
        if os.path.isdir(ckpt_dir) else []
    if not files:
        raise FileNotFoundError(f"No checkpoints in {os.path.abspath(ckpt_dir)}")
    chosen = sorted([f for f in files if "best" in f.lower()] or files)[0]
    print(f"  [WARN] '{exact_name}' not found; falling back to {chosen}. If this is not "
          "the fold that held these years out, the result is contaminated.")
    return os.path.join(ckpt_dir, chosen)


def load_generator(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    arch = ck.get("arch")
    if arch is None:
        raise RuntimeError(f"{path} has no 'arch' key -- retrain with the current Train.py.")
    net = build_generator(arch, dev)
    net.load_state_dict(ck.get("ema_state_dict") or ck["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if ck.get("phase") != "gan":
        print(f"  [WARN] {path} has phase='{ck.get('phase')}', not 'gan' -- this is a "
              "Phase-1-only (pretrain) checkpoint, not the GAN-fine-tuned deliverable. "
              "Evaluating it anyway, but this is an ablation checkpoint, not the model "
              "this package's README describes as the SRGAN result.")
    return net, ck


# ------------------------------------------------------------------------------
# EVALUATION OF ONE FOLD
# ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(net, ds, idx, topo_full, pt, dev, amp_dtype, batch, tile):
    per_day = {"crps": [], "mae": [], "mse": [], "spread": []}
    ranks = np.zeros(2)   # M=1 -> 2-bin rank histogram (under/over)
    spec_p = spec_t = None
    n_spec = 0
    wet_p = wet_t = 0.0
    all_p, all_t = [], []

    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        hr = ds.hr[b].to(dev)
        B = hr.shape[0]
        topo = topo_full.expand(B, -1, -1, -1)

        lr = F.avg_pool2d(hr, kernel_size=DS_FACTOR, stride=DS_FACTOR)
        target = hr[:, PRECIP_CH:PRECIP_CH + 1].float()

        tg = TiledGenerator(net, topo, DS_FACTOR, tile_hr=tile, amp_dtype=amp_dtype)
        pred = tg(lr)

        # ens has a size-1 "member" axis purely so crps_map/rank_histogram's
        # shared-with-corrdiff_fm formulas apply unmodified; there is exactly
        # one realization here, always.
        ens = denorm_precip_mmday(pred, pt).unsqueeze(0)
        obs = denorm_precip_mmday(target, pt)

        c, s = crps_map(ens, obs)
        em = ens.mean(0)
        for k in range(B):
            per_day["crps"].append(float(c[k].mean()))
            per_day["mae"].append(float((em[k] - obs[k]).abs().mean()))
            per_day["mse"].append(float(((em[k] - obs[k]) ** 2).mean()))
            per_day["spread"].append(float(s[k].mean()))

        ranks += rank_histogram(ens, obs, 2)

        sp, st = radial_spectrum(em), radial_spectrum(obs)
        spec_p = sp if spec_p is None else spec_p + sp
        spec_t = st if spec_t is None else spec_t + st
        n_spec += 1

        wet_p += float((em > 1.0).float().mean()) * B
        wet_t += float((obs > 1.0).float().mean()) * B
        all_p.append(em.flatten()[::37].cpu()); all_t.append(obs.flatten()[::37].cpu())

    n = len(per_day["crps"])
    rmse = float(np.sqrt(np.mean(per_day["mse"])))
    spread = float(np.mean(per_day["spread"]))          # identically 0, M=1
    ssr = spread / max(rmse, 1e-9)                        # identically 0, M=1
    rh = ranks / max(ranks.sum(), 1)
    chi2 = float(((rh - 1.0 / len(rh)) ** 2).sum() * len(rh) * ranks.sum())

    p_all = torch.cat(all_p).numpy(); t_all = torch.cat(all_t).numpy()
    return {
        "n_days": n,
        "crps": float(np.mean(per_day["crps"])),   # == mae, by construction (see docstring)
        "mae": float(np.mean(per_day["mae"])),
        "rmse": rmse,
        "spread_skill": float(ssr),                  # identically 0 -- see docstring
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
    ap.add_argument("--ckpt-dir", default=GENERATOR_CKPT_DIR)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--tile", type=int, default=None,
                    help="Default: the training patch recorded in the checkpoint.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or "outputs/eval_srgan.json"
    dev = torch.device(args.device)
    amp_dtype, _ = pick_amp(dev)

    ds = ClimateDataset(HR_FILES, ORO_8KM, rank=0)
    pt = ds.get_precip_transform_meta()
    oro = torch.as_tensor(ds.oro, dtype=torch.float32, device=dev)
    while oro.dim() < 4:
        oro = oro.unsqueeze(0)
    topo_full = expand_topo(oro)

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)
    sel = args.folds if args.folds is not None else list(range(len(folds)))

    print("\nArm: srgan")
    print("Ensemble size:  1 (deterministic generator -- CRPS degenerates to MAE, "
          "spread/spread_skill are always 0; see module docstring)")
    print(f"Eval seed:      {EVAL_SEED} (recorded for parity with corrdiff_fm's JSON schema; "
          "this package's forward pass has no randomness for it to seed)\n")

    results = {}
    for fi in sel:
        fold = folds[fi]
        name, idx = fold["name"], fold["test_idx"]
        print(f"{'='*78}\nFold {name}  --  {len(idx)} held-out test days\n{'='*78}")

        ck_path = os.path.join(args.ckpt_dir, f"Generator_{name}_best.pth")
        if not os.path.exists(ck_path):
            ck_path = _find_ckpt(args.ckpt_dir, f"Generator_{name}_best.pth")

        net, ck = load_generator(ck_path, dev)
        assert_precip_transform_compatible(ck, pt, ck_path, 0, "SRGAN generator")
        tile = args.tile or ck.get("patch") or PATCH or min(ds.H, ds.W)
        print(f"  [srgan] {ck_path}  tile={tile}  val_rmse={ck.get('val_rmse_mmday'):.4f}  "
              f"val_spectrum_logratio={ck.get('val_spectrum_logratio')}")

        m = evaluate(net, ds, idx, topo_full, pt, dev, amp_dtype, args.batch, tile)
        print(f"  MAE/CRPS {m['mae']:8.4f} | RMSE {m['rmse']:8.4f} | "
              f"spec {m['spectrum_logratio']:5.3f} | "
              f"wet {m['wet_frac_pred']:.3f}/{m['wet_frac_target']:.3f} | "
              f"P99 {m['p99_pred']:7.2f}/{m['p99_target']:7.2f}")

        results[name] = {"default": m}
        del net
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"param": "srgan", "nfe": 1, "steps": 1, "members": 1,
                   "eval_seed": EVAL_SEED, "folds": results}, f, indent=2)
    print(f"\nWrote {out_path}")
    print("Note: 'nfe'/'steps'/'members' are fixed at 1 -- this is a single deterministic "
          "forward pass per day, not an iterative sampler. Compare against corrdiff_fm with:")
    print(f"  python Compare.py outputs/eval_edm.json {out_path}")


if __name__ == "__main__":
    main()
