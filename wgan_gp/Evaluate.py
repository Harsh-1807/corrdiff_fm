# -*- coding: utf-8 -*-
"""
Evaluate.py -- score the WGAN-GP Generator on the held-out test years.
====================================================================
Writes a JSON with EXACTLY the schema keys corrdiff_fm's Evaluate.py writes
(crps, mae, rmse, spread_skill, rank_chi2, spectrum_logratio, wet_frac_pred/
target, p99_pred/target, n_days, _crps_per_day), so this package's output is
directly usable by Compare.py against corrdiff_fm's eval_edm.json/eval_fm.json
-- and, once they exist, the sibling srgan/srdrn packages' own eval JSONs.
The metric FUNCTIONS below (crps_map, rank_histogram, radial_spectrum,
spectrum_score, paired_bootstrap) are copied unchanged from corrdiff_fm's
Evaluate.py: none of them reference a diffusion sampler, only ens/obs tensors,
so there is nothing arm-specific to adapt.

WHAT DIFFERS FROM corrdiff_fm's Evaluate.py
--------------------------------------------
There is no frozen Stage-1 regressor to load and no NFE-per-member sampler
loop -- the "ensemble" here is built the same way Train.py's
`ensemble_validate` builds it: hold (lr, topo) fixed for a given day and
resample the Generator's noise input z, drawing `--members` independent
forward passes. Each ensemble member costs exactly ONE Generator forward pass
(tiled), so `nfe` is recorded as 1 per member in the output JSON -- this is
expected to differ from the diffusion arms' `nfe` (their sampler needs
`2*steps-1` network evaluations per member), and Compare.py's "NFE MISMATCH"
warning will correctly fire whenever this package's JSON is compared against
a diffusion arm's. That warning is doing its job, not reporting a bug: the
two model families spend their compute budgets in fundamentally different
places (one big feed-forward network vs. many evaluations of an iterative
denoiser), and comparing raw NFE across families is not a meaningful
apples-to-apples measure to begin with -- CRPS at a fixed wall-clock or
fixed-parameter budget would be the fairer comparison, and is left to the
reader to construct from these JSONs' metrics.

Usage
-----
  python Evaluate.py --members 16
  python Evaluate.py --folds 0 2 4 --members 32
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
from Train import build_generator, assert_precip_transform_compatible, coarsen


# ------------------------------------------------------------------------------
# METRICS -- copied unchanged from corrdiff_fm/Evaluate.py (pure tensor math,
# nothing arm-specific)
# ------------------------------------------------------------------------------

def crps_map(ens, obs):
    """Fair (unbiased) ensemble CRPS, returned per grid point so it can be
    aggregated or bootstrapped by day. ens [M,B,1,H,W], obs [B,1,H,W]."""
    M = ens.shape[0]
    mae = (ens - obs.unsqueeze(0)).abs().mean(0)
    if M > 1:
        pair = (ens.unsqueeze(0) - ens.unsqueeze(1)).abs().sum(dim=(0, 1))
        spread = pair / (2.0 * M * (M - 1))
    else:
        spread = torch.zeros_like(mae)
    return mae - spread, spread


def rank_histogram(ens, obs, n_bins=None):
    """Where the observation falls within the sorted ensemble. A calibrated
    ensemble gives a flat histogram; a U shape means under-dispersion (the
    truth keeps landing outside the ensemble), a dome means over-dispersion."""
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
    the Generator's sub-grid detail is supposed to be synthesizing. 0 is perfect."""
    n = len(targ_spec)
    lo = max(1, n // 4)
    a = pred_spec[lo:n].clamp_min(1e-12)
    b = targ_spec[lo:n].clamp_min(1e-12)
    return float((torch.log10(a / b)).abs().mean())


def paired_bootstrap(diff_per_day, n_boot=5000, seed=0):
    """95% CI on the mean of a paired difference."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff_per_day, dtype=np.float64)
    if len(d) < 2:
        return float(d.mean()) if len(d) else 0.0, 0.0, 0.0
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------------------
# CHECKPOINT LOADING
# ------------------------------------------------------------------------------

def _find_ckpt(ckpt_dir, exact_name):
    """Same fallback policy as corrdiff_fm's Evaluate.py/Inference.py: prefer
    the fold-matched checkpoint, but warn loudly rather than crash if absent."""
    cand = os.path.join(ckpt_dir, exact_name)
    if os.path.exists(cand):
        return cand
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith((".pt", ".pth"))]) \
        if os.path.isdir(ckpt_dir) else []
    if not files:
        raise FileNotFoundError(f"No checkpoints in {os.path.abspath(ckpt_dir)}")
    chosen = sorted([f for f in files if "best" in f.lower()] or files)[0]
    print(f"  [WARN] '{exact_name}' not found; falling back to {chosen}. The comparison "
          "stays internally consistent, but this checkpoint may not be the one held out "
          "for these test years -- do not quote the absolute numbers.")
    return os.path.join(ckpt_dir, chosen)


def load_generator(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    arch = ck.get("arch")
    if arch is None:
        raise RuntimeError(f"{path} has no 'arch' key -- retrain with the current Train.py.")
    if ck.get("algo", "wgan_gp") != "wgan_gp":
        raise RuntimeError(f"{path} was trained with algo='{ck.get('algo')}', "
                           "this package implements 'wgan_gp'.")
    net = build_generator(arch, dev, dropout=0.0)
    net.load_state_dict(ck.get("ema_state_dict") or ck["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, ck


# ------------------------------------------------------------------------------
# EVALUATION OF ONE FOLD
# ------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(net, ds, idx, topo_full, pt, dev, members, batch, tile):
    per_day = {"crps": [], "mae": [], "mse": [], "spread": []}
    ranks = np.zeros(members + 1)
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
        obs = denorm_precip_mmday(target, pt)

        # Same seed policy as corrdiff_fm's Evaluate.py: seeded by (EVAL_SEED,
        # start) so day i sees the same noise draws across repeated runs, and
        # so this arm's JSON can be paired day-for-day against another arm's
        # JSON in Compare.py.
        gen_rng = torch.Generator(device=dev).manual_seed(EVAL_SEED + start)
        tiled = TiledGenerator(net, topo, DS_FACTOR, tile_hr=tile, amp_dtype=torch.float32)

        ens = []
        for _ in range(members):
            fake = tiled(lr, generator=gen_rng)
            ens.append(denorm_precip_mmday(fake, pt))
        ens = torch.stack(ens, 0)

        c, s = crps_map(ens, obs)
        em = ens.mean(0)
        for k in range(B):
            per_day["crps"].append(float(c[k].mean()))
            per_day["mae"].append(float((em[k] - obs[k]).abs().mean()))
            per_day["mse"].append(float(((em[k] - obs[k]) ** 2).mean()))
            per_day["spread"].append(float(s[k].mean()))

        ranks += rank_histogram(ens, obs, members + 1)

        sp, st = radial_spectrum(ens[0]), radial_spectrum(obs)
        spec_p = sp if spec_p is None else spec_p + sp
        spec_t = st if spec_t is None else spec_t + st
        n_spec += 1

        wet_p += float((em > 1.0).float().mean()) * B
        wet_t += float((obs > 1.0).float().mean()) * B
        all_p.append(em.flatten()[::37].cpu()); all_t.append(obs.flatten()[::37].cpu())

    n = len(per_day["crps"])
    rmse = float(np.sqrt(np.mean(per_day["mse"])))
    spread = float(np.mean(per_day["spread"]))
    ssr = spread / max(rmse, 1e-9) * np.sqrt(1.0 + 1.0 / members)
    rh = ranks / max(ranks.sum(), 1)
    chi2 = float(((rh - 1.0 / len(rh)) ** 2).sum() * len(rh) * ranks.sum())

    p_all = torch.cat(all_p).numpy(); t_all = torch.cat(all_t).numpy()
    return {
        "n_days": n,
        "crps": float(np.mean(per_day["crps"])),
        "mae": float(np.mean(per_day["mae"])),
        "rmse": rmse,
        "spread_skill": float(ssr),
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
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--tile", type=int, default=None,
                    help="Default: the training patch recorded in the checkpoint.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or "outputs/eval_wgan_gp.json"
    dev = torch.device(args.device)

    ds = ClimateDataset(HR_FILES, ORO_8KM, rank=0)
    pt = ds.get_precip_transform_meta()
    oro = torch.as_tensor(ds.oro, dtype=torch.float32, device=dev)
    while oro.dim() < 4:
        oro = oro.unsqueeze(0)
    topo_full = expand_topo(oro)

    folds = get_climate_kfolds(ds, k=KFOLD_K, val_ratio=VAL_RATIO)
    sel = args.folds if args.folds is not None else list(range(len(folds)))

    print("\nArm: wgan_gp")
    print(f"NFE per member: 1 (single Generator forward pass, tiled)")
    print(f"Ensemble size:  {args.members}")
    print(f"Eval seed:      {EVAL_SEED} (shared across arms -> paired comparison)\n")

    results = {}
    for fi in sel:
        fold = folds[fi]
        name, idx = fold["name"], fold["test_idx"]
        print(f"{'='*78}\nFold {name}  --  {len(idx)} held-out test days\n{'='*78}")

        ck_path = _find_ckpt(args.ckpt_dir, f"Generator_{name}_best.pth")
        net, ck = load_generator(ck_path, dev)
        assert_precip_transform_compatible(ck, pt, ck_path, 0, "Generator (wgan_gp)")
        tile = args.tile or ck.get("patch") or PATCH or min(ds.H, ds.W)
        print(f"  [wgan_gp] {ck_path}  tile={tile}  val_crps={ck.get('crps'):.4f}")

        m = evaluate(net, ds, idx, topo_full, pt, dev, args.members, args.batch, tile)
        print(f"  [default] CRPS {m['crps']:8.4f} | MAE {m['mae']:8.4f} | "
              f"RMSE {m['rmse']:8.4f} | S/S {m['spread_skill']:5.3f} | "
              f"spec {m['spectrum_logratio']:5.3f} | "
              f"wet {m['wet_frac_pred']:.3f}/{m['wet_frac_target']:.3f} | "
              f"P99 {m['p99_pred']:7.2f}/{m['p99_target']:7.2f}")

        results[name] = {"default": m}
        del net
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"param": "wgan_gp", "nfe": 1, "steps": 1,
                   "members": args.members, "eval_seed": EVAL_SEED,
                   "folds": results}, f, indent=2)
    print(f"\nWrote {out_path}")
    print("Compare against another arm's JSON with:  python Compare.py "
          f"{out_path} outputs/eval_edm.json")


if __name__ == "__main__":
    main()
