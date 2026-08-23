# -*- coding: utf-8 -*-
"""
Evaluate.py -- score THIS package's arm on the held-out test years.
====================================================================
IDENTICAL IN BOTH PACKAGES. Writes a JSON of per-fold, per-day metrics. Run it
in each package, then use Compare.py to pair the two JSONs into a head-to-head.

Kept as two independent runs plus a merge, rather than one script that loads
both arms, so that neither package has to import the other's sampler. The
pairing still works exactly, because both arms see the same test days in the
same order with the same generator seed (Config.EVAL_SEED) -- so day i in the
EDM JSON and day i in the FM JSON are the same day with the same initial noise.

METRICS, AND WHY THESE ONES
---------------------------
  CRPS            The headline. A proper scoring rule -- unlike RMSE it cannot
                  be gamed by a model that hedges toward the conditional mean,
                  which is exactly the failure mode to worry about here.
  MAE / RMSE      Of the ensemble mean. Expect the more deterministic model to
                  win RMSE while losing CRPS; the paper sees precisely this
                  (Table 1). If that happens, the RMSE winner is under-dispersed,
                  not better.
  Spread/skill    Ensemble spread over ensemble-mean RMSE, adjusted by
                  sqrt(1 + 1/M) so 1.0 means calibrated (paper Fig. 3).
  Rank histogram  Chi-square against uniform. Catches bias that spread/skill
                  averages away; a U shape means the truth keeps landing outside
                  the ensemble.
  Log-spectrum    Mean |log10 power ratio| vs target over the top two octaves.
    ratio         This is the one CRPS cannot see -- a model can score well on
                  CRPS while producing blurry fields.
  Wet fraction    Distributional realism, including at the tail that matters.
  and P99

Usage
-----
  python Evaluate.py --members 16 --steps 18
  python Evaluate.py --churn-sweep 0 10 40      # sampler-stochasticity sweep
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from Config import (HR_FILES, ORO_8KM, REGRESSOR_CKPT_DIR, KFOLD_K, VAL_RATIO,
                    EVAL_SEED, PRECIP_CH, PATCH)
from Dataset import ClimateDataset, get_climate_kfolds, DS_FACTOR
from Network import expand_topo, denorm_precip_mmday
from Tiling import regress_tiled
from Param import (PARAM, CKPT_DIR, SAMPLE_STEPS as DEFAULT_STEPS,
                   sample_residual, config_from_dict, with_stochasticity, nfe_count)
from TrainStage2 import (load_frozen_regressor, assert_precip_transform_compatible,
                         pick_amp, build_unet, coarsen)


# ------------------------------------------------------------------------------
# METRICS
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
    the diffusion step is supposed to be synthesizing. 0 is perfect."""
    n = len(targ_spec)
    lo = max(1, n // 4)
    a = pred_spec[lo:n].clamp_min(1e-12)
    b = targ_spec[lo:n].clamp_min(1e-12)
    return float((torch.log10(a / b)).abs().mean())


def paired_bootstrap(diff_per_day, n_boot=5000, seed=0):
    """95% CI on the mean of a paired difference. Paired because both arms saw
    the same days -- Var(x_i - y_i) << Var(x_i), so this is far tighter than
    comparing two independent means."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff_per_day, dtype=np.float64)
    if len(d) < 2:
        return float(d.mean()) if len(d) else 0.0, 0.0, 0.0
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------------------
# EVALUATION OF ONE ARM
# ------------------------------------------------------------------------------

def _find_ckpt(ckpt_dir, exact_name):
    """Same fallback policy as Inference.py: prefer the fold-matched checkpoint,
    but warn loudly rather than crash if it is absent -- and say why it matters,
    because a mismatched Stage 1 leaks the held-out years into both arms."""
    cand = os.path.join(ckpt_dir, exact_name)
    if os.path.exists(cand):
        return cand
    files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith((".pt", ".pth"))]) \
        if os.path.isdir(ckpt_dir) else []
    if not files:
        raise FileNotFoundError(f"No checkpoints in {os.path.abspath(ckpt_dir)}")
    chosen = sorted([f for f in files if "best" in f.lower()] or files)[0]
    print(f"  [WARN] '{exact_name}' not found; falling back to {chosen}. Both arms get "
          "the same fallback so the COMPARISON stays paired and fair, but the absolute "
          "numbers are contaminated -- do not quote them as test-set scores.")
    return os.path.join(ckpt_dir, chosen)


def load_arm(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    arch = ck.get("arch")
    if arch is None:
        raise RuntimeError(f"{path} has no 'arch' key -- retrain with the current script.")
    net = build_unet(arch, dev, dropout=0.0)
    net.load_state_dict(ck.get("ema_state_dict") or ck["model_state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    param = ck.get("param", PARAM)
    if param != PARAM:
        raise RuntimeError(f"{path} has param='{param}', this package implements '{PARAM}'.")
    cfg = config_from_dict(ck.get("edm_cfg") or ck.get("fm_cfg"))
    return net, ck, param, cfg


@torch.no_grad()
def evaluate(net, param, cfg, ck, regressor, ds, idx, topo_full, pt, dev,
             amp_dtype, members, steps, batch, tile, churn_override=None):
    if churn_override is not None:
        cfg = with_stochasticity(cfg, churn_override)

    res_mean, res_std = ck["res_mean"], ck["res_std"]
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
        mu = regress_tiled(regressor, lr, topo, DS_FACTOR, tile_hr=tile, amp_dtype=amp_dtype)
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear", align_corners=False)
        cond = torch.cat([mu, lr_up.float()], 1)

        # Same seed for both arms on the same day => paired comparison.
        gen = torch.Generator(device=dev).manual_seed(EVAL_SEED + start)
        ens = []
        for _ in range(members):
            r = sample_residual(net, cond, topo, (B, 1, *target.shape[-2:]),
                                dev, cfg, n_steps=steps, generator=gen, tile=tile,
                                amp_dtype=amp_dtype)
            ens.append(denorm_precip_mmday(mu + res_mean + res_std * r, pt))
        ens = torch.stack(ens, 0)
        obs = denorm_precip_mmday(target, pt)

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
    rmse = math_sqrt(np.mean(per_day["mse"]))
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


def math_sqrt(x):
    return float(np.sqrt(x))


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help="Both arms must use the same value. NFE = 2*steps-1 either way.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--tile", type=int, default=None,
                    help="Default: the training patch recorded in the checkpoint.")
    ap.add_argument("--churn-sweep", type=float, nargs="+", default=None,
                    help="Score at each of these stochasticity levels (EDM S_churn / "
                         "FM churn). The gap between 0 and the default separates the "
                         "contribution of the sampler from that of the parameterization.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"outputs/eval_{PARAM}.json"
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
    sweep = args.churn_sweep if args.churn_sweep is not None else [None]

    print(f"\nArm: {PARAM}")
    print(f"NFE per member: {nfe_count(args.steps)} ({args.steps} steps)")
    print(f"Ensemble size:  {args.members}")
    print(f"Eval seed:      {EVAL_SEED} (shared across arms -> paired comparison)\n")

    results = {}
    for fi in sel:
        fold = folds[fi]
        name, idx = fold["name"], fold["test_idx"]
        print(f"{'='*78}\nFold {name}  --  {len(idx)} held-out test days\n{'='*78}")

        reg_path = os.path.join(REGRESSOR_CKPT_DIR, f"Regressor_{name}_best.pth")
        if not os.path.exists(reg_path):
            print(f"  [skip] missing Stage-1 checkpoint {reg_path}")
            continue
        ck_path = os.path.join(args.ckpt_dir, f"Stage2_{name}_best.pth")
        if not os.path.exists(ck_path):
            print(f"  [skip] missing Stage-2 checkpoint {ck_path}")
            continue

        regressor = load_frozen_regressor(reg_path, dev, pt, rank=0)
        net, ck, param, cfg = load_arm(ck_path, dev)
        assert_precip_transform_compatible(ck, pt, ck_path, 0, f"Stage-2 {PARAM}")
        tile = args.tile or ck.get("patch") or PATCH or min(ds.H, ds.W)
        print(f"  [{PARAM}] {ck_path}  tile={tile}  val_crps={ck.get('crps'):.4f}")

        fold_res = {}
        for churn in sweep:
            tag = "default" if churn is None else f"churn={churn:g}"
            m = evaluate(net, param, cfg, ck, regressor, ds, idx, topo_full, pt,
                         dev, amp_dtype, args.members, args.steps, args.batch,
                         tile, churn_override=churn)
            print(f"  [{tag}] CRPS {m['crps']:8.4f} | MAE {m['mae']:8.4f} | "
                  f"RMSE {m['rmse']:8.4f} | S/S {m['spread_skill']:5.3f} | "
                  f"spec {m['spectrum_logratio']:5.3f} | "
                  f"wet {m['wet_frac_pred']:.3f}/{m['wet_frac_target']:.3f} | "
                  f"P99 {m['p99_pred']:7.2f}/{m['p99_target']:7.2f}")
            fold_res[tag] = m

        results[name] = fold_res
        del regressor, net
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"param": PARAM, "nfe": nfe_count(args.steps), "steps": args.steps,
                   "members": args.members, "eval_seed": EVAL_SEED,
                   "folds": results}, f, indent=2)
    print(f"\nWrote {out_path}")
    print("Run the other package's Evaluate.py, then:  python Compare.py "
          f"outputs/eval_edm.json outputs/eval_fm.json")


if __name__ == "__main__":
    main()
