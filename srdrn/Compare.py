# -*- coding: utf-8 -*-
"""
Compare.py -- pair two Evaluate.py JSONs into a head-to-head table.
====================================================================
Copied verbatim from corrdiff_fm/Compare.py. Pure standard library plus numpy:
no torch, no model loading, so it runs anywhere and doesn't need to import any
package's network code -- it only ever looks at the JSON that each package's
own Evaluate.py already produced. That is what lets this exact file be shared,
unmodified, by corrdiff_fm (edm/fm arms), this package (srdrn), and any sibling
wgan_gp/srgan-style package: as long as an Evaluate.py JSON has the schema this
file expects (crps, mae, rmse, spread_skill, rank_chi2, spectrum_logratio,
wet_frac_pred/target, p99_pred/target, n_days, _crps_per_day), it can be
compared against any other package's JSON.

The pairing is valid because both arms scored the same held-out test days in
the same order with the same generator seed (Config.EVAL_SEED). That matters
more than it sounds: Var(x_i - y_i) is far smaller than Var(x_i), so a paired
test has much more power than comparing two independent means.

For SRDRN specifically: it is a deterministic, single-shot network, so its
JSON always has members=1, spread=0, and CRPS collapses to MAE (see
Evaluate.py's own note on this degeneracy). Compare.py does not need to know
that -- it just reads whatever crps/mae/rmse numbers are in the JSON -- but
you should, when reading the printed spread/skill column: 0.000 for the SRDRN
side of a comparison is the expected, structural result of having no ensemble,
not a sign that something is broken.

Usage
-----
  python Compare.py outputs/eval_edm.json outputs/eval_srdrn.json
  python Compare.py outputs/eval_edm.json outputs/eval_srdrn.json --markdown
"""

import json
import argparse

import numpy as np


def paired_bootstrap(d, n_boot=10000, seed=0):
    """95% CI on the mean of a paired difference."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 2:
        return (float(d.mean()) if len(d) else 0.0), 0.0, 0.0
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(a, b, key, width=10, prec=4, lower_better=True):
    va, vb = a.get(key), b.get(key)
    if va is None or vb is None:
        return f"{'--':>{width}} {'--':>{width}}"
    better = (va < vb) if lower_better else (va > vb)
    sa = f"{va:.{prec}f}" + ("*" if better else " ")
    sb = f"{vb:.{prec}f}" + (" " if better else "*")
    return f"{sa:>{width}} {sb:>{width}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_a", help="usually outputs/eval_edm.json")
    ap.add_argument("json_b", help="usually outputs/eval_srdrn.json")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    A = json.load(open(args.json_a))
    B = json.load(open(args.json_b))
    na, nb = A["param"], B["param"]

    print("=" * 82)
    print(f"{na.upper()}  vs  {nb.upper()}")
    print("=" * 82)

    if A["nfe"] != B["nfe"]:
        print(f"\n  *** WARNING: NFE MISMATCH -- {na}={A['nfe']}, {nb}={B['nfe']}. ***")
        print("  These runs are not comparable. Re-run both with the same --steps.\n")
    else:
        print(f"NFE per member: {A['nfe']}   ensemble size: {A['members']}   "
              f"eval seed: {A.get('eval_seed')}")

    if A.get("members") != B.get("members"):
        print(f"  *** NOTE: ensemble sizes differ ({A['members']} vs {B['members']}). "
              "CRPS is bias-corrected so this is survivable, but spread/skill is not "
              "directly comparable -- expected and structural if one side is SRDRN "
              "(members=1, deterministic). ***")
    if A.get("eval_seed") != B.get("eval_seed"):
        print("  *** WARNING: different eval seeds -- the paired test below is invalid. "
              "Treat the CI as unpaired and much wider than shown. ***")

    all_d = []
    for fold in sorted(set(A["folds"]) & set(B["folds"])):
        for tag in sorted(set(A["folds"][fold]) & set(B["folds"][fold])):
            a, b = A["folds"][fold][tag], B["folds"][fold][tag]
            print(f"\n--- fold {fold}  [{tag}]  {a.get('n_days')} days "
                  f"---------------------------------")
            print(f"{'metric':<22}{na:>11}{nb:>11}    (* = better)")
            print(f"{'CRPS (mm/day)':<22}{fmt(a,b,'crps')}")
            print(f"{'MAE  (mm/day)':<22}{fmt(a,b,'mae')}")
            print(f"{'RMSE (mm/day)':<22}{fmt(a,b,'rmse')}")
            print(f"{'spread/skill':<22}"
                  f"{a['spread_skill']:>10.3f} {b['spread_skill']:>10.3f}   "
                  f"(1.0 = calibrated)")
            print(f"{'spectrum log-ratio':<22}{fmt(a,b,'spectrum_logratio',prec=3)}")
            print(f"{'rank chi-square':<22}{fmt(a,b,'rank_chi2',prec=1)}")
            wet_lbl = "wet frac (tgt {:.3f})".format(a['wet_frac_target'])
            p99_lbl = "P99 (tgt {:.1f})".format(a['p99_target'])
            print(f"{wet_lbl:<22}{a['wet_frac_pred']:>10.3f} {b['wet_frac_pred']:>10.3f}")
            print(f"{p99_lbl:<22}{a['p99_pred']:>10.2f} {b['p99_pred']:>10.2f}")

            da, db = a.get("_crps_per_day"), b.get("_crps_per_day")
            if da and db and len(da) == len(db):
                d = np.array(da) - np.array(db)
                all_d.append(d)
                mean, lo, hi = paired_bootstrap(d)
                verdict = (f"{na} better" if hi < 0 else
                           f"{nb} better" if lo > 0 else "no significant difference")
                print(f"  paired dCRPS ({na} - {nb}) = {mean:+.4f}  "
                      f"[{lo:+.4f}, {hi:+.4f}] 95% CI  ->  {verdict}")
                print(f"  {na} lower CRPS on {int((d < 0).sum())}/{len(d)} days")
            else:
                print("  [no per-day CRPS stored -- cannot pair this fold]")

    if all_d:
        d = np.concatenate(all_d)
        mean, lo, hi = paired_bootstrap(d)
        verdict = (f"{na} better" if hi < 0 else
                   f"{nb} better" if lo > 0 else "no significant difference")
        print("\n" + "=" * 82)
        print(f"POOLED ACROSS ALL FOLDS AND DAYS  (n = {len(d)})")
        print(f"  paired dCRPS ({na} - {nb}) = {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}] 95% CI")
        print(f"  -> {verdict}")
        print(f"  {na} lower CRPS on {int((d < 0).sum())}/{len(d)} days "
              f"({100*(d < 0).mean():.1f}%)")
        print("=" * 82)

    print("""
How to read this
----------------
CRPS is the headline; the paired CI is what makes it a claim rather than an
anecdote. If one arm wins RMSE but loses CRPS it is under-dispersed, not better
-- check spread/skill and the rank chi-square before believing the RMSE. If CRPS
is close but the spectrum log-ratio is not, the fine-scale structure differs in
a way CRPS is not sensitive to, and that is usually the more interesting result.

When one side is SRDRN (deterministic, members=1): CRPS degenerates exactly to
MAE and spread/skill is always 0 by construction, not because the model failed
to disperse -- there is no ensemble to disperse. The informative comparisons
against a probabilistic arm (CorrDiff-EDM/FM) are then MAE/RMSE (does the
deterministic mean estimate beat or lose to the diffusion ensemble mean?),
the spectrum log-ratio (does SRDRN's output look blurry relative to the
target's fine-scale power, which a purely MSE/MAE-trained network is
structurally prone to?), and P99 (does SRDRN reproduce precipitation extremes,
which is exactly what WMAE was designed to help with?).

One caveat on external validity: this is one region, one set of folds, one
data pipeline. It is evidence about your setup, not a general claim about the
methods. Run all five folds before concluding anything.""")


if __name__ == "__main__":
    main()
