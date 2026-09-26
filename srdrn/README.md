# SRDRN precipitation downscaling — a purely supervised, deterministic package

A 16-residual-block, feed-forward Super-Resolution Deep Residual Network for
32 km → 8 km precipitation downscaling, trained end-to-end with a plain
supervised loss (MSE, MAE, or weighted MAE). Implements the architecture
described in *SRDRN: Super-Resolution Deep Residual Network* (IOP Publishing,
[doi:10.1088/2752-5295/ae6885](https://iopscience.iop.org/article/10.1088/2752-5295/ae6885)).

**This is a sibling package to `corrdiff_fm`** (residual-diffusion EDM/flow-
matching downscaling) and to any other package under `/home/ylale/extras/h/`
(e.g. `wgan_gp`, `srgan`). Same task, same data, same k-fold protocol, same
evaluation schema — a genuinely different, non-adversarial, deterministic
architecture family, so that a head-to-head via `Compare.py` measures
architecture, not incidental pipeline differences.

---

## SRDAN → SRDRN: a naming correction, stated plainly

This package was originally requested under the name **"SRDAN"**, citing the
DOI above. Having fetched and read that paper in full, its actual published
title and architecture name is **SRDRN** ("Super-Resolution Deep Residual
Network"), and the paper is explicit that it is **non-adversarial**: it
contains no discriminator and no adversarial loss anywhere, and its own
selling point (paper abstract / Sec. 3) is *"stable and reproducible training
without issues such as mode collapse"* — a benefit the authors attribute
directly to training with plain supervised losses (MSE / MAE / weighted-MAE)
rather than a GAN objective. Building an adversarial "SRDAN" while citing this
paper would misrepresent it. After confirming this with the user, this
package implements SRDRN exactly as published (subject to the two documented
adaptations below) — a plain, deterministic, residual super-resolution
network. **There is no discriminator, no adversarial loss, and no noise input
anywhere in this codebase.** Every file, class, and checkpoint in this package
is named `SRDRN`/`srdrn`, not `SRDAN`, and this paragraph exists so that
distinction survives however many times this README gets skimmed rather than
read.

---

## Run order

| # | Command | GPUs | What it does |
|---|---|---|---|
| 0 | `qsub job_0_prepare.sh` | 0 | Fix precip units at the source, run pre-flight checks (**run at most once across every package under `/home/ylale/extras/h/`, not once per package** — see below) |
| 1 | `qsub job_1_train.sh` | 2 | Train SRDRN (single stage — no Stage 1/Stage 2 split) |
| 2 | `qsub job_2_evaluate.sh` | 1 | Score on held-out test years → JSON |
| 3 | `qsub -J 0-7 job_3_inference.sh` | 1×8 | 8 km → 2 km cascade, sharded by year |

Then, once this package **and** at least one sibling package have finished
their evaluation step:

```bash
python Compare.py outputs/eval_edm.json outputs/eval_srdrn.json
```

**Step 0 only needs running ONCE across every package that shares this ROOT**
(`corrdiff_fm`'s `edm`/`fm` arms, this `srdrn` package, and any sibling
`wgan_gp`/`srgan`-style package). `PrepareData.py` corrects the precip file's
units at the source and validates the shared static inputs; it knows nothing
about SRDRN, CorrDiff, or any other downstream architecture, and running it
again elsewhere just re-diagnoses a file that's already correct. Point every
package's `Config.HR_FILES`/`Config.ORO_8KM` at the same corrected file rather
than re-running the correction per package.

Unlike `corrdiff_fm`, there is **no shared Stage-1 checkpoint** to coordinate:
SRDRN is single-stage, so `job_1_train.sh` is the only training job in this
package, full stop.

---

## Before you start

Edit **`Config.py`** — every path and cross-run constant lives there, mirroring
`corrdiff_fm`'s own convention (so there is exactly one place to change and
nothing to keep in sync across scripts):

```python
ROOT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh"   # SAME value as corrdiff_fm/Config.py
CKPT_DIR = "checkpoints/srdrn/"                        # this package's own, not shared
PATCH = 128                                            # see below
NUM_RES_BLOCKS = 16                                    # faithful to the paper
LOSS_KIND = "wmae"                                     # "mse" | "mae" | "wmae"
```

Also edit the two `conda` lines at the top of each `job_*.sh`.

`ROOT`, `HR_FILES`, `ORO_8KM`, `ORO_2KM`, `VAR_MAP`, `DS_FACTOR`, `PRECIP_CH`,
`IN_CH_LR`, `PATCH`, `PATCH_OVERLAP`, `KFOLD_K`, `VAL_RATIO`, `SEED`, and
`EVAL_SEED` are all held at the **same values** as `corrdiff_fm/Config.py` on
purpose — see that file's docstring copy in this package's `Config.py` for
why. If you repoint one package's `ROOT` you almost certainly want to repoint
every sibling package's the same way, or the comparability this whole
directory structure is built around breaks silently.

### `PATCH` still matters here, for a different (weaker) reason than in `corrdiff_fm`

In `corrdiff_fm`, `PATCH` pins the network to a fixed grid size because two of
its blocks (`SpectralConv2d`, `SelfAttn2d`) are not resolution-agnostic — a
larger canvas is a genuinely different operator there. **SRDRN has neither
block**: every layer (conv, BatchNorm, PReLU, PixelShuffle) is strictly local
and translation-equivariant, so feeding it a much larger domain in one shot
would, modulo edge effects and memory, produce the same result as tiling and
blending. `PATCH`/tiling is still worth keeping here, but for smaller, more
honestly-stated reasons — see `Tiling.py`'s module docstring for the full
argument, and the "Tiling caveat" section below for the short version.

---

## Deviations from the published SRDRN, and why

Two adaptations were made deliberately, per an explicit user decision to keep
this package directly comparable — via `Compare.py` — with `corrdiff_fm` and
any other sibling package under `/home/ylale/extras/h/`, rather than replicate
the paper's own narrower India-region setup. Everything else (16 residual
blocks, local **and** global skip connections via elementwise addition,
BatchNorm+PReLU inside every residual and upsampling block, feature
extraction entirely in LR space, a final conv to 1 channel) is kept faithful
to the paper.

1. **8x → 4x upsampling.** The published SRDRN targets 0.8°→0.1° (8x) over
   the Indian subcontinent, via 3 upsampling blocks. This package targets
   32 km→8 km (4x, `Config.DS_FACTOR=4`), via 2 upsampling blocks
   (`Config.N_UP_BLOCKS=2`), because that is the resolution ratio every other
   package under `/home/ylale/extras/h/` already trains on. `Network.SRDRN`
   asserts `2**N_UP_BLOCKS == DS_FACTOR`, mirroring `CorrDiffRegressor`'s own
   `channel_mult` assertion in `corrdiff_fm/Network.py` — the same style of
   guard against silently mismatched upsampling depth and target factor.

2. **3-channel paper input → 4-channel + HR topography.** The paper's input
   is a 3-channel LR tensor: LR precipitation, LR daily climatology, and LR
   orography, all crudely downsampled together. This package instead uses
   the sibling packages' 4-channel LR input (`huss`, `mslp`, `tas`, `precip`
   — see `Config.IN_CH_LR` and `Dataset.ClimateDataset`), and fuses
   topography separately, at HR resolution, via the richer 8-channel
   `expand_topo` featurization (elevation, slope, aspect, curvature,
   land-sea mask, coordinates) — the same convention `corrdiff_fm`'s
   `CorrDiffRegressor` uses, and for the same reason: `expand_topo`'s
   derivatives are Sobel gradients that are only meaningful at the
   resolution they're computed at, so it is both cheaper and more physically
   direct to compute them once at HR and fuse them in after upsampling than
   to downsample orography into the LR trunk and lose that resolution.
   Concretely, this moves the topography fusion point from "concatenated
   into the LR input at the very start" (the paper) to "concatenated with
   the upsampled trunk features just before the final output conv" (this
   package) — see `Network.SRDRN.forward`, step 5, and the `_TopoBranch`
   class docstring for exactly where and why.

Neither change touches the parts of the architecture the paper actually
describes in detail — the residual-block design, the double level of skip
connections, or the normalization/activation choices — and both are recorded
in every checkpoint's `arch` sub-dict so a reader can always tell exactly what
was trained, not just what the current `Config.py` happens to say.

---

## Tiling caveat: honest about severity, not overclaiming it

`Tiling.py` still tiles the domain at inference for memory and consistency
reasons, but **the justification is genuinely weaker than `corrdiff_fm`'s**,
and the module docstring says so explicitly rather than reusing that
package's stronger language verbatim. In brief:

- `corrdiff_fm` tiles because two of its blocks (`SpectralConv2d`,
  `SelfAttn2d`) tie the network's behaviour to the absolute grid size, so a
  larger canvas is a **different operator**, not the same one applied more
  times. SRDRN's trunk has neither block — it is fully local and
  translation-equivariant.
- SRDRN is tiled anyway because (1) a domain-wide forward pass through 16
  BatchNorm-heavy residual blocks at HR resolution can simply exceed GPU
  memory even though it would be numerically fine given infinite memory —
  this is the dominant, and most honest, reason; (2) keeping the inference
  tile size equal to the training patch size keeps the feature-activation
  distribution BatchNorm's running statistics were calibrated against
  reasonably matched, a real but secondary effect since BatchNorm-eval
  applies fixed running stats regardless of input size; and (3) conv padding
  at tile edges introduces seams that tiling itself creates, which
  overlap-blending (kept from `corrdiff_fm`'s `blend_window`/`tile_origins`)
  then removes.

Read `Tiling.py`'s module docstring for the full argument if you're deciding
whether to change `PATCH` or the tile size at inference.

---

## Files

| File | |
|---|---|
| `Config.py` | Every path and constant — same values as `corrdiff_fm/Config.py` for the shared ones, plus SRDRN's own architecture/loss constants |
| `PrepareData.py` | Unit correction + pre-flight validation (shared infra — run once across all packages) |
| `Dataset.py` | NetCDF loading, normalization, k-fold splits (copied essentially verbatim from `corrdiff_fm`) |
| `Network.py` | `SRDRN` generator, reused topo/utility functions, the WMAE weighting reconstruction |
| `Tiling.py` | Overlapping-tile machinery, adapted (simplified — no per-sampler-step blending, since there's no sampler) |
| `Train.py` | Single training script: k-fold loop, EMA, checkpoint with `arch` sub-dict, mm/day validation |
| `Inference.py` | 8 km → 2 km cascade, sharded by year |
| `Evaluate.py` | Held-out scoring → JSON (same schema as `corrdiff_fm`'s) |
| `Compare.py` | Pairs two JSONs into a head-to-head (copied verbatim — pure stdlib+numpy) |

---

## Why WMAE specifically, and the honesty caveat about the reconstructed weighting

The paper trains three separate models — SRDRN-MSE, SRDRN-MAE, SRDRN-WMAE —
and reports WMAE as the best performer for reproducing precipitation
extremes. `Config.LOSS_KIND` defaults to `"wmae"` for that reason, and all
three are implemented behind the same switch (`Network.srdrn_loss`).

**The paper's abstract does not state the exact WMAE weighting formula.**
`Network.wmae_pixel_weight` is **this implementation's reconstruction of the
paper's stated intent** — "emphasize precipitation-intensity extremes" — not
a verified reproduction of the authors' exact function. The reconstruction
used is:

```
weight = 1.0 + ALPHA * intensity
```

where `intensity` is the **denormalized mm/day** target value, divided by the
current batch's own 99th-percentile precipitation intensity and clamped to
`[0, 1]` — denormalized because "intensity extremes" is a physical-space
statement (log1p is a variance-stabilizing transform, so weighting in
normalized space would suppress exactly the tail this loss exists to
emphasize), and per-batch-quantile-scaled rather than a fixed global constant
so no single extreme-tail outlier day can blow up the weight and swamp every
other pixel's gradient. `Config.WMAE_ALPHA` (default `2.0`) is a **tunable
hyperparameter, not a settled constant** — treat it with exactly the same
epistemic status `corrdiff_fm`'s README gives its own unverified `sigma_max`
calibration: state clearly what you tuned it against if you change it and
report results.

---

## Outputs

One NetCDF per held-out test year, in `outputs/downscaled_2km_srdrn/`:

- `precip_mean` — the (only) deterministic downscaled precipitation field, mm/day

**There is no `precip_std` and no `precip_members`.** Both would be
meaningless for a model with no stochastic component: `precip_std` would be a
column of exact zeros, and `precip_members` would just be `precip_mean`
repeated. Rather than emit degenerate placeholders that a downstream script
might mistake for real uncertainty information, `Inference.py` omits them
entirely and records `deterministic=1` / `ensemble_members=1` in the file's
global attributes so this is discoverable from the NetCDF itself, not only
from this README.

---

## Caveats the code cannot fix

- **SRDRN is a conditional-mean estimator by construction** (whichever of
  MSE/MAE/WMAE it's trained with is still a pixelwise loss minimized in
  expectation), so expect its output to be systematically **smoother** than
  any individual real 2 km realization — it cannot add back genuine sub-grid
  variability the way a stochastic arm (`corrdiff_fm`) can in principle do.
  Watch `Evaluate.py`'s `spectrum_logratio` metric for this directly.
- An 8 km RCM field is close to, but not exactly, the area mean of a 2 km
  field. Mitigated by `Train.py`'s `AUG_P`/`AUG_MAX` coarse-intensity jitter
  during training; not eliminated.
- Normalization statistics are the 8 km ones. Fine fields have heavier tails,
  so expect the extreme tail to be conservative at 2 km — this is exactly the
  failure mode WMAE is meant to push back against, not eliminate outright.
- The WMAE weighting formula is a documented reconstruction, not a verified
  reproduction of the paper's own (unstated) formula — see the section above.

Treat the 2 km output as a physically plausible refinement, not a validated
product, until you have 2 km ground truth to score it against — the same
standard `corrdiff_fm`'s README holds its own cascade to.

---

## Sanity checks worth watching

- **Train/val loss ratio.** SRDRN has no analogue of `corrdiff_fm`'s
  "Stage 2 loss ≈ 1.0 at epoch 1" unit-variance check (its target is
  normalized log1p precip, not a unit-variance residual), but a normalized
  training loss that keeps falling sharply while validation RMSE/MAE (mm/day)
  plateaus or worsens is the same overfitting signature it always is —
  watch both, not just the training curve.
- **WMAE vs. plain-MAE divergence.** `Train.py`'s `validate()` always reports
  plain, unweighted MAE/RMSE in mm/day regardless of `LOSS_KIND`. If you train
  with `"wmae"`, watch this alongside the (weighted) training loss: if the
  weighted loss keeps improving while plain MAE stagnates or worsens, the
  weighting is trading mean accuracy for tail accuracy more aggressively than
  intended, and `Config.WMAE_ALPHA` is the first thing to turn down.
- **P99 (predicted vs. target), in `Evaluate.py`'s JSON.** This is the most
  direct read on whether WMAE is doing its stated job of reproducing
  precipitation extremes — if SRDRN-WMAE's P99 isn't visibly closer to the
  target's than SRDRN-MSE's, the weighting reconstruction likely needs
  retuning (or, per the honesty caveat above, may simply differ from
  whatever the paper's authors actually used).
- **`spectrum_logratio` vs. a stochastic sibling package's, via `Compare.py`.**
  Expect SRDRN to be blurrier at the fine scales — this is the structural
  cost of a deterministic conditional-mean estimator, not a bug to chase away
  with more training.
