# CorrDiff precipitation downscaling — EDM package

Residual corrective diffusion for 32 km → 8 km training, applied as a cascade to
generate 2 km fields. Implements Mardani et al. 2024 (arXiv:2309.15214v4).

**This is the `edm` arm.** The companion ``fm`` package is byte-identical
except for two files — `Param.py` and ``Diffusion.py`` — so any difference you measure
between them is attributable to the generative parameterization rather than to a
hundred incidental implementation choices.

---

## Run order

| # | Command | GPUs | What it does |
|---|---|---|---|
| 0 | `qsub job_0_prepare.sh` | 0 | Fix precip units at the source, run pre-flight checks |
| 1 | `qsub job_1_regressor.sh` | 2 | Stage 1: conditional mean, **L2 loss** |
| 2 | `qsub job_2_stage2.sh` | 2 | Stage 2: the EDM residual corrector |
| 3 | `qsub job_3_evaluate.sh` | 1 | Score on held-out test years → JSON |
| 4 | `qsub -J 0-7 job_4_inference.sh` | 1×8 | 8 km → 2 km cascade, sharded by year |

Then, once **both** packages have finished step 3:

```bash
python Compare.py outputs/eval_edm.json outputs/eval_fm.json
```

**Steps 0 and 1 only need running once, in one package.** Both arms share the
corrected precip file and the same Stage-1 checkpoints. Training a separate
Stage 1 per arm would confound the comparison entirely, since each arm would
then be correcting a different mean. Point `Config.REGRESSOR_CKPT_DIR` in both
packages at the same directory.

---

## Before you start

Edit **`Config.py`** — it holds every path and every cross-stage constant, so
there is exactly one place to change and nothing to keep in sync:

```python
ROOT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh"
REGRESSOR_CKPT_DIR = "checkpoints/regressor/"     # SHARED between arms
PATCH = 128                                       # see below
```

Also edit the two `conda` lines at the top of each `job_*.sh`.

### `PATCH` is the one constant to understand

It is the grid size the networks see in training **and** the tile size at
inference, and **both stages must use the same value.**

The 8 km → 2 km cascade works because the learned operator is scale-*relative*:
"given an area mean and fine topography, restore 4× of sub-grid structure."
Nothing in that names a resolution. But two blocks in `Network.py` are not
resolution-agnostic — `SpectralConv2d` ties learned weights to absolute FFT mode
indices, and `SelfAttn2d` switches to pooled attention past a token budget. So
handing a network a 4× larger canvas is a genuinely different operator, not the
same one applied more times. A shape assertion will not catch this: the output
is still exactly 4× the input, it is just wrong.

Pinning the grid size, and tiling the large domain, is what makes the cascade
legitimate.

---

## Files

| File | |
|---|---|
| `Config.py` | Every path and cross-stage constant |
| `PrepareData.py` | Unit correction + pre-flight validation |
| `Dataset.py` | NetCDF loading, normalization, k-fold splits |
| `Network.py` | `CorrDiffRegressor` (Stage 1) and `UNet` (Stage 2) |
| `Tiling.py` | Overlapping-tile machinery, shared by both stages |
| **``Diffusion.py``** | **EDM core — differs from the other package** |
| **`Param.py`** | **Parameterization adapter — differs from the other package** |
| `Regressor.py` | Stage-1 training (L2) |
| `TrainStage2.py` | Stage-2 training |
| `Inference.py` | 8 km → 2 km cascade |
| `Evaluate.py` | Held-out scoring → JSON |
| `Compare.py` | Pairs two JSONs into a head-to-head |

---

## Why Stage 1 uses L2, specifically

Not a stylistic choice. The whole decomposition

```
x = E[x|y] + (x − E[x|y])
    Stage 1      Stage 2
```

rests on Stage 1 estimating the conditional **mean**, because that is what makes
the residual zero-mean, and a zero-mean residual is what gives the variance
reduction (paper Eq. 2) that makes the diffusion problem easier than modelling
p(x) directly.

The squared-error minimizer *is* the conditional mean. An L1 loss converges to
the conditional **median** instead — which for precipitation, with its huge point
mass at zero and long right tail, is far below the mean and often exactly zero.
The residual would be strongly positively biased and the assumption underpinning
the method would simply be false. Resist the temptation to "robustify" this
stage with Huber.

The loss is computed in normalized log1p space, not mm/day; squared error on raw
precipitation would be dominated by a handful of extreme wet points.

---

## Outputs

One NetCDF per held-out test year, in `outputs/downscaled_2km_edm/`:

- `precip_mean` — ensemble mean, mm/day
- `precip_std` — ensemble spread (uncertainty), mm/day
- `precip_stage1_mean` — Stage-1-only deterministic baseline
- `precip_members` — individual members (with `--save-members`)

Each file records the arm, tile size, NFE and sampler settings in its global
attributes.

---

## Caveats the code cannot fix

These are inherent to the cascade, and are surfaced rather than hidden:

- An 8 km RCM field is close to, but not exactly, the area mean of a 2 km field.
  Mitigated by the `LR_AUG` jitter during training; not eliminated.
- Sub-grid precipitation variance is **not** scale-invariant. The residual
  amplitude learned at 32→8 km is not guaranteed correct at 8→2 km. Exposed as
  `--res-std-scale` (default 1.0) rather than left unstated. Tune only against
  held-out 2 km data, and report the value you used.
- Normalization statistics are the 8 km ones. Fine fields have heavier tails, so
  expect the extreme tail to be conservative.

Treat the 2 km output as a physically plausible refinement, not a validated
product, until you have 2 km ground truth to score it against.

---

## EDM specifics

Implements the paper's Sec. 5.3.2 exactly: `ln σ ~ N(0, 1.2²)` (i.e. `P_mean = 0.0`,
**not** the −1.2 image default), `σ_max = 800`, `σ_min = 0.002`, `ρ = 7`, 18 steps,
and the **stochastic** second-order sampler (Karras Algorithm 2, with churn — not
the deterministic ODE).

`sigma_data = 1.0` because the residual is explicitly rescaled to unit variance
before the denoiser sees it. The loss is taken in preconditioned space with
**uniform** weight: dividing the target by `c_out` already *is* Karras' λ(σ), and
applying λ(σ) again on top squares it.

### One calibration confound to know about

Measured against an *exact analytic denoiser* on unit-variance data — so any
deviation is pure integrator error, no network involved:

| steps | σmax=800, churn=0 | σmax=800, churn=40 | σmax=80, churn=40 |
|---|---|---|---|
| 8 | 1.718 | 2.044 | 1.502 |
| **18** | **1.106** | **1.210** | **1.111** |
| 32 | 1.029 | 1.082 | 1.047 |
| 64 | 1.006 | 1.038 | 1.021 |

against a true std of 1.000. At the paper's 18 steps EDM over-disperses by
10–21% for purely numerical reasons. It converges away, so it is discretization
error, not a bug. The cause is not ρ (7 is already near-optimal) — it is that
with `σ_data = 1.0`, `σ_max = 800` stretches an 18-point schedule so thin that
only 2 of 18 levels land near σ ≈ σ_data.

**The fix is free.** `σ_max` appears *only* in the sampling schedule — training
draws σ from the log-normal and never consults it — so it can be changed without
retraining. `job_4_inference.sh` uses `--sigma-max-sample 200 --steps 32`; drop
those two flags to reproduce the paper literally.

This lands directly on the spread/skill ratio you are calibrating, so decide
deliberately, and do not compare 18-step EDM against 18-step flow matching and
attribute the spread difference to the parameterization.

---

## Sanity checks worth watching

- **Stage 2 loss ≈ 1.0 at epoch 1.** Both arms are constructed to have a
  unit-variance target, so a very different starting loss means something is
  mis-scaled.
- **Spread/skill ratio**, printed each validation. 1.0 is calibrated. Expect
  under-dispersion — the paper reports its own model as under-dispersive and
  calls calibration an open problem (Sec. 3.4). Below ~0.4 means the sampler
  stochasticity is not doing its job.
- **CRPS, not the training loss**, is what model selection uses. For a
  probabilistic downscaler the denoising loss is only a surrogate, and the two
  diverge: loss keeps falling while spread quietly collapses.
