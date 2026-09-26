# WGAN-GP precipitation downscaling

Single-stage adversarial 32 km -> 8 km training, applied as a cascade to
generate 2 km fields. Implements Gulrajani et al. 2017, "Improved Training of
Wasserstein GANs" (arXiv:1704.00028), building on Arjovsky et al. 2017's WGAN
(the Wasserstein-distance objective itself; Gulrajani's own contribution is
replacing WGAN's weight-clipped critic with a differentiable gradient
penalty).

**This is a sibling package to `../corrdiff_fm`** (CorrDiff flow-matching/EDM
residual-diffusion downscaler): same task, same data, same k-fold protocol,
same tiling convention -- but a genuinely different, single-stage adversarial
architecture family rather than a two-stage regressor + residual-diffusion
cascade. Reading `../corrdiff_fm/README.md` first is not required but will
make every "unlike corrdiff_fm" note below land with more context.

---

## Run order

| # | Command | GPUs | What it does |
|---|---|---|---|
| 0 | `qsub job_0_prepare.sh` | 0 | Fix precip units at the source, run pre-flight checks |
| 1 | `qsub job_1_train.sh` | 2 | Train Generator + Critic end-to-end, per fold |
| 2 | `qsub job_2_evaluate.sh` | 1 | Score on held-out test years -> JSON |
| 3 | `qsub -J 0-7 job_3_inference.sh` | 1x8 | 8 km -> 2 km cascade, sharded by year |

Then, once this package and (any of) `../corrdiff_fm`'s arms have finished
their evaluation step:

```bash
python Compare.py outputs/eval_wgan_gp.json ../corrdiff_fm/outputs/eval_edm.json
```

**Step 0 only needs running ONCE, across EVERY package under
`/home/ylale/extras/h/`** -- not once per package. `PrepareData.py` here is a
byte-for-byte copy of `corrdiff_fm`'s: it corrects the same source precip
file at the same path (`Config.HR_FILES` points at an identical path in every
package) and runs the same pre-flight checks. If `corrdiff_fm/job_0_prepare.sh`
has already produced the corrected file, running this package's `job_0_prepare.sh`
again is harmless (it detects the corrected file, skips the rewrite, and only
re-validates) but is not necessary. There is no analogue of corrdiff_fm's "train
Stage 1 once, share it" step here, because WGAN-GP has no frozen Stage-1 model
to share -- see "Why no two-stage decomposition" below.

---

## Before you start

Edit **`Config.py`**:

```python
ROOT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh"
GENERATOR_CKPT_DIR = "checkpoints/generator_wgan_gp/"   # this package's own
CRITIC_CKPT_DIR = "checkpoints/critic_wgan_gp/"         # this package's own
PATCH = 128            # training crop size AND inference tile size -- see below
```

`ROOT`, `HR_FILES`, `ORO_8KM`, `ORO_2KM`, `VAR_MAP`, `DS_FACTOR`, `PRECIP_CH`,
`IN_CH_LR`, `PATCH`, `PATCH_OVERLAP`, `KFOLD_K`, `VAL_RATIO`, `SEED` and
`EVAL_SEED` are the SAME values as `corrdiff_fm/Config.py` -- deliberately, so
this package trains on the same data with the same folds and the same tiling
convention, and any measured difference against `corrdiff_fm` is attributable
to the architecture, not to a silently different split or patch size. Do not
change these unless you are also changing them in `corrdiff_fm`, or you are no
longer running a comparable experiment.

Also edit the two `conda` lines at the top of each `job_*.sh`.

### Does the PATCH / tiling caveat still apply here?

Partially, and for a **different and materially weaker** reason than
`corrdiff_fm`.

`corrdiff_fm`'s caveat is about the network's WEIGHTS being tied to an
absolute grid size: `SpectralConv2d` learns coefficients on absolute FFT mode
indices, and `SelfAttn2d` switches from exact to pooled attention past a
token budget, so handing either network a 4x-larger canvas is a **genuinely
different operator**, not the same one applied more times.

This package's `Generator` (see `Network.py`'s module docstring) has neither
block. It is a plain residual-convolutional stack: every layer is a local
convolution or a channel-wise operation (GroupNorm, SE-gate), and none of them
have any notion of the overall canvas size baked into their learned weights.
A conv kernel computes the same function on a 128x128 crop and a 944x944
domain. So the specific failure mode that makes tiling **mandatory** for
`corrdiff_fm` (silently wrong output, not even a shape error) does not apply
here in the same way.

Two weaker effects remain, and are worth knowing about rather than assuming
away:

1. **Ordinary convolutional padding seams.** Every conv here uses zero (or,
   at the domain edge, replicate) padding. A pixel near an artificial tile
   boundary sees a border that does not exist in the real, continuous domain,
   so two independently-run tiles can each answer slightly differently near
   their shared edge. This is why tiling (`Tiling.py`) is still used -- not to
   avoid a wrong operator, but to avoid a visible seam. See `Tiling.py`'s
   module docstring for the full argument, including why this is a strictly
   *simpler* problem than `corrdiff_fm`'s (no per-tile independent randomness
   to reconcile, because the noise field `z` is drawn once over the whole
   domain and sliced consistently across tiles).
2. **GroupNorm statistics computed at training-patch extent vs. full-domain
   extent.** Every `GroupNorm` in this Generator normalizes over whatever
   spatial extent it is handed. A model whose norm layers only ever saw
   128x128 (or smaller) activations during training computes a genuinely
   different set of normalization statistics if handed a 944x944 activation
   map in one shot. This is a real, if smaller, train/inference mismatch. It
   does not corrupt the OPERATOR the way `corrdiff_fm`'s issue does (there is
   no absolute-position dependence learned into a weight), but it is still a
   distributional shift the network never saw during training, and tiled
   inference at the training patch size sidesteps it by construction.

There is a third, more architecture-specific point worth being honest about:
the **Critic** that shaped this Generator during adversarial training never
saw anything but `PATCH`-sized real/fake pairs. A Critic with no opinion
about a 944x944 canvas cannot have taught the Generator anything about
producing coherence at that scale -- but since the Critic is discarded after
training and inference only runs the Generator, this is really a
*training-time* scope limitation (the adversarial pressure was local, patch-
scale realism) rather than an *inference-time* correctness issue. Tiled
inference at `PATCH` size keeps the Generator operating exactly within the
scope the Critic ever trained it for.

**Net assessment:** tiling is retained here for real reasons, but the
strongest of `corrdiff_fm`'s three reasons (weights tied to an absolute grid
size) genuinely does not apply to this architecture. Do not carry that
specific claim over uncritically if you are asked to compare the two
packages' caveats side by side.

---

## Why no two-stage decomposition

`corrdiff_fm` decomposes `x = E[x|y] + (x - E[x|y])` because that
decomposition is what makes its **diffusion** problem tractable: a Stage-1
regressor (trained with L2, which is what makes it converge to the
conditional MEAN rather than the median) makes the Stage-2 residual
zero-mean, and a zero-mean residual is what gives the variance reduction
(corrdiff paper Eq. 2) that makes modelling the residual with a denoiser
easier than modelling `p(x)` directly.

None of that machinery is needed for an adversarial framework. WGAN-GP's
critic has no analogous variance-reduction argument to exploit -- it just
needs SOME sample from the Generator's output distribution to judge against a
real one. So the Generator here is trained end-to-end to output the FULL
field directly (`Network.Generator`, `lr + topo + z -> full HR precip`), with
the adversarial (Wasserstein) term supplying the "this looks like a real
downscaled field" pressure that Stage-2's denoising loss supplies in
`corrdiff_fm`. Introducing a frozen Stage-1 mean here would buy nothing while
adding exactly the two-checkpoint bookkeeping complexity `Regressor.py` exists
to manage in the other package. One network, one training script (`Train.py`),
replacing `corrdiff_fm`'s `Regressor.py` + `TrainStage2.py` split.

---

## Why the noise input (and why the SRGAN/SRDRN siblings will not have one)

WGAN-GP's identity is the Wasserstein critic + gradient penalty -- that is
orthogonal to whether the Generator is stochastic or deterministic. But this
package is meant to produce an ensemble/uncertainty product comparable to
`corrdiff_fm`'s spread output, so the Generator is given an explicit
low-dimensional spatial noise input: `z ~ N(0, I)` drawn as a
`[B, Z_CH, H/4, W/4]` tensor (`Z_CH=4` by default, `Config.Z_CH`), concatenated
to the LR input channels before the stem convolution. Holding `(lr, topo)`
fixed and resampling `z` draws an ensemble, mirroring how `corrdiff_fm`
resamples its sampler's initial noise.

**This is a deliberate design choice specific to this package, not an
inherent property of WGAN-GP.** The (planned) sibling `srgan` and `srdrn`
packages are deterministic super-resolution networks with no noise input at
all -- there is nothing in either architecture family that requires one. See
`Config.py`'s `Z_CH` docstring for the reasoning behind the specific channel
count.

**A caveat worth repeating from `Inference.py`'s module docstring:** resampling
`z` and forward-passing gives you the empirical variability the Generator
*learned* to produce, pressured by the Wasserstein objective toward matching
the true conditional distribution's marginal statistics, but with none of a
diffusion sampler's step-wise theoretical grounding in an (approximate)
reverse-time SDE/ODE. GAN-family models are well documented to be prone to
**mode collapse** in exactly this dimension: a Generator that has learned to
mostly ignore `z` (because the Critic never effectively punished it for doing
so) produces an ensemble that can look calibrated in an aggregate metric like
spread/skill while actually just repeating a near-identical field per member.
Watch `precip_std` in the inference output, and the per-fold spread/skill
ratio during training, with this specific failure mode in mind.

---

## Files

| File | |
|---|---|
| `Config.py` | Every path and constant; shares data/task constants with `corrdiff_fm`, has its own architecture/algorithm constants |
| `PrepareData.py` | Unit correction + pre-flight validation (shared infrastructure, copied from `corrdiff_fm`) |
| `Dataset.py` | NetCDF loading, normalization, k-fold splits (copied from `corrdiff_fm`, unmodified) |
| `Network.py` | `Generator` and `Critic`, plus reused topo/utility functions |
| `Tiling.py` | `TiledGenerator` -- overlapping-tile machinery, adapted (not copied) from `corrdiff_fm/Tiling.py` |
| `Train.py` | WGAN-GP training: replaces `corrdiff_fm`'s `Regressor.py` + `TrainStage2.py` with one script |
| `Inference.py` | 8 km -> 2 km cascade |
| `Evaluate.py` | Held-out scoring -> JSON, schema-compatible with `corrdiff_fm`'s |
| `Compare.py` | Pairs two JSONs into a head-to-head (copied verbatim; pure stdlib + numpy, model-agnostic) |

---

## Why WGAN-GP specifically, and why these losses

**Critic, not "discriminator".** The network in `Network.Critic` has an
unbounded scalar output and NO sigmoid. In the original (Arjovsky et al.
2017) WGAN formulation its value approximates a Kantorovich potential -- an
estimate of (a multiple of) the Wasserstein-1 distance between the real and
generated distributions -- not a real/fake probability. This matters
practically, not just terminologically: a sigmoid-bounded discriminator's
gradient vanishes as it saturates towards confident correct classification,
which is exactly the failure mode (vanishing gradients / mode collapse) the
Wasserstein reformulation exists to fix.

**Gradient penalty, not weight clipping.** Arjovsky's original WGAN enforces
the 1-Lipschitz constraint the Kantorovich-Rubinstein duality requires by
clipping the critic's weights to a small range, which the WGAN-GP paper shows
tends to push the critic towards learning overly simple functions (clipping
biases the critic towards its extremes) and to gradient explosion/vanishing
depending on the clipping threshold. Gulrajani's fix, implemented in
`Train.py`'s `gradient_penalty()`, instead penalizes the critic directly for
having a gradient norm away from 1 at random points between real and fake
samples:

```
x_hat = eps*real + (1-eps)*fake,  eps ~ U(0,1) per sample
GP = E[(||grad_{x_hat} D(x_hat)||_2 - 1)^2]
L_D = E[D(fake)] - E[D(real)] + LAMBDA_GP * GP        (LAMBDA_GP = 10, paper value)
```

**`n_critic = 5` critic updates per generator update** (`Config.N_CRITIC`,
paper default): the critic needs to stay close to the optimal Kantorovich
potential for the current generator before the generator's gradient (which
depends on the critic being close to optimal) is trustworthy.

**Adam betas = (0.0, 0.9), for BOTH networks -- not Adam's own default
(0.9, 0.999), and not the original WGAN's RMSProp.** The gradient penalty
differentiates the critic a SECOND time (`torch.autograd.grad(...,
create_graph=True)` inside a term that is itself later `.backward()`'d), and
Gulrajani et al. Sec 4 report training instability with Adam's default
momentum once this second-order term is added; lowering both betas removes
that stale momentum. This is not what the original (weight-clipped) WGAN
uses either -- the two papers solve different critic-stabilization problems
with different optimizers, and the fix for one does not transfer to the
other automatically. See `Config.py`'s `BETAS` docstring for the full
argument.

**Content (L1) loss term, `LAMBDA_CONTENT * L1(fake, target)`, added to the
Generator's loss -- NOT part of the original (unconditional) WGAN-GP.** In
Gulrajani's unconditional setting there is no "correct" target for a given
`z` to hit; the adversarial term alone is the whole training signal because
only the marginal output distribution needs to look real. This task is
CONDITIONAL (`lr -> a SPECIFIC paired HR field`), and `E[D(G(lr,topo,z))]`
by itself only pressures the marginal distribution of outputs, with no
mechanism tying a PARTICULAR `G(lr)` to the PARTICULAR truth paired with that
`lr`. Conditioning the Critic on `(lr, topo)` (see `Network.Critic`'s
docstring) recovers part of this correspondence, but only through the
Critic's own capacity and training dynamics -- a much weaker and noisier
constraint, especially early in training, than a direct pixel-space anchor.
L1 (not L2) for the same reason `corrdiff_fm/Regressor.py` gives for NOT using
L2 on ITS Stage 2: here we want the Generator to produce plausible individual
REALIZATIONS (the adversarial term is what pushes towards realizations rather
than an averaged mean), and L2 pixel-wise pulls every realization back toward
the conditional mean -- precisely the blurring failure mode adversarial
training exists to avoid. `LAMBDA_CONTENT = 50.0` is a judgment call,
justified in `Config.py`'s docstring against the expected relative magnitude
of the two loss terms at initialization (sane range 10-100; see that
docstring for the reasoning against the extremes).

**No BatchNorm anywhere in the Critic**, `GroupNorm(1 group)` (a LayerNorm
equivalent -- normalizes each sample independently over all its own channels
and spatial positions) used instead where normalization is wanted, first
layer un-normalized. Standard, load-bearing WGAN-GP guidance (paper Sec 4):
BatchNorm computes statistics ACROSS the samples in a minibatch, which makes
the per-sample gradient-penalty interpretation (each interpolated sample's
own gradient norm close to 1) ill-defined, since one sample's critic output
would then depend on every other sample sharing its batch.

---

## Outputs

One NetCDF per held-out test year, in `outputs/downscaled_2km_wgan_gp/`:

- `precip_mean` -- ensemble mean (over resampled `z`), mm/day
- `precip_std` -- ensemble spread (uncertainty, from resampling `z`), mm/day
- `precip_members` -- individual members (with `--save-members`)

There is no `precip_stage1_mean` analogue (no Stage 1 exists in this
package). Each file records the algorithm, tile size and ensemble size in its
global attributes.

---

## Caveats the code cannot fix

- An 8 km RCM field is close to, but not exactly, the area mean of a 2 km
  field. Mitigated by `Train.py`'s `LR_AUG` jitter during training; not
  eliminated. Identical rationale to `corrdiff_fm`.
- Sub-grid precipitation variance is **not** scale-invariant, and unlike
  `corrdiff_fm` there is **no exposed rescaling knob** (no analogue of
  `--res-std-scale`): the Generator's sensitivity to `z` is baked into its
  trained weights rather than living in a separate multiplicative scalar, so
  there is nothing equivalent to expose. If the 8->2 km ensemble spread turns
  out to be miscalibrated, the only recourse is retraining or a downstream
  post-hoc calibration step, not a CLI flag.
- Normalization statistics are the 8 km ones. Fine fields have heavier tails,
  so expect the extreme tail to be conservative.
- The ensemble's calibration has **weaker theoretical grounding** than
  `corrdiff_fm`'s diffusion ensemble -- see "Why the noise input" above.
  Watch for mode collapse (see sanity checks below), not just spread/skill.
- Comparing this package's `nfe` (recorded as 1, a single feed-forward pass
  per ensemble member) against a diffusion arm's `nfe` (`2*steps-1` network
  evaluations per member) via `Compare.py` is not a meaningful
  apples-to-apples compute comparison, even though `Compare.py` will (and
  should) flag the mismatch. The two families spend their compute budgets in
  fundamentally different places.

Treat the 2 km output as a physically plausible refinement, not a validated
product, until you have 2 km ground truth to score it against.

---

## Sanity checks worth watching

- **Wasserstein distance estimate (`W-est` in the training log,
  `E[D(real)] - E[D(fake)]`) should trend upward and then plateau**, not
  oscillate wildly or diverge. A steadily INCREASING estimate mid-training
  that never plateaus, or one that goes strongly negative, usually means the
  Generator and Critic have fallen out of the balance the `n_critic` ratio is
  supposed to maintain.
- **Gradient penalty (`GP` in the training log) should stay small (order
  0.01-1), not grow.** A GP that climbs steadily means the Critic is
  violating its intended 1-Lipschitz constraint somewhere in the interpolated
  region, and `LAMBDA_GP` is not successfully reining it in -- often a sign
  the Critic's capacity or learning rate needs revisiting before anything
  else.
- **The Critic should not saturate.** If `real_score` and `fake_score`
  rapidly separate to very large magnitudes early in training and then barely
  move, the Critic has likely found a trivial way to tell real from fake
  (e.g. a normalization artefact) rather than learning the actual
  distributional structure, and the Generator's adversarial gradient
  (`-fake_score.mean()`) will be uninformative even though it is technically
  nonzero.
- **CRPS, not the training losses**, is what model selection uses (see
  `Train.py`'s module docstring for why: `L_D` and `L_G` are both moving
  targets in a two-player game, and neither is a proper scoring rule against
  ground truth). Watch it fall over training, and watch spread/skill
  alongside it -- a falling CRPS with collapsing spread/skill is the mode-
  collapse signature described above, not a genuine improvement.
- **Content loss (`L1` term in the `G` breakdown) should fall steadily**,
  especially early on when it dominates the Generator's loss by construction
  (see `LAMBDA_CONTENT`'s docstring). If it plateaus at a high value while
  the adversarial term swings around, the pixel anchor is not doing its job
  and the Generator may be exploiting the Critic instead of learning the
  actual conditional mapping.
