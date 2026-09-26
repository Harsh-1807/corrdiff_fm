# SRGAN precipitation downscaling package

Single-stage, deterministic, adversarially-trained super-resolution for 32 km
&rarr; 8 km training, applied as a cascade to generate 2 km fields. Implements
Ledig et al. 2017 CVPR, "Photo-Realistic Single Image Super-Resolution Using a
Generative Adversarial Network" (arXiv:1609.04802).

**This is a sibling package to `corrdiff_fm`**, not an independent project:
same NetCDF sources, same 8 km training grid / 2 km inference cascade, same
k-fold split, same evaluation seed, same output JSON schema. Everything that
differs between the two packages is the ARCHITECTURE FAMILY itself --
corrdiff_fm's two-stage deterministic-mean + residual-diffusion decomposition
versus this package's single generator trained end to end against a
discriminator. That is deliberate: a comparison between architecture families
is only informative if everything else is held fixed.

---

## Run order

| # | Command | GPUs | What it does |
|---|---|---|---|
| 0 | `qsub job_0_prepare.sh` | 0 | Fix precip units at the source, run pre-flight checks |
| 1 | `qsub job_1_train.sh` | 2 | **Both** training phases: MSE pretrain, then GAN fine-tune |
| 2 | `qsub job_2_evaluate.sh` | 1 | Score on held-out test years &rarr; JSON |
| 3 | `qsub -J 0-7 job_3_inference.sh` | 1&times;8 | 8 km &rarr; 2 km cascade, sharded by year |

Then, once this package and any other (e.g. `corrdiff_fm`) have both finished
step 2:

```bash
python Compare.py ../corrdiff_fm/outputs/eval_edm.json outputs/eval_srgan.json
```

**Step 0 only needs running ONCE across every package in `/home/ylale/extras/h/`,
not once per package.** `corrdiff_fm`'s `edm`/`fm` arms, this `srgan` package,
and the `wgan_gp`/`srdrn` siblings all train on the exact same corrected
precipitation file and the exact same static orography. Run `PrepareData.py`
from any one package's directory; every package's `Config.HR_FILES` already
points at the same corrected file on disk. Running it again here is harmless
(it detects the corrected file already exists and only re-validates) but is
never required if you already ran it elsewhere.

**Step 1 is a single script covering both training phases**, unlike
corrdiff_fm's separate Stage-1/Stage-2 jobs -- there is no frozen Stage-1
model here to train once and share; see "Why SRGAN specifically" below for
why this architecture has no such split.

---

## Before you start

Edit **`Config.py`** -- it holds every path and every cross-file constant:

```python
ROOT = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh"
GENERATOR_CKPT_DIR = "checkpoints/generator_srgan/"       # this package's OWN dir
PATCH = 128                                               # see below
PRETRAIN_EPOCHS = 600     # Phase 1 budget
GAN_EPOCHS = 400          # Phase 2 budget
ADV_WEIGHT = 1e-3         # Ledig et al. Eq. 3 -- do not casually retune
USE_VGG_PERCEPTUAL = False   # see "Why VGG-vs-not" below
```

Also edit the two `conda` lines at the top of each `job_*.sh`.

`ROOT`, `HR_FILES`, `ORO_8KM`, `ORO_2KM`, `VAR_MAP`, `DS_FACTOR`, `PRECIP_CH`,
`IN_CH_LR`, `PATCH`, `PATCH_OVERLAP`, `KFOLD_K`, `VAL_RATIO`, `SEED`,
`EVAL_SEED` all carry the SAME VALUES as `corrdiff_fm/Config.py` (copied, not
imported, so this package stands alone). If you change one of these here,
change it in every sibling package too, or the architecture comparison stops
being apples-to-apples.

### Does the PATCH/tiling caveat still apply here? -- an honest answer

Yes, but **weaker** than in corrdiff_fm, and for a partly different reason.
Read `Tiling.py`'s module docstring for the full argument; the short version:

corrdiff_fm's Stage-2 U-Net contains `SpectralConv2d` (ties learned weights to
*absolute* FFT mode indices) and `SelfAttn2d` (silently switches from exact to
pooled attention past a token budget) -- both make that network's behaviour a
genuinely different OPERATOR at a different grid size, not the same operator
applied to more pixels. **This package's `Generator` has neither block.** It
is entirely convolutional with a fixed, finite receptive field, and its
BatchNorm layers use FROZEN RUNNING STATISTICS in eval mode -- feeding it a
much larger 2 km canvas does not change what those statistics do to the
affine transform per pixel. So handing this generator a full, untiled 2 km
domain would **not** silently swap in a different learned function the way it
would for corrdiff_fm's Stage 2.

What tiling still fixes here: (1) finite-receptive-field **boundary effects**
-- every conv pads at its own edges, and a tile boundary is an artificial edge
the network rarely saw during training (training crops are drawn from deep
inside the domain far more often than at its actual physical edge), which
produces visible seams in a naive quilt of independently-generated tiles; (2)
**memory** -- a 2 km domain is `DS_FACTOR^2 = 16x` the pixel count of the 8 km
training domain, and 16 residual blocks over the whole thing in one forward
pass may simply not fit on one GPU. Both are real reasons to tile; neither is
"the network computes a fundamentally different function," which is the
stronger claim corrdiff_fm has to make about its own architecture. Say this
plainly rather than reusing corrdiff_fm's language unchanged -- it would
overstate this package's actual resolution-transfer risk.

---

## Files

| File | |
|---|---|
| `Config.py` | Every path and cross-file constant |
| `PrepareData.py` | Unit correction + pre-flight validation (**shared across all packages**) |
| `Dataset.py` | NetCDF loading, normalization, k-fold splits (**near-verbatim from corrdiff_fm**) |
| `Network.py` | `Generator` (SRResNet backbone), `Discriminator` (VGG-style), `ContentLoss`, `VGGFeatureExtractor`, plus reused topography utilities |
| `Tiling.py` | `TiledGenerator` -- one-shot tiled inference wrapper (adapted, simplified, from corrdiff_fm's `Tiling.py`) |
| `Train.py` | Two-phase training: MSE pretrain, then adversarial fine-tune, inside the same k-fold loop structure as corrdiff_fm |
| `Inference.py` | 8 km &rarr; 2 km cascade |
| `Evaluate.py` | Held-out scoring &rarr; JSON, same schema as corrdiff_fm's |
| `Compare.py` | Pairs two JSONs into a head-to-head (**verbatim from corrdiff_fm**) |

---

## Why SRGAN specifically, and why these particular numbers

### The task framing -- and why it is genuinely different from corrdiff_fm

corrdiff_fm decomposes `precip = E[precip|LR,topo] + residual`, trains a
Stage-1 regressor on the mean and a Stage-2 diffusion model on the residual,
and composes them at inference. That decomposition exists because it makes
the residual's distribution close to zero-mean and low-variance, which is
what makes a *diffusion* model's job easier (see corrdiff_fm's README/
Regressor.py docstring). **SRGAN's own paper does not use any such
decomposition** -- Fig. 4's generator maps input to output directly, trained
once, end to end. Reproducing that decomposition here would not be "a more
careful SRGAN" -- it would be a different, uncited architecture wearing
SRGAN's name. So this package has ONE generator, `Generator`, mapping (LR
climate stack, HR topo) &rarr; HR precip directly, and no Stage-1 checkpoint
at all.

### Why the generator has no noise input

Per the paper, SRGAN's generator is **deterministic**: nothing in Fig. 4
concatenates a latent anywhere. The "generative" character of SRGAN comes
entirely from the *adversarial training signal* reshaping the distribution of
G's outputs (over the training set) toward one indistinguishable from real HR
fields by D -- not from sampling a per-example latent at inference. This
contrasts sharply with the sibling `wgan_gp` package's generator, which DOES
take a noise input, for a documented reason specific to that package (it
needs a controllable source of sample-to-sample variability to produce an
ensemble). Nothing analogous exists here. The direct consequence: this
package produces a single deterministic field, with no `precip_std` /
`precip_members` at inference and CRPS degenerating to MAE at evaluation (see
`Evaluate.py`'s docstring) -- not a bug, an architectural fact.

### The generator architecture (Ledig et al. Fig. 4, faithfully)

9x9 stem conv + PReLU &rarr; 16 residual blocks (conv3x3-BN-PReLU-conv3x3-BN,
identity skip) &rarr; conv3x3+BN &rarr; **global skip** adding back the post-stem
feature map (the paper's "elementwise sum" before upsampling -- easy to drop
by accident, and doing so measurably hurts convergence since the residual
stack would otherwise have to re-learn an identity path through 16 stacked BN
layers) &rarr; 2 PixelShuffle(2) blocks (exactly `2**n == DS_FACTOR=4`, the
paper's own 4x configuration -- no adaptation needed, unlike the `srdrn`
sibling package) &rarr; topography fusion (concatenate the upsampled trunk
features with a small conv+GroupNorm+SiLU topo branch, mirroring
corrdiff_fm's `CorrDiffRegressor`) &rarr; 9x9 output conv to 1 channel
(zero-initialized, since "predict zero" is already correctly scaled in this
normalized log1p space).

### The discriminator architecture (Ledig et al. Fig. 4, faithfully)

8 conv layers, 3x3 kernels, stride alternating 1/2, channels doubling
64-64-128-128-256-256-512-512, LeakyReLU(0.2) throughout, BatchNorm on every
layer except the first. Ends in the paper's literal dense(1024)-LeakyReLU-
dense(1)-sigmoid head (safe here because D is only ever invoked on
fixed-size `PATCH` training crops, never at an arbitrary inference tile size
the way G is -- see `Network.Discriminator`'s docstring). Output: a
real-valued probability the input is a real HR field, trained with **standard
binary cross-entropy**, real label 1, fake label 0 -- the paper's original,
non-Wasserstein formulation. No critic, no Lipschitz constraint, no gradient
penalty; that machinery belongs to the `wgan_gp` sibling package.

D is **conditioned** on the same upsampled-LR + topo context G sees
(concatenated as extra input channels) -- a deliberate, necessary deviation
from the literal paper, whose discriminator is unconditional. Natural-image
super-resolution has essentially one plausible LR&rarr;HR mapping direction to
judge; precipitation downscaling does not (the same 8 km mean can correspond
to very different, and very differently *plausible*, 2 km fields depending on
topography), so an unconditional D would learn "is this a plausible field IN
GENERAL" rather than "given THIS context" -- the same reasoning the
`wgan_gp` sibling package's critic uses.

### The two-phase training protocol (Ledig et al. Sec. 3.2)

**Phase 1 (`PRETRAIN_EPOCHS`, default 600):** train G alone on pixel MSE loss,
matching corrdiff_fm's Stage-1 rationale for why MSE targets the conditional
mean. The paper trains its "SRResNet" baseline for a large number of updates
before ever touching the adversarial loss specifically to avoid an
adversarial signal computed against an equally-untrained discriminator
dominating gradients from a randomly-initialized generator.

**Phase 2 (`GAN_EPOCHS`, default 400):** fine-tune the SAME generator weights
with `content_loss + 1e-3 * adversarial_loss` (Ledig et al. Eq. 3). **The
`1e-3` is not a casual knob.** `adversarial_loss`'s gradient magnitude is set
by how confidently D currently rejects G's output -- a property of D's
current state, not of how far G's output is from the target in any content
sense. Right after Phase 1, D has never seen G's now-much-better outputs and
can be very confidently correct that they are fake, so the un-weighted
adversarial gradient can dominate the content term by orders of magnitude
exactly in the epochs where you most need content fidelity preserved. `1e-3`
keeps content loss dominant throughout, letting the adversarial term act as a
texture/realism regularizer on top rather than a competing objective.

### Why VGG-vs-not, specifically

`content_loss` has two mutually exclusive implementations
(`Config.USE_VGG_PERCEPTUAL`):

- **`False` (default):** pixel L1 + a Sobel-gradient-magnitude L1 term.
  Physically motivated: precipitation fronts and convective cells ARE sharp
  spatial gradients, the physically meaningful high-frequency structure this
  whole exercise exists to recover, and rewarding gradient-magnitude fidelity
  directly penalizes the blur a pure pixel loss under-penalizes -- without
  any pretrained network in the loop.
- **`True`:** literal paper fidelity. A frozen ImageNet-pretrained VGG19,
  truncated at conv5_4 **pre-activation** (the paper's "VGG54", chosen over
  the commonly-used shallower "VGG22" alternative because it is the paper's
  headline configuration), replicate the single precipitation channel to 3,
  MSE on those features, plus a small L1 pixel anchor (VGG features are
  approximately contrast/scale-invariant by design, so without an anchor
  there is very little gradient pushing the predicted ABSOLUTE intensity
  toward the target's). **Domain-mismatch caveat, stated plainly:** VGG19 was
  trained on ImageNet photographs. A single-channel, log1p-normalized
  precipitation field is not a natural image in any sense VGG's filters were
  built to exploit, and there is no reason its edge/texture/color-opponent
  detectors pick out anything physically meaningful in a rainfall field. This
  path exists for readers who want literal fidelity to the paper's recipe; it
  is not claimed to be the physically better option, and the default
  Sobel-gradient path is the recommended one for this data modality. If
  `torchvision`'s pretrained VGG19 weights cannot be downloaded (no internet
  on a compute node, a firewalled cluster), `Network.VGGFeatureExtractor`
  fails loudly with a message pointing you at `USE_VGG_PERCEPTUAL = False`
  rather than silently substituting a different loss.

### Optimizer settings, and the contrast with `wgan_gp`

Adam, `lr=1e-4`, `betas=(0.9, 0.999)` -- the paper's exact setting, for both G
and D, in both phases. Contrast with the `wgan_gp` sibling package's very
different `betas=(0.0, 0.9)`: THAT package's critic loss includes a
gradient-penalty term whose interaction with Adam's first-moment (heavy-ball)
momentum is known to destabilize training unless `beta1` is reduced toward 0
(Gulrajani et al. 2017, Sec. 4). This package's discriminator has no gradient
penalty -- it is a plain sigmoid+BCE classifier -- so there is no such
interaction, and the paper's own, higher-momentum Adam setting is correct
here.

`EMA` on G is kept for checkpointing/eval, following corrdiff_fm's
convention. **This is this package's own addition, not the paper's** -- it
gives a lower-variance checkpoint to evaluate/deploy than the raw training
weights, particularly valuable once the adversarial loss is active and the
raw weights get noisier step to step.

### Model selection

Since there is no ensemble here, CRPS collapses exactly to MAE (see
`Evaluate.py`). That degeneracy has a real consequence for **Phase-2 model
selection specifically**: RMSE/MAE are minimized by the conditional mean, and
the entire point of the adversarial term is to push the generator's output
AWAY from a blurry conditional-mean-like solution -- a strictly harder
criterion that a plain RMSE-minimizer resists. Selecting purely on RMSE during
Phase 2 would systematically prefer checkpoints that partially undid the GAN
fine-tuning. So Phase-2 selection uses a composite score,
`rmse * (1 + SPECTRUM_SELECTION_WEIGHT * |spectrum_logratio|)`, where
`spectrum_logratio` is the same high-frequency-spectrum metric
`Evaluate.py` reports -- specifically the one thing a GAN buys you over a
plain regressor, and specifically the thing RMSE alone cannot see. Phase-1
selection is plain validation RMSE (there is no adversarial term yet to fight
against).

---

## Outputs

One NetCDF per held-out test year, in `outputs/downscaled_2km_srgan/`:

- `precip_mean` -- the generator's single deterministic output, mm/day.

**There is no `precip_std` and no `precip_members`.** This is not an omission:
this package's generator has no noise input (see above), so there is nothing
to sample repeatedly and nothing to compute a spread over. The variable is
still named `precip_mean` only for output-schema parity with corrdiff_fm; each
file's global attributes state this explicitly so a reader with only the
NetCDF, not this README, cannot mistake the missing variables for a bug.

Each file records the architecture, tile size, and fold in its global
attributes.

---

## Caveats the code cannot fix

- An 8 km RCM field is close to, but not exactly, the area mean of a 2 km
  field. Mitigated by the `LR_AUG` jitter during training; not eliminated.
- Sub-grid precipitation variance is **not** scale-invariant. Unlike
  corrdiff_fm, there is no `--res-std-scale`-style knob here to correct for
  this post hoc -- the generator's output IS the field, not a residual added
  to a separately-scaled baseline. Any systematic amplitude bias observed at
  2 km is evidence of this, not a bug to patch with a nonexistent flag.
- Normalization statistics are the 8 km ones. Fine fields have heavier tails,
  so expect the extreme tail to be conservative.
- The Phase-1-only checkpoint (`Generator_{fold}_pretrain.pth`) is kept for
  ablation/diagnostics (e.g. to see exactly what the adversarial fine-tuning
  changed), but is NOT the deliverable model -- `Evaluate.py`/`Inference.py`
  always load the Phase-2 (`_best.pth`) checkpoint by default and print a
  warning if asked to score a `phase != "gan"` checkpoint.

Treat the 2 km output as a physically plausible refinement, not a validated
product, until you have 2 km ground truth to score it against.

---

## Sanity checks worth watching

- **Discriminator accuracy staying near 50%, not collapsing to 0%/100%.**
  Printed every Phase-2 epoch as `D acc`. A sustained ~100% means D has "won"
  and G's adversarial gradient has likely collapsed to near-zero (the
  training loop will keep running, but Phase 2 is no longer doing anything
  useful beyond Phase 1). A sustained ~0% means G has "won" (or D has stopped
  learning) and the adversarial term is equally uninformative from the other
  direction. Neither extreme is fatal on its own for a few epochs, but a
  SUSTAINED extreme means Phase 2 is not doing its job.
- **Content loss not exploding when Phase 2 starts.** It should start close to
  where Phase 1 left off (plus whatever the Sobel/VGG term adds on top of
  plain MSE) and drift only gradually as the adversarial term reshapes the
  output. A sharp jump right at the phase transition usually means the
  `1e-3` weighting is not doing its job (check `ADV_WEIGHT` was not
  accidentally changed) or that Phase-1 weights were not loaded correctly
  into Phase 2.
- **Phase-1 val RMSE curve** should look like any ordinary supervised
  regression training curve (monotone-ish decrease, early-stopped on
  patience) -- if it does not converge before Phase 2 begins, Phase 2 is
  fine-tuning from a bad starting point and everything downstream is
  compromised.
- **Phase-2 `spectrum_logratio`** should visibly IMPROVE (move toward 0)
  relative to what the Phase-1-only checkpoint achieves on the same
  validation set -- that is the entire empirical claim the adversarial phase
  is making. If it does not, something in the adversarial loop is not
  contributing (check the discriminator-accuracy sanity check above first).
