# -*- coding: utf-8 -*-
"""
Network.py -- Generator and Critic for adversarial (WGAN-GP) precipitation
downscaling.
=============================================================================
Single-stage architecture: one Generator maps LR coarse fields + HR
topography (+ a spatial noise draw) directly to HR precipitation. There is no
Stage-1/Stage-2 split here -- see the package README, "Why no two-stage
decomposition", for why that split is a diffusion-specific device that an
adversarial framework does not need.

Contract with Dataset.py (identical to corrdiff_fm, NOT modified here):
  * hr  : [4, H, W]  huss, mslp, tas, precip -- ALL already per-channel normalized
                     (precip additionally log1p'd before normalization)
  * lr  : [4, H/4, W/4] = F.avg_pool2d(hr, 4)          (normalized space)
  * oro : [1, H, W]  RAW orography in metres, NaN/fill -> 0.0

REUSED VERBATIM FROM corrdiff_fm/Network.py (these are generic, physically-
grounded utilities, not diffusion machinery -- see this package's README,
"What was reused near-verbatim"):
  _g, expand_topo (+ TOPO_CHANNELS/TOPO_ELEV_CLIP/TOPO_SLOPE_Q/LAND_THRESHOLD_M/
  _sobel_kernels/_robust_scale), SEBlock, PixelShuffleUp, denorm_precip_mmday.

NOT REUSED, WRITTEN FRESH FOR THIS PACKAGE: ResConv/FiLM/SelfAttn2d/
SpectralConv2d/DilatedBottleneck (corrdiff_fm's diffusion-era building blocks)
are replaced below by a single plain residual block with no timestep
embedding and no per-level topo FiLM, because:
  * There is no timestep/noise-level input to condition on -- the Generator
    is a single feed-forward pass, not an iterative denoiser, so every block
    in TrainStage2.py's UNet that exists to mix in a sigma/t embedding has no
    counterpart here.
  * Per-level FiLM conditioning on topography is not required for this
    architecture: topo is concatenated ONCE, after the last upsample stage
    (mirroring CorrDiffRegressor's own `fuse` step), which is sufficient
    because by that point the tensor is already at full HR resolution and a
    single fusion block has direct spatial access to every topo channel.
  * Self-attention (SelfAttn2d) and spectral convolution (SpectralConv2d) are
    explicitly NOT used, per the task brief: a plain residual-conv generator
    is the standard WGAN-GP-family choice, and -- more importantly for this
    codebase's running theme of resolution transfer -- both of those blocks
    are exactly the two components that make corrdiff_fm's networks NOT
    resolution-agnostic (SpectralConv2d ties weights to absolute FFT mode
    indices; SelfAttn2d switches to pooled attention past a token budget).
    Leaving them out means THIS Generator has no analogous absolute-grid-size
    dependency baked into its weights. See "Does the PATCH/tiling caveat
    still apply here?" in the README for the (weaker, but not zero) residual
    caveat that remains: GroupNorm statistics and what the Critic ever saw
    during training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 0. Small utilities (verbatim from corrdiff_fm/Network.py)
# ----------------------------------------------------------------------------

def _g(ch, mx=32, min_per_group=8):
    """GroupNorm group count: largest divisor <= mx keeping >= min_per_group ch/group.

    Fatter groups than a naive "largest divisor <= 32" would give -- e.g. 24
    groups of 2 channels for ch=48 normalizes over almost nothing per group
    and is close to InstanceNorm, which is noticeably noisier.
    """
    best = 1
    for g in range(1, min(mx, ch) + 1):
        if ch % g == 0 and ch // g >= min_per_group:
            best = g
    return best


# ----------------------------------------------------------------------------
# 1. Topography featurizer (verbatim from corrdiff_fm/Network.py)
# ----------------------------------------------------------------------------
# Channel layout of expand_topo() output (all in ~[-1, 1]):
#   0 elevation (standardized, clipped)
#   1 slope magnitude (robustly scaled)
#   2 sin(aspect) * slope_confidence
#   3 cos(aspect) * slope_confidence
#   4 curvature / Laplacian (robustly scaled)   <- orographic uplift proxy
#   5 land-sea mask (+1 land, -1 water)         <- critical for coastal precip
#   6 y coordinate
#   7 x coordinate
TOPO_CHANNELS = 8

TOPO_ELEV_CLIP = 3.0       # in standard deviations of the domain elevation
TOPO_SLOPE_Q = 0.99        # robust quantile used to scale slope / curvature
LAND_THRESHOLD_M = 0.5     # metres; Dataset.load_oro maps ocean/fill to 0.0


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 8.0
    lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    return torch.stack([kx, ky, lap]).unsqueeze(1).to(device=device, dtype=dtype)


def _robust_scale(x, q=TOPO_SLOPE_Q):
    """Per-sample division by the q-quantile of |x| (deterministic, self-calibrating)."""
    B = x.shape[0]
    flat = x.reshape(B, -1).abs().float()
    s = torch.quantile(flat, q, dim=1).view(B, 1, 1, 1).clamp_min(1e-6)
    return x / s


@torch.no_grad()
def expand_topo(oro):
    """Raw orography -> rich normalized topographic conditioning tensor.

    Args:  oro [B,1,H,W] or [B,H,W] raw metres (as produced by Dataset.load_oro)
    Returns: [B, TOPO_CHANNELS, H, W] float32 in ~[-1, 1]

    Notes
    -----
    * Uses replicate padding, so the domain border does not look like a cliff
      (zero padding produces spurious slope/curvature rings otherwise).
    * Aspect is encoded as (sin, cos) rather than angle/pi: angle/pi has a hard
      discontinuity at +/-pi, which a CNN cannot represent smoothly.
    * Aspect is damped by slope confidence so flat/ocean pixels get 0 rather
      than an arbitrary direction from numerical noise.
    * Run this ONCE per domain (orography is static) and expand along batch.
    """
    if oro.dim() == 3:
        oro = oro.unsqueeze(1)
    e_raw = oro[:, :1].float()
    B, _, H, W = e_raw.shape

    land = torch.where(e_raw > LAND_THRESHOLD_M, 1.0, -1.0)

    m = e_raw.mean(dim=(2, 3), keepdim=True)
    s = e_raw.std(dim=(2, 3), keepdim=True).clamp_min(1e-3)
    e = (e_raw - m) / s

    k = _sobel_kernels(e.device, e.dtype)
    grad = F.conv2d(F.pad(e, (1, 1, 1, 1), mode="replicate"), k)
    dx, dy, lap = grad[:, 0:1], grad[:, 1:2], grad[:, 2:3]

    slope = torch.sqrt(dx * dx + dy * dy + 1e-12)
    slope_s = _robust_scale(slope)
    conf = slope_s.clamp(0.0, 1.0)                 # 0 on flat ground, 1 on steep
    slope_c = conf * 2.0 - 1.0

    ang = torch.atan2(dy, dx)
    sin_a, cos_a = torch.sin(ang) * conf, torch.cos(ang) * conf

    curv = _robust_scale(lap).clamp(-1.0, 1.0)
    elev_c = (e / TOPO_ELEV_CLIP).clamp(-1.0, 1.0)

    yg = torch.linspace(-1, 1, H, device=e.device).view(1, 1, H, 1).expand(B, 1, H, W)
    xg = torch.linspace(-1, 1, W, device=e.device).view(1, 1, 1, W).expand(B, 1, H, W)

    return torch.cat([elev_c, slope_c, sin_a, cos_a, curv, land, yg, xg], dim=1)


# ----------------------------------------------------------------------------
# 2. Building blocks
# ----------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-excite channel gate (verbatim from corrdiff_fm/Network.py)."""

    def __init__(self, ch, r=8, drop=0.0):
        super().__init__()
        h = max(4, ch // r)
        self.f = nn.Sequential(
            nn.Linear(ch, h, bias=False), nn.SiLU(), nn.Dropout(drop),
            nn.Linear(h, ch, bias=False), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.f(x.mean(dim=(2, 3)).view(b, c)).view(b, c, 1, 1)
        return x * w


class PixelShuffleUp(nn.Module):
    """x2 sub-pixel upsampling with ICNR-ish init (conv then shuffle).

    Verbatim from corrdiff_fm/Network.py. Two of these compose to the x4
    (DS_FACTOR) upsample, exactly like CorrDiffRegressor's decoder.
    """

    def __init__(self, ic, oc, drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc * 4, 3, padding=1)
        self.shuf = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(_g(oc), oc)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        return self.drop(F.silu(self.norm(self.shuf(self.conv(x)))))


class GenResBlock(nn.Module):
    """Pre-norm residual conv block for the Generator.

    Deliberately simpler than corrdiff_fm's ResConv: no timestep embedding
    (this is not an iterative denoiser) and no per-level FiLM (topo enters
    once, after the last upsample -- see module docstring).
    """

    def __init__(self, ic, oc, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(_g(ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.n2 = nn.GroupNorm(_g(oc), oc)
        self.drop = nn.Dropout2d(dropout)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.se = SEBlock(oc, drop=dropout * 0.5)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.drop(self.n2(h))))
        return self.skip(x) + self.se(h)


# ----------------------------------------------------------------------------
# 3. GENERATOR
# ----------------------------------------------------------------------------

class Generator(nn.Module):
    """(LR 4ch normalized, concatenated with spatial noise z) + HR topo
    -> HR precip mean (1ch, normalized log1p space).

    Design lineage: this is CorrDiffRegressor's good ideas -- trunk work at LR
    resolution, PixelShuffleUp x2 for the x4 upsample, zero-init output layer,
    a global LR-precip skip connection -- with the diffusion-era conditioning
    machinery (per-level FiLM, self-attention, spectral convolution) removed,
    and a noise input added. See the module and README docstrings for why
    each of those choices was made.

    len(channel_mult) - 1 must equal log2(ds_factor), exactly as in
    CorrDiffRegressor, for the same reason: two PixelShuffleUp stages (x2 each)
    must compose to exactly x4.
    """

    def __init__(self, lr_channels=4, z_channels=4, out_channels=1,
                 base_channels=96, channel_mult=(1, 2, 4), num_blocks=2,
                 topo_channels=TOPO_CHANNELS, dropout=0.10, ds_factor=4,
                 precip_lr_ch=3, use_lr_skip=True):
        super().__init__()
        chs = [base_channels * m for m in channel_mult]
        n_up = len(chs) - 1
        assert 2 ** n_up == ds_factor, (
            f"len(channel_mult)-1 ({n_up}) must satisfy 2**n == ds_factor ({ds_factor})")

        self.z_channels = z_channels
        self.ds_factor = ds_factor
        self.precip_lr_ch = precip_lr_ch
        self.use_lr_skip = use_lr_skip
        self.topo_channels = topo_channels

        lr_ch = chs[-1]
        self.stem = nn.Conv2d(lr_channels + z_channels, lr_ch, 3, padding=1)
        self.trunk = nn.ModuleList([GenResBlock(lr_ch, lr_ch, dropout) for _ in range(num_blocks)])

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        c = lr_ch
        for oc in reversed(chs[:-1]):
            self.ups.append(PixelShuffleUp(c, oc, dropout))
            self.up_blocks.append(GenResBlock(oc, oc, dropout))
            c = oc

        # Topo enters ONCE here, after the Generator is already at full HR
        # resolution -- mirrors CorrDiffRegressor's `fuse` step (see module
        # docstring for why per-level FiLM is not needed).
        self.topo_stem = nn.Sequential(
            nn.Conv2d(topo_channels, c, 3, padding=1),
            nn.GroupNorm(_g(c), c), nn.SiLU(),
        )
        self.fuse = GenResBlock(c * 2, c, dropout)
        self.hr_blocks = nn.ModuleList([GenResBlock(c, c, dropout) for _ in range(num_blocks)])

        self.out = nn.Sequential(
            nn.GroupNorm(_g(c), c), nn.SiLU(), nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(c, out_channels, 3, padding=1),
        )
        # Zero-init output: the Generator starts by predicting exactly the
        # bilinearly-upsampled LR precip skip (see forward()) and a residual
        # of identically zero, so training starts from a sane, bias-free
        # baseline rather than random noise.
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def sample_noise(self, lr, generator=None):
        """Draw z ~ N(0, I) at LR resolution, matching `lr`'s batch/device/dtype.

        Exposed as a separate method (rather than only inlined in forward())
        so Train.py / Inference.py can draw a FIXED lr/topo pair and resample
        z repeatedly to build an ensemble, exactly mirroring how the
        diffusion siblings resample their sampler's initial noise.
        """
        B, _, h, w = lr.shape
        return torch.randn(B, self.z_channels, h, w, device=lr.device,
                            dtype=lr.dtype, generator=generator)

    def forward(self, lr, topo, z=None, generator=None):
        if z is None:
            z = self.sample_noise(lr, generator=generator)
        x = torch.cat([lr, z.to(lr.dtype)], dim=1)

        h = self.stem(x)
        for b in self.trunk:
            h = b(h)
        for up, blk in zip(self.ups, self.up_blocks):
            h = up(h)
            h = blk(h)

        if h.shape[-2:] != topo.shape[-2:]:
            h = F.interpolate(h, size=topo.shape[-2:], mode="bilinear", align_corners=False)

        t = self.topo_stem(topo.to(h.dtype))
        h = self.fuse(torch.cat([h, t], dim=1))
        for b in self.hr_blocks:
            h = b(h)

        delta = self.out(h)
        if self.use_lr_skip:
            base = F.interpolate(
                lr[:, self.precip_lr_ch:self.precip_lr_ch + 1].to(delta.dtype),
                size=delta.shape[-2:], mode="bilinear", align_corners=False)
            return base + delta
        return delta


# ----------------------------------------------------------------------------
# 4. CRITIC (Wasserstein critic -- NOT a "discriminator")
# ----------------------------------------------------------------------------

class Critic(nn.Module):
    """Conditional Wasserstein critic: unbounded scalar output, NO sigmoid.

    "Critic" and not "discriminator" is not a cosmetic choice of word: in the
    original (unbounded, sigmoid-free) WGAN formulation the network's output
    approximates a Kantorovich potential and its value is an estimate of (a
    multiple of) the Wasserstein-1 distance between the real and generated
    distributions, not a real/fake probability. Slapping a sigmoid back on
    would reintroduce the vanishing-gradient failure mode WGAN was designed
    to fix.

    CONDITIONAL: concatenates the bilinearly-upsampled LR fields and HR topo
    to the HR precip field being judged. An unconditional critic could not
    penalize a Generator for producing a plausible-looking field that ignores
    its LR/topo input entirely -- it can only tell "real-looking" from
    "fake-looking" over the marginal distribution of precip fields, with no
    way to check that a specific G(lr) corresponds to THIS lr. Conditioning
    the critic is what makes the adversarial signal actually enforce
    lr-to-precip correspondence rather than just generic photorealism.

    NO BATCHNORM ANYWHERE -- standard, load-bearing WGAN-GP guidance (paper
    Sec 4): BatchNorm computes normalization statistics ACROSS the samples in
    a minibatch, which (a) makes the per-sample gradient penalty ill-defined,
    since D(x_hat_i) for one interpolated sample i would then depend on every
    OTHER sample sharing its minibatch, breaking the "per-sample gradient norm
    close to 1" interpretation the penalty relies on, and (b) correlates the
    critic's judgement of different samples within a batch, which the
    Wasserstein-distance estimate is not supposed to have. GroupNorm(1 group)
    is used instead where normalization is still wanted (a LayerNorm
    equivalent: it normalizes each sample independently over ALL of its own
    channels and spatial positions, with no interaction with other samples in
    the batch), and the first layer skips normalization entirely (standard
    DCGAN/WGAN practice -- normalizing the raw conditioned input throws away
    absolute-intensity information that later layers can still use).

    OUTPUT HEAD: fully-convolutional PatchGAN-style -- a final 1x1-equivalent
    conv reduces to a single-channel score MAP, which is then spatially
    averaged to one scalar per sample (as opposed to global-pool-then-linear).
    See Config.py's CRITIC_* docstring for why this was the chosen option: it
    keeps the whole network free of any fixed-size linear layer (so it never
    silently breaks if handed a different tile shape) and it gives every
    spatial location its own gradient contribution to the score rather than
    pooling the feature map away before scoring it.
    """

    def __init__(self, precip_channels=1, cond_channels=4, topo_channels=TOPO_CHANNELS,
                 base_channels=64, channel_mult=(1, 2, 4, 8), dropout=0.0):
        super().__init__()
        in_ch = precip_channels + cond_channels + topo_channels
        chs = [base_channels * m for m in channel_mult]

        layers = []
        c = in_ch
        for i, oc in enumerate(chs):
            layers.append(nn.Conv2d(c, oc, kernel_size=4, stride=2, padding=1))
            if i > 0:   # first layer: no normalization (see class docstring)
                layers.append(nn.GroupNorm(1, oc))
            # inplace=False, deliberately: the gradient penalty differentiates
            # THROUGH this network a second time (autograd.grad(..., create_graph=
            # True) inside a term that is itself later .backward()'d), and an
            # in-place activation can overwrite an activation autograd needs to
            # keep around for that second differentiation, which either raises
            # "modified by an in-place operation" or -- worse -- silently
            # computes the wrong gradient penalty.
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            c = oc
        self.features = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(c, 1, kernel_size=3, padding=1)

    def forward(self, precip, lr_up, topo):
        """precip [B,1,H,W], lr_up [B,cond_channels,H,W] (already resized to
        match precip's H,W), topo [B,topo_channels,H,W]. Returns [B] scalar
        Wasserstein-potential estimates (higher = "more real-looking")."""
        x = torch.cat([precip, lr_up, topo], dim=1)
        h = self.features(x)
        patch_scores = self.out_conv(h)
        return patch_scores.mean(dim=[1, 2, 3])


# ----------------------------------------------------------------------------
# 5. Precipitation denormalization utility (verbatim from corrdiff_fm/Network.py)
# ----------------------------------------------------------------------------

def denorm_precip_mmday(precip_norm, precip_transform_meta):
    """Model-normalized precip -> physical mm/day, using checkpoint meta.

    Exact inverse of Dataset.py's forward transform:
        raw_mm -> * precip_scale -> log1p -> (x - norm_mean) / norm_std

    The clamp is applied in log1p space (log1p(p) >= 0  <=>  p >= 0), which is
    the correct place for it: it enforces non-negative precipitation without
    distorting any physically valid value. `precip_scale` is divided back out
    -- it is 1.0 in the current pipeline, but omitting the division silently
    breaks any checkpoint written under an old PRECIP_SCALE=24.0 convention.
    """
    mean = precip_transform_meta.get("norm_mean", precip_transform_meta.get("precip_log_mean", 0.0))
    std = precip_transform_meta.get("norm_std", precip_transform_meta.get("precip_log_std", 1.0))
    scale = float(precip_transform_meta.get("precip_scale", 1.0)) or 1.0

    # Upper clamp guards the exponential: expm1 of a large normalized value
    # overflows to inf, and a single inf poisons every downstream mean, std and
    # CRPS for the whole batch. 5000 mm/day is ~2.5x the highest daily rainfall
    # ever recorded on Earth, so this never touches a physical value -- it only
    # stops an unconverged or mis-scaled model from producing NaNs instead of
    # obviously-wrong numbers.
    max_mmday = float(precip_transform_meta.get("max_mmday", 5000.0))
    hi = math.log1p(max_mmday * scale)
    x_log = precip_norm * std + mean
    return torch.expm1(torch.clamp(x_log, min=0.0, max=hi)) / scale
