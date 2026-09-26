# -*- coding: utf-8 -*-
"""
Network.py -- SRGAN architecture for precipitation downscaling.
=================================================================
Implements Ledig et al. 2017 CVPR, "Photo-Realistic Single Image Super-
Resolution Using a Generative Adversarial Network" (arXiv:1609.04802), Fig. 4,
adapted for a 4-channel climate LR input + topography conditioning instead of
an RGB natural image.

Contract with Dataset.py (which must NOT be modified):
  hr  : [4, H, W]     huss, mslp, tas, precip -- ALL already per-channel
                      normalized (precip additionally log1p'd before
                      normalization)
  lr  : [4, H/4, W/4] = F.avg_pool2d(hr, 4)          (normalized space)
  oro : [1, H, W]     RAW orography in metres, NaN/fill -> 0.0

TASK FRAMING -- READ THIS BEFORE CHANGING ANYTHING
---------------------------------------------------
This package does NOT use corrdiff_fm's two-stage decomposition (deterministic
mean + residual diffusion correction). Instead a single generator `Generator`
maps LR + topo directly to HR precip, end to end, trained adversarially against
a `Discriminator`. There is no Stage-1 regressor to freeze and no residual
target to normalize against -- the whole "mean vs. residual" split that
corrdiff_fm's math depends on (see its Regressor.py docstring) is simply a
different way of parameterizing the same conditional distribution, and SRGAN's
paper does not use it: SRResNet is one network, trained once, on the full
target.

WHY THE GENERATOR HAS NO NOISE INPUT
-------------------------------------
Per Ledig et al., SRGAN's generator is DETERMINISTIC given its input -- there is
no latent/noise concatenated anywhere in Fig. 4. The "generative" character of
the model comes entirely from the *adversarial training signal*, which shapes
the distribution of G's outputs (over the training set) toward one a
discriminator cannot distinguish from real HR fields, not from sampling a
per-example latent at inference time. A single forward pass of a trained G on a
given LR+topo input always returns the same field.

This is a real, sharp architectural contrast with the sibling `wgan_gp`
package, whose generator DOES concatenate a noise channel (for a documented
reason specific to that package: it needs a controllable source of sample-to-
sample variability to produce an ensemble at inference, the way corrdiff_fm's
diffusion sampler does by drawing different sigma-trajectories). Nothing
analogous exists here. The direct, unavoidable consequence is that this
package's Inference.py writes a single `precip_mean` field with no
`precip_std` / `precip_members` -- there is nothing stochastic to average over.
See Evaluate.py for how this also makes CRPS degenerate to MAE.

WHAT IS REUSED FROM corrdiff_fm, AND WHY THAT IS SAFE
------------------------------------------------------
`expand_topo`, `TOPO_CHANNELS`, `TOPO_ELEV_CLIP`, `TOPO_SLOPE_Q`,
`LAND_THRESHOLD_M`, `_sobel_kernels`, `_robust_scale`, `_g`, and
`denorm_precip_mmday` are copied verbatim (only the module-level docstring
context differs) from corrdiff_fm/Network.py. None of them is diffusion- or
regressor-specific: they are a physically-grounded topography featurizer and a
handful of generic numerical utilities that any architecture consuming the same
Dataset.py contract needs identically. Re-deriving them here with different
names or slightly different constants would be the kind of incidental
divergence that makes an architecture comparison meaningless.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 0. Small utilities (verbatim from corrdiff_fm/Network.py)
# ----------------------------------------------------------------------------

def _g(ch, mx=32, min_per_group=8):
    """GroupNorm group count: largest divisor <= mx keeping >= min_per_group ch/group."""
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
    * Uses replicate padding, so the domain border does not look like a cliff.
    * Aspect is encoded as (sin, cos) rather than angle/pi: angle/pi has a hard
      discontinuity at +/-pi, which a CNN cannot represent smoothly.
    * Aspect is damped by slope confidence so flat/ocean pixels get 0 rather
      than an arbitrary direction from numerical noise.
    * Run this ONCE per domain (orography is static) and expand along batch;
      see Train.py / Inference.py.
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
# 2. Precipitation denormalization utility (verbatim from corrdiff_fm/Network.py)
# ----------------------------------------------------------------------------

def denorm_precip_mmday(precip_norm, precip_transform_meta):
    """Model-normalized precip -> physical mm/day, using checkpoint meta.

    Exact inverse of Dataset.py's forward transform:
        raw_mm -> * precip_scale -> log1p -> (x - norm_mean) / norm_std
    """
    mean = precip_transform_meta.get("norm_mean", precip_transform_meta.get("precip_log_mean", 0.0))
    std = precip_transform_meta.get("norm_std", precip_transform_meta.get("precip_log_std", 1.0))
    scale = float(precip_transform_meta.get("precip_scale", 1.0)) or 1.0

    # Upper clamp guards the exponential the same way corrdiff_fm's version
    # does: expm1 of a large normalized value overflows to inf, and a single
    # inf poisons every downstream mean/std/metric for the whole batch. 5000
    # mm/day is ~2.5x the highest daily rainfall ever recorded on Earth, so
    # this never touches a physical value -- it only stops an unconverged or
    # mis-scaled model from producing NaNs instead of obviously-wrong numbers.
    max_mmday = float(precip_transform_meta.get("max_mmday", 5000.0))
    hi = math.log1p(max_mmday * scale)
    x_log = precip_norm * std + mean
    return torch.expm1(torch.clamp(x_log, min=0.0, max=hi)) / scale


# ----------------------------------------------------------------------------
# 3. Generic edge-detection utility, for the non-VGG content loss
# ----------------------------------------------------------------------------

def sobel_gradient_magnitude(x):
    """Edge-magnitude map |grad(x)| via Sobel operators (replicate padding).

    Used by ContentLoss's default (non-VGG) path. Precipitation fronts and
    convective cells are, physically, sharp spatial gradients -- exactly the
    structure an L1/L2 pixel loss alone under-rewards, because a blurred
    field that gets the *coarse* mean right pays only a small pixel-loss
    penalty for smearing out a front. Matching gradient MAGNITUDE (not simply
    adding a higher weight on large values) directly penalizes that smearing
    without needing any pretrained feature extractor -- this is the role
    VGG features play in the paper's natural-image setting (rewarding
    perceptually sharp texture), reimplemented here with a filter that has
    a literal physical meaning for this field instead of an ImageNet prior.
    """
    k = _sobel_kernels(x.device, x.dtype)[:2]  # kx, ky only; drop the Laplacian
    g = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k)
    gx, gy = g[:, 0:1], g[:, 1:2]
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


# ----------------------------------------------------------------------------
# 4. VGG perceptual loss (literal-paper option, USE_VGG_PERCEPTUAL=True)
# ----------------------------------------------------------------------------

class VGGFeatureExtractor(nn.Module):
    """Frozen ImageNet-pretrained VGG19, truncated at conv5_4 PRE-activation.

    This is the paper's "VGG54" content loss (Ledig et al. Sec 3.2 / Table 2):
    features are taken BEFORE the ReLU of the 4th conv in the 5th block,
    because the paper found post-activation features are (1) very sparsely
    active on much of the image (ReLU throws away everything a filter
    considers "not present"), which weakens the gradient signal, and (2) have
    magnitudes that vary a lot across layers/images. `layer_idx=35` on
    `torchvision.models.vgg19().features` is the conv output at index 34 --
    features[:35] therefore includes that conv but excludes the ReLU at index
    35. (A commonly-used *alternative* in reimplementations is the shallower
    "VGG22" -- features[:9], after the 2nd conv of block 2 -- which rewards
    lower-level texture rather than high-level content; conv5_4 is the
    paper's headline configuration, so that is the default here.)

    DOMAIN-MISMATCH CAVEAT -- READ BEFORE ENABLING USE_VGG_PERCEPTUAL
    --------------------------------------------------------------------
    VGG19 was trained on ImageNet: 3-channel photographs of natural objects,
    normalized to ImageNet's RGB statistics. A single-channel, log1p-
    normalized precipitation field is not a natural image in any sense VGG's
    filters were built to exploit -- there is no reason to expect its early
    conv filters (edge/color-opponent/texture detectors tuned for photographs)
    to pick out features that are physically meaningful for a rainfall field,
    and every reason to expect some filters to respond to structure that is
    an artefact of how this code maps a scalar field into a fake "image"
    (replication to 3 identical channels, and the ad hoc squashing below)
    rather than anything about the rain itself. This path exists for literal
    fidelity to the paper's recipe and for readers who want to reproduce that
    exact choice -- it is not claimed to be the physically better option.
    The default (USE_VGG_PERCEPTUAL=False) Sobel-gradient content loss in
    ContentLoss below has no such mismatch and is the recommended path for
    this data modality.

    The single-channel -> VGG-input mapping: normalized log1p precip lives
    roughly in the range a per-channel standardization produces (call it
    ~[-3, 3] for the overwhelming majority of pixels, given the log1p
    transform pulls in the long right tail before normalization). We clamp to
    that range and rescale linearly to [0, 1] (an image intensity range),
    replicate to 3 channels, then apply ImageNet's per-channel mean/std -- the
    same normalization VGG19 was trained with, since its BatchNorm-free conv
    stack has no other built-in scale invariance.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, layer_idx=35):
        super().__init__()
        try:
            from torchvision.models import vgg19, VGG19_Weights
        except ImportError as e:
            raise RuntimeError(
                "USE_VGG_PERCEPTUAL=True requires torchvision to be installed. "
                "Set Config.USE_VGG_PERCEPTUAL = False to use the physically-"
                "motivated Sobel-gradient content loss instead (no external "
                "dependency, no pretrained-weight download)."
            ) from e
        try:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except Exception as e:
            # Typically: no internet access on a compute node, or torchvision's
            # weight-hosting URL is unreachable from behind a cluster firewall.
            # Fail LOUDLY here rather than silently falling back to a different
            # loss -- a silent fallback would mean whatever paper-fidelity claim
            # a run makes is simply false, discovered only much later.
            raise RuntimeError(
                "Could not download/load ImageNet-pretrained VGG19 weights "
                "(no internet access on this node, a firewalled cluster, or "
                "torchvision's weight URL is unreachable). Set "
                "Config.USE_VGG_PERCEPTUAL = False to use the Sobel-gradient "
                "content loss instead, which needs no pretrained weights."
            ) from e

        layers = list(vgg.features.children())[:layer_idx]
        self.features = nn.Sequential(*layers).eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _to_vgg_input(self, x):
        x = x.clamp(-3.0, 3.0)
        x = (x + 3.0) / 6.0                    # -> [0, 1]
        x = x.repeat(1, 3, 1, 1)                # 1ch precip -> fake 3ch "image"
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        with torch.no_grad():
            ft = self.features(self._to_vgg_input(target))
        fp = self.features(self._to_vgg_input(pred))
        return F.mse_loss(fp, ft)


class ContentLoss(nn.Module):
    """Generator content-fidelity term l^SR_X (Ledig et al. Eq. 3's first term).

    Two mutually exclusive implementations, selected by `use_vgg`:

      use_vgg=False (Config.USE_VGG_PERCEPTUAL default)
        pixel L1  +  sobel_weight * Sobel-gradient-magnitude L1.
        Physically motivated: rewards getting both the field VALUES and its
        EDGES (precipitation fronts, the physically meaningful high-frequency
        structure this whole exercise is trying to recover) right, with no
        pretrained natural-image network anywhere in the loop.

      use_vgg=True
        VGGFeatureExtractor MSE (paper's literal VGG54 content loss)
          + a small pixel_anchor_weight * L1 term.
        The anchor is necessary in practice, not just a nicety: VGG features
        are approximately contrast/scale-invariant by design (that is what
        makes them good *perceptual* features for natural images), so a pure
        VGG-feature loss has very little gradient pushing the predicted
        ABSOLUTE precipitation intensity toward the target's -- two fields
        that "look the same shape" to VGG but differ by a multiplicative
        factor would score similarly. The pixel anchor supplies that missing
        intensity signal. See VGGFeatureExtractor's docstring for the
        domain-mismatch caveat this path carries regardless.
    """

    def __init__(self, use_vgg=False, vgg_layer_idx=35,
                 pixel_anchor_weight=0.1, sobel_weight=0.5):
        super().__init__()
        self.use_vgg = use_vgg
        self.pixel_anchor_weight = pixel_anchor_weight
        self.sobel_weight = sobel_weight
        self.vgg = VGGFeatureExtractor(layer_idx=vgg_layer_idx) if use_vgg else None

    def forward(self, pred, target):
        if self.use_vgg:
            return self.vgg(pred, target) + self.pixel_anchor_weight * F.l1_loss(pred, target)
        pixel = F.l1_loss(pred, target)
        grad = F.l1_loss(sobel_gradient_magnitude(pred), sobel_gradient_magnitude(target))
        return pixel + self.sobel_weight * grad


# ----------------------------------------------------------------------------
# 5. GENERATOR -- SRResNet backbone (Ledig et al. Fig. 4)
# ----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """The paper's exact residual block: conv(3x3)-BN-PReLU-conv(3x3)-BN,
    added to the block's input. No SE gating, no dilation, no attention --
    those are corrdiff_fm's Network.py's own additions for ITS architecture
    family (a conditional U-Net denoiser) and have no place in a faithful
    SRResNet reimplementation; adding them here would no longer be measuring
    what the paper's architecture does.
    """

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.act = nn.PReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return x + y


class PixelShuffleBlock(nn.Module):
    """conv(3x3) -> PixelShuffle(2) -> PReLU sub-pixel upsampler (Ledig et al.
    Fig. 4's own upsampling block for the paper's 4x configuration). Two of
    these compose to exactly DS_FACTOR=4x -- the paper's own 4x setup, so
    unlike the sibling `srdrn` package no adaptation of the upsampling depth
    is required here."""

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class Generator(nn.Module):
    """SRResNet generator (Ledig et al. Fig. 4), adapted for climate downscaling.

    Deviations from the literal paper, and why each is necessary:
      * Input is the 4-channel normalized LR climate stack (huss, mslp, tas,
        precip), not a 3-channel RGB image -- so the stem conv's in_channels
        is 4, not 3.
      * Topography (8ch, HR resolution) is fused right before the final
        output conv via a small conv+GroupNorm+SiLU branch, concatenated with
        the upsampled trunk features -- mirroring corrdiff_fm's
        CorrDiffRegressor fusion pattern. The paper has no topography input
        at all (natural images carry no analogous side channel).
      * No final tanh / [-1,1] squash. The paper's output is a natural image
        living in [-1,1]; this package's target is already a roughly-N(0,1)
        normalized log1p precipitation field (Dataset.py's per-channel
        standardization), so squashing it through tanh would just add a
        needless saturating nonlinearity fighting both loss terms for no
        benefit.
      * Output conv is zero-initialized: in this normalized space the
        per-channel mean is zero by construction, so "predict zero" is
        already a sane, correctly-scaled starting guess -- the same
        reasoning corrdiff_fm's CorrDiffRegressor applies to its own output
        layer.

    Everything else is the paper's Fig. 4 exactly: 9x9 stem conv + PReLU,
    B=NUM_RES_BLOCKS residual blocks, a post-residual conv+BN, a GLOBAL skip
    connection adding back the post-stem feature map (the paper's
    "elementwise sum" before upsampling -- easy to drop by accident, and
    doing so measurably hurts convergence since the residual blocks would
    otherwise have to re-learn an identity path through 16 stacked BN layers),
    then n_up = log2(ds_factor) PixelShuffle blocks.

    No noise input anywhere -- see the module docstring's "WHY THE GENERATOR
    HAS NO NOISE INPUT" section. This network is a deterministic function of
    (lr, topo).
    """

    def __init__(self, in_channels=4, out_channels=1, base_channels=64,
                 num_res_blocks=16, ds_factor=4, topo_channels=TOPO_CHANNELS):
        super().__init__()
        n_up = round(math.log2(ds_factor))
        assert 2 ** n_up == ds_factor, f"ds_factor={ds_factor} must be a power of 2"
        self.ds_factor = ds_factor

        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=9, padding=4)
        self.stem_act = nn.PReLU()

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_res_blocks)])

        self.post_res_conv = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.post_res_bn = nn.BatchNorm2d(base_channels)

        self.upsample = nn.Sequential(
            *[PixelShuffleBlock(base_channels) for _ in range(n_up)])

        # Topography fusion branch. GroupNorm (not BatchNorm) here on purpose:
        # this branch runs on a single, static, per-domain topo tensor merely
        # broadcast across the batch dimension (see Train.py/Inference.py --
        # `topo.expand(B, -1, -1, -1)`), so a "batch" of B identical maps has
        # no meaningful batch statistics for BatchNorm to estimate; GroupNorm
        # normalizes within each sample and is well-defined regardless.
        self.topo_stem = nn.Sequential(
            nn.Conv2d(topo_channels, base_channels, 3, padding=1),
            nn.GroupNorm(_g(base_channels), base_channels),
            nn.SiLU(),
        )
        self.fuse = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.fuse_bn = nn.BatchNorm2d(base_channels)
        self.fuse_act = nn.PReLU()

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=9, padding=4)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, lr, topo):
        h0 = self.stem_act(self.stem(lr))         # post-stem feature map, kept for the global skip
        h = self.res_blocks(h0)
        h = self.post_res_bn(self.post_res_conv(h))
        h = h + h0                                 # Fig. 4's "elementwise sum" before upsampling
        h = self.upsample(h)                       # now at HR (topo) resolution

        if h.shape[-2:] != topo.shape[-2:]:
            # Guards against off-by-one edge cases (e.g. a cropped domain not
            # exactly ds_factor-divisible); should be a no-op in the normal
            # PATCH-aligned training/inference path.
            h = F.interpolate(h, size=topo.shape[-2:], mode="bilinear", align_corners=False)

        t = self.topo_stem(topo.to(h.dtype))
        h = self.fuse_act(self.fuse_bn(self.fuse(torch.cat([h, t], dim=1))))
        return self.out_conv(h)


# ----------------------------------------------------------------------------
# 6. DISCRIMINATOR -- VGG-style classifier (Ledig et al. Fig. 4)
# ----------------------------------------------------------------------------

class _DiscBlock(nn.Module):
    def __init__(self, ic, oc, stride, use_bn=True):
        super().__init__()
        # No bias when followed by BatchNorm -- BN's own beta makes a
        # preceding conv bias redundant and it would just be an extra set of
        # parameters BN immediately cancels out.
        self.conv = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(oc) if use_bn else None
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)


class Discriminator(nn.Module):
    """VGG-style discriminator (Ledig et al. Fig. 4): 8 conv layers, 3x3
    kernels, stride alternating 1/2, channels doubling 64-64-128-128-256-256-
    512-512, LeakyReLU(0.2) throughout, BatchNorm on every layer EXCEPT the
    very first (paper convention -- BN on the input-facing layer tends to
    introduce artefacts correlated with the exact per-batch input statistics,
    which is undesirable right where the network is looking at raw pixel/
    precip values rather than learned features).

    Ends in the paper's literal dense(1024)-LeakyReLU-dense(1)-sigmoid head
    rather than a global-average-pool alternative. This is safe here (and
    would NOT be safe for a network meant to run on arbitrary tile sizes,
    which is why Generator has no such fixed-size head): D is invoked ONLY
    during training, exclusively on PATCH-sized crops (`Config.DISC_PATCH`,
    fixed for a given training run) -- it is never asked to score a 2 km
    inference tile of a different size, unlike G, which Tiling.py's
    TiledGenerator does run at inference tile sizes. If you ever want to
    resume/fine-tune D at a different PATCH, you must retrain it from
    scratch; the dense head's dimensions are baked into its state_dict.

    CONDITIONING -- A DELIBERATE DEVIATION FROM THE LITERAL PAPER
    ----------------------------------------------------------------
    The paper's discriminator is UNCONDITIONAL: for a natural photograph,
    "is this a real high-resolution image" is a well-posed question on its
    own, because there is essentially one plausible LR->HR mapping direction
    being judged (does this look like a real photo at this resolution).
    Precipitation downscaling has no such single answer -- a physically
    plausible 2 km rainfall field for one 8 km-mean/topography context can be
    wildly implausible for another (e.g. concentrated on a windward slope
    that does not exist in the second context). An unconditional D would
    therefore end up learning "is this a plausible precip field IN GENERAL"
    rather than "is this a plausible precip field GIVEN THIS LR+topo
    context", and could be fooled by (or unfairly penalize) a G that produces
    realistic-looking but contextually wrong structure. So D is conditioned
    on the same upsampled-LR + topo context G sees, by concatenation as extra
    input channels -- exactly the same reasoning the `wgan_gp` sibling
    package's critic uses.

    NON-WASSERSTEIN, STANDARD GAN -- THE OTHER DELIBERATE CONTRAST WITH
    THE `wgan_gp` SIBLING PACKAGE
    ----------------------------------------------------------------------
    This is the paper's original formulation: a binary classifier with a
    sigmoid output, trained with binary cross-entropy (real label 1, fake
    label 0), and the generator trained to push D(fake) toward 1. There is no
    Lipschitz constraint, no gradient penalty, and no Wasserstein critic here
    -- that machinery belongs to the `wgan_gp` package, which exists
    specifically to be a different adversarial-training recipe on the same
    task. Mixing the two within one package would make neither package a
    faithful implementation of anything.
    """

    def __init__(self, in_channels, base_channels=64, patch=128):
        super().__init__()
        # (out_channels, stride, use_batchnorm) for each of the 8 conv layers.
        layout = [
            (base_channels, 1, False),
            (base_channels, 2, True),
            (base_channels * 2, 1, True),
            (base_channels * 2, 2, True),
            (base_channels * 4, 1, True),
            (base_channels * 4, 2, True),
            (base_channels * 8, 1, True),
            (base_channels * 8, 2, True),
        ]
        blocks = []
        ic = in_channels
        for oc, stride, use_bn in layout:
            blocks.append(_DiscBlock(ic, oc, stride, use_bn=use_bn))
            ic = oc
        self.conv_stack = nn.Sequential(*blocks)

        n_down = sum(1 for _, s, _ in layout if s == 2)
        assert patch % (2 ** n_down) == 0, (
            f"patch={patch} must be divisible by {2 ** n_down} "
            f"({n_down} stride-2 layers in the discriminator)")
        feat_hw = patch // (2 ** n_down)
        flat_dim = ic * feat_hw * feat_hw

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, precip, lr_up, topo):
        """precip [B,1,H,W] (real or fake HR precip), lr_up [B,IN_CH_LR,H,W]
        (bilinearly upsampled LR climate stack), topo [B,TOPO_CHANNELS,H,W].
        Returns a [B,1] probability in (0,1) that `precip` is real."""
        x = torch.cat([precip, lr_up, topo], dim=1)
        x = self.conv_stack(x)
        logit = self.head(x)
        return torch.sigmoid(logit)
