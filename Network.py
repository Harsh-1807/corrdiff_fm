# -*- coding: utf-8 -*-
"""
Network.py -- architectures for two-stage CorrDiff precipitation downscaling.

Stage 1  CorrDiffRegressor : deterministic conditional mean  E[precip_HR | LR, topo]
Stage 2  UNet              : flow-matching (or EDM) network on the *normalized
                             residual*  r = (precip_HR - reg_mean) / residual_std

Contract with Dataset.py (which must NOT be modified):
  * hr  : [4, H, W]  huss, mslp, tas, precip -- ALL already per-channel normalized
                     (precip additionally log1p'd before normalization)
  * lr  : [4, H/4, W/4] = F.avg_pool2d(hr, 4)          (normalized space)
  * oro : [1, H, W]  RAW orography in metres, NaN/fill -> 0.0

IMPORTANT FIX vs. the previous Network.py
-----------------------------------------
`compute_slope_aspect` was written for *standardized* elevation (TOPO_ELEV_CLIP=2.0)
but Dataset.load_oro hands back raw metres. Every land pixel therefore saturated
the clip and the elevation channel collapsed to a constant 1.0 -- i.e. the model
was effectively trained with no usable topography. `expand_topo` below now
standardizes internally and derives richer, self-calibrating descriptors.

FIX (2026-08-06): the Stage-2 `UNet` class referenced by TrainDiffusion.py was
missing entirely -- the file was truncated inside a stub `Upsample.__init__`
(`super().__init__` with no call and no `forward`), which is what produced
`ImportError: cannot import name 'UNet' from 'Network'` at import time (not a
distributed-launch problem -- all 4 ranks failed on the same import). `Upsample`
is now completed and a full conditional `UNet` has been added, wired to match
exactly how TrainDiffusion.py calls it:
  * input  x    : [B, in_channels, H, W] = cat(noisy residual[1], reg_mean[1],
                  bilinearly-upsampled LR[4])  (see fm_build_input / edm cat)
  * input  t    : [B] raw timestep (t*1000 for flow matching, or c_noise for EDM)
  * input  topo : [B, TOPO_CHANNELS, H, W] -- resized internally per-level by FiLM
  * kwarg  cfg_drop : optional [B] bool -- zeroes conditioning channels + topo for
                  the unconditional branch used in classifier-free guidance
                  (see fm_sample / edm_sample's `guidance != 1.0` path)
  * output      : [B, out_channels, H, W] predicted velocity (fm) or D_theta (edm)

FIX (2026-08-23): `timestep_embedding` was starved of dynamic range under EDM
(its top frequency is 1.0, while c_noise = ln(sigma)/4 only spans ~[-2, 2], so
nearly every embedding channel was constant across the whole noise range and
the denoiser was effectively blind to sigma). Replaced by `FourierEmbedding`,
an EDM-style frozen random-Fourier feature map that resolves both the EDM
c_noise range and the [0, 1] flow-matching range. `t` is now passed RAW -- do
NOT multiply by 1000 any more.

RESOLUTION TRANSFER -- READ THIS BEFORE RUNNING 8 km -> 2 km
-----------------------------------------------------------
Two blocks in this file are NOT resolution-agnostic, despite everything being
fully convolutional:

  * `SpectralConv2d` learns weights on absolute FFT mode indices. Mode k on a
    176x176 grid and mode k on a 704x704 grid correspond to different physical
    wavelengths, so a model trained at one grid size is being asked a different
    question at the other.
  * `SelfAttn2d` silently switches from exact attention to attention over an
    adaptively pooled grid once H*W > max_tokens. Crossing that threshold
    between training and inference changes the operator itself.

Neither is a bug per se, but together they mean the cascade only holds if the
network sees the SAME pixel grid size at train and at inference. The supported
way to do that is: train on PATCH-sized crops, then run inference tiled at
exactly that tile size (see Diffusion.py's TiledDenoiser and Inference.py).
Do not train full-domain at 8 km and then hand the model a full 2 km domain.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 0. Small utilities
# ----------------------------------------------------------------------------

def _g(ch, mx=32, min_per_group=8):
    """GroupNorm group count: largest divisor <= mx keeping >= min_per_group ch/group.

    The old version returned the largest divisor <= 32 outright, giving e.g.
    24 groups of 2 channels for ch=48 -- normalizing over 2 channels is close to
    InstanceNorm and is noticeably noisier. This targets fatter groups.
    """
    best = 1
    for g in range(1, min(mx, ch) + 1):
        if ch % g == 0 and ch // g >= min_per_group:
            best = g
    return best


def _zero(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ----------------------------------------------------------------------------
# 1. Topography featurizer
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
      (zero padding produced spurious slope/curvature rings before).
    * Aspect is encoded as (sin, cos) rather than angle/pi: angle/pi has a hard
      discontinuity at +/-pi, which a CNN cannot represent smoothly.
    * Aspect is damped by slope confidence so flat/ocean pixels get 0 rather
      than an arbitrary direction from numerical noise.
    * Run this ONCE per domain (orography is static) and expand along batch;
      see the training scripts.
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
# 2. Coarse-intensity augmentation (Stage 1 only)
# ----------------------------------------------------------------------------

def augment_coarse_intensity(fine, ds_factor=4, alpha_max=0.20, p=0.15):
    """Perturb the coarse field towards local maxima.

    At training time LR = avg_pool(HR_truth); at deployment LR is a genuine
    coarse (32 km) field whose sub-grid intensity distribution differs. Blending
    a little max-pool into the average makes Stage 1 less brittle to that shift.
    Set p=0.0 to reproduce the exact Dataset.__getitem__ LR definition.
    """
    avg = F.avg_pool2d(fine, kernel_size=ds_factor, stride=ds_factor)
    if alpha_max <= 0.0 or p <= 0.0 or torch.rand(()).item() > p:
        return avg
    mx = F.max_pool2d(fine, kernel_size=ds_factor, stride=ds_factor)
    alpha = torch.rand(fine.shape[0], 1, 1, 1, device=fine.device) * alpha_max
    return (1.0 - alpha) * avg + alpha * mx


# ----------------------------------------------------------------------------
# 3. Building blocks
# ----------------------------------------------------------------------------

class SEBlock(nn.Module):
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


class FiLM(nn.Module):
    """Zero-initialized spatial FiLM conditioned on the topo tensor.

    Starts as an exact identity, so topography is introduced gradually and
    cannot destabilize early training.
    """

    def __init__(self, topo_ch, out_ch, scale=0.5):
        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(topo_ch, out_ch * 2, 3, padding=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h, topo):
        if topo is None:
            return h
        if topo.shape[-2:] != h.shape[-2:]:
            topo = F.interpolate(topo, size=h.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.proj(topo.to(h.dtype)).chunk(2, dim=1)
        return h * (1.0 + self.scale * torch.tanh(gamma)) + self.scale * beta


class ResConv(nn.Module):
    """Pre-norm residual conv block, optional topo FiLM. No timestep input."""

    def __init__(self, ic, oc, drop=0.1, topo_ch=0):
        super().__init__()
        self.n1 = nn.GroupNorm(_g(ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.n2 = nn.GroupNorm(_g(oc), oc)
        self.film = FiLM(topo_ch, oc) if topo_ch else None
        self.drop = nn.Dropout2d(drop)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.se = SEBlock(oc, drop=drop * 0.5)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, topo=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.n2(h)
        if self.film is not None:
            h = self.film(h, topo)
        h = self.c2(F.silu(self.drop(h)))
        return self.skip(x) + self.se(h)


class SelfAttn2d(nn.Module):
    """Self-attention that adapts to the feature-map size.

    If H*W <= max_tokens it is exact full self-attention (via SDPA, so it uses
    flash/mem-efficient kernels). Otherwise it degrades to attention over an
    adaptively pooled grid, which is what the old `BnAttn` always did -- that
    hard-coded 8x8 pooling threw away most of the bottleneck resolution even
    when full attention was affordable.
    """

    def __init__(self, ch, head_dim=64, drop=0.1, max_tokens=4096):
        super().__init__()
        heads = max(1, ch // head_dim)
        while ch % heads != 0 and heads > 1:
            heads -= 1
        self.h = heads
        self.max_tokens = max_tokens
        self.norm = nn.GroupNorm(_g(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.Dropout2d(drop))
        nn.init.zeros_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        
        z = self.norm(x)
        
        pooled = H * W > self.max_tokens
        if pooled:
            s = max(4, int(math.sqrt(self.max_tokens)))
            sh, sw = min(H, s), min(W, s)
            z = F.adaptive_avg_pool2d(z, (sh, sw))
        else:
            sh, sw = H, W
            
        q, k, v = self.qkv(z).reshape(B, 3, self.h, C // self.h, sh * sw).unbind(1)
        q, k, v = (t.transpose(-2, -1).contiguous() for t in (q, k, v))  # B,h,N,d
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(-2, -1).reshape(B, C, sh, sw)
        o = self.proj(o)
        
        if pooled:
            o = F.interpolate(o, (H, W), mode="bilinear", align_corners=False)
            
        return x + o


class SpectralConv2d(nn.Module):
    """FNO-style low-mode spectral convolution (replaces the old FourierFilter).

    The old block applied ONE complex channel-mixing matrix identically to every
    frequency, which (for a real input passed back through irfft) collapses to a
    1x1 conv plus a Hilbert-like term -- almost no spectral selectivity, and it
    also silently broke under autocast because complex half is unsupported.

    This version learns per-mode weights on the lowest (my, mx) modes only,
    inside a channel-reduced subspace so the parameter count stays sane, and it
    forces float32 for the FFT. Zero-initialized output projection.
    """

    def __init__(self, ch, fdim=64, modes=(16, 16), drop=0.0):
        super().__init__()
        fdim = min(fdim, ch)
        self.my, self.mx = modes
        self.fdim = fdim
        self.norm = nn.GroupNorm(_g(ch), ch)
        self.down = nn.Conv2d(ch, fdim, 1, bias=False)
        # Two mode blocks: low positive-frequency rows, and wrapped negative rows.
        # Kept 3-D on purpose -- a 5-D parameter makes
        # `model.to(memory_format=torch.channels_last)` raise, and channels_last
        # is worth real throughput for the rest of the conv stack.
        sh = (fdim, fdim, self.my * self.mx * 2)
        self.w_lo = nn.Parameter(torch.randn(*sh) * 0.02)
        self.w_hi = nn.Parameter(torch.randn(*sh) * 0.02)
        self.up = nn.Sequential(nn.Conv2d(fdim, ch, 1), nn.Dropout2d(drop))
        nn.init.zeros_(self.up[0].weight)
        nn.init.zeros_(self.up[0].bias)

    def _modes(self, w):
        w = w.float().reshape(self.fdim, self.fdim, self.my, self.mx, 2).contiguous()
        return torch.view_as_complex(w)

    def forward(self, x):
        B, C, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            # FIXED: Cast x to float32 before passing to the normalisation layer 
            # to resolve BFloat16/Float mismatch during AMP
            x32 = x.float()
            z = self.down(self.norm(x32))
            f = torch.fft.rfft2(z, norm="ortho")
            my = min(self.my, max(1, H // 2))
            mx = min(self.mx, f.shape[-1])
            out = torch.zeros_like(f)
            wl = self._modes(self.w_lo)[:, :, :my, :mx]
            wh = self._modes(self.w_hi)[:, :, :my, :mx]
            out[:, :, :my, :mx] = torch.einsum("bixy,ioxy->boxy", f[:, :, :my, :mx], wl)
            out[:, :, -my:, :mx] = torch.einsum("bixy,ioxy->boxy", f[:, :, -my:, :mx], wh)
            z = torch.fft.irfft2(out, s=(H, W), norm="ortho")
        return x + self.up(z.to(x.dtype))


class DilatedBottleneck(nn.Module):
    """Multi-rate dilated context, zero-initialized residual."""

    def __init__(self, in_ch, mid_ch=256, drop=0.1, rates=(1, 2, 4, 8)):
        super().__init__()
        b = max(8, mid_ch // len(rates))
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, b, 3, padding=r, dilation=r, bias=False),
                nn.GroupNorm(_g(b), b), nn.SiLU(), nn.Dropout2d(drop),
            ) for r in rates
        ])
        self.project = nn.Sequential(
            nn.Conv2d(b * len(rates), in_ch, 1, bias=False),
            nn.GroupNorm(_g(in_ch), in_ch),
        )
        nn.init.zeros_(self.project[0].weight)

    def forward(self, x):
        return x + self.project(torch.cat([b(x) for b in self.branches], 1))


class PixelShuffleUp(nn.Module):
    """x2 sub-pixel upsampling with ICNR-ish init (conv then shuffle)."""

    def __init__(self, ic, oc, drop=0.0):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc * 4, 3, padding=1)
        self.shuf = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(_g(oc), oc)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        return self.drop(F.silu(self.norm(self.shuf(self.conv(x)))))


# ----------------------------------------------------------------------------
# 4. STAGE 1 -- deterministic regressor (conditional mean)
# ----------------------------------------------------------------------------

class CorrDiffRegressor(nn.Module):
    """LR (4ch, normalized) + topo -> HR precip mean (1ch, normalized log space).

    Design changes vs. the previous version:
      * All heavy trunk work happens at LR resolution (H/4), then sub-pixel
        upsampling to HR. The old model bilinear-upsampled the input x4 first
        and ran a full two-branch U-Net at HR -- ~16x the FLOPs for no gain.
      * No spatial downsampling anywhere: the LR grid is already small
        (H/4), so downsampling risks 1x1 feature maps on small domains.
        Global context comes from self-attention + dilated convs instead, which
        makes the model resolution-agnostic (needed for patch training and for
        the later 8 km -> 2 km stage).
      * Global skip: predicts a *correction* to the bilinearly interpolated LR
        precip channel, so it starts from a sane baseline.
      * Topography enters through zero-init FiLM at every scale.

    len(channel_mult) - 1 must equal log2(ds_factor).
    """

    def __init__(self, in_channels=4, out_channels=1, base_channels=64,
                 channel_mult=(1, 2, 4), num_blocks=2, topo_channels=TOPO_CHANNELS,
                 dropout=0.10, ds_factor=4, precip_lr_ch=3, use_lr_skip=True, **kw):
        super().__init__()
        chs = [base_channels * m for m in channel_mult]
        n_up = len(chs) - 1
        assert 2 ** n_up == ds_factor, (
            f"len(channel_mult)-1 ({n_up}) must satisfy 2**n == ds_factor ({ds_factor})")

        self.ds_factor = ds_factor
        self.precip_lr_ch = precip_lr_ch
        self.use_lr_skip = use_lr_skip and (in_channels > precip_lr_ch)
        self.topo_channels = topo_channels

        lr_ch = chs[-1]
        self.lr_stem = nn.Conv2d(in_channels, lr_ch, 3, padding=1)
        self.lr_pre = nn.ModuleList([ResConv(lr_ch, lr_ch, dropout) for _ in range(num_blocks)])
        self.lr_dil = DilatedBottleneck(lr_ch, min(256, lr_ch), dropout)
        self.lr_attn = SelfAttn2d(lr_ch, drop=dropout)
        self.lr_post = nn.ModuleList([ResConv(lr_ch, lr_ch, dropout) for _ in range(num_blocks)])

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        c = lr_ch
        for oc in reversed(chs[:-1]):
            self.ups.append(PixelShuffleUp(c, oc, dropout))
            self.up_blocks.append(ResConv(oc, oc, dropout, topo_ch=topo_channels))
            c = oc

        self.topo_stem = nn.Sequential(
            nn.Conv2d(topo_channels, c, 3, padding=1),
            nn.GroupNorm(_g(c), c), nn.SiLU(),
        )
        self.topo_block = ResConv(c, c, dropout)

        self.fuse = ResConv(c * 2, c, dropout, topo_ch=topo_channels)
        self.hr_blocks = nn.ModuleList(
            [ResConv(c, c, dropout, topo_ch=topo_channels) for _ in range(num_blocks)])

        self.out = nn.Sequential(
            nn.GroupNorm(_g(c), c), nn.SiLU(), nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(c, out_channels, 3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, topo):
        h = self.lr_stem(x)
        for b in self.lr_pre:
            h = b(h)
        h = self.lr_attn(self.lr_dil(h))
        for b in self.lr_post:
            h = b(h)

        for up, blk in zip(self.ups, self.up_blocks):
            h = up(h)
            h = blk(h, topo)

        if h.shape[-2:] != topo.shape[-2:]:
            h = F.interpolate(h, size=topo.shape[-2:], mode="bilinear", align_corners=False)

        t = self.topo_block(self.topo_stem(topo.to(h.dtype)))
        h = self.fuse(torch.cat([h, t], 1), topo)
        for b in self.hr_blocks:
            h = b(h, topo)

        delta = self.out(h)
        if self.use_lr_skip:
            base = F.interpolate(
                x[:, self.precip_lr_ch:self.precip_lr_ch + 1].to(delta.dtype),
                size=delta.shape[-2:], mode="bilinear", align_corners=False)
            return base + delta
        return delta


# ----------------------------------------------------------------------------
# 5. STAGE 2 -- conditional U-Net (flow matching / EDM)
# ----------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block with timestep embedding and topo FiLM."""

    def __init__(self, ic, oc, ec, topo_channels=TOPO_CHANNELS, dropout=0.1):
        super().__init__()
        self.n1 = nn.GroupNorm(_g(ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.ep = nn.Sequential(nn.SiLU(), nn.Linear(ec, oc))
        self.n2 = nn.GroupNorm(_g(oc), oc)
        self.film = FiLM(topo_channels, oc) if topo_channels else None
        self.drop = nn.Dropout2d(dropout)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.se = SEBlock(oc, drop=dropout * 0.5)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, emb, topo=None):
        h = self.c1(F.silu(self.n1(x))) + self.ep(emb).to(x.dtype)[:, :, None, None]
        h = self.n2(h)
        if self.film is not None:
            h = self.film(h, topo)
        h = self.c2(F.silu(self.drop(h)))
        return self.skip(x) + self.se(h)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """Nearest + conv (no ConvTranspose checkerboard artifacts).

    FIX: the previous version's __init__ never called super().__init__()
    (missing parens -- `super().__init__` is just a bound-method reference,
    not a call) and had no conv/forward at all, so the class was unusable and
    the file was truncated right after it. Completed below.
    """

    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


def timestep_embedding(t, dim, max_period=10000.0):
    """Standard sinusoidal embedding (Transformer / DDPM style).

    RETAINED ONLY FOR BACKWARD COMPATIBILITY -- do not use with EDM.

    Why: `freqs` spans 1.0 down to 1/max_period. EDM feeds
    c_noise = ln(sigma)/4, which lives in roughly [-2, 2]. With the highest
    frequency equal to 1.0, `args` never leaves [-2, 2] either, so cos(args)
    is nearly flat and the overwhelming majority of the embedding channels
    are numerically constant across the entire noise range. The network then
    has almost no usable sigma conditioning, which is precisely the failure
    mode where an EDM model trains to a plausible-looking loss and samples
    blurry mush. Use `FourierEmbedding` instead.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FourierEmbedding(nn.Module):
    """Random-Fourier noise-level embedding (Karras et al., EDM).

    Frequencies are drawn once at construction from N(0, scale^2) and frozen
    as a buffer, so the embedding is part of the checkpoint and is identical
    between training and sampling. `scale=16` resolves inputs of order 1e-2
    to 1e1 in the argument, which covers both:

      * EDM   : t = c_noise = ln(sigma)/4  in roughly [-2, 2]
      * FM    : t = the flow time          in [0, 1]

    so the same UNet class serves both parameterizations without the caller
    having to invent an arbitrary multiplier like `t * 1000`.
    """

    def __init__(self, dim, scale=16.0):
        super().__init__()
        half = dim // 2
        self.dim = dim
        self.register_buffer("freqs", torch.randn(half) * scale, persistent=True)

    def forward(self, t):
        args = 2.0 * math.pi * t.float().view(-1, 1) * self.freqs.view(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class _CondBlock(nn.Module):
    """ResBlock + optional self-attention + optional spectral conv, one call.

    Used identically on the down- and up- paths of `UNet` so level-building
    logic doesn't have to special-case attention/spectral wiring twice.
    """

    def __init__(self, ic, oc, ec, topo_channels, dropout, attn=False, spectral=False,
                 spectral_modes=(16, 16)):
        super().__init__()
        self.res = ResBlock(ic, oc, ec, topo_channels, dropout)
        self.attn = SelfAttn2d(oc, drop=dropout) if attn else None
        self.spec = SpectralConv2d(oc, modes=spectral_modes, drop=dropout) if spectral else None

    def forward(self, x, emb, topo=None):
        x = self.res(x, emb, topo)
        if self.attn is not None:
            x = self.attn(x)
        if self.spec is not None:
            x = self.spec(x)
        return x


class UNet(nn.Module):
    """Conditional U-Net for the Stage-2 flow-matching / EDM residual model.

    Matches how TrainDiffusion.py calls it:
      * `x`    : [B, in_channels, H, W]. Channel 0 is the noisy state (the
                 quantity actually being denoised/integrated); the remaining
                 channels are conditioning (Stage-1 reg_mean + upsampled LR),
                 per `fm_build_input` / the `torch.cat([c_in*xx, cond], 1)`
                 assembly in `edm_sample`.
      * `t`    : [B] raw timestep -- t*1000 for flow matching, c_noise for EDM.
      * `topo` : [B, topo_channels, H, W] at HR resolution; each ResBlock's
                 FiLM resizes it to that level's feature-map size internally,
                 so the same full-res tensor is passed unchanged at every depth.
      * `cfg_drop` : optional [B] bool. Where True, zeroes every conditioning
                 channel (everything in `x` after channel 0) and `topo`,
                 producing the unconditional forward pass used for
                 classifier-free guidance in fm_sample/edm_sample. This mirrors
                 CFG_DROP_P conditioning dropout applied at training time by
                 the (external) training loop.

    Standard ADM/guided-diffusion skip-connection bookkeeping: the down path
    pushes one skip per residual block plus one per downsample (all levels
    except the last also get a downsample skip); the up path always pops
    `num_res_blocks + 1` skips per level. The totals balance exactly because
    the extra "+1" at the innermost level is paid for by the stem's initial
    skip push -- this is the same pattern used by OpenAI's guided-diffusion /
    Stable Diffusion UNets, just re-expressed with this file's ResBlock/FiLM.
    """

    def __init__(self, in_channels, out_channels=1, base_channels=64,
                 channel_mult=(1, 2, 2, 4), num_res_blocks=2, dropout=0.1,
                 attn_levels=(2, 3), use_spectral=True, topo_channels=TOPO_CHANNELS,
                 spectral_modes=(16, 16), noise_embed_scale=16.0):
        super().__init__()
        self.topo_channels = topo_channels
        self.temb_dim = base_channels
        ec = base_channels * 4
        self.temb = FourierEmbedding(base_channels, scale=noise_embed_scale)
        self.temb_mlp = nn.Sequential(
            nn.Linear(base_channels, ec), nn.SiLU(), nn.Linear(ec, ec),
        )

        chs = [base_channels * m for m in channel_mult]
        self.stem = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        # ---- encoder ----
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        skip_chs = [chs[0]]
        c = chs[0]
        for lvl, oc in enumerate(chs):
            level = nn.ModuleList()
            for _ in range(num_res_blocks):
                attn = lvl in attn_levels
                level.append(_CondBlock(c, oc, ec, topo_channels, dropout,
                                         attn=attn, spectral=use_spectral and attn,
                                         spectral_modes=spectral_modes))
                c = oc
                skip_chs.append(c)
            self.down.append(level)
            if lvl != len(chs) - 1:
                self.downsample.append(Downsample(c))
                skip_chs.append(c)
            else:
                self.downsample.append(None)

        # ---- bottleneck ----
        self.mid = nn.ModuleList([
            _CondBlock(c, c, ec, topo_channels, dropout, attn=True,
                       spectral=use_spectral, spectral_modes=spectral_modes),
            _CondBlock(c, c, ec, topo_channels, dropout, attn=False, spectral=False),
        ])

        # ---- decoder ----
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for lvl, oc in reversed(list(enumerate(chs))):
            level = nn.ModuleList()
            attn = lvl in attn_levels
            for _ in range(num_res_blocks + 1):
                ic = c + skip_chs.pop()
                level.append(_CondBlock(ic, oc, ec, topo_channels, dropout,
                                         attn=attn, spectral=use_spectral and attn,
                                         spectral_modes=spectral_modes))
                c = oc
            self.up.append(level)
            self.upsample.append(Upsample(c) if lvl != 0 else None)

        self.out_norm = nn.GroupNorm(_g(c), c)
        self.out_conv = nn.Conv2d(c, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, topo=None, cfg_drop=None):
        if cfg_drop is not None:
            mask = cfg_drop.view(-1, 1, 1, 1).to(x.dtype)
            state, cond = x[:, :1], x[:, 1:]
            x = torch.cat([state, cond * (1.0 - mask)], dim=1)
            if topo is not None:
                topo = topo * (1.0 - mask)

        emb = self.temb_mlp(self.temb(t))

        h = self.stem(x)
        skips = [h]
        for lvl, level in enumerate(self.down):
            for blk in level:
                h = blk(h, emb, topo)
                skips.append(h)
            if self.downsample[lvl] is not None:
                h = self.downsample[lvl](h)
                skips.append(h)

        for blk in self.mid:
            h = blk(h, emb, topo)

        for i, level in enumerate(self.up):
            for blk in level:
                h = blk(torch.cat([h, skips.pop()], dim=1), emb, topo)
            if self.upsample[i] is not None:
                h = self.upsample[i](h)

        return self.out_conv(F.silu(self.out_norm(h)))


# ============================================================================
# 6. Precipitation Denormalization Utility
# ============================================================================

def denorm_precip_mmday(precip_norm, precip_transform_meta):
    """Model-normalized precip -> physical mm/day, using checkpoint meta.

    Exact inverse of Dataset.py's forward transform:
        raw_mm -> * precip_scale -> log1p -> (x - norm_mean) / norm_std

    The clamp is applied in log1p space (log1p(p) >= 0  <=>  p >= 0), which is
    the correct place for it: it enforces non-negative precipitation without
    distorting any physically valid value. `precip_scale` is divided back out
    -- it is 1.0 in the current pipeline, but omitting the division silently
    breaks any checkpoint written under the old PRECIP_SCALE=24.0 convention.
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
