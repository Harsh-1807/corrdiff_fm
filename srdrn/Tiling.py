# -*- coding: utf-8 -*-
"""
Tiling.py -- overlapping-tile machinery for SRDRN inference.
==============================================================
Adapted, not copied blindly, from corrdiff_fm/Tiling.py. `blend_window` and
`tile_origins` are kept verbatim -- they are pure geometry (a raised-cosine
partition-of-unity blend and a tile-origin generator) with nothing
diffusion-specific about them. What is NOT kept is corrdiff_fm's generic
`TiledOperator` class and its Stage-2-specific "blend the sampler's per-step
output, not the finished sample" logic: that machinery exists to solve a
problem (independent noise draws per tile producing mutually inconsistent
storm placements at tile seams) that simply does not arise here. SRDRN has no
noise input and no sampler loop -- it is a single deterministic forward pass,
so there is exactly one output per tile to blend, not one per sampler step.
That is a large simplification, made possible by SRDRN being purely
feed-forward, not an oversight.

WHY TILE AT ALL, THEN? (an honest, weaker case than corrdiff_fm's)
--------------------------------------------------------------------
corrdiff_fm's README states its tiling necessity in strong terms: `SpectralConv2d`
ties learned weights to absolute FFT mode indices and `SelfAttn2d` switches to
pooled attention past a token budget, so handing that network a different grid
size is a GENUINELY DIFFERENT OPERATOR, not the same one applied more times.

SRDRN's trunk contains neither block. Every layer in `Network.SRDRN` -- conv,
BatchNorm, PReLU, PixelShuffle -- is a strictly local, translation-equivariant
operator whose per-pixel output does not depend on the overall canvas size.
Feeding it a full 8000x8000 domain in one shot would, MODULO EDGE EFFECTS AND
MEMORY, produce the identical result to tiling it and blending the tiles: this
is a genuinely weaker justification for tiling than corrdiff_fm has, and this
module says so rather than overclaiming the same severity. Three real, but
smaller, reasons to tile anyway:

  1. MEMORY. A domain-wide forward pass through 16 BatchNorm-heavy residual
     blocks plus two upsampling blocks, at HR (4x) resolution, over a
     multi-decade, multi-region NetCDF domain, can exceed a single GPU's
     memory even though it would be numerically fine given infinite memory.
     Tiling turns an "impossible in one shot" forward pass into a sequence of
     small ones. This is the DOMINANT real reason to tile SRDRN, honestly --
     a practical memory constraint, not a correctness requirement the way
     corrdiff_fm's is.

  2. BATCHNORM STATISTIC MATCHING. SRDRN's BatchNorm layers run in eval mode
     at inference, using running statistics (mean/var) accumulated during
     training over PATCH-sized (Config.PATCH) crops. Those running statistics
     describe the distribution of per-channel feature ACTIVATIONS the network
     produced when looking at patch-sized fields. A full-domain forward pass
     still uses the same fixed running stats (BatchNorm-eval does not
     recompute statistics from the current input), so this does not change
     the FORMULA the network applies -- but if the actual feature-activation
     distribution over a huge domain differs systematically from what a
     PATCH-sized crop looks like (e.g. a domain-spanning tile mixes very wet
     and very dry sub-regions whose combined statistics no single training
     crop ever exhibited), the network is being asked to extrapolate slightly
     outside the input distribution its BatchNorm affine parameters were
     tuned against. Keeping the inference-time tile size equal to the
     training patch size removes this concern by construction, at the cost of
     needing to blend tiles back together. This is real but strictly smaller
     than corrdiff_fm's issue: there, mismatched grid size changes the
     OPERATOR itself; here, it can only shift an eval-mode normalization
     layer's input distribution slightly off-training-distribution.

  3. SEAMS FROM CONV PADDING. Every 3x3 conv in this network pads its input,
     so a pixel near a TILE boundary sees zero/replicate-padded context
     instead of the true neighbouring field one tile over. This is an
     artifact tiling ITSELF introduces (a full single-shot forward pass over
     the true domain has no such artificial boundaries, only the real domain
     edge) -- overlap-blending several offset tilings is the standard fix,
     trading a bit of redundant compute for removing the seam.

In short: tile SRDRN mainly so a large domain fits in memory at all, and
blend the tiles' outputs (reason 3) mainly to undo the seam that tiling itself
creates, while reason 2 is a secondary correctness argument for keeping the
tile size pinned to the training patch size specifically (rather than any
memory-feasible size).
"""

import math
import torch
from torch.amp import autocast


def blend_window(h, w, overlap, device):
    """Separable raised-cosine ramp: 1 in the tile interior, tapering smoothly
    to ~0 across the overlap band on each edge.

    Weights are accumulated and divided out by the caller, so the taper only has
    to be smooth -- it does not have to sum to one by itself. That makes the
    blend exact for any tile layout, including the flush-to-edge final tile.
    """
    def ramp(n):
        v = torch.ones(n, device=device, dtype=torch.float32)
        k = min(overlap, n // 2)
        if k > 0:
            t = torch.linspace(0, math.pi, 2 * k + 2, device=device)[1:k + 1]
            edge = (1.0 - torch.cos(t)) * 0.5
            v[:k] = edge
            v[-k:] = edge.flip(0)
        return v
    return (ramp(h).view(1, 1, h, 1) * ramp(w).view(1, 1, 1, w)).clamp_min(1e-6)


def tile_origins(total, tile, stride):
    """Tile start offsets covering [0, total).

    The last offset is snapped flush to the far edge so the domain boundary is
    never left unmodelled -- a plain range() would drop up to `stride-1` columns
    off the end, which at fine resolution is real physical distance of
    coastline or terrain left un-downscaled.
    """
    if tile >= total:
        return [0]
    outs = list(range(0, total - tile + 1, stride))
    if outs[-1] != total - tile:
        outs.append(total - tile)
    return outs


# =============================================================================
# SRDRN TILED REGRESSION
# =============================================================================
# Named `regress_tiled` to mirror corrdiff_fm's own function of the same name
# for its Stage-1 CorrDiffRegressor -- SRDRN plays exactly that structural
# role here (a deterministic LR+topo -> HR conditional-mean network, just a
# deeper, paper-faithful one), so the same name signals the same job rather
# than inventing new vocabulary for an equivalent thing.

@torch.no_grad()
def regress_tiled(model, lr, topo_hr, ds_factor, tile_hr=None, overlap=None,
                  amp_dtype=torch.float32):
    """SRDRN forward pass, tiled to bound memory and to keep the inference-time
    grid size equal to the training patch size (see module docstring).

    `lr`      [B,Cin,h,w]  coarse input at its own resolution
    `topo_hr` [B,Ct,H,W]   topography on the OUTPUT grid, H = h*ds_factor
    `tile_hr` tile size in OUTPUT pixels (must be a multiple of ds_factor);
              None (or a tile at least as large as the domain) runs the whole
              domain in a single forward pass.

    Unlike corrdiff_fm's Stage-2 sampler-blending, there is only ONE quantity
    to blend per tile here (SRDRN's single deterministic output), so this
    reduces to: run the network on each overlapping tile, weight each tile's
    output by a raised-cosine window, accumulate, and divide by the summed
    window -- ordinary MultiDiffusion-style blending with the "diffusion"
    part removed, since there is no sampler loop to blend across.
    """
    H, W = topo_hr.shape[-2:]
    use_amp = amp_dtype != torch.float32

    if tile_hr is None or (H <= tile_hr and W <= tile_hr):
        with autocast(device_type=lr.device.type, dtype=amp_dtype, enabled=use_amp):
            return model(lr, topo_hr).float()

    assert tile_hr % ds_factor == 0, "tile_hr must be divisible by ds_factor"
    ov = int(overlap if overlap is not None else max(8, tile_hr // 4))
    ov -= ov % ds_factor
    th = min(tile_hr, H - H % ds_factor)
    tw = min(tile_hr, W - W % ds_factor)
    stride = max(ds_factor, th - ov)
    oy = [o - o % ds_factor for o in tile_origins(H, th, stride)]
    ox = [o - o % ds_factor for o in tile_origins(W, tw, stride)]

    window = blend_window(th, tw, ov, lr.device)
    num = torch.zeros((lr.shape[0], 1, H, W), device=lr.device, dtype=torch.float32)
    den = torch.zeros((1, 1, H, W), device=lr.device, dtype=torch.float32)

    for i in oy:
        for j in ox:
            il, jl = i // ds_factor, j // ds_factor
            lr_t = lr[..., il:il + th // ds_factor, jl:jl + tw // ds_factor]
            tp_t = topo_hr[..., i:i + th, j:j + tw]
            with autocast(device_type=lr.device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(lr_t, tp_t).float()
            num[..., i:i + th, j:j + tw] += out * window
            den[..., i:i + th, j:j + tw] += window

    return num / den


class TiledRegressor:
    """Thin object wrapper around `regress_tiled`, mirroring corrdiff_fm's
    naming convention (Config-driven, callable) for callers that prefer an
    object with bound tile/overlap settings over passing them at every call
    site. Train.py, Evaluate.py and Inference.py all use the plain
    `regress_tiled` function directly (matching corrdiff_fm's own Regressor.py
    usage pattern); this class exists purely as an ergonomic alternative and
    is not itself exercised by this package's scripts.
    """

    def __init__(self, model, ds_factor, tile_hr=None, overlap=None,
                 amp_dtype=torch.float32):
        self.model = model
        self.ds_factor = ds_factor
        self.tile_hr = tile_hr
        self.overlap = overlap
        self.amp_dtype = amp_dtype

    def __call__(self, lr, topo_hr):
        return regress_tiled(self.model, lr, topo_hr, self.ds_factor,
                              tile_hr=self.tile_hr, overlap=self.overlap,
                              amp_dtype=self.amp_dtype)
