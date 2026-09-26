# -*- coding: utf-8 -*-
"""
Tiling.py -- overlapping-tile machinery for the SRGAN generator.
==================================================================
Adapted from corrdiff_fm/Tiling.py, not copied verbatim: that file's
`TiledOperator` exists to blend an ITERATIVE sampler's per-step output across
tiles (many denoiser/velocity calls per tile, one global noise draw, blend
every step so adjacent tiles integrate the same trajectory). SRGAN's generator
needs none of that -- it is a deterministic, single-shot feed-forward network,
so there is exactly ONE network call per tile, and the wrapper below
(`TiledGenerator`) is correspondingly simpler: geometry + one forward pass +
blend, once.

WHY TILING IS STILL WORTH DOING HERE AT ALL
--------------------------------------------
This is a WEAKER version of corrdiff_fm's resolution-transfer problem, and it
is worth being precise about exactly how much weaker, rather than either
overstating it (as if nothing here needed tiling) or ignoring it (as if the
same argument applied at full force):

  * corrdiff_fm's Stage-2 U-Net contains `SpectralConv2d` (learns weights on
    ABSOLUTE FFT mode indices -- mode k means a different physical wavelength
    on a different-sized grid) and `SelfAttn2d` (silently switches from exact
    to pooled attention once H*W crosses a token budget). Both make the
    network's behaviour a function of input GRID SIZE, not just of the local
    field -- a strictly different operator on a larger canvas, not the same
    operator applied more times.

  * This package's Generator has NEITHER block. It is entirely convolutional
    with a fixed, finite receptive field (9x9 stem + 16x(3x3+3x3) residual
    blocks + 3x3 fusion convs + 9x9 output = a large but FIXED number of
    pixels, independent of input size) and BatchNorm layers whose behaviour at
    inference (`model.eval()`) uses the FROZEN RUNNING statistics recorded
    during training, not the statistics of whatever canvas you hand it. A
    BatchNorm layer in eval mode applies `(x - running_mean) / sqrt(running_var
    + eps) * gamma + beta` identically regardless of the spatial extent of x --
    feeding it a much larger 2 km canvas does not change what that affine
    transform does to any given pixel's activations. So handing this generator
    a full 2 km domain in one forward pass would NOT silently swap in a
    different learned operator the way it would for corrdiff_fm's Stage 2.

  * What tiling DOES still fix here: finite-receptive-field boundary effects.
    Every conv in this network zero/replicate-pads at its own edges. On a
    tile, "the edge" is a real domain boundary maybe 1-in-many times, but for
    every interior seam introduced by splitting a large domain into tiles, it
    is an ARTIFICIAL edge the network was never trained to see (training crops
    are drawn from the interior of the full domain far more often than they
    coincide with the actual domain boundary). A naive quilt of independently
    generated tiles would show visible seams at every tile boundary from this
    edge-effect mismatch, exactly the way any patch-based conv network does.
    The raised-cosine blend below is what removes that, by feathering
    independently-computed tiles together in their overlap band rather than
    concatenating them at a hard edge.

  * Also, plainly: a 2 km domain is DS_FACTOR^2 = 16x the pixel COUNT of the
    8 km domain the network trained on (4x the linear extent). Running the
    whole thing through 16 residual blocks at once in one forward pass may
    simply not fit in GPU memory, independent of any of the above -- tiling is
    also just how you make inference feasible at all.

So: tile because of boundary seams and memory, not because the network computes
a different function at a different grid size (it doesn't, to first order).
That is a real difference in KIND from corrdiff_fm's cascade justification, not
merely a smaller version of the same number -- say so plainly rather than
recycling that file's language unchanged.
"""

import math
import torch
import torch.nn.functional as F


def blend_window(h, w, overlap, device):
    """Separable raised-cosine ramp: 1 in the tile interior, tapering smoothly
    to ~0 across the overlap band on each edge. Identical to corrdiff_fm's
    (pure geometry, nothing parameterization-specific about it).

    Weights are accumulated and divided out by the caller, so the taper only
    has to be smooth -- it does not have to sum to one by itself. That makes
    the blend exact for any tile layout, including the flush-to-edge final
    tile.
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
    never left unmodelled -- a plain range() would drop up to `stride-1`
    columns off the end, which at 2 km is tens of kilometres of coastline.
    """
    if tile >= total:
        return [0]
    outs = list(range(0, total - tile + 1, stride))
    if outs[-1] != total - tile:
        outs.append(total - tile)
    return outs


class TiledGenerator:
    """One-shot tiled wrapper around `Generator.forward(lr, topo_hr)`.

    Mirrors corrdiff_fm's `regress_tiled` (its Stage-1 conditional-mean
    regressor has exactly the same `forward(lr, topo)` signature and the same
    "coarse input at LR resolution, topo at HR resolution" contract), renamed
    to make it obvious this is the SRGAN package's tiled inference path and
    simplified for a single-shot generator: no autocast dtype plumbing beyond
    what the generator call itself needs, no multi-step sampler loop, no
    per-tile aux/conditioning tensors beyond `topo` -- everything the network
    needs besides `lr` lives in `topo` (already includes coordinate channels)
    plus whatever the caller passed as `lr` itself.

    Usage
    -----
        tg = TiledGenerator(generator, topo_hr, ds_factor=4, tile_hr=128)
        precip_hr_norm = tg(lr)      # lr: [B, IN_CH_LR, H/ds, W/ds]
    """

    def __init__(self, generator, topo_hr, ds_factor, tile_hr=None, overlap=None,
                 amp_dtype=torch.float32):
        self.generator = generator
        self.topo = topo_hr
        self.ds_factor = int(ds_factor)
        self.tile_hr = tile_hr
        self.overlap = overlap
        self.amp_dtype = amp_dtype

    @torch.no_grad()
    def __call__(self, lr):
        H, W = self.topo.shape[-2:]
        ds = self.ds_factor
        use_amp = self.amp_dtype != torch.float32
        B = lr.shape[0]
        topo = self.topo.expand(B, -1, -1, -1) if self.topo.shape[0] == 1 else self.topo

        if self.tile_hr is None or (H <= self.tile_hr and W <= self.tile_hr):
            with torch.autocast(device_type=lr.device.type, dtype=self.amp_dtype, enabled=use_amp):
                return self.generator(lr, topo).float()

        tile_hr = self.tile_hr
        assert tile_hr % ds == 0, "tile_hr must be divisible by ds_factor"
        ov = int(self.overlap if self.overlap is not None else max(8, tile_hr // 4))
        ov -= ov % ds
        th = min(tile_hr, H - H % ds)
        tw = min(tile_hr, W - W % ds)
        stride = max(ds, th - ov)
        oy = [o - o % ds for o in tile_origins(H, th, stride)]
        ox = [o - o % ds for o in tile_origins(W, tw, stride)]

        window = blend_window(th, tw, ov, lr.device)
        num = torch.zeros((B, 1, H, W), device=lr.device, dtype=torch.float32)
        den = torch.zeros((1, 1, H, W), device=lr.device, dtype=torch.float32)

        for i in oy:
            for j in ox:
                il, jl = i // ds, j // ds
                lr_t = lr[..., il:il + th // ds, jl:jl + tw // ds]
                tp_t = topo[..., i:i + th, j:j + tw]
                with torch.autocast(device_type=lr.device.type, dtype=self.amp_dtype, enabled=use_amp):
                    out = self.generator(lr_t, tp_t).float()
                num[..., i:i + th, j:j + tw] += out * window
                den[..., i:i + th, j:j + tw] += window

        return num / den
