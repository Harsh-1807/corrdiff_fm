# -*- coding: utf-8 -*-
"""
Tiling.py -- overlapping-tile machinery shared by Stage 1 and Stage 2.
=======================================================================
Identical in the EDM and flow-matching packages. Nothing here is specific to
either parameterization: it is geometry plus a partition-of-unity blend.

WHY TILING IS NOT OPTIONAL FOR THE 8 km -> 2 km CASCADE
--------------------------------------------------------
The cascade works because the learned operator is scale-RELATIVE: "given an area
mean and fine topography, restore 4x of sub-grid structure". Nothing in that
statement names a resolution, and the networks are fully convolutional.

Except that two blocks in Network.py are not resolution-agnostic:

  * SpectralConv2d learns weights on ABSOLUTE FFT mode indices. Mode k on a
    176x176 grid and mode k on a 704x704 grid are different physical
    wavelengths.
  * SelfAttn2d silently switches from exact attention to attention over an
    adaptively pooled grid once H*W exceeds its token budget.

So the network's behaviour is a function of the input GRID SIZE, not only of the
local field, and handing it a 4x-larger canvas is a genuinely different operator
rather than the same one applied more times. A shape assertion will not catch
this -- the output is still exactly 4x the input; it is just wrong.

The fix is to pin the grid size the network sees to the training patch size and
cover the large domain with overlapping tiles.

WHAT GETS BLENDED, AND WHY IT MATTERS
--------------------------------------
For Stage 2 the naive approach -- sample each tile independently, then feather
the results together -- fails badly for a stochastic model. Adjacent tiles draw
independent noise, commit to different and mutually inconsistent storm
placements, and feathering two disagreeing realizations produces a blurred seam
exactly where you least want one.

Instead, callers keep ONE global latent with ONE global noise draw and blend the
network's OUTPUT at every sampler step (D_theta for EDM, v_theta for flow
matching). Both are linear in the quantity the integrator consumes, so averaging
them in the overlaps is well-posed. Every tile then integrates the same
trajectory and the result is a single globally coherent realization.
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
    off the end, which at 2 km is tens of kilometres of coastline.
    """
    if tile >= total:
        return [0]
    outs = list(range(0, total - tile + 1, stride))
    if outs[-1] != total - tile:
        outs.append(total - tile)
    return outs


class TiledOperator:
    """Generic overlapping-tile wrapper around a network call.

    Subclasses supply `apply(x_tiles, aux, cond_tiles, topo_tiles)`; everything
    else -- geometry, batching of tiles, blending -- lives here so the EDM and
    flow-matching versions cannot drift apart.
    """

    def __init__(self, net, cond, topo, cfg, tile, overlap=None,
                 amp_dtype=torch.float32, guidance=1.0, max_tiles_per_batch=8):
        self.net, self.cond, self.topo, self.cfg = net, cond, topo, cfg
        self.tile = int(tile)
        self.overlap = int(overlap if overlap is not None else max(8, self.tile // 4))
        self.amp_dtype, self.guidance = amp_dtype, guidance
        self.max_tiles = int(max_tiles_per_batch)

        H, W = cond.shape[-2:]
        self.th, self.tw = min(self.tile, H), min(self.tile, W)
        self.oy = tile_origins(H, self.th, max(1, self.th - self.overlap))
        self.ox = tile_origins(W, self.tw, max(1, self.tw - self.overlap))
        self.window = blend_window(self.th, self.tw, self.overlap, cond.device)
        self.n_tiles = len(self.oy) * len(self.ox)

    def apply(self, xs, aux, cs, tp):
        raise NotImplementedError

    def __call__(self, x, aux):
        B = x.shape[0]
        num = torch.zeros_like(x, dtype=torch.float32)
        den = torch.zeros((1, 1, *x.shape[-2:]), device=x.device, dtype=torch.float32)

        if not torch.is_tensor(aux):
            aux = torch.as_tensor(float(aux), device=x.device)

        coords = [(i, j) for i in self.oy for j in self.ox]
        chunk = max(1, self.max_tiles // max(1, B))

        for c0 in range(0, len(coords), chunk):
            group = coords[c0:c0 + chunk]
            xs = torch.cat([x[..., i:i + self.th, j:j + self.tw] for i, j in group], 0)
            cs = torch.cat([self.cond[..., i:i + self.th, j:j + self.tw] for i, j in group], 0)
            tp = torch.cat([self.topo[..., i:i + self.th, j:j + self.tw] for i, j in group], 0)
            a = aux.expand(xs.shape[0]) if aux.dim() == 0 else aux.repeat(len(group))

            out = self.apply(xs, a, cs, tp)

            for k, (i, j) in enumerate(group):
                num[..., i:i + self.th, j:j + self.tw] += out[k * B:(k + 1) * B] * self.window
                den[..., i:i + self.th, j:j + self.tw] += self.window

        return num / den


# =============================================================================
# STAGE-1 TILED REGRESSION
# =============================================================================

@torch.no_grad()
def regress_tiled(regressor, lr, topo_hr, ds_factor, tile_hr=None, overlap=None,
                  amp_dtype=torch.float32):
    """Stage-1 conditional mean, tiled for the same reason as Stage 2.

    `lr`      [B,Cin,h,w]  coarse input at its own resolution
    `topo_hr` [B,Ct,H,W]   topography on the OUTPUT grid, H = h*ds_factor
    `tile_hr` tile size in OUTPUT pixels (must be a multiple of ds_factor)

    CorrDiffRegressor carries self-attention and a dilated bottleneck at the LR
    resolution, so its effective receptive field -- and, past the token budget,
    its attention operator -- also depend on input grid size. Same disease as
    Stage 2, same cure.
    """
    H, W = topo_hr.shape[-2:]
    use_amp = amp_dtype != torch.float32

    if tile_hr is None or (H <= tile_hr and W <= tile_hr):
        with autocast(device_type=lr.device.type, dtype=amp_dtype, enabled=use_amp):
            return regressor(lr, topo_hr).float()

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
                out = regressor(lr_t, tp_t).float()
            num[..., i:i + th, j:j + tw] += out * window
            den[..., i:i + th, j:j + tw] += window

    return num / den
