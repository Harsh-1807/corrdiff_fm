# -*- coding: utf-8 -*-
"""
Tiling.py -- overlapping-tile machinery for the WGAN-GP Generator.
=====================================================================
Adapted from corrdiff_fm/Tiling.py, not copied verbatim: `blend_window` and
`tile_origins` are pure geometry and are unchanged, but the wrapper class is
rewritten because the underlying problem it solves is genuinely simpler here.

WHY TILING IS STILL NEEDED, EVEN THOUGH THE FAILURE MODE IS DIFFERENT
-----------------------------------------------------------------------
corrdiff_fm tiles because its networks are not resolution-agnostic at all
(SpectralConv2d's absolute FFT modes, SelfAttn2d's token-budget switch) --
handing them a differently-sized canvas is a genuinely different operator.
This package's Generator has neither block (see Network.py's module
docstring), so that specific failure mode does not apply here, and the
"Does PATCH pin the network to one grid size?" answer is materially weaker
for this architecture -- a plain fully-convolutional residual stack really is
close to resolution-agnostic. See the README section "Does the PATCH/tiling
caveat still apply here?" for the honest, non-oversold version of that claim.

But tiling is still not optional, for a boring and completely ordinary
convolutional-network reason: every conv in this Generator uses zero (or, at
the domain border, replicate) padding, which means a pixel near a tile edge
sees an artificial boundary that does not exist in the real, continuous 2 km
domain. Two independently-run tiles can therefore each produce a slightly
different answer for what should be the same underlying field near their
shared border, and naively concatenating the tiles side-by-side leaves a
visible seam exactly on that border. Overlapping tiles, blended with the same
raised-cosine window `blend_window` used in corrdiff_fm, fix this: every
pixel near a border gets a weighted combination of what BOTH overlapping
tiles produced for it, so the seam is smoothed away rather than left as a
hard discontinuity.

WHY THIS IS A STRICTLY SIMPLER PROBLEM THAN corrdiff_fm's
-------------------------------------------------------------
corrdiff_fm's Stage-2 sampler is iterative and stochastic: two adjacent tiles,
sampled independently, draw independent noise at EVERY denoising step and can
therefore commit to mutually inconsistent storm placements, so corrdiff_fm's
`TiledOperator` has to keep ONE global latent and blend the network's output
at every intermediate step to keep the whole domain on a single coherent
trajectory.

This Generator is a SINGLE feed-forward pass (see Network.py -- there is no
iterative refinement to keep synchronized), so there are no intermediate
steps to blend across tiles in the first place. The only trace of the
diffusion siblings' "independent randomness" concern that survives here is
the noise input z: if every tile drew its OWN independent z, adjacent tiles
could still place their sub-grid noise-driven structure inconsistently right
at the shared border. The fix is the same idea in miniature: `generate_tiled`
draws (or accepts) ONE noise field over the FULL LR-resolution domain and
slices it per tile exactly like the LR conditioning itself, so every tile
sees a spatially consistent piece of the same global noise draw rather than
an independent one. There is no risk of two tiles "disagreeing on storm
placement" in the corrdiff_fm sense, because there is no per-tile independent
randomness left to disagree -- z is shared, and blending only ever has to
smooth an ordinary convolutional border effect, not reconcile two competing
realizations.
"""

import math
import torch


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


class TiledGenerator:
    """Overlapping-tile wrapper around a `Network.Generator`.

    Unlike corrdiff_fm's `TiledOperator` (which exists to blend an iterative
    sampler's per-STEP output across tiles), this class blends a single
    feed-forward call per tile -- there is no step loop, because the
    Generator has no steps. `__call__` therefore does exactly one thing:
    slice the domain into overlapping tiles, run the Generator once on each,
    and blend the results with the same raised-cosine window used throughout
    this codebase family.

    Geometry (tile origins, blend window) is computed ONCE at construction and
    reused across repeated `__call__`s -- this matters because Inference.py
    calls it once per ensemble member (resampling z each time) and recomputing
    tile offsets and the blend window from scratch on every one of, say, 16
    ensemble draws would be pure waste.
    """

    def __init__(self, generator, topo_hr, ds_factor, tile_hr=None, overlap=None,
                 amp_dtype=torch.float32, max_tiles_per_batch=8):
        self.net = generator
        self.topo = topo_hr
        self.ds_factor = int(ds_factor)
        self.amp_dtype = amp_dtype
        self.max_tiles = int(max_tiles_per_batch)

        H, W = topo_hr.shape[-2:]
        self.H, self.W = H, W
        self.full_domain = tile_hr is None or (H <= tile_hr and W <= tile_hr)

        if self.full_domain:
            self.th, self.tw = H, W
            self.oy, self.ox = [0], [0]
            self.overlap = 0
            self.window = None
        else:
            assert tile_hr % self.ds_factor == 0, "tile_hr must be divisible by ds_factor"
            ov = int(overlap if overlap is not None else max(8, tile_hr // 4))
            ov -= ov % self.ds_factor
            self.overlap = ov
            self.th = min(tile_hr, H - H % self.ds_factor)
            self.tw = min(tile_hr, W - W % self.ds_factor)
            stride = max(self.ds_factor, self.th - ov)
            self.oy = [o - o % self.ds_factor for o in tile_origins(H, self.th, stride)]
            self.ox = [o - o % self.ds_factor for o in tile_origins(W, self.tw, stride)]
            self.window = blend_window(self.th, self.tw, ov, topo_hr.device)

        self.n_tiles = len(self.oy) * len(self.ox)

    @torch.no_grad()
    def __call__(self, lr, z=None, generator=None):
        """lr [B,Cin,h,w] coarse input at ITS OWN resolution (h = H/ds_factor).
        z, if given, must be a FULL-DOMAIN LR-resolution noise field
        [B,Z_CH,h,w] -- see module docstring for why sharing one global noise
        draw across tiles is what keeps tiled inference seam-free. If z is
        None, one is drawn here, once, over the full LR-resolution domain
        (not per-tile), using `generator` (a torch.Generator) if supplied for
        reproducibility.

        Returns HR precip [B, 1, H, W] in normalized log1p space.
        """
        use_amp = self.amp_dtype != torch.float32
        h_lr, w_lr = lr.shape[-2:]

        if z is None:
            z = self.net.sample_noise(lr, generator=generator)

        if self.full_domain:
            with torch.autocast(device_type=lr.device.type, dtype=self.amp_dtype, enabled=use_amp):
                return self.net(lr, self.topo, z=z).float()

        num = torch.zeros((lr.shape[0], 1, self.H, self.W), device=lr.device, dtype=torch.float32)
        den = torch.zeros((1, 1, self.H, self.W), device=lr.device, dtype=torch.float32)

        coords = [(i, j) for i in self.oy for j in self.ox]
        chunk = max(1, self.max_tiles // max(1, lr.shape[0]))

        for c0 in range(0, len(coords), chunk):
            group = coords[c0:c0 + chunk]
            il_list = [i // self.ds_factor for i, _ in group]
            jl_list = [j // self.ds_factor for _, j in group]
            th_lr, tw_lr = self.th // self.ds_factor, self.tw // self.ds_factor

            lr_batch = torch.cat(
                [lr[..., il:il + th_lr, jl:jl + tw_lr] for il, jl in zip(il_list, jl_list)], 0)
            z_batch = torch.cat(
                [z[..., il:il + th_lr, jl:jl + tw_lr] for il, jl in zip(il_list, jl_list)], 0)
            topo_batch = torch.cat(
                [self.topo[..., i:i + self.th, j:j + self.tw] for i, j in group], 0)

            with torch.autocast(device_type=lr.device.type, dtype=self.amp_dtype, enabled=use_amp):
                out = self.net(lr_batch, topo_batch, z=z_batch).float()

            B = lr.shape[0]
            for k, (i, j) in enumerate(group):
                num[..., i:i + self.th, j:j + self.tw] += out[k * B:(k + 1) * B] * self.window
                den[..., i:i + self.th, j:j + self.tw] += self.window

        return num / den


@torch.no_grad()
def generate_tiled(generator, lr, topo_hr, ds_factor, z=None, tile_hr=None,
                   overlap=None, amp_dtype=torch.float32, max_tiles_per_batch=8,
                   rng=None):
    """One-shot convenience wrapper around `TiledGenerator` for callers (e.g.
    Evaluate.py) that need exactly one tiled forward pass and do not benefit
    from caching the tile geometry across repeated calls. Prefer constructing
    `TiledGenerator` directly and reusing it when you will call it more than
    once with the same topo/tile_hr (e.g. once per ensemble member).
    """
    tiled = TiledGenerator(generator, topo_hr, ds_factor, tile_hr=tile_hr,
                          overlap=overlap, amp_dtype=amp_dtype,
                          max_tiles_per_batch=max_tiles_per_batch)
    return tiled(lr, z=z, generator=rng)
