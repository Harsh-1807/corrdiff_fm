# -*- coding: utf-8 -*-
"""
Diffusion.py -- EDM core for the CorrDiff residual corrector.
=============================================================
Single source of truth for the diffusion mathematics, shared verbatim by
TrainDiffusion.py and Inference.py. Keeping it in one file is not cosmetic:
every bug found in the previous version came from the training script and the
inference script disagreeing about a constant.

WHAT THE PAPER ACTUALLY SPECIFIES (Mardani et al. 2024, Sec. 5.2.2 / 5.3.2)
---------------------------------------------------------------------------
  residual        r = x - mu_hat        (mu_hat = frozen Stage-1 regression)
  training noise  ln(sigma) ~ N(0, 1.2^2)      i.e. P_mean = 0.0, P_std = 1.2
  sampler         EDM second-order STOCHASTIC sampler (Karras Algorithm 2)
  schedule        18 steps, sigma_max = 800, sigma_min = 0.002, rho = 7
  conditioning    coarse input channels + the Stage-1 mean, concatenated
                  with the noisy state over the channel dimension

Note P_mean = 0.0, NOT the -1.2 that EDM uses for natural images. The paper is
explicit about why: sigma_max must be large enough to "completely destruct the
large data intensity", which for a heavy-tailed geophysical residual means
pushing the noise distribution up, not down. Using -1.2 concentrates ~90% of
training samples below sigma = 1 and the model essentially never learns the
high-noise regime that the first few sampler steps live in.

CONVENTIONS FIXED HERE
----------------------
  * sigma_data = 1.0 because the residual is explicitly rescaled to unit
    variance before it ever reaches this module (see res_std in
    TrainDiffusion.py). sigma_data is the standard deviation of the data the
    denoiser sees; asserting 0.5 while feeding unit-variance data mis-scales
    c_skip, c_out and c_in simultaneously.
  * The loss is computed in PRECONDITIONED space (network output vs.
    (r - c_skip*z)/c_out) with UNIFORM weight 1.0. That is already exactly
    Karras' lambda(sigma) = 1/c_out^2 weighting expressed in the other basis.
    Multiplying by lambda(sigma) on top of dividing by c_out applies the
    weighting twice, which blows up the effective loss at small sigma by
    1/c_out^2 -- factors of 1e5 and up at the low end of the schedule.
"""

import math
import torch
from torch.amp import autocast

from Tiling import TiledOperator, regress_tiled  # noqa: F401  (re-exported)


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

class EDMConfig:
    """All EDM constants in one place; serialized into every checkpoint."""

    def __init__(self,
                 sigma_data=1.0,      # residual is pre-normalized to unit variance
                 sigma_min=0.002,     # paper 5.3.2
                 sigma_max=800.0,     # paper 5.3.2 (NOT the 80.0 image default)
                 rho=7.0,             # paper: "rest of hyperparameters from EDM"
                 P_mean=0.0,          # paper: ln(sigma) ~ N(0, 1.2^2)
                 P_std=1.2,
                 # Sampling-time sigma_max, if it should differ from the value
                 # above. None = use sigma_max.
                 #
                 # WHY THIS EXISTS. Note that sigma_max appears ONLY in
                 # karras_sigma_schedule -- training draws sigma from the
                 # log-normal and never consults it. So sigma_max is purely a
                 # sampler choice and can be changed without retraining.
                 #
                 # That matters, because with sigma_data = 1.0 the paper's
                 # sigma_max = 800 spreads an 18-step schedule so thin that
                 # only 2 of the 18 levels land in the decisive band around
                 # sigma ~ sigma_data. Measured against an exact analytic
                 # denoiser on unit-variance data (probe_steps.py):
                 #
                 #   steps  smax=800/churn0  smax=800/churn40  smax=80/churn40
                 #      18       1.106            1.210            1.111
                 #      32       1.029            1.082            1.047
                 #      64       1.006            1.038            1.021
                 #
                 # against a true std of 1.000. That is a 10-21% ensemble
                 # OVER-dispersion of purely numerical origin, which lands
                 # directly on the spread/skill ratio you are trying to
                 # calibrate. It is discretization error, not a bug -- it
                 # converges away with steps.
                 #
                 # Options, in order of preference:
                 #   * sigma_max_sample = 200 with 32 steps  (well under 3%)
                 #   * keep 800 and raise steps to 64
                 #   * keep the paper's 800/18 for fidelity to the reference,
                 #     and simply KNOW that a few percent of your spread is
                 #     the integrator rather than the model
                 sigma_max_sample=None,
                 # Algorithm 2 stochasticity. S_churn = 0 degrades to the
                 # deterministic Heun ODE solver; the paper uses the stochastic
                 # variant, which is what restores the small-scale variance the
                 # spectra in Fig. 2 depend on. Note from the table above that
                 # churn AMPLIFIES the discretization error, because noise
                 # injected late has fewer steps left in which to be removed.
                 S_churn=40.0,
                 S_min=0.05,
                 S_max=50.0,
                 S_noise=1.003):
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_max_sample = None if sigma_max_sample is None else float(sigma_max_sample)
        self.S_churn = float(S_churn)
        self.S_min = float(S_min)
        self.S_max = float(S_max)
        self.S_noise = float(S_noise)

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        base = cls().__dict__
        return cls(**{k: v for k, v in (d or {}).items() if k in base})

    def __repr__(self):
        return f"EDMConfig({self.to_dict()})"


# =============================================================================
# 2. PRECONDITIONING AND NOISE DISTRIBUTION
# =============================================================================

def edm_precond(sigma, sigma_data):
    """Karras et al. Table 1, 'EDM' row.

    Returns c_skip, c_out, c_in shaped [B,1,1,1] for broadcasting against
    [B,C,H,W], and c_noise shaped [B] for the embedding.
    """
    s = sigma.view(-1, 1, 1, 1).float()
    sd2 = sigma_data ** 2
    c_skip = sd2 / (s ** 2 + sd2)
    c_out = s * sigma_data / torch.sqrt(s ** 2 + sd2)
    c_in = 1.0 / torch.sqrt(s ** 2 + sd2)
    c_noise = torch.log(sigma.view(-1).float()) / 4.0
    return c_skip, c_out, c_in, c_noise


def sample_sigma_lognormal(n, device, cfg):
    """ln(sigma) ~ N(P_mean, P_std^2)  ->  sigma = exp(P_mean + P_std * eps).

    This is the literal reading of the paper's
    "sigma ~ lognormal(mu = 0.0, sigma = 1.2)": the parameters are those of
    the underlying NORMAL, not of the lognormal itself. With P_mean = 0 the
    median sigma is 1.0 and the interquartile range is roughly [0.44, 2.25],
    with a tail reaching sigma ~ 40 at four standard deviations -- which is
    what lets the model cover a schedule that starts at sigma_max = 800.

    Deliberately NOT clamped to [sigma_min, sigma_max]: clamping piles up
    probability mass on the two endpoints and biases the denoiser there.
    """
    eps = torch.randn(n, device=device)
    return torch.exp(cfg.P_mean + cfg.P_std * eps)


def karras_sigma_schedule(n_steps, device, cfg):
    """Time-step discretization, Karras Eq. (5). Returns n_steps+1 values,
    monotonically decreasing, with a hard 0 appended as the final level."""
    i = torch.arange(n_steps, device=device, dtype=torch.float64)
    smax = cfg.sigma_max_sample if cfg.sigma_max_sample is not None else cfg.sigma_max
    a = smax ** (1.0 / cfg.rho)
    b = cfg.sigma_min ** (1.0 / cfg.rho)
    s = (a + i / max(1, n_steps - 1) * (b - a)) ** cfg.rho
    return torch.cat([s, torch.zeros(1, device=device, dtype=torch.float64)]).float()


# =============================================================================
# 3. THE DENOISER D_theta
# =============================================================================

def edm_denoise(net, x, sigma, cond, topo, cfg, amp_dtype=torch.float32,
                guidance=1.0):
    """D_theta(x; sigma, cond) -- the preconditioned denoiser.

    `x`     [B,1,H,W] noisy residual
    `sigma` scalar tensor or [B] tensor of noise levels
    `cond`  [B,Ccond,H,W] conditioning stack (Stage-1 mean + upsampled LR)
    """
    B = x.shape[0]
    if sigma.dim() == 0:
        sigma = sigma.expand(B)
    c_skip, c_out, c_in, c_noise = edm_precond(sigma, cfg.sigma_data)

    inp = torch.cat([(c_in * x).to(x.dtype), cond], dim=1)
    use_amp = amp_dtype != torch.float32
    with autocast(device_type=x.device.type, dtype=amp_dtype, enabled=use_amp):
        raw = net(inp, c_noise, topo).float()
        if guidance != 1.0:
            drop = torch.ones(B, dtype=torch.bool, device=x.device)
            raw_u = net(inp, c_noise, topo, cfg_drop=drop).float()
            raw = raw_u + guidance * (raw - raw_u)
    return c_skip * x.float() + c_out * raw


def edm_training_target(r, sigma, cfg):
    """Returns (noisy_state, network_target, c_in) for one training step.

    The network is asked to predict  F_theta = (r - c_skip * z) / c_out,
    and the loss on F_theta is taken with UNIFORM weight. See the module
    docstring for why no extra lambda(sigma) belongs here.
    """
    noise = torch.randn_like(r)
    s = sigma.view(-1, 1, 1, 1)
    z = r + noise * s
    c_skip, c_out, c_in, c_noise = edm_precond(sigma, cfg.sigma_data)
    target = (r - c_skip * z) / c_out
    return z, target, c_in, c_noise


# =============================================================================
# 4. SAMPLERS
# =============================================================================

@torch.no_grad()
def edm_stochastic_sample(denoise_fn, shape, device, cfg, n_steps=18,
                          generator=None, dtype=torch.float32):
    """EDM Algorithm 2 -- second-order stochastic sampler. Verbatim.

    `denoise_fn(x, sigma_scalar) -> D_theta(x; sigma)` is injected so that the
    identical integrator drives both the plain single-tile path and the tiled
    MultiDiffusion path used at 2 km. There is exactly one implementation of
    the integrator in this codebase, and this is it.

    The churn step (gamma > 0) is what makes this the *stochastic* sampler:
    each iteration first pushes the sample back UP to a slightly higher noise
    level by injecting fresh Gaussian noise, then denoises down past where it
    started. That re-injection is the mechanism by which the sampler keeps
    generating new fine-scale structure instead of collapsing onto the
    conditional mean, and it is why the deterministic ODE variant (S_churn=0)
    produces visibly under-dispersed, over-smooth precipitation fields.
    """
    sig = karras_sigma_schedule(n_steps, device, cfg)
    x = torch.randn(shape, device=device, generator=generator, dtype=torch.float32) * sig[0]

    gamma_base = min(cfg.S_churn / max(1, n_steps), math.sqrt(2.0) - 1.0)

    for i in range(n_steps):
        s_cur, s_next = sig[i], sig[i + 1]

        # --- stochastic churn: increase noise level s_cur -> s_hat ---------
        gamma = gamma_base if (cfg.S_min <= s_cur <= cfg.S_max) else 0.0
        s_hat = s_cur * (1.0 + gamma)
        if gamma > 0:
            eps = torch.randn(x.shape, device=device, generator=generator,
                              dtype=torch.float32) * cfg.S_noise
            x = x + torch.sqrt(torch.clamp(s_hat ** 2 - s_cur ** 2, min=0.0)) * eps

        # --- Euler step at s_hat -------------------------------------------
        d_cur = (x - denoise_fn(x, s_hat)) / s_hat
        x_next = x + (s_next - s_hat) * d_cur

        # --- 2nd-order (Heun) correction, skipped on the final step --------
        if s_next > 0:
            d_prime = (x_next - denoise_fn(x_next, s_next)) / s_next
            x_next = x + (s_next - s_hat) * 0.5 * (d_cur + d_prime)

        x = x_next

    return x


@torch.no_grad()
def edm_deterministic_sample(denoise_fn, shape, device, cfg, n_steps=18,
                             generator=None, dtype=torch.float32):
    """Algorithm 1 (probability-flow ODE, Heun). Provided for ablation only --
    use it to demonstrate the spread collapse, not to produce ensembles."""
    ablation = EDMConfig(**{**cfg.to_dict(), "S_churn": 0.0})
    return edm_stochastic_sample(denoise_fn, shape, device, ablation,
                                 n_steps=n_steps, generator=generator, dtype=dtype)


# =============================================================================
# 5. TILED (MULTIDIFFUSION) DENOISER -- the 8 km -> 2 km enabler
# =============================================================================

class TiledDenoiser(TiledOperator):
    """D_theta evaluated in training-sized tiles and blended across overlaps.

    See Tiling.py for the full rationale. In brief: the network must see the
    same grid size it trained on, and what gets averaged is the DENOISER OUTPUT
    at every sampler step -- never independently sampled tiles, which would
    disagree with each other and blur at the seams.
    """

    def apply(self, xs, sigma, cs, tp):
        return edm_denoise(self.net, xs, sigma, cs, tp, self.cfg,
                           amp_dtype=self.amp_dtype, guidance=self.guidance)


def make_denoise_fn(net, cond, topo, cfg, tile=None, overlap=None,
                    amp_dtype=torch.float32, guidance=1.0,
                    max_tiles_per_batch=8):
    """Returns `denoise_fn(x, sigma)`, tiled if and only if needed.

    When the conditioning grid already matches the training tile, the closure
    calls the network directly -- so training-time validation and
    deployment-time tiled inference are the same code path with the same
    numerics, and there is nothing to drift out of sync.
    """
    H, W = cond.shape[-2:]
    if tile is None or (H <= tile and W <= tile):
        def fn(x, sigma):
            if not torch.is_tensor(sigma):
                sigma = torch.as_tensor(float(sigma), device=x.device)
            return edm_denoise(net, x, sigma, cond, topo, cfg,
                               amp_dtype=amp_dtype, guidance=guidance)
        return fn
    return TiledDenoiser(net, cond, topo, cfg, tile, overlap=overlap,
                         amp_dtype=amp_dtype, guidance=guidance,
                         max_tiles_per_batch=max_tiles_per_batch)


def nfe_count(n_steps):
    """Network evaluations per ensemble member.

    Algorithm 2 costs 2 per step except the last (which skips the corrector),
    and the churn step is network-free. Identical to Heun flow matching at equal
    n_steps -- print it in any results table, because an NFE-unmatched
    comparison is not a comparison.
    """
    return 2 * n_steps - 1
