# -*- coding: utf-8 -*-
"""
Param.py -- EDM parameterization adapter.
==========================================
THIS FILE AND `Diffusion.py` ARE THE ONLY THINGS THAT DIFFER BETWEEN THE EDM AND
FLOW-MATCHING PACKAGES. Everything else -- Config, Dataset, Network, Tiling,
Regressor, TrainStage2, Inference, Evaluate -- is byte-identical.

That is the point. A comparison between two generative parameterizations is only
worth as much as the amount of code the two arms hold in common, and here the
shared surface is everything except a noise schedule and a loss target.

WHAT THIS ARM IMPLEMENTS
------------------------
Karras et al. EDM, as specified in Mardani et al. Sec. 5.3.2:
    ln(sigma) ~ N(0, 1.2^2)          -- P_mean = 0.0, NOT the -1.2 image default
    18 steps, sigma_max = 800, sigma_min = 0.002, rho = 7
    second-order STOCHASTIC sampler (Algorithm 2, with churn)
"""

import math
import torch

from Diffusion import (EDMConfig, sample_sigma_lognormal, edm_training_target,
                       edm_stochastic_sample, make_denoise_fn, nfe_count)

PARAM = "edm"
CKPT_DIR = "checkpoints/stage2_edm/"

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
# sigma_data = 1.0 because the residual is explicitly rescaled to unit variance
# before it reaches the denoiser (see res_std in TrainStage2.py). c_skip, c_out
# and c_in are all functions of sigma_data; setting it to anything else while
# feeding unit-variance data mis-scales the entire preconditioning at once.
CFG = EDMConfig(
    sigma_data=1.0,
    sigma_min=0.002,
    sigma_max=800.0,        # paper Sec. 5.3.2
    rho=7.0,
    P_mean=0.0,             # paper: ln(sigma) ~ N(0, 1.2^2)
    P_std=1.2,
    # Sampling-time sigma_max. sigma_max appears ONLY in the schedule -- training
    # draws sigma from the log-normal and never consults it -- so this is purely
    # a sampler choice, changeable without retraining.
    #
    # Leave at None to reproduce the paper exactly. Set to 200.0 (with
    # SAMPLE_STEPS raised to 32) if you care more about calibrated ensemble
    # spread than about literal fidelity to the reference: measured against an
    # exact analytic denoiser on unit-variance data, 800/18 over-disperses by
    # ~21% from Heun discretization alone, 200/32 by ~6%, 800/64 by ~4%.
    sigma_max_sample=None,
    # Algorithm 2 stochasticity. S_churn = 0 degrades to the deterministic ODE,
    # which visibly collapses ensemble spread -- that is the control, not the
    # product.
    S_churn=40.0, S_min=0.05, S_max=50.0, S_noise=1.003,
)

SAMPLE_STEPS = 18           # paper Sec. 5.3.2. NFE = 2*18 - 1 = 35.
CFG_KEY = "edm_cfg"


# ------------------------------------------------------------------------------
# UNIFORM INTERFACE
# ------------------------------------------------------------------------------

def build_train_batch(r, cfg, device):
    """-> (state_to_concat_with_cond, time_input, target).

    The network predicts F_theta = (r - c_skip*z)/c_out and the loss on it is
    taken with UNIFORM weight. Dividing the target by c_out IS Karras'
    lambda(sigma); applying lambda(sigma) again on top of that squares it, which
    weights small-sigma samples by up to ~6e4 relative to large-sigma ones.
    """
    sigma = sample_sigma_lognormal(r.shape[0], device, cfg)
    z, target, c_in, c_noise = edm_training_target(r, sigma, cfg)
    return (c_in * z).to(r.dtype), c_noise, target


def sample_residual(net, cond, topo, shape, device, cfg, n_steps,
                    generator=None, tile=None, overlap=None,
                    amp_dtype=torch.float32, guidance=1.0, max_tiles_per_batch=8):
    """One ensemble member of the normalized residual."""
    fn = make_denoise_fn(net, cond, topo, cfg, tile=tile, overlap=overlap,
                         amp_dtype=amp_dtype, guidance=guidance,
                         max_tiles_per_batch=max_tiles_per_batch)
    return edm_stochastic_sample(fn, shape, device, cfg, n_steps=n_steps,
                                 generator=generator)


def config_from_dict(d):
    return EDMConfig.from_dict(d)


def with_stochasticity(cfg, value):
    """Override the sampler's noise re-injection (S_churn here, churn in the FM
    package) so both arms expose the same knob under the same name."""
    return EDMConfig(**{**cfg.to_dict(), "S_churn": float(value)})


def with_sigma_max_sample(cfg, value):
    return EDMConfig(**{**cfg.to_dict(), "sigma_max_sample": float(value)})
