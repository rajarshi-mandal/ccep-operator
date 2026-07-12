"""Loss terms for the joint causal DAG-SSM objective (spec 2.5, 4).

Total objective optimised under the augmented-Lagrangian schedule:

    L(W, theta) = -L_obs  +  lambda_int * L_int  +  (alpha * h + rho/2 * h^2)

where ``h = h(A)`` is the acyclicity functional. The trainer owns alpha/rho; this module
provides the three differentiable pieces:

  * ``obs_nll``     — negative mean Kalman log-likelihood of resting fMRI windows (L_obs).
  * ``intervention_loss`` — mismatch between the model's predicted downstream response to a
    ``do(stim)`` and the measured region TEP. Compared as a *spatial topography*: per-region
    response energy, normalised. We deliberately avoid aligning the SSM's abstract rollout
    steps to EEG milliseconds (that mapping is unidentified); the causal claim we supervise
    is "stimulating region i drives THIS pattern of downstream regions", which is what the
    held-out-site Pearson-r metric in Experiment 1 also scores.
  * ``acyclicity_penalty`` — the ``alpha*h + rho/2 h^2`` augmented-Lagrangian term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------
# Model-upgrade losses (additive, opt-in — see reports/MODEL_UPGRADES.md).
# Each targets one audited gap. The DEFAULT objective (obs_nll + intervention_loss +
# acyclicity_penalty) is unchanged, so existing checkpoints/tests are unaffected.
# --------------------------------------------------------------------------------
def resample_time(traj: torch.Tensor, T_out: int) -> torch.Tensor:
    """Linearly resample an SSM rollout ``[B, steps, d]`` onto ``T_out`` time points.

    The SSM's abstract ``steps`` and the measured EEG ``T`` live on different (unidentified)
    scales. Rather than discarding the time axis (the old energy collapse), we *calibrate*
    by stretching the rollout onto the measured grid so predicted and measured waveforms are
    directly comparable. Returns ``[B, T_out, d]``.
    """
    B, steps, d = traj.shape
    x = traj.permute(0, 2, 1)                       # [B, d, steps]
    x = F.interpolate(x, size=T_out, mode="linear", align_corners=True)
    return x.permute(0, 2, 1)                        # [B, T_out, d]


def _unit_time(x: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Zero-mean, unit-norm each region's waveform along ``time_dim`` (shape-only)."""
    x = x - x.mean(dim=time_dim, keepdim=True)
    return x / x.norm(dim=time_dim, keepdim=True).clamp_min(1e-12)


def waveform_loss(model, stim_idx: torch.Tensor, region_tep: torch.Tensor,
                  amplitude: float = 1.0, steps: int | None = None,
                  hold: bool = False, inject: bool = False,
                  energy_weight: float = 0.5) -> torch.Tensor:
    """GAP 4 — temporal-shape interventional loss (recovers the discarded time axis).

    Rolls do(stim) forward, resamples the prediction onto the measured EEG grid, and scores
    BOTH the per-region temporal *shape* (1 - cos of unit waveforms) and the spatial energy
    topography (the old target). ``energy_weight`` blends them. Unlike ``intervention_loss``
    this supervises *when* each region responds, not only *how much*. ``inject`` (GAP 5) routes
    the stimulus through the learned spread map ``ssm.B``.
    """
    B, d, T = region_tep.shape
    steps = steps or min(T, 64)
    pred = model.predict_intervention(stim_idx, amplitude, steps, hold=hold,
                                      inject=inject)            # [B, steps, d]
    pred_T = resample_time(pred, T)                                           # [B, T, d]
    pred_w = _unit_time(pred_T, time_dim=1)                                   # [B, T, d]
    meas_w = _unit_time(region_tep.transpose(1, 2), time_dim=1)               # [B, T, d]
    shape_cos = (pred_w * meas_w).sum(dim=1)                                  # [B, d]
    l_shape = (1.0 - shape_cos).mean()
    l_energy = topography_loss(response_energy(pred, dim=1),
                               response_energy(region_tep, dim=-1))
    return (1.0 - energy_weight) * l_shape + energy_weight * l_energy


def deflated_topography_loss(pred_topo: torch.Tensor, meas_topo: torch.Tensor,
                             shared_mode: torch.Tensor) -> torch.Tensor:
    """GAP 1 — score the topography AFTER projecting out the dominant shared spatial mode.

    ``shared_mode``: unit ``[d]`` (e.g. first PC of the training topographies = the
    site-/subject-invariant pattern). Deflating it removes the component a group-mean
    template already explains, so the loss rewards only structure beyond the shared mode.
    Both topographies ``[B, d]``.
    """
    m = shared_mode / shared_mode.norm().clamp_min(1e-12)
    p = pred_topo - (pred_topo @ m).unsqueeze(-1) * m
    q = meas_topo - (meas_topo @ m).unsqueeze(-1) * m
    return topography_loss(p, q)


def obs_matrix_locality_penalty(C: torch.Tensor, centroids: torch.Tensor,
                                sigma_mm: float = 40.0) -> torch.Tensor:
    """GAP 1 — anatomical prior on a learned observation matrix ``C`` ``[obs, d]``.

    Penalises C weights that couple an observed parcel to a *spatially distant* latent region
    (Gaussian-distance weighting on MNI centroids). Encourages a local/near-diagonal C so the
    learned latent space stays interpretable and the dominant non-local bridge artifact mode is
    discouraged. ``centroids``: ``[d, 3]`` MNI. Returns a scalar.
    """
    D = torch.cdist(centroids, centroids)                       # [d, d] mm
    w = 1.0 - torch.exp(-(D / sigma_mm) ** 2)                    # 0 near-diagonal, ->1 far
    return (C.pow(2) * w).sum() / C.pow(2).sum().clamp_min(1e-12)


# --------------------------------------------------------------------------------
# L_obs
# --------------------------------------------------------------------------------
def obs_nll(model, y: torch.Tensor, u: torch.Tensor | None = None,
            per_element: bool = True) -> torch.Tensor:
    """Negative Kalman log-likelihood of observational windows ``y`` ``[B, T, obs]``.

    ``per_element=True`` divides by ``T * obs_dim`` so L_obs is O(1) and comparable in
    scale to the O(1) interventional cosine loss — otherwise the raw NLL (~thousands)
    swamps L_int and the interventional/causal supervision is effectively ignored.
    """
    nll = -model.obs_log_likelihood(y, u=u).mean()
    if per_element:
        _, T, obs = y.shape
        nll = nll / (T * obs)
    return nll


# --------------------------------------------------------------------------------
# L_int helpers
# --------------------------------------------------------------------------------
def response_energy(traj: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Per-region response energy (RMS over time) -> ``[..., d]`` spatial topography.

    ``traj``: predicted ``[B, steps, d]`` or measured ``[B, d, T]``. ``dim`` is the time
    axis. The result is the timescale-invariant magnitude of each region's response.
    """
    return traj.pow(2).mean(dim=dim).clamp_min(1e-12).sqrt()


def _normalize(v: torch.Tensor) -> torch.Tensor:
    """Unit-norm each row (so loss scores pattern, not amplitude)."""
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def topography_loss(pred_topo: torch.Tensor, meas_topo: torch.Tensor) -> torch.Tensor:
    """Cosine-distance between normalised predicted/measured response topographies.

    Both ``[B, d]``. Returns scalar ``mean(1 - cos)`` in ``[0, 2]``; 0 == identical pattern.
    Equivalent to ``1 - Pearson`` after centring, which matches the Exp-1 success metric.
    """
    p = _normalize(pred_topo - pred_topo.mean(dim=-1, keepdim=True))
    m = _normalize(meas_topo - meas_topo.mean(dim=-1, keepdim=True))
    cos = (p * m).sum(dim=-1)
    return (1.0 - cos).mean()


# --------------------------------------------------------------------------------
# L_int
# --------------------------------------------------------------------------------
def intervention_loss(model, stim_idx: torch.Tensor, region_tep: torch.Tensor,
                      amplitude: float = 1.0, steps: int | None = None,
                      hold: bool = False) -> torch.Tensor:
    """Interventional supervision: predicted vs measured downstream response topography.

    ``stim_idx``: ``[B]`` stimulated parcels. ``region_tep``: ``[B, d, T]`` measured TEP in
    region space. Rolls the do-operation forward, reduces both predicted and measured
    responses to per-region energy, and scores the spatial-pattern mismatch.
    """
    B, d, T = region_tep.shape
    steps = steps or min(T, 32)
    pred = model.predict_intervention(stim_idx, amplitude, steps, hold=hold)  # [B, steps, d]
    pred_topo = response_energy(pred, dim=1)            # [B, d]
    meas_topo = response_energy(region_tep, dim=-1)     # [B, d]
    return topography_loss(pred_topo, meas_topo)


# --------------------------------------------------------------------------------
# Acyclicity (augmented Lagrangian)
# --------------------------------------------------------------------------------
def acyclicity_penalty(h: torch.Tensor, alpha: float, rho: float) -> torch.Tensor:
    """Augmented-Lagrangian acyclicity term ``alpha*h + rho/2 * h^2``."""
    return alpha * h + 0.5 * rho * h * h


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.causal_dag_ssm import CausalDAGSSM

    torch.manual_seed(0)
    d = 100
    model = CausalDAGSSM(d)

    y = torch.randn(4, 30, d)
    print("obs_nll:", float(obs_nll(model, y)))

    stim = torch.tensor([40, 13])
    tep = torch.randn(2, d, 600)
    li = intervention_loss(model, stim, tep)
    print("intervention_loss:", float(li), "(in [0,2], 0=perfect pattern match)")

    h = model.acyclicity()
    print("acyclicity_penalty(alpha=0,rho=1):", float(acyclicity_penalty(h, 0.0, 1.0)))

    total = obs_nll(model, y) + li + acyclicity_penalty(h, 0.0, 1.0)
    total.backward()
    print("joint backward -> W.grad nonzero:", bool(model.W.grad.abs().sum() > 0))
