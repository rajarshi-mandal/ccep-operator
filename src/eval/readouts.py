"""Temporal/spatiotemporal readouts for the bake-off arms (eval-only, no model change).

The headline Exp-1B metric collapses each region's TMS response to a single energy scalar
(RMS over time). That static topography is ~87% site-invariant, so a group-mean template ties
the causal model (H2 null) *by construction* — the population mean is the Bayes-optimal
group-level predictor under L2/correlation loss.

These readouts score the axis the energy metric throws away — **when** each region responds —
where the causal graph's propagation structure can carry information a static template cannot.
All functions are plain tensor ops so they can be applied to the *already-trained* Exp-1B fold
checkpoints (Arm B = latency, Arm C = early->late forecast) without retraining.

Time-axis convention (matches the rest of the codebase):
  * predicted do-rollout trajectories are ``[B, steps, d]`` (time axis = 1),
  * measured region TEPs are ``[B, d, T]`` or ``[d, T]`` (time axis = -1).
Predicted "steps" are abstract SSM iterations; measured time is EEG samples. The two scales are
NOT identified (documented in training/losses.py), so latency is compared by **rank**
(Spearman) — only the *ordering* of regional activation is claimed, not absolute milliseconds.
"""
from __future__ import annotations

import torch


def temporal_com(traj: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Energy-weighted temporal centre-of-mass per region -> latency proxy.

    ``com[region] = sum_t (t * x_t^2) / sum_t x_t^2``. Smoother and far more robust than
    argmax for sparse/decaying responses (a near-silent region gets a stable, if uninformative,
    value rather than a noise-driven argmax). Returns the COM with the time axis reduced.
    """
    x2 = traj.pow(2)
    T = traj.shape[time_dim]
    t = torch.arange(T, device=traj.device, dtype=traj.dtype)
    shape = [1] * traj.dim()
    shape[time_dim] = T
    t = t.reshape(shape)
    num = (t * x2).sum(dim=time_dim)
    den = x2.sum(dim=time_dim).clamp_min(1e-12)
    return num / den


def peak_latency(traj: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Per-region time index of peak |response| (argmax over time). Float tensor."""
    return traj.abs().argmax(dim=time_dim).to(traj.dtype)


def windowed_energy(tep: torch.Tensor, t0: int, t1: int, time_dim: int) -> torch.Tensor:
    """Response-energy topography (RMS) restricted to time window ``[t0, t1)``."""
    sl = [slice(None)] * tep.dim()
    sl[time_dim] = slice(t0, t1)
    seg = tep[tuple(sl)]
    return seg.pow(2).mean(dim=time_dim).clamp_min(1e-12).sqrt()


def _rank(v: torch.Tensor) -> torch.Tensor:
    """Average-free ordinal ranks of a 1-D tensor (argsort of argsort)."""
    order = v.argsort()
    ranks = torch.empty_like(v)
    ranks[order] = torch.arange(v.numel(), device=v.device, dtype=v.dtype)
    return ranks


def pearsonr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors (matches experiments/exp1_held_out_tms)."""
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def spearmanr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation = Pearson on ranks. Robust to the unidentified step<->ms scale."""
    return pearsonr(_rank(a), _rank(b))


def downstream_mask(d: int, stim_parcel: int, device=None) -> torch.Tensor:
    """Boolean keep-mask of length ``d`` with the stimulated parcel removed."""
    keep = torch.ones(d, dtype=torch.bool, device=device)
    keep[stim_parcel] = False
    return keep
