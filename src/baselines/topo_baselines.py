"""Non-causal / weak baselines for the Exp-1B LOSO held-out-subject TMS prediction.

Every baseline maps to a *predicted region response topography* (a length-``d`` non-negative
vector), so the Exp-1B harness can score them all identically: Pearson r of the predicted vs
measured response-energy topography, with the stimulated parcel excluded ("downstream r").

The headline causal model must beat these to earn any causal claim. Ranked roughly by
strength:

  * ``mean_topography``   — predict the held subject with the mean of the training subjects'
    measured topographies. This is the strong, non-causal ceiling (the topography is ~87%
    site-invariant, so this scores high without any causal structure).
  * ``fc_propagation``    — predict the response to stimulating parcel ``p`` as the magnitude
    of the resting-state functional-connectivity row of ``p``. "Regions correlated with M1 at
    rest are the ones that respond to M1 stimulation." A genuinely causal-flavoured but still
    correlational baseline (no interventional training, no acyclicity).
  * ``distance_decay``    — response decays with Euclidean distance from the stimulated parcel
    centroid (a pure-geometry null: nearer regions respond more).
  * ``untrained_model``   — the do(stim) prediction of an *untrained* CausalDAGSSM (random W).
    This is the floor the trained model is launched from.

``residual_over_mean`` and ``dcm`` are documented stubs (see ``NotImplemented*`` markers) —
they require training / external tooling and are scoped as next steps, not run here.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def _energy(tep: torch.Tensor) -> torch.Tensor:
    """Region response-energy topography from a ``[d, T]`` TEP: sqrt(mean over time of x^2)."""
    return torch.sqrt((tep.float() ** 2).mean(dim=-1))


def measured_topography(record) -> torch.Tensor:
    """Measured response-energy topography ``[d]`` of one interventional record."""
    return _energy(record["region_tep"])


def mean_topography(train_records: Sequence[dict]) -> torch.Tensor:
    """Strong non-causal baseline: mean measured topography over training subjects ``[d]``."""
    return torch.stack([measured_topography(r) for r in train_records]).mean(0)


def functional_connectivity(obs_fmri: np.ndarray) -> np.ndarray:
    """Group resting-state FC ``[d, d]``: mean over subjects of per-subject Pearson FC.

    ``obs_fmri``: ``[n_subjects, T, d]`` z-scored BOLD. Diagonal is zeroed so a parcel's
    self-correlation does not dominate its predicted topography.
    """
    n, _, d = obs_fmri.shape
    acc = np.zeros((d, d), dtype=np.float64)
    for s in range(n):
        x = obs_fmri[s]                       # [T, d]
        c = np.corrcoef(x, rowvar=False)      # [d, d]
        acc += np.nan_to_num(c)
    fc = acc / max(n, 1)
    np.fill_diagonal(fc, 0.0)
    return fc


def fc_propagation(fc: np.ndarray, stim_parcel: int) -> torch.Tensor:
    """Predict topography from the |FC| row of the stimulated parcel ``[d]``."""
    return torch.from_numpy(np.abs(fc[stim_parcel]).astype(np.float32))


def distance_decay(centroids: np.ndarray, stim_parcel: int,
                   length_scale: float | None = None) -> torch.Tensor:
    """Pure-geometry null: response ∝ exp(-dist/length_scale) from the stimulated centroid.

    ``centroids``: ``[d, 3]`` MNI coordinates. ``length_scale`` defaults to the median
    pairwise nearest-neighbour distance so the decay is data-scaled, not arbitrary.
    """
    c = centroids.astype(np.float64)
    dist = np.linalg.norm(c - c[stim_parcel], axis=1)   # [d]
    if length_scale is None:
        # median of all pairwise distances / 4 → a moderate, data-derived scale
        length_scale = float(np.median(dist[dist > 0])) / 2.0 or 1.0
    resp = np.exp(-dist / length_scale)
    return torch.from_numpy(resp.astype(np.float32))


@torch.no_grad()
def untrained_model_topography(model, stim_parcel: int, steps: int,
                               amplitude: float = 1.0) -> torch.Tensor:
    """do(stim) response-energy topography of a (typically untrained) CausalDAGSSM ``[d]``."""
    pred = model.predict_intervention(torch.tensor([stim_parcel]), amplitude, steps)  # [1,steps,d]
    return torch.sqrt((pred[0] ** 2).mean(dim=0))


# --------------------------------------------------------------------------------
# Documented stubs (scoped as next steps; intentionally not executed in the harness)
# --------------------------------------------------------------------------------
def residual_over_mean(*_args, **_kwargs):
    """STUB. Model the residual ABOVE the mean topography rather than predicting from scratch.

    Rationale (RESEARCH_HYPOTHESES H2 fallback): the mean-topo baseline is strong because the
    topography is site-invariant; the causal signal lives in the per-site *deviation*. A proper
    implementation trains the DAG-SSM to predict ``measured − mean_topo`` and adds the mean back
    at eval. Requires a training loop change, so it is left as a next step, not a passive
    baseline.
    """
    raise NotImplementedError("residual_over_mean is a scoped next step, not a passive baseline")


def dcm(*_args, **_kwargs):
    """STUB. Dynamic Causal Modelling comparison (SPM/MATLAB or a bilinear-DCM reimpl).

    A fair DCM baseline needs the bilinear state equation + EM inversion; out of scope for the
    autonomous run (external tooling / heavy reimplementation). Documented as a comparison to
    add before publication.
    """
    raise NotImplementedError("dcm baseline requires external tooling; documented as next step")
