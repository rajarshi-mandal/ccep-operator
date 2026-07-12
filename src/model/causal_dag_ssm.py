"""Full Causal DAG-SSM model — assembles the three validated pieces (spec 2, 9.4).

This is the object the trainer optimises. It owns:

  * ``W``  — the learnable weighted adjacency (the causal graph). ``A = W o M`` with the
    diagonal masked out, so ``A[i, j]`` = "region j influences region i".
  * ``ssm`` — the linear Gaussian SSM backbone (B, C, noise covariances) that turns A into
    a likelihood over observational fMRI via an exact Kalman filter (the ``L_obs`` term).
  * ``dag`` — the NOTEARS/DAGMA acyclicity functional ``h(W)`` driving A toward a DAG.
  * ``intervention`` — Pearl ``do()`` via graph surgery, predicting the TMS-evoked region
    response that the interventional loss ``L_int`` regresses against the measured TEP.

The trainer combines three signals:  maximise ``obs_log_likelihood`` (fit resting fMRI),
minimise interventional prediction error against TEPs, and drive ``acyclicity`` to zero
under an augmented-Lagrangian schedule. Everything here is differentiable in ``W``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:  # works as a package (trainer/tests) and as a direct script
    from .dag_constraint import DAGConstraint
    from .intervention import InterventionModule
    from .ssm import LinearGaussianSSM
except ImportError:  # pragma: no cover - script execution path
    from model.dag_constraint import DAGConstraint
    from model.intervention import InterventionModule
    from model.ssm import LinearGaussianSSM


class CausalDAGSSM(nn.Module):
    def __init__(self, d: int, variant: str = "notears", dagma_s: float = 1.0,
                 input_dim: int = 1, obs_dim: int | None = None,
                 learn_C: bool = False, init_scale: float = 0.1):
        super().__init__()
        self.d = d
        # Learnable weighted adjacency. Small init keeps A near zero (acyclic) at start.
        self.W = nn.Parameter(init_scale * torch.randn(d, d))
        self.dag = DAGConstraint(d, variant=variant, dagma_s=dagma_s)
        self.ssm = LinearGaussianSSM(d, input_dim=input_dim, obs_dim=obs_dim,
                                     learn_C=learn_C)
        self.intervention = InterventionModule(self.ssm)

    # ------------------------------------------------------------------ adjacency
    @property
    def A(self) -> torch.Tensor:
        """Masked transition matrix used everywhere (diagonal zeroed)."""
        return self.dag.apply_mask(self.W)

    def acyclicity(self) -> torch.Tensor:
        """Scalar NOTEARS/DAGMA acyclicity ``h(A)`` — zero iff A is a DAG."""
        return self.dag.h(self.A)

    # ------------------------------------------------------------------ L_obs
    def obs_log_likelihood(self, y: torch.Tensor,
                           u: torch.Tensor | None = None) -> torch.Tensor:
        """Exact Kalman data log-likelihood of observational windows ``y`` ``[B, T, obs]``.

        Returns ``[B]``. The trainer maximises the mean of this (the ``L_obs`` term).
        """
        return self.ssm.kalman_log_likelihood(y, self.A, u=u)

    # ------------------------------------------------------------------ L_int
    def predict_intervention(self, stim_idx: torch.Tensor,
                             amplitude: torch.Tensor | float, steps: int,
                             hold: bool = False, inject: bool = False) -> torch.Tensor:
        """Predict region trajectories after a ``do(stim)`` intervention.

        ``stim_idx``: ``[B]`` stimulated parcels. Returns ``[B, steps, obs_dim]`` — the
        model's TEP prediction, compared against the measured region TEP in ``L_int``.
        ``inject`` (GAP 5): also drive the stimulus through the learned spread map ``ssm.B``.
        """
        return self.intervention.do(self.A, stim_idx, amplitude, steps, hold=hold,
                                    inject=inject)

    # ------------------------------------------------------------------ readout
    @torch.no_grad()
    def extract_dag(self, tau: float) -> torch.Tensor:
        """Threshold |A| < tau to recover the final discrete DAG after training."""
        return self.dag.threshold(self.A, tau)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config

    cfg = load_config()
    d = cfg.parcellation.d
    torch.manual_seed(0)

    model = CausalDAGSSM(
        d,
        variant=cfg.dag.variant,
        input_dim=cfg.model.input_dim,
        init_scale=cfg.model.init_state_scale,
    )
    print(f"CausalDAGSSM: d={d}, variant={cfg.dag.variant}")
    print("  W params:", model.W.numel(), "| total trainable:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # L_obs path: a batch of fake observational windows.
    y = torch.randn(4, 30, d)
    ll = model.obs_log_likelihood(y)
    print("  obs_log_likelihood:", tuple(ll.shape), "mean=%.1f" % ll.mean().item())

    # L_int path: stimulate two parcels.
    stim = torch.tensor([40, 13])   # M1_L, parietal_L parcels
    pred = model.predict_intervention(stim, amplitude=1.0, steps=8)
    print("  predict_intervention:", tuple(pred.shape))

    # Acyclicity and end-to-end differentiability through ALL three terms.
    h = model.acyclicity()
    loss = -ll.mean() + pred.pow(2).mean() + 10.0 * h
    loss.backward()
    g = model.W.grad
    print("  acyclicity h(A)=%.3e" % h.item())
    print("  joint loss backward -> W.grad finite:", bool(torch.isfinite(g).all()),
          "| nonzero:", bool(g.abs().sum() > 0))

    dag = model.extract_dag(cfg.dag.threshold)
    print("  extracted DAG edges (|A|>%.2f):" % cfg.dag.threshold,
          int((dag.abs() > 0).sum().item()))
