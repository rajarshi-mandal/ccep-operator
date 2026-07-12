"""DAG acyclicity constraint — the core novelty (spec 2.3).

The SSM transition matrix is parameterised as ``A = W * M`` (Hadamard with a mask M,
default all-ones except a zeroed diagonal) and constrained to encode a directed acyclic
graph. Two continuous acyclicity functionals are provided; both satisfy ``h(W) = 0``
iff the weighted adjacency ``W`` is a DAG:

  * NOTEARS (Zheng et al. 2018):  h(W) = tr(exp(W * W)) - d
  * DAGMA   (Bello et al. 2022):  h(W) = -logdet(s*I - W * W) + d*log(s)
    (often numerically better-behaved; used as a drop-in fallback)

Edge convention (project-wide): ``A[i, j] != 0`` means region *j influences i* — i.e.
column j -> row i. ``W * W`` is convention-agnostic for the acyclicity test, so the same
h(W) works regardless; the convention only matters for intervention graph surgery.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def notears_h(W: torch.Tensor) -> torch.Tensor:
    """NOTEARS acyclicity: ``tr(exp(W o W)) - d``. Zero iff W is a DAG."""
    d = W.shape[-1]
    WW = W * W
    return torch.matrix_exp(WW).diagonal(dim1=-2, dim2=-1).sum(-1) - d


def dagma_h(W: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """DAGMA log-det acyclicity: ``-logdet(sI - W o W) + d log s``.

    Requires ``sI - W o W`` to be an M-matrix (positive definite); valid near the DAG
    region. More stable gradients than the matrix exponential for larger d.
    """
    d = W.shape[-1]
    WW = W * W
    eye = torch.eye(d, dtype=W.dtype, device=W.device)
    M = s * eye - WW
    sign, logabsdet = torch.linalg.slogdet(M)
    return -logabsdet + d * torch.log(torch.tensor(s, dtype=W.dtype, device=W.device))


class DAGConstraint(nn.Module):
    """Holds the masked weight matrix W and exposes the acyclicity functional."""

    def __init__(self, d: int, variant: str = "notears", dagma_s: float = 1.0,
                 mask_diagonal: bool = True):
        super().__init__()
        self.d = d
        self.variant = variant
        self.dagma_s = dagma_s
        # Mask removes self-loops (diagonal) by default; off-diagonal is learnable.
        mask = torch.ones(d, d)
        if mask_diagonal:
            mask.fill_diagonal_(0.0)
        self.register_buffer("mask", mask)

    def h(self, W: torch.Tensor) -> torch.Tensor:
        if self.variant == "dagma":
            return dagma_h(W, self.dagma_s)
        return notears_h(W)

    def apply_mask(self, W: torch.Tensor) -> torch.Tensor:
        return W * self.mask

    @staticmethod
    def threshold(W: torch.Tensor, tau: float) -> torch.Tensor:
        """Zero out |w| < tau to extract the final discrete DAG after training."""
        return torch.where(W.abs() < tau, torch.zeros_like(W), W)


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 6
    # A strictly upper-triangular W is acyclic by construction -> h should be ~0.
    Wdag = torch.triu(torch.randn(d, d), diagonal=1)
    # A matrix with a 2-cycle -> h should be > 0.
    Wcyc = Wdag.clone()
    Wcyc[2, 0] = 0.9
    Wcyc[0, 2] = 0.9
    for name, fn in [("notears", notears_h), ("dagma", dagma_h)]:
        print(f"{name}:  h(DAG) = {fn(Wdag).item():.3e}   h(cyclic) = {fn(Wcyc).item():.3e}")
