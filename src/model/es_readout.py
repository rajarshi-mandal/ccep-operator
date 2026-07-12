"""Trained subject-conditioned do()-readout for es-fMRI (roadmap §12 Route-B, amortized).

Predicts the evoked topography of stimulating parcel p in subject s by propagating an impulse
through a subject-specific propagation operator and reading out per-parcel response energy:

    A_subj = A_group + ΔA_subj(FC_s)          # ΔA low-rank, modulated by the subject's rest FC
    h_0 = e_p;  energy = Σ_t (A_subj^t e_p)^2;  ŷ = sqrt(energy)

The t=0 term is the stimulation locality (response at the site); t>0 terms are network
propagation through A_subj — exactly the decomposition Stage-4b showed matters. Trained
end-to-end with the topography (1 − cosine) loss. Ablation = drop ΔA_subj (A_subj = A_group).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ESReadout(nn.Module):
    def __init__(self, d: int, rank: int = 8, steps: int = 6, enc_hidden: int = 64,
                 spectral_cap: float = 0.9, cond_mode: str = "encoder"):
        super().__init__()
        self.d, self.steps, self.cap, self.cond_mode = d, steps, spectral_cap, cond_mode
        self.W_group = nn.Parameter(0.01 * torch.randn(d, d))
        if cond_mode == "encoder":
            self.U = nn.Parameter(0.05 * torch.randn(d, rank))
            self.V = nn.Parameter(0.05 * torch.randn(d, rank))
            self.enc = nn.Sequential(nn.Linear(d, enc_hidden), nn.Tanh(),
                                     nn.Linear(enc_hidden, rank))
        elif cond_mode == "fc_direct":
            # subject structure comes from data (FC deviation); only a scalar gain is learned,
            # so personalization can't overfit — the §12 strong-shrinkage prior, in the extreme.
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(cond_mode)
        self.register_buffer("eye", torch.eye(d))

    def _scale(self, A):
        # keep the operator contractive so the impulse rollout stays finite
        with torch.no_grad():
            sr = torch.linalg.matrix_norm(A, ord=2)
        return A * (self.cap / sr.clamp_min(self.cap)) if sr > self.cap else A

    def A(self, cond: torch.Tensor, ablate: bool = False) -> torch.Tensor:
        Ag = self.W_group * (1 - self.eye)
        if ablate or cond is None:
            return self._scale(Ag)
        if self.cond_mode == "encoder":
            dA = (self.U * self.enc(cond)) @ self.V.t()      # cond = FC feature [d]
        else:
            dA = self.beta * cond                            # cond = FC deviation [d,d]
        return self._scale(Ag + dA * (1 - self.eye))

    def predict(self, stim_idx: int, fc_feat: torch.Tensor, ablate: bool = False) -> torch.Tensor:
        A = self.A(fc_feat, ablate)
        h = torch.zeros(self.d, device=A.device, dtype=A.dtype)
        h[stim_idx] = 1.0
        energy = torch.zeros_like(h)
        for _ in range(self.steps):
            energy = energy + h * h
            h = A @ h
        return torch.sqrt(energy + 1e-8)            # [d] predicted topography


def topo_loss(pred: torch.Tensor, meas: torch.Tensor) -> torch.Tensor:
    """1 − cosine of mean-centred topographies (== 1 − Pearson r), the Exp-1 success metric."""
    p = pred - pred.mean(); m = meas - meas.mean()
    cos = (p * m).sum() / (p.norm().clamp_min(1e-8) * m.norm().clamp_min(1e-8))
    return 1.0 - cos


class ESReadout2(nn.Module):
    """Enhanced do()-readout (improvements #1/#3): a DIRECTED group operator with a NOTEARS
    acyclicity penalty and LEARNABLE per-step energy weights (so the model learns the
    locality↔network mix instead of fixing it). ``predict`` accepts an optional per-subject ΔA
    (improvement #1: fit from the subject's own stim→response pairs)."""

    def __init__(self, d: int, steps: int = 8, spectral_cap: float = 0.9):
        super().__init__()
        self.d, self.steps, self.cap = d, steps, spectral_cap
        self.W = nn.Parameter(0.01 * torch.randn(d, d))      # directed, asymmetric
        self.step_logits = nn.Parameter(torch.zeros(steps))  # softmax -> per-step energy weights
        self.register_buffer("eye", torch.eye(d))

    def _scale(self, A):
        with torch.no_grad():
            sr = torch.linalg.matrix_norm(A, ord=2)
        return A * (self.cap / sr.clamp_min(self.cap)) if sr > self.cap else A

    def group_A(self):
        return self.W * (1 - self.eye)

    def acyclicity(self):
        """NOTEARS h(A) = tr(exp(A∘A)) − d; zero iff A is a DAG."""
        A = self.group_A()
        return torch.trace(torch.matrix_exp(A * A)) - self.d

    def predict(self, stim_idx: int, dA: torch.Tensor | None = None) -> torch.Tensor:
        A = self.group_A()
        if dA is not None:
            A = A + dA * (1 - self.eye)
        A = self._scale(A)
        w = torch.softmax(self.step_logits, 0)
        h = torch.zeros(self.d, device=A.device, dtype=A.dtype); h[stim_idx] = 1.0
        energy = torch.zeros_like(h)
        for t in range(self.steps):
            energy = energy + w[t] * h * h
            h = A @ h
        return torch.sqrt(energy + 1e-8)
