"""Do-calculus intervention module — graph surgery (spec 2.4).

To predict the effect of TMS to region i, we simulate Pearl's ``do(h_i = a)``:

  1. Graph surgery: sever region i from its normal causes by zeroing the *incoming*
     edges to i. With the project convention ``A[i, j]`` = "j influences i", incoming
     edges to i are ROW i of A, so we zero row i.
  2. Clamp: set the initial latent state of region i to the stimulus amplitude.
  3. Propagate: roll the modified dynamics forward to predict the downstream response
     at every other region.

Everything is a differentiable masked operation (no in-place writes that break autograd),
so gradients from the interventional loss flow back into A, B and the noise params.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def graph_surgery(A: torch.Tensor, stim_idx: torch.Tensor) -> torch.Tensor:
    """Zero the incoming edges (rows) of the stimulated nodes.

    ``A``: ``[d, d]`` (shared) or ``[B, d, d]`` (per-sample). ``stim_idx``: ``[B]`` long.
    Returns ``[B, d, d]`` with row ``stim_idx[b]`` zeroed for each b. Differentiable.
    """
    d = A.shape[-1]
    B = stim_idx.shape[0]
    if A.dim() == 2:
        A = A.unsqueeze(0).expand(B, d, d)
    # row mask: 0 on the stimulated row, 1 elsewhere
    row_mask = torch.ones(B, d, device=A.device, dtype=A.dtype)
    row_mask = row_mask.scatter(1, stim_idx.unsqueeze(1), 0.0)   # [B, d]
    return A * row_mask.unsqueeze(-1)                            # broadcast over columns


class InterventionModule(nn.Module):
    """Predicts the downstream region response of a do-operation."""

    def __init__(self, ssm):
        super().__init__()
        self.ssm = ssm

    def do(self, A: torch.Tensor, stim_idx: torch.Tensor, amplitude: torch.Tensor,
           steps: int, hold: bool = False, inject: bool = False) -> torch.Tensor:
        """Predict region trajectories after stimulating ``stim_idx``.

        ``A``: ``[d, d]`` or ``[B, d, d]``. ``stim_idx``: ``[B]``. ``amplitude``: scalar
        or ``[B]``. ``steps``: rollout length. ``hold``: if True, re-clamp the stimulated
        region every step (sustained do); else clamp only at t=0 (TMS impulse).

        ``inject`` (GAP 5): also drive the stimulus through the learned input map ``ssm.B``
        as an impulse at t=0, so the otherwise-dead ``B`` becomes a trainable *stimulus-spread*
        map (TMS does not excite a single parcel cleanly — current spreads to a focal patch).
        ``B`` is shared across sites (a single spread kernel), which is all 2 sites can identify.

        Returns observed region trajectory ``[B, steps, obs_dim]``.
        """
        d = A.shape[-1]
        B = stim_idx.shape[0]
        dev, dt = A.device, A.dtype
        A_do = graph_surgery(A, stim_idx)                       # [B, d, d]

        if not torch.is_tensor(amplitude):
            amplitude = torch.tensor(float(amplitude), device=dev, dtype=dt)
        amp = amplitude.expand(B) if amplitude.dim() == 0 else amplitude  # [B]

        # one-hot of the stimulated region, for clamping
        onehot = torch.zeros(B, d, device=dev, dtype=dt).scatter(
            1, stim_idx.unsqueeze(1), 1.0)
        clamp_vec = onehot * amp.unsqueeze(1)                   # [B, d]
        keep = 1.0 - onehot                                     # zero on stim region

        # optional learned stimulus spread injected at the stimulated row only
        spread0 = None
        if inject:
            # B:[d, input_dim]; broadcast the amp impulse through the first input channel,
            # gated onto the stimulated region's surgered row so it respects graph surgery.
            spread = (self.ssm.B[:, 0].to(dt)).unsqueeze(0) * amp.unsqueeze(1)   # [B, d]
            spread0 = spread * keep                              # don't overwrite the clamp node

        # clamp initial state at the stimulus amplitude (rest start from 0)
        h = clamp_vec if spread0 is None else clamp_vec + spread0
        traj = []
        for _ in range(steps):
            # h_next[b] = A_do[b] @ h[b]
            h = torch.einsum("bij,bj->bi", A_do, h)
            if hold:
                # re-impose the do-value on the stimulated region each step
                h = h * keep + clamp_vec
            traj.append(h)
        H = torch.stack(traj, dim=1)                            # [B, steps, d]
        return self.ssm.observe(H)                              # [B, steps, obs_dim]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.ssm import LinearGaussianSSM

    torch.manual_seed(0)
    d = 6
    # chain 0->1->2->...; A[i, i-1] = influence of (i-1) on i
    A = torch.zeros(d, d)
    for i in range(1, d):
        A[i, i - 1] = 0.8
    ssm = LinearGaussianSSM(d)

    iv = InterventionModule(ssm)
    stim = torch.tensor([0, 3])           # stimulate region 0 (batch 0), region 3 (batch 1)
    out = iv.do(A, stim, amplitude=1.0, steps=5)   # [2, 5, d]
    print("do() output:", tuple(out.shape))

    # Surgery check: stimulating node 3 must zero its incoming edge from node 2,
    # so upstream regions (0,1,2) stay silent; only 3,4,5 light up.
    resp_b1 = out[1].abs().sum(0)         # total response per region for batch item 1
    print("response per region (stim=3):", resp_b1.round(decimals=3).tolist())
    upstream_silent = torch.allclose(resp_b1[:3], torch.zeros(3), atol=1e-6)
    downstream_active = resp_b1[3:].sum() > 0
    print("upstream silent:", upstream_silent, "| downstream active:", bool(downstream_active))

    # Differentiability check
    Ap = A.clone().requires_grad_(True)
    loss = iv.do(Ap, stim, 1.0, 5).pow(2).sum()
    loss.backward()
    print("grad finite:", bool(torch.isfinite(Ap.grad).all()), "| grad nonzero:",
          bool(Ap.grad.abs().sum() > 0))
