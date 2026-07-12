"""Unit tests for the DAG acyclicity constraint (spec 9.2 — risky piece).

Silent bugs here invalidate every downstream causal claim, so we verify the
mathematical contract directly: h(W) == 0 iff W is a DAG, and gradients flow.
"""
import pytest
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model.dag_constraint import DAGConstraint, dagma_h, notears_h  # noqa: E402


def _random_dag_weights(d, density=0.5, seed=0):
    """Random weighted adjacency that is a DAG (random permutation of upper-triangular)."""
    rng = np.random.default_rng(seed)
    W = np.triu(rng.normal(size=(d, d)), k=1)
    W *= (rng.random((d, d)) < density)
    perm = rng.permutation(d)
    W = W[perm][:, perm]  # still acyclic under joint row/col permutation
    return torch.tensor(W, dtype=torch.float64)


def test_notears_zero_on_dags():
    for seed in range(10):
        W = _random_dag_weights(8, seed=seed)
        assert abs(notears_h(W).item()) < 1e-8, f"DAG should give h=0 (seed {seed})"


def test_dagma_zero_on_dags():
    for seed in range(10):
        W = _random_dag_weights(8, seed=seed)
        assert abs(dagma_h(W, s=1.0).item()) < 1e-6, f"DAG should give h~0 (seed {seed})"


def test_positive_on_cycles():
    # Any cycle gives strictly positive h; magnitude shrinks with cycle length and
    # sub-unit weights, so we only require strict positivity here ...
    for d in (3, 5, 8):
        W = torch.zeros(d, d, dtype=torch.float64)
        for i in range(d):  # a full directed cycle 0->1->...->d-1->0
            W[(i + 1) % d, i] = 0.8
        assert notears_h(W).item() > 0
        assert dagma_h(W).item() > 0
    # ... and a strong short cycle must be clearly positive.
    W2 = torch.zeros(4, 4, dtype=torch.float64)
    W2[1, 0] = W2[0, 1] = 1.2  # 2-cycle
    assert notears_h(W2).item() > 1e-2
    assert dagma_h(W2, s=2.0).item() > 1e-2


def test_self_loop_is_cyclic():
    W = torch.zeros(4, 4, dtype=torch.float64)
    W[1, 1] = 0.5  # self loop = cycle of length 1
    assert notears_h(W).item() > 1e-6


def test_gradient_flows_and_decreases_h():
    """One gradient step on h(W) should reduce h for a cyclic W."""
    torch.manual_seed(0)
    # Mild-scale weights so matrix_exp stays well-conditioned for the finite step.
    W = (0.1 * torch.randn(6, 6, dtype=torch.float64)).requires_grad_(True)
    h0 = notears_h(W)
    h0.backward()
    assert W.grad is not None and torch.isfinite(W.grad).all()
    with torch.no_grad():
        W2 = W - 1e-3 * W.grad  # gradient descent on h
    assert notears_h(W2).item() < h0.item()


def test_mask_removes_diagonal():
    dc = DAGConstraint(d=5, mask_diagonal=True)
    W = torch.ones(5, 5)
    masked = dc.apply_mask(W)
    assert torch.allclose(masked.diagonal(), torch.zeros(5))


def test_threshold():
    W = torch.tensor([[0.0, 0.2, 0.5], [0.1, 0.0, 0.4], [0.35, 0.05, 0.0]])
    out = DAGConstraint.threshold(W, tau=0.3)
    assert (out.abs() < 0.3).sum() == (out == 0).sum()  # all small entries zeroed
    assert out[0, 2].item() == pytest.approx(0.5) and out[2, 0].item() == pytest.approx(0.35)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
