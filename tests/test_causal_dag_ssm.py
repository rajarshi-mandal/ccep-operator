"""Integration tests for the assembled CausalDAGSSM (spec 9.4).

These check the three loss paths share one parameter set and stay jointly differentiable:
the observational Kalman likelihood, the interventional do-prediction, and the acyclicity
functional all push gradients into the single weighted adjacency ``W``.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402


def test_A_diagonal_masked():
    m = CausalDAGSSM(8)
    assert torch.allclose(m.A.diagonal(), torch.zeros(8))


def test_obs_likelihood_shape_and_grad():
    m = CausalDAGSSM(6)
    y = torch.randn(4, 20, 6)
    ll = m.obs_log_likelihood(y)
    assert ll.shape == (4,) and torch.isfinite(ll).all()
    ll.mean().backward()
    assert m.W.grad is not None and torch.isfinite(m.W.grad).all()


def test_intervention_prediction_shape():
    m = CausalDAGSSM(10)
    pred = m.predict_intervention(torch.tensor([3, 7]), amplitude=1.0, steps=5)
    assert pred.shape == (2, 5, 10)


def test_acyclicity_nonnegative_and_zero_for_dag():
    m = CausalDAGSSM(6)
    # Force W to a strictly upper-triangular (acyclic) matrix.
    with torch.no_grad():
        m.W.copy_(torch.triu(torch.randn(6, 6), diagonal=1))
    assert m.acyclicity().item() < 1e-4
    # A 2-cycle makes it strictly positive.
    with torch.no_grad():
        m.W[0, 2] = 0.9
        m.W[2, 0] = 0.9
    assert m.acyclicity().item() > 1e-3


def test_joint_loss_backward_all_paths():
    """One backward through obs + int + acyclicity must touch W with a finite grad."""
    m = CausalDAGSSM(7)
    y = torch.randn(3, 15, 7)
    ll = m.obs_log_likelihood(y)
    pred = m.predict_intervention(torch.tensor([1, 4, 5]), 1.0, 6)
    loss = -ll.mean() + pred.pow(2).mean() + 5.0 * m.acyclicity()
    loss.backward()
    g = m.W.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_extract_dag_thresholds():
    m = CausalDAGSSM(5)
    with torch.no_grad():
        m.W.fill_(0.5)
    dense = m.extract_dag(0.3)
    sparse = m.extract_dag(0.7)
    assert (dense.abs() > 0).sum() > (sparse.abs() > 0).sum()
    assert torch.allclose(sparse, torch.zeros(5, 5))  # all 0.5 < 0.7 -> empty


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
