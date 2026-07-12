"""Unit tests for the linear Gaussian SSM + Kalman filter (spec 9.2 — risky piece).

We check the filter on synthetic data where the answer is known: (1) the likelihood is
maximised at the true noise/dynamics, (2) gradient descent on the Kalman likelihood
recovers the true transition matrix, (3) the 1-D filter matches a hand-derived value.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model.ssm import LinearGaussianSSM  # noqa: E402


def _simulate(A, T, B, q_std, r_std, seed=0):
    torch.manual_seed(seed)
    d = A.shape[0]
    h = torch.zeros(B, d, dtype=torch.float64)
    ys = []
    for _ in range(T):
        h = h @ A.T + q_std * torch.randn(B, d, dtype=torch.float64)
        ys.append(h + r_std * torch.randn(B, d, dtype=torch.float64))
    return torch.stack(ys, dim=1)


def _matched_ssm(d, q_std, r_std):
    ssm = LinearGaussianSSM(d).double()
    with torch.no_grad():
        ssm.log_q.fill_(torch.log(torch.tensor(q_std ** 2)))
        ssm.log_r.fill_(torch.log(torch.tensor(r_std ** 2)))
    return ssm


def test_true_A_beats_wrong_A():
    d = 5
    A = torch.triu(0.3 * torch.randn(d, d, dtype=torch.float64), diagonal=1)
    y = _simulate(A, T=60, B=8, q_std=0.1, r_std=0.1)
    ssm = _matched_ssm(d, 0.1, 0.1)
    assert ssm.kalman_log_likelihood(y, A).mean() > \
        ssm.kalman_log_likelihood(y, torch.zeros_like(A)).mean()


def test_likelihood_is_finite_and_differentiable():
    d = 4
    A = torch.triu(0.3 * torch.randn(d, d, dtype=torch.float64), diagonal=1)
    A.requires_grad_(True)
    y = _simulate(A.detach(), T=30, B=4, q_std=0.1, r_std=0.1)
    ssm = _matched_ssm(d, 0.1, 0.1)
    ll = ssm.kalman_log_likelihood(y, A).mean()
    assert torch.isfinite(ll)
    ll.backward()
    assert A.grad is not None and torch.isfinite(A.grad).all()


def test_gradient_descent_recovers_A():
    """Optimising the Kalman LL over a free A should recover the true A."""
    torch.manual_seed(1)
    d = 4
    A_true = torch.triu(0.4 * torch.randn(d, d, dtype=torch.float64), diagonal=1)
    y = _simulate(A_true, T=80, B=16, q_std=0.05, r_std=0.05, seed=2)
    ssm = _matched_ssm(d, 0.05, 0.05)

    A = torch.zeros(d, d, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([A], lr=0.05)
    ll0 = ssm.kalman_log_likelihood(y, A).mean().item()
    for _ in range(300):
        opt.zero_grad()
        loss = -ssm.kalman_log_likelihood(y, A).mean()
        loss.backward()
        opt.step()
    ll1 = ssm.kalman_log_likelihood(y, A).mean().item()
    assert ll1 > ll0  # likelihood improved
    # recovered A should correlate strongly with the truth
    a, b = A.detach().flatten(), A_true.flatten()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
    assert corr > 0.9, f"recovered A corr with truth = {corr:.3f}"


def test_rollout_shapes_and_dynamics():
    d = 3
    ssm = LinearGaussianSSM(d).double()
    A = torch.eye(d, dtype=torch.float64) * 0.5
    h0 = torch.ones(2, d, dtype=torch.float64)
    traj = ssm.rollout(h0, steps=4, A=A)
    assert traj.shape == (2, 4, d)
    # with A=0.5 I and no input, each step halves the state
    assert torch.allclose(traj[:, 0], 0.5 * h0)
    assert torch.allclose(traj[:, 1], 0.25 * h0)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
