"""Unit tests for the do-calculus intervention module (spec 9.2 — risky piece).

Verifies the two properties that make the interventional claim valid: graph surgery
severs the stimulated node from its causes (so upstream regions stay silent), and the
whole do-operation is differentiable (no autograd-breaking in-place ops).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model.intervention import InterventionModule, graph_surgery  # noqa: E402
from model.ssm import LinearGaussianSSM  # noqa: E402


def _chain(d, w=0.8):
    """Directed chain 0->1->...->d-1 under convention A[i, i-1] = weight."""
    A = torch.zeros(d, d)
    for i in range(1, d):
        A[i, i - 1] = w
    return A


def test_surgery_zeros_incoming_row():
    A = torch.ones(4, 4)
    stim = torch.tensor([2])
    out = graph_surgery(A, stim)[0]
    assert torch.allclose(out[2], torch.zeros(4))             # row 2 (incoming to 2) zeroed
    assert torch.allclose(out[torch.arange(4) != 2], A[torch.arange(4) != 2])


def test_surgery_per_sample_batched():
    A = torch.ones(5, 5)
    stim = torch.tensor([0, 3, 4])
    out = graph_surgery(A, stim)
    assert out.shape == (3, 5, 5)
    for b, s in enumerate(stim.tolist()):
        assert torch.allclose(out[b, s], torch.zeros(5))


def test_upstream_silent_downstream_active():
    d = 6
    A = _chain(d)
    iv = InterventionModule(LinearGaussianSSM(d))
    out = iv.do(A, torch.tensor([3]), amplitude=1.0, steps=5)[0]  # [steps, d]
    resp = out.abs().sum(0)
    # stimulating node 3 cuts its dependence on node 2 -> 0,1,2 never activate
    assert torch.allclose(resp[:3], torch.zeros(3), atol=1e-6)
    # propagation reaches downstream nodes 4,5
    assert resp[4] > 0 and resp[5] > 0


def test_different_stim_sites_differ():
    d = 6
    A = _chain(d)
    iv = InterventionModule(LinearGaussianSSM(d))
    out = iv.do(A, torch.tensor([0, 3]), amplitude=1.0, steps=5)
    # the two stimulation sites must produce different downstream patterns
    assert not torch.allclose(out[0], out[1])


def test_differentiable():
    d = 5
    A = _chain(d).requires_grad_(True)
    iv = InterventionModule(LinearGaussianSSM(d))
    loss = iv.do(A, torch.tensor([1]), amplitude=1.0, steps=4).pow(2).sum()
    loss.backward()
    assert A.grad is not None and torch.isfinite(A.grad).all()
    assert A.grad.abs().sum() > 0


def test_hold_sustains_stimulation():
    d = 4
    A = _chain(d)
    iv = InterventionModule(LinearGaussianSSM(d))
    # with hold=True the stimulated region is re-clamped every step
    out = iv.do(A, torch.tensor([0]), amplitude=1.0, steps=3, hold=True)[0]
    assert torch.allclose(out[:, 0], torch.ones(3))  # region 0 held at amplitude


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
