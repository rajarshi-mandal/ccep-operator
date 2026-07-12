"""Tests for the audited model-upgrade additions (all opt-in, default objective unchanged)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from training.losses import (resample_time, waveform_loss, deflated_topography_loss,  # noqa: E402
                             obs_matrix_locality_penalty, topography_loss, response_energy)


def test_resample_time_endpoints_and_shape():
    traj = torch.zeros(2, 8, 5)
    traj[:, 0] = 1.0       # first step
    traj[:, -1] = 3.0      # last step
    out = resample_time(traj, T_out=40)
    assert out.shape == (2, 40, 5)
    # linear resample with aligned corners preserves the endpoints
    assert torch.allclose(out[:, 0], torch.full((2, 5), 1.0), atol=1e-5)
    assert torch.allclose(out[:, -1], torch.full((2, 5), 3.0), atol=1e-5)


def test_waveform_loss_runs_and_is_bounded():
    torch.manual_seed(0)
    d = 12
    model = CausalDAGSSM(d)
    stim = torch.tensor([3, 7])
    tep = torch.randn(2, d, 50)
    l = waveform_loss(model, stim, tep, energy_weight=0.5)
    assert torch.isfinite(l) and 0.0 <= float(l.detach()) <= 2.0
    l.backward()                                   # differentiable into W
    assert model.W.grad is not None and model.W.grad.abs().sum() > 0


def test_deflation_removes_shared_mode():
    d = 6
    mode = torch.zeros(d); mode[0] = 1.0           # shared mode = region 0
    # two topographies identical on the shared mode, differing on the residual
    pred = torch.tensor([[5.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    meas = torch.tensor([[5.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    full = topography_loss(pred, meas)
    deflated = deflated_topography_loss(pred, meas, mode)
    # after removing the dominant shared component the residual mismatch is exposed (larger)
    assert deflated > full


def test_locality_penalty_prefers_local_C():
    torch.manual_seed(0)
    d = 8
    centroids = torch.randn(d, 3) * 30.0
    local = torch.eye(d)                            # reads each parcel from itself
    nonlocal_C = torch.ones(d, d)                   # couples everything to everything
    assert obs_matrix_locality_penalty(local, centroids) < \
           obs_matrix_locality_penalty(nonlocal_C, centroids)


def test_inject_activates_B_gradient():
    torch.manual_seed(0)
    d = 10
    model = CausalDAGSSM(d)
    with torch.no_grad():
        model.ssm.B.fill_(0.2)                      # nonzero spread so injection has effect
    stim = torch.tensor([4])
    pred = model.predict_intervention(stim, 1.0, steps=6, inject=True)
    loss = pred.pow(2).sum()
    loss.backward()
    # the otherwise-dead B receives gradient only via the injection path
    assert model.ssm.B.grad is not None and model.ssm.B.grad.abs().sum() > 0


def test_inject_changes_prediction():
    torch.manual_seed(1)
    d = 10
    model = CausalDAGSSM(d)
    with torch.no_grad():
        model.ssm.B.copy_(0.3 * torch.randn(d, 1))
    stim = torch.tensor([4])
    base = model.predict_intervention(stim, 1.0, steps=6, inject=False)
    inj = model.predict_intervention(stim, 1.0, steps=6, inject=True)
    assert not torch.allclose(base, inj)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
