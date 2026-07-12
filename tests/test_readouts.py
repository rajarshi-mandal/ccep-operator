"""Tests for the bake-off temporal readouts (eval-only metrics, no model change)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.readouts import (temporal_com, peak_latency, windowed_energy,  # noqa: E402
                           spearmanr, pearsonr, downstream_mask)


def test_temporal_com_orders_early_vs_late():
    T, d = 20, 2
    traj = torch.zeros(1, T, d)
    traj[0, 2, 0] = 1.0     # region 0 peaks early
    traj[0, 17, 1] = 1.0    # region 1 peaks late
    com = temporal_com(traj, time_dim=1)[0]
    assert com[0] < com[1]
    assert abs(com[0].item() - 2.0) < 1e-5 and abs(com[1].item() - 17.0) < 1e-5


def test_peak_latency_argmax():
    traj = torch.zeros(1, 10, 3)
    traj[0, 1, 0] = -5.0    # uses |.|, so the negative peak counts
    traj[0, 7, 1] = 2.0
    traj[0, 4, 2] = 3.0
    lat = peak_latency(traj, time_dim=1)[0]
    assert lat.tolist() == [1.0, 7.0, 4.0]


def test_windowed_energy_restricts_window():
    tep = torch.zeros(1, 2, 12)
    tep[0, 0, 0:3] = 2.0     # energy only in the early window for region 0
    tep[0, 1, 8:11] = 2.0    # energy only in the late window for region 1
    early = windowed_energy(tep, 0, 4, time_dim=-1)[0]
    late = windowed_energy(tep, 8, 12, time_dim=-1)[0]
    assert early[0] > early[1] and late[1] > late[0]


def test_spearman_monotonic():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(spearmanr(a, a * 3.0) - 1.0) < 1e-6        # any increasing map -> +1
    assert abs(spearmanr(a, -a) + 1.0) < 1e-6             # decreasing -> -1


def test_pearson_and_mask():
    a = torch.tensor([0.0, 1.0, 2.0])
    assert abs(pearsonr(a, a) - 1.0) < 1e-6
    keep = downstream_mask(4, 2)
    assert keep.tolist() == [True, True, False, True]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
