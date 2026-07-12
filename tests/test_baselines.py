"""Tests for src/baselines/topo_baselines.py — shapes, sanity, and stub contracts."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines.topo_baselines import (measured_topography, mean_topography,  # noqa: E402
                                      functional_connectivity, fc_propagation,
                                      distance_decay, residual_over_mean, dcm)


def _rec(d=6, T=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"region_tep": torch.randn(d, T, generator=g), "stim_parcel": 0}


def test_measured_topography_is_rms_over_time():
    tep = torch.tensor([[3.0, 4.0], [0.0, 0.0]])  # row0 rms = sqrt((9+16)/2)
    topo = measured_topography({"region_tep": tep})
    assert topo.shape == (2,)
    assert abs(float(topo[0]) - np.sqrt(12.5)) < 1e-5
    assert float(topo[1]) == 0.0


def test_mean_topography_averages_records():
    recs = [_rec(seed=i) for i in range(4)]
    mt = mean_topography(recs)
    manual = torch.stack([measured_topography(r) for r in recs]).mean(0)
    assert torch.allclose(mt, manual)


def test_fc_zero_diagonal_and_shape():
    obs = np.random.RandomState(0).randn(3, 100, 6)
    fc = functional_connectivity(obs)
    assert fc.shape == (6, 6)
    assert np.allclose(np.diag(fc), 0.0)
    assert np.allclose(fc, fc.T, atol=1e-8)  # correlation is symmetric


def test_fc_propagation_shape():
    fc = np.random.RandomState(1).randn(6, 6)
    p = fc_propagation(fc, 2)
    assert p.shape == (6,)
    assert (p >= 0).all()  # magnitude


def test_distance_decay_monotone_and_peaks_at_stim():
    # centroids on a line; response must be max at the stimulated parcel and decay with distance
    cents = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]], dtype=float)
    r = distance_decay(cents, stim_parcel=0)
    assert int(torch.argmax(r)) == 0
    assert r[0] >= r[1] >= r[2] >= r[3]


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        residual_over_mean()
    with pytest.raises(NotImplementedError):
        dcm()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
