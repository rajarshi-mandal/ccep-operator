"""Phase-1 pipeline unit tests (fast — no BOLD I/O except the optional cache check)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import ds005498_pipeline as P  # noqa: E402


def test_parse_coil_mni_signs():
    assert np.allclose(P.parse_coil_mni("stim34x6x62"), [34, 6, 62])
    assert np.allclose(P.parse_coil_mni("stimMinus38x22x48"), [-38, 22, 48])
    assert np.allclose(P.parse_coil_mni("stim40xMinus18x64"), [40, -18, 64])
    assert np.allclose(P.parse_coil_mni("stimMinus24x60xMinus2"), [-24, 60, -2])


def test_parse_coil_mni_bad():
    with pytest.raises(ValueError):
        P.parse_coil_mni("stim34x6")


def test_coil_to_parcel_nearest():
    centroids = np.array([[0, 0, 0], [50, 0, 0], [-50, 0, 0]], float)
    assert P.coil_to_parcel(np.array([48.0, 1, -1]), centroids) == 1
    assert P.coil_to_parcel(np.array([-2.0, 1, 0]), centroids) == 0


def test_zscore_flat_parcel_stays_zero():
    ts = np.zeros((50, 3), np.float32)
    ts[:, 0] = np.random.randn(50)
    z = P.zscore(ts)
    assert abs(z[:, 0].mean()) < 1e-5 and abs(z[:, 0].std() - 1) < 1e-3
    assert np.allclose(z[:, 1], 0)  # flat -> not NaN


def test_evoked_response_shapes_and_reliability():
    rng = np.random.default_rng(0)
    d, T, tr = 100, 167, 2.4
    onsets = np.sort(rng.uniform(5, T * tr - 10, 68))
    # build a parcel ts with a reproducible stimulus-locked bump on a few parcels
    ts = rng.standard_normal((T, d)).astype(np.float32) * 0.3
    ft = tr * np.arange(T)
    from nilearn.glm.first_level import compute_regressor
    cond = np.vstack([onsets, np.full_like(onsets, 0.3), np.ones_like(onsets)])
    reg = compute_regressor(cond, "glover", ft, oversampling=16)[0][:, 0]
    ts[:, :20] += 4.0 * reg[:, None]
    ev = P.evoked_response(ts, onsets, tr)
    assert ev["topo"].shape == (d,)
    assert ev["fir"].shape[0] == d
    assert ev["n_pulses"] == 68
    assert ev["reliability"] > 0.5   # strong shared signal -> high split-half r


def test_evoked_response_too_few_pulses():
    assert P.evoked_response(np.zeros((10, 100), np.float32), np.array([1.0, 2, 3]), 2.4) is None


@pytest.mark.skipif(not Path("data/processed/ds005498/manifest.json").exists(),
                    reason="cache not built")
def test_cache_loader_contract():
    c = P.DS005498Cache(qc_filter=True)
    assert len(c) > 0
    r = c.records[0]
    d = c.centroids.shape[0]
    assert r.region_tep.shape == (d, 1)
    assert r.topo.shape == (d,)
    assert r.subject_rest.ndim == 2 and r.subject_rest.shape[1] == d
    assert 0 <= r.stim_parcel < d
    # LOSO-WS yields train sets that exclude the test record's site for that subject
    for test, train in c.loso_ws():
        assert all(t.subject == test.subject for t in train)
        assert all(t is not test for t in train)           # identity (dataclass holds arrays)
        assert all(t.site_name != test.site_name or t is not test for t in train)
        break
