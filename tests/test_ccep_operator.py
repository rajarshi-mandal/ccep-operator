"""Tests for the CCEP effective-connectivity operator (experiments/ccep_operator_v2.py).

Guards the core modeling result: (1) the operator NESTS the distance kernel (steps=0 == distance),
(2) it is amplitude-preserving / spectrally normalised, and (3) when genuine network structure is
planted, propagation beats the pure distance kernel — the property the v1 operator lacked.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from data.ccep_pipeline import CCEPSubject  # noqa: E402
import ccep_operator_v2 as O  # noqa: E402
import ccep_loso as L  # noqa: E402


def _synthetic(n_contacts=40, n_sites=25, network_strength=1.0, seed=0):
    """A subject whose responses = local decay + a planted long-range network link.

    Contacts lie on a line; each stim site sits on a contact. The response to stimulating site s
    is a Gaussian around s (locality) PLUS, for the 'hub' half of contacts, a copy of the response
    at a distant partner contact (a planted network edge distance alone cannot capture).
    """
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n_contacts, 3))
    xyz[:, 0] = np.linspace(0, 100, n_contacts)        # 1-D line, 0..100 mm
    site_contacts = rng.choice(n_contacts, size=n_sites, replace=False)
    partner = (np.arange(n_contacts) + n_contacts // 2) % n_contacts  # long-range pairing
    responses = np.full((n_sites, n_contacts), np.nan)
    stim_xyz = np.zeros((n_sites, 3))
    stim_idx = np.full((n_sites, 2), -1, dtype=int)
    for i, sc in enumerate(site_contacts):
        D = np.linalg.norm(xyz - xyz[sc][None], axis=1)
        local = np.exp(-(D ** 2) / (2 * 12.0 ** 2))
        net = network_strength * local[partner]         # response also appears at the partner site
        r = local + net + 0.01 * rng.standard_normal(n_contacts)
        r[sc] = np.nan                                   # exclude the stimulated contact
        responses[i] = r
        stim_xyz[i] = xyz[sc]
        stim_idx[i, 0] = sc
    rel = np.ones(n_sites)
    return CCEPSubject(
        subject="synthetic", contacts=[f"c{i}" for i in range(n_contacts)], contact_xyz=xyz,
        sites=[f"s{i}" for i in range(n_sites)], responses=responses,
        responses_signed=responses, n2=responses, stim_xyz=stim_xyz, stim_idx=stim_idx,
        reliability=rel, n_trials=np.full(n_sites, 10), fs=2048.0,
    )


def test_operator_nests_distance():
    """steps=0 (or alpha=0) must reduce exactly to the nested distance kernel."""
    cs = _synthetic()
    train = list(range(1, len(cs.sites)))
    P = O._build_operator(cs, train, "symmetric")
    seed = O.predict_operator_v2(cs, 0, train, sigma=15, alpha=0.0, steps=0, mode="symmetric", P=P)
    dist = L.predict_distance(cs, 0, sigma=15)
    assert np.allclose(seed, dist), "operator_v2 with no propagation must equal the distance kernel"


def test_operator_spectral_norm():
    """The propagation operator is spectrally normalised (radius ~1), preserving amplitude ratios
    rather than row-normalising them away (the v1 bug)."""
    cs = _synthetic()
    train = list(range(len(cs.sites)))
    P = O._build_operator(cs, train, "symmetric")
    assert np.allclose(P, P.T, atol=1e-8), "symmetric mode must yield a symmetric operator"
    sr = np.abs(np.linalg.eigvalsh(P)).max()
    assert sr == pytest.approx(1.0, abs=1e-6), f"spectral radius should be ~1, got {sr}"


def test_propagation_beats_distance_with_planted_network():
    """With a planted long-range network edge, propagation (steps>0) beats the distance kernel;
    with no network it should not hurt. This is exactly the property v1 lacked."""
    # strong planted network
    cs = _synthetic(network_strength=1.0, seed=1)
    res = O.eval_subject(cs)
    assert res is not None
    sc, _ = res
    assert sc["operator_v2"] > sc["distance"] + 0.02, (
        f"propagation should beat distance when network structure exists: "
        f"op2={sc['operator_v2']:.3f} vs dist={sc['distance']:.3f}")

    # no network -> operator_v2 should not be meaningfully worse than distance (nests it)
    cs0 = _synthetic(network_strength=0.0, seed=2)
    sc0, _ = O.eval_subject(cs0)
    assert sc0["operator_v2"] >= sc0["distance"] - 0.05
