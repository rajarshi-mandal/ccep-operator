"""Tests for Exp-1B scoring: downstream r excludes the stimulated parcel; dataset carries
subject ids. Data-dependent tests skip cleanly if processed caches are absent."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
PROC = ROOT / "data" / "processed"


def test_pearsonr_basic():
    from exp1_held_out_tms import pearsonr
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert abs(pearsonr(a, a) - 1.0) < 1e-6
    assert abs(pearsonr(a, -a) + 1.0) < 1e-6
    assert abs(pearsonr(a, torch.zeros(4))) < 1e-6  # zero-variance -> 0


def test_downstream_excludes_stim_changes_score():
    from exp3_baselines_exp1b import downstream_r
    # craft a case where the stim parcel agrees but downstream disagrees
    meas = torch.tensor([10.0, 1.0, 2.0, 3.0])
    pred = torch.tensor([10.0, 3.0, 2.0, 1.0])
    full = downstream_r(pred, meas, stim=0)        # parcel 0 dropped either way here
    # dropping a DIFFERENT parcel (the agreeing big one) must change r vs keeping it
    keep_all = float(((pred - pred.mean()) * (meas - meas.mean())).sum() /
                     ((pred - pred.mean()).norm() * (meas - meas.mean()).norm()))
    assert full != pytest.approx(keep_all)


@pytest.mark.skipif(not (PROC / "interventional_region.npz").exists(),
                    reason="processed interventional cache not present")
def test_dataset_carries_subjects():
    from data.dataset import InterventionalDataset
    ds = InterventionalDataset(PROC, site_filter={"M1_L"})
    assert len(ds) > 0
    subs = [ds[i]["subject"] for i in range(len(ds))]
    assert all(isinstance(s, str) and s for s in subs), "every M1 record needs a subject id"
    assert len(set(subs)) == len(subs), "LOSO requires unique subject ids per fold"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
