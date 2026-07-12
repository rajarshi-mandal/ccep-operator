"""GATE for ds002799 — is the es-fMRI evoked signal real, localized, and HRF-shaped?

This is the analog of GATE 0 for the new dataset, but on already-preprocessed (fMRIPrep) BOLD.
Unlike ds005498 (raw single-pulse -> artifact), this should look like real neural BOLD:
  1. split-half spatial reliability (high = reproducible evoked map)
  2. stim-parcel localization — does the response peak NEAR the stimulated electrode?
     (intracranial stim should drive a strong local response; percentile -> high)
  3. group-mean FIR vs canonical HRF (corr -> positive)
If these pass, the model can actually be tested here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.ds005498_pipeline import DS005498Cache, FIR_DELAYS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/processed/ds002799")
    args = ap.parse_args()
    c = DS005498Cache(cache_dir=args.cache_dir, qc_filter=False)
    recs = c.records
    d = c.centroids.shape[0]
    print(f"[gate] {len(recs)} records, {len(c.subjects())} subjects, d={d}")

    rel = [r.reliability for r in recs if np.isfinite(r.reliability)]
    # localization: percentile of stim parcel's |response| (1.0 = response peaks at stim site)
    pct = []
    for r in recs:
        t = np.abs(r.topo)
        pct.append(np.argsort(np.argsort(t))[r.stim_parcel] / (d - 1))
    # group-mean FIR at stim parcel vs canonical HRF
    firs = np.stack([r.fir[r.stim_parcel] for r in recs])
    gm = firs.mean(0)
    from nilearn.glm.first_level import compute_regressor
    lags = [round(2.4 * k, 1) for k in FIR_DELAYS]
    ftc = np.arange(0, 20, 0.1)
    g = compute_regressor(np.array([[0.], [0.3], [1.]]), "glover", ftc)[0][:, 0]
    canon = np.interp(lags, ftc, g); canon = canon - canon.mean(); canon /= np.linalg.norm(canon) + 1e-9
    gmc = gm - gm.mean(); hrf_corr = float(gmc @ canon / (np.linalg.norm(gmc) + 1e-9))

    print(f"\n(1) split-half reliability : median {np.median(rel):+.3f} "
          f"[IQR {np.percentile(rel,25):+.3f},{np.percentile(rel,75):+.3f}] (n={len(rel)})")
    print(f"(2) stim-parcel localization: median percentile {np.median(pct):.2f} "
          f"(1.0 = response peaks at stim site; ds005498 was 0.41)")
    print(f"(3) group FIR vs canonical HRF: corr {hrf_corr:+.3f}  (ds005498 was -0.65)")
    print(f"    group FIR @ stim parcel {np.round(gm,2)} (lags {lags}s)")
    good = int(np.median(rel) > 0.3) + int(np.median(pct) > 0.6) + int(hrf_corr > 0.3)
    print(f"\nGATE: {good}/3 criteria pass — "
          + ("signal is real & localized; run the model (phase2_loso_ws + phase2b)."
             if good >= 2 else "weak; inspect before modeling."))


if __name__ == "__main__":
    main()
