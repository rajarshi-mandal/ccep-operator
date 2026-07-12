"""Distance-stratified noise ceiling vs achieved r (the gate for all further modelling).

The flat 0.961 ceiling is inflated by easy near-field contacts. This decomposes BOTH the ceiling
(corr of half-split topographies) AND the achieved model r into distance bins, so we can see where
the recoverable headroom actually is:
  * if the far-field ceiling is high but achieved r is low  -> real headroom, Class A worth it;
  * if the far-field ceiling itself collapses               -> the far-field is noise, we're near done.

Achieved model = `combo` recomputed here (locality + network residual, nested-CV) so the comparison
is apples-to-apples with the ceiling on the same contacts/bins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402
from ccep_loso import (  # noqa: E402
    all_caches, _valid_mask, predict_distance, predict_stim_knn, predict_combo,
    _resid, _z, REL_MIN, SIGMA_GRID, TAU_GRID, BETA_GRID, _score_param,
)

BINS = [(0, 20), (20, 40), (40, 1e9)]    # mm: near / mid / far
BIN_NAMES = ["near(0-20)", "mid(20-40)", "far(40+)"]


def binned_r(pred, meas, mask, dist):
    """Pearson r between pred and meas within each distance bin (masked, NaN-safe)."""
    out = []
    for lo, hi in BINS:
        b = mask & (dist >= lo) & (dist < hi) & np.isfinite(pred) & np.isfinite(meas)
        if b.sum() < 4:
            out.append(np.nan); continue
        p, m = pred[b] - pred[b].mean(), meas[b] - meas[b].mean()
        den = np.linalg.norm(p) * np.linalg.norm(m)
        out.append(float((p @ m) / den) if den > 1e-12 else np.nan)
    return out


def main():
    caches = all_caches()
    # accumulate per-subject mean (over sites) of ceiling and combo r, per bin
    ceil_subj = {b: [] for b in BIN_NAMES}
    combo_subj = {b: [] for b in BIN_NAMES}
    frac_subj = {b: [] for b in BIN_NAMES}    # fraction of contacts in each bin
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        if cs.responses_h1 is None:
            print(f"{cs.subject}: no half-split data (rebuild needed)"); return
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
        if len(keep) < 6:
            continue
        cw = {b: [] for b in BIN_NAMES}; mw = {b: [] for b in BIN_NAMES}
        fw = {b: [] for b in BIN_NAMES}
        for test_i in keep:
            train_idx = [t for t in keep if t != test_i]
            mask = _valid_mask(cs, test_i, train_idx)
            dist = np.linalg.norm(cs.contact_xyz - cs.stim_xyz[test_i][None], axis=1)
            # ceiling: half1 vs half2
            cvals = binned_r(cs.responses_h1[test_i], cs.responses_h2[test_i], mask, dist)
            # combo prediction (nested-CV sigma, tau, beta)
            sig = max(SIGMA_GRID, key=lambda s: _score_param(
                cs, train_idx, lambda j, tr, s=s: predict_distance(cs, j, s)))
            tau = max(TAU_GRID, key=lambda tt: _score_param(
                cs, train_idx, lambda j, tr, tt=tt: predict_stim_knn(cs, j, tr, tt)))
            beta = max(BETA_GRID, key=lambda bb: _score_param(
                cs, train_idx,
                lambda j, tr, bb=bb: predict_combo(cs, j, tr, sig, tau, bb, _valid_mask(cs, j, tr))))
            pred = predict_combo(cs, test_i, train_idx, sig, tau, beta, mask)
            mvals = binned_r(pred, cs.responses[test_i], mask, dist)
            for k, name in enumerate(BIN_NAMES):
                if np.isfinite(cvals[k]):
                    cw[name].append(cvals[k])
                if np.isfinite(mvals[k]):
                    mw[name].append(mvals[k])
                lo, hi = BINS[k]
                fw[name].append(float(((mask) & (dist >= lo) & (dist < hi)).sum() / max(mask.sum(), 1)))
        for name in BIN_NAMES:
            if cw[name]:
                ceil_subj[name].append(np.mean(cw[name]))
            if mw[name]:
                combo_subj[name].append(np.mean(mw[name]))
            frac_subj[name].append(np.mean(fw[name]))

    print("Distance-stratified ceiling (half-split r) vs combo achieved r — subject-level mean\n")
    print(f"{'bin':12s} {'%contacts':>10s} {'ceiling':>9s} {'combo':>9s} {'gap':>8s} {'%realised':>10s}")
    for name in BIN_NAMES:
        ce = np.array(ceil_subj[name]); mo = np.array(combo_subj[name]); fr = np.array(frac_subj[name])
        ceil_m = ce.mean(); combo_m = mo.mean(); gap = ceil_m - combo_m
        print(f"{name:12s} {fr.mean()*100:9.0f}% {ceil_m:9.3f} {combo_m:9.3f} {gap:+8.3f} "
              f"{combo_m/ceil_m*100:9.0f}%")
    print("\nInterpretation:")
    far_ceil = np.mean(ceil_subj["far(40+)"]); far_combo = np.mean(combo_subj["far(40+)"])
    if far_ceil > 0.5 and (far_ceil - far_combo) > 0.12:
        print(f"  Far-field ceiling {far_ceil:.2f} >> combo {far_combo:.2f}: REAL recoverable headroom "
              f"in the network-driven far contacts -> Class A (new features) is worth it.")
    elif far_ceil < 0.4:
        print(f"  Far-field ceiling only {far_ceil:.2f}: the distant response is largely trial-noise "
              f"/ idiosyncratic -> little to recover; the geometry+network model is near practical max.")
    else:
        print(f"  Far-field: ceiling {far_ceil:.2f}, combo {far_combo:.2f} — modest headroom.")


if __name__ == "__main__":
    main()
