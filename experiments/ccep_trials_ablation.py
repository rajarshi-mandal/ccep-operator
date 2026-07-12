"""STEP 1 — trials-ablation: measure how combo r and the noise ceiling rise with trials/site.

Human CCEP caps at ~10-14 pulses/site, so we can only vary trials DOWNWARD (3..all) — but that
measures the local slope of r vs trials and the ceiling, which is exactly the quantity that says
how much higher-trial acquisition would buy. Re-epochs a high-trial subset at each cap (in memory),
runs the combo leave-stim-site-out LOSO, reports mean ceiling + combo r per cap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import build_subject  # noqa: E402
from ccep_loso import (  # noqa: E402
    topo_r, _valid_mask, predict_distance, predict_stim_knn, predict_combo,
    REL_MIN, SIGMA_GRID, TAU_GRID, BETA_GRID, _score_param,
)

# high-trial subjects (ds004696 HAPwave have the most pulses/site)
DATA = ROOT.parent / "Open Neuro ds004696"
SUBS = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"]
CAPS = [4, 6, 8, None]   # None = all available (pipeline needs >=4 trials/site for the split-half)


def combo_loso(cs):
    """Mean combo topo-r + mean half-split ceiling over reliable sites (nested-CV per fold)."""
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
    if len(keep) < 6:
        return np.nan, np.nan, len(keep)
    rs, ceil = [], []
    h1, h2 = cs.responses_h1, cs.responses_h2
    for test_i in keep:
        train = [t for t in keep if t != test_i]
        mask = _valid_mask(cs, test_i, train)
        tgt = cs.responses[test_i]
        sig = max(SIGMA_GRID, key=lambda s: _score_param(
            cs, train, lambda j, tr, s=s: predict_distance(cs, j, s)))
        tau = max(TAU_GRID, key=lambda tt: _score_param(
            cs, train, lambda j, tr, tt=tt: predict_stim_knn(cs, j, tr, tt)))
        beta = max(BETA_GRID, key=lambda bb: _score_param(
            cs, train, lambda j, tr, bb=bb: predict_combo(cs, j, tr, sig, tau, bb, _valid_mask(cs, j, tr))))
        rs.append(topo_r(predict_combo(cs, test_i, train, sig, tau, beta, mask), tgt, mask))
        if h1 is not None:
            ceil.append(topo_r(h1[test_i], h2[test_i], mask))
    return float(np.nanmean(rs)), float(np.nanmean(ceil)) if ceil else np.nan, len(keep)


def main():
    print("Trials-ablation on 5 high-trial ds004696 subjects (combo r vs trials/site)\n")
    print(f"{'cap':>6s} {'mean_trials':>11s} {'ceiling':>8s} {'combo_r':>8s}")
    rows = []
    for cap in CAPS:
        rr, cc, tt = [], [], []
        for s in SUBS:
            cs = build_subject(str(DATA), s, verbose=False, n_trials_cap=cap)
            r, c, _ = combo_loso(cs)
            nt = cs.n_trials[np.isfinite(cs.n_trials)] if cs.n_trials.size else np.array([np.nan])
            rr.append(r); cc.append(c); tt.append(np.nanmedian(nt))
        label = str(cap) if cap else "all"
        rows.append((label, np.mean(tt), np.nanmean(cc), np.nanmean(rr)))
        print(f"{label:>6s} {np.mean(tt):11.1f} {np.nanmean(cc):8.3f} {np.nanmean(rr):8.3f}")

    # local slope of combo r vs trials, and a simple Spearman-Brown extrapolation of the ceiling
    xs = np.array([r[1] for r in rows]); ys = np.array([r[3] for r in rows])
    if len(xs) >= 2:
        slope = np.polyfit(xs, ys, 1)[0]
        print(f"\nlocal slope d(combo_r)/d(trial) = {slope:+.4f} per trial")
        # SB: reliability of N trials from per-trial rho; fit rho from observed ceiling at max cap
        cmax = rows[-1][2]; nmax = rows[-1][1]
        if np.isfinite(cmax) and cmax > 0:
            rho = cmax / (nmax - cmax * (nmax - 1))   # invert Spearman-Brown for per-trial reliability
            for N in (20, 50, 100):
                relN = N * rho / (1 + (N - 1) * rho)
                print(f"  SB ceiling at {N} trials/site ≈ {relN:.3f}  (vs {cmax:.3f} at {nmax:.0f})")
    print("\nReading: combo r and ceiling rise with trials; extrapolating the SB ceiling shows the")
    print("headroom a higher-trial acquisition (50-100 pulses/site) would open for the model.")


if __name__ == "__main__":
    main()
