"""Stage 4b (ds002799) — does network/causal propagation add ANYTHING beyond spatial locality?

Phase 4 found the site-specific deviation is predictable (r~0.44, p=1e-4) but the causal readout
only TIES a spatial-Gaussian-at-stim prior — i.e. it might all be "response near the stim site."
This is the decisive test: residualize the target deviation against the locality prior, then ask
whether a connectivity-based readout (functional-connectivity 1-step / multi-step propagation, and
the rest-VAR causal impulse) captures the LOCALITY-RESIDUAL deviation. If a network readout's
incremental deviation-r is significantly > 0, network propagation carries real site-specific
information beyond locality -> the causal mechanism is worth building. If ~0, it's pure locality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ds005498_pipeline import DS005498Cache  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test  # noqa: E402
from phase2_loso_ws import fit_group_A, fit_subject_A, impulse_topo, spatial_r  # noqa: E402

SIGMA = 25.0
STEPS = 8


def fc(rest):
    C = np.corrcoef(rest.T)
    return np.nan_to_num(C)


def fc_impulse(C, p, steps):
    """Propagate an impulse at p through |FC| as a (symmetric) spread operator."""
    A = np.abs(C); np.fill_diagonal(A, 0.0)
    A = A / (A.sum(1, keepdims=True) + 1e-9)          # row-normalise -> diffusion
    h = np.zeros(A.shape[0]); h[p] = 1.0
    energy = np.zeros_like(h)
    for _ in range(steps):
        energy += h * h
        h = A @ h
    return np.sqrt(energy)


def resid(x, basis):
    """Residual of x after removing its projection on (centered, unit) basis."""
    b = basis - basis.mean(); nb = np.linalg.norm(b)
    if nb < 1e-9:
        return x - x.mean()
    b = b / nb; xc = x - x.mean()
    return xc - (xc @ b) * b


def main():
    cache = DS005498Cache(cache_dir="data/processed/ds002799", qc_filter=True)
    cents = cache.centroids
    Dmm = np.linalg.norm(cents[:, None] - cents[None], axis=-1)
    by_sub = {}
    for r in cache.records:
        by_sub.setdefault(r.subject, []).append(r)
    subjects = [s for s in sorted(by_sub) if len(by_sub[s]) >= 2]
    rests = [by_sub[s][0].subject_rest for s in subjects]
    A_group = fit_group_A(rests, 50.0)
    A_subj = {s: fit_subject_A(by_sub[s][0].subject_rest, A_group, 200.0) for s in subjects}
    FC = {s: fc(by_sub[s][0].subject_rest) for s in subjects}
    print(f"[4b] {sum(len(by_sub[s]) for s in subjects)} records, {len(subjects)} subjects")

    nets = ["fc_1step", "fc_diffuse", "causal_var"]
    raw_dev = {m: [] for m in ["spatial_gauss"] + nets}     # per-subject mean deviation-r
    inc_dev = {m: [] for m in nets}                         # incremental over locality
    for s in subjects:
        srecs = by_sub[s]
        rd = {m: [] for m in raw_dev}; idv = {m: [] for m in nets}
        for i, test in enumerate(srecs):
            others = [srecs[j] for j in range(len(srecs)) if j != i]
            p = test.stim_parcel
            wmean = np.mean([o.topo for o in others], axis=0)
            tgt = test.topo - wmean
            loc = np.exp(-(Dmm[p] ** 2) / (2 * SIGMA ** 2))
            preds = {
                "spatial_gauss": loc,
                "fc_1step": np.abs(FC[s][:, p]),
                "fc_diffuse": fc_impulse(FC[s], p, STEPS),
                "causal_var": impulse_topo(A_subj[s], p, STEPS),
            }
            loc_dev = loc - wmean
            for m, pr in preds.items():
                rd[m].append(spatial_r(pr - wmean, tgt))
            # incremental: does the network predictor explain the locality-RESIDUAL of the target?
            tgt_r = resid(tgt, loc_dev)
            for m in nets:
                idv[m].append(spatial_r(resid(preds[m] - wmean, loc_dev), tgt_r))
        for m in raw_dev:
            raw_dev[m].append(float(np.mean(rd[m])))
        for m in nets:
            inc_dev[m].append(float(np.mean(idv[m])))

    print("\n=== deviation-r (capture of site-specific deviation; spatial_gauss = locality ref) ===")
    for m in ["spatial_gauss"] + nets:
        v = np.array(raw_dev[m]); _, lo, hi = bootstrap_ci(v.tolist())
        print(f"  {m:13s} {v.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("\n=== INCREMENTAL deviation-r OVER spatial locality (the decisive test) ===")
    for m in nets:
        v = np.array(inc_dev[m]); _, lo, hi = bootstrap_ci(v.tolist())
        p0 = paired_permutation_test(v.tolist(), [0.0] * len(v))
        print(f"  {m:13s} incremental {v.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p(vs0)={p0:.3g} "
              f"frac>0 {float((v>0).mean())*100:.0f}%")
    best = max(nets, key=lambda m: np.mean(inc_dev[m]))
    bv = np.array(inc_dev[best]); bp = paired_permutation_test(bv.tolist(), [0.0] * len(bv))
    print(f"\nVERDICT: best network readout = {best}, incremental-over-locality {bv.mean():+.3f} "
          f"p={bp:.3g} -> "
          + ("network propagation ADDS beyond locality — the causal mechanism has value; build it."
             if bv.mean() > 0.05 and bp < 0.1 else
             "no signal beyond spatial locality — es-fMRI site-specificity = locality; "
             "the causal-mechanism claim is NOT earned."))


if __name__ == "__main__":
    main()
