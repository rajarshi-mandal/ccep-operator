"""Two SNR-targeted enhancements on the n=13 CCEP LOSO (the methods most likely to lift r):

  METHOD 1 — waveform-shape (CRP) target: predict the canonical-response-parameterization amplitude
             (full-trace SVD matched filter) instead of the single N1 peak. Higher target SNR.
  METHOD 2 — cross-site low-rank denoising: low-rank (truncated-SVD) denoise the TRAINING response
             matrix before building predictors (no test leakage — the held-out target is never
             denoised). Shares structure across sites to clean each training topography.

For each target (N1 peak vs CRP) and each predictor source (raw vs low-rank-denoised training), run
leave-stim-site-out with within_mean / distance / combo, subject-level. Compare to the 0.728 baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa: E402
from ccep_loso import (  # noqa: E402
    all_caches, topo_r, _valid_mask, predict_distance, predict_stim_knn, predict_combo,
    _z, _resid, REL_MIN, SIGMA_GRID, TAU_GRID, BETA_GRID, _score_param,
)

RANK_GRID = [3, 5, 8, 12, 20]   # truncated-SVD ranks to CV for low-rank denoising


def lowrank_denoise(M, rank):
    """Truncated-SVD denoise of a [n_sites, n_contacts] matrix (NaN-filled by column mean)."""
    X = M.copy()
    col = np.nanmean(X, axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col, inds[1])
    mu = X.mean(0)
    U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    r = min(rank, len(S))
    return (U[:, :r] * S[:r]) @ Vt[:r] + mu


def get_target(cs, which):
    if which == "crp" and cs.responses_crp is not None:
        return cs.responses_crp, cs.reliability_crp
    return cs.responses, cs.reliability


def eval_subject(cs, target="crp", denoise=False):
    R, rel = get_target(cs, target)
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(rel)) & (rel >= REL_MIN)]
    if len(keep) < 6:
        return None
    rows = {m: [] for m in ["ceiling", "within_mean", "distance", "combo"]}
    # ceiling from the matching target's half-splits
    h1 = cs.crp_h1 if (target == "crp" and cs.crp_h1 is not None) else cs.responses_h1
    h2 = cs.crp_h2 if (target == "crp" and cs.crp_h2 is not None) else cs.responses_h2

    # a CCEPSubject-like view whose .responses is the chosen (optionally denoised-on-train) target,
    # built per fold below. predict_* read cs.responses, so we swap it per fold.
    orig = cs.responses
    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        mask = _valid_mask(cs, test_i, train_idx)
        tgt = R[test_i]
        if h1 is not None and h2 is not None:
            rows["ceiling"].append(topo_r(h1[test_i], h2[test_i], mask))

        Rtrain = R.copy()
        if denoise:
            # denoise using ONLY training rows (entry-wise CV rank), leave the test row untouched
            sub = R[train_idx]
            den = lowrank_denoise(sub, cv_rank(sub))
            for j, t in enumerate(train_idx):
                Rtrain[t] = den[j]
        cs.responses = Rtrain  # predictors read training rows from here

        Rt = Rtrain[train_idx]
        wm = np.nansum(Rt, 0) / (np.sum(np.isfinite(Rt), 0) + 1e-9)
        rows["within_mean"].append(topo_r(wm, tgt, mask))
        sig = max(SIGMA_GRID, key=lambda s: _score_param(
            cs, train_idx, lambda j, tr, s=s: predict_distance(cs, j, s)))
        rows["distance"].append(topo_r(predict_distance(cs, test_i, sig), tgt, mask))
        tau = max(TAU_GRID, key=lambda tt: _score_param(
            cs, train_idx, lambda j, tr, tt=tt: predict_stim_knn(cs, j, tr, tt)))
        beta = max(BETA_GRID, key=lambda bb: _score_param(
            cs, train_idx,
            lambda j, tr, bb=bb: predict_combo(cs, j, tr, sig, tau, bb, _valid_mask(cs, j, tr))))
        rows["combo"].append(topo_r(predict_combo(cs, test_i, train_idx, sig, tau, beta, mask), tgt, mask))
    cs.responses = orig
    return {m: float(np.nanmean(v)) for m, v in rows.items()}, len(keep)


def cv_rank(sub):
    """Pick truncated-SVD rank by entry-wise hold-out reconstruction error (matrix-completion CV)."""
    rng = np.random.default_rng(0)
    finite = np.argwhere(np.isfinite(sub))
    if len(finite) < 30 or sub.shape[0] < max(RANK_GRID) + 1:
        return min(RANK_GRID[1], sub.shape[0] - 1)
    sel = rng.choice(len(finite), size=max(10, int(0.15 * len(finite))), replace=False)
    hold = finite[sel]
    Xtr = sub.copy()
    Xtr[hold[:, 0], hold[:, 1]] = np.nan
    best, berr = RANK_GRID[0], np.inf
    for rk in RANK_GRID:
        if rk >= sub.shape[0]:
            continue
        rec = lowrank_denoise(Xtr, rk)
        err = np.mean((rec[hold[:, 0], hold[:, 1]] - sub[hold[:, 0], hold[:, 1]]) ** 2)
        if err < berr:
            berr, best = err, rk
    return best


def run(target, denoise, caches):
    rows = {m: [] for m in ["ceiling", "within_mean", "distance", "combo"]}
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        res = eval_subject(cs, target=target, denoise=denoise)
        if res is None:
            continue
        s, _ = res
        for m in rows:
            rows[m].append(s[m])
    return rows


def main():
    caches = all_caches()
    configs = [
        ("N1-peak  raw       ", "n1", False),
        ("CRP      raw       ", "crp", False),
        ("CRP      lowrank   ", "crp", True),
    ]
    results = {}
    print(f"{'config':20s} {'ceiling':>8s} {'within':>8s} {'distance':>9s} {'combo':>8s}")
    for name, tgt, den in configs:
        r = run(tgt, den, caches)
        results[name] = r
        print(f"{name:20s} {np.mean(r['ceiling']):8.3f} {np.mean(r['within_mean']):8.3f} "
              f"{np.mean(r['distance']):9.3f} {np.mean(r['combo']):8.3f}")

    base = results["N1-peak  raw       "]["combo"]
    print("\n=== combo r vs the N1-peak baseline (paired across 13 subjects) ===")
    print(f"  N1-peak raw combo   {np.mean(base):.3f}  (baseline)")
    for name in ["CRP      raw       ", "CRP      lowrank   "]:
        v = results[name]["combo"]
        p = paired_permutation_test(v, base); d = cohens_d_paired(v, base)
        mn, lo, hi = bootstrap_ci(v)
        print(f"  {name} {mn:.3f} [{lo:.3f},{hi:.3f}]  Δ={np.mean(v)-np.mean(base):+.3f}  p={p:.3g}  d={d:+.2f}")


if __name__ == "__main__":
    main()
