"""Operator v2 — make the effective-connectivity OPERATOR beat the distance baseline ALONE.

The reviewer's sharpest critique: the raw `operator` (0.622) underperforms the `distance`
locality kernel (0.641), which undercuts the "directed propagation" novelty. Diagnosis of the
original `predict_operator` (ccep_loso.py):
  1. ROW-NORMALISES W (Wn = W / rowsum) -> throws away response *amplitude*, the very decay-with-
     distance signal the distance kernel exploits.
  2. Seeds a delta on the nearest stim contacts and does a random walk that DIFFUSES MASS AWAY
     from the stim site, while the strongest CCEP N1 is AT/NEAR the site.
  3. Linear step-sum readout, no tuned self/propagation mix.

A distance kernel is just the *isotropic, zero-propagation special case* of an effective-
connectivity operator. operator_v2 fixes the implementation so the operator NESTS distance and
adds anisotropic network propagation, using only the held-out stim COORDINATE as input (same
information the distance baseline gets) plus the subject's measured connectome:

    seed  = exp(-D(contact, stim)^2 / 2 sigma^2)            # locality seed, geometry only (t=0)
    A     = spectral_norm( symmetrise( measured connectivity from TRAIN sites ) )   # amplitude kept
    y*    = sum_{t=0..T} alpha^t * (A^t @ seed)             # heat-kernel diffusion on the connectome

t=0 is the distance kernel; t>=1 is network propagation through the subject's own effective
connectivity. Tuned by nested inner-LOO on TRAIN sites only (sigma, alpha, T, mode). No held-out
row enters A (no leakage); the held-out prediction uses only its coordinate + the connectome.

Run:  python experiments/ccep_operator_v2.py [--fast]   (--fast = ds004774+ds004696 only, n=13)
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
import ccep_loso as L  # noqa: E402  (reuse predict_distance / predict_operator / topo_r / masks)

SIGMA_GRID = [5, 10, 15, 20, 30, 50]          # mm, locality seed width
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]      # propagation gain (0 == pure distance kernel)
STEP_GRID = [0, 1, 2, 3]                       # diffusion steps (0 == pure distance kernel)
MODE_GRID = ["symmetric", "forward"]          # connectivity direction
REL_MIN = L.REL_MIN


def _spec_norm(P, sym, iters=40):
    """Fast spectral radius (sym) / 2-norm (forward) via power iteration — for the inner-CV hot
    loop. O(iters * n^2) vs eigvalsh's O(n^3). Exact eigvalsh is kept for the final prediction."""
    n = P.shape[0]
    if n == 0:
        return 0.0
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v) + 1e-12
    M = P if sym else None
    for _ in range(iters):
        w = (P @ v) if sym else (P.T @ (P @ v))
        nv = np.linalg.norm(w)
        if nv < 1e-12:
            return 0.0
        v = w / nv
    return float(np.linalg.norm(P @ v))   # sym: |lambda|max ; forward: sigma_max


def _build_operator(cs, train_idx, mode="symmetric", fast=False):
    """Amplitude-preserving, spectral-normalised effective-connectivity operator from TRAIN sites.

    W[a,:] = response when contact a was stimulated (forward, a->others). Rows are AVERAGED (not
    overwritten) when a contact was stimulated in several sites. The propagation matrix P maps a
    direct-activation vector to its network spread: P[c,c'] = "c responds when c' is active".
    Since W[c',c] = response at c when c' stimulated = connectivity c'->c, forward P = W.T.
    Symmetric P = (W+W.T)/2 (more contacts participate: most contacts are never stim sites, so
    pure W.T only reads seed mass at stim contacts). Spectral-normalised to radius 1 (kept
    contractive at readout time via alpha<=1) so amplitude RATIOS survive (the row-norm bug fix).
    """
    n_c = len(cs.contacts)
    W = np.zeros((n_c, n_c))
    cnt = np.zeros(n_c)
    for s in train_idx:
        r = np.nan_to_num(cs.responses[s])
        for a in cs.stim_idx[s]:
            if a >= 0:
                W[a] += r
                cnt[a] += 1
    nz = cnt > 0
    W[nz] /= cnt[nz, None]
    if mode == "forward":
        P = W.T
    elif mode == "symmetric":
        P = (W + W.T) / 2.0
    else:
        raise ValueError(mode)
    # spectral normalise (keep amplitude ratios — the v1 row-norm bug fix). Exact eigvalsh/norm-2
    # for the final prediction; fast power iteration inside the nested-CV hot loop.
    if fast:
        sr = _spec_norm(P, sym=(mode == "symmetric"))
    elif mode == "symmetric":
        sr = np.abs(np.linalg.eigvalsh(P)).max() if n_c else 0.0
    else:
        sr = np.linalg.norm(P, 2) if n_c else 0.0
    if sr > 1e-9:
        P = P / sr
    return P


def predict_operator_v2(cs, test_i, train_idx, sigma, alpha, steps, mode="symmetric", P=None):
    """Heat-kernel diffusion of a distance-seeded impulse through the measured connectome."""
    if P is None:
        P = _build_operator(cs, train_idx, mode)
    D = np.linalg.norm(cs.contact_xyz - cs.stim_xyz[test_i][None], axis=1)
    seed = np.exp(-(D ** 2) / (2 * sigma ** 2))
    y = seed.astype(float).copy()
    cur = seed.astype(float).copy()
    for _ in range(int(steps)):
        cur = P @ cur
        y = y + alpha * cur
    return y


def _best_params(cs, train_idx):
    """Nested inner-LOO over TRAIN sites to pick (sigma, alpha, steps, mode).

    Efficient: for each inner fold and mode, build P ONCE (rebuilt excluding the inner-test site,
    no peeking) and precompute the propagation powers A^t @ seed; every (alpha, steps) is then a
    cheap linear combination. This collapses the grid from ~240 P-builds/fold to ~2.
    """
    maxstep = max(STEP_GRID)
    agg = {}  # (sigma, alpha, steps, mode) -> [r over inner folds]
    for j in train_idx:
        inner = [t for t in train_idx if t != j]
        if len(inner) < 3:
            continue
        mask = L._valid_mask(cs, j, inner)
        tgt = cs.responses[j]
        D = np.linalg.norm(cs.contact_xyz - cs.stim_xyz[j][None], axis=1)
        for mode in MODE_GRID:
            P = _build_operator(cs, inner, mode, fast=True)
            for sigma in SIGMA_GRID:
                seed = np.exp(-(D ** 2) / (2 * sigma ** 2))
                powers = [seed]
                cur = seed.copy()
                for _ in range(maxstep):
                    cur = P @ cur
                    powers.append(cur)
                cum = np.cumsum(np.array(powers[1:]), axis=0) if maxstep else None  # sum_{1..t}
                for steps in STEP_GRID:
                    for alpha in (ALPHA_GRID if steps > 0 else [0.0]):
                        y = powers[0] + (alpha * cum[steps - 1] if steps > 0 else 0.0)
                        r = L.topo_r(y, tgt, mask)
                        agg.setdefault((sigma, alpha, steps, mode), []).append(r)
    best, best_r = (15, 0.5, 1, "symmetric"), -2.0
    for key, rs in agg.items():
        rs = [r for r in rs if np.isfinite(r)]
        m = np.mean(rs) if rs else -1.0
        if m > best_r:
            best_r, best = m, key
    return best


def eval_subject(cs):
    sites = np.arange(len(cs.sites))
    rel = cs.reliability
    keep = sites[(np.isfinite(rel)) & (rel >= REL_MIN)]
    if len(keep) < 6:
        return None
    fold = {m: [] for m in ["within_mean", "distance", "operator_v1", "operator_v2"]}
    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        tgt = cs.responses[test_i]
        mask = L._valid_mask(cs, test_i, train_idx)

        # within_mean
        R = cs.responses[train_idx]
        wmean = np.nansum(R, axis=0) / (np.sum(np.isfinite(R), axis=0) + 1e-9)
        fold["within_mean"].append(L.topo_r(wmean, tgt, mask))

        # distance (nested-CV sigma) — identical protocol to ccep_loso
        sig = max(L.SIGMA_GRID, key=lambda s: L._score_param(
            cs, train_idx, lambda j, tr, s=s: L.predict_distance(cs, j, s)))
        fold["distance"].append(L.topo_r(L.predict_distance(cs, test_i, sig), tgt, mask))

        # operator v1 (original) for reference
        st = max(L.STEP_GRID, key=lambda st: L._score_param(
            cs, train_idx, lambda j, tr, st=st: L.predict_operator(cs, j, tr, 3, st)))
        fold["operator_v1"].append(L.topo_r(L.predict_operator(cs, test_i, train_idx, 3, st), tgt, mask))

        # operator v2 (nested-CV sigma, alpha, steps, mode)
        sg, al, stp, md = _best_params(cs, train_idx)
        P = _build_operator(cs, train_idx, md)
        op2 = predict_operator_v2(cs, test_i, train_idx, sg, al, stp, md, P=P)
        fold["operator_v2"].append(L.topo_r(op2, tgt, mask))
        _PARAMS.append((sg, al, stp, md))

    return {m: float(np.nanmean(v)) for m, v in fold.items()}, len(keep)


_PARAMS = []  # (sigma, alpha, steps, mode) chosen per outer fold — to prove propagation is USED


def main(fast=False):
    caches = L.all_caches()
    if fast:
        caches = [(d, p) for (d, p) in caches if d in ("ds004774", "ds004696")]
    if not caches:
        print("no caches"); return
    models = ["within_mean", "distance", "operator_v1", "operator_v2"]
    rows = {m: [] for m in models}
    print(f"{'subject':18s} {'nsites':>6s} " + " ".join(f"{m:>12s}" for m in models))
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        res = eval_subject(cs)
        if res is None:
            continue
        sc, nk = res
        for m in models:
            rows[m].append(sc[m])
        tag = f"{ds[-4:]}/{cs.subject}"
        print(f"{tag:18s} {nk:6d} " + " ".join(f"{sc[m]:>+12.3f}" for m in models))

    print(f"\n=== SUBJECT-LEVEL topography-r (n={len(rows['distance'])}, bootstrap 95% CI) ===")
    for m in models:
        mean, lo, hi = bootstrap_ci(rows[m])
        print(f"  {m:12s} {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== operator_v2 vs distance (the critique) — paired across subjects ===")
    for ref in ["distance", "operator_v1", "within_mean"]:
        v, b = rows["operator_v2"], rows[ref]
        diff = np.mean(v) - np.mean(b)
        p = paired_permutation_test(v, b)
        d = cohens_d_paired(v, b)
        win = sum(1 for a, q in zip(v, b) if a > q)
        flag = "  <-- operator_v2 WINS" if diff > 0 and p < 0.1 else ""
        print(f"  operator_v2 vs {ref:12s} delta={diff:+.3f}  p={p:.3g}  d={d:+.2f}  ({win}/{len(v)}){flag}")

    # prove the win is from PROPAGATION, not a degenerate reduction to the distance kernel
    if _PARAMS:
        steps = np.array([p[2] for p in _PARAMS])
        alphas = np.array([p[1] for p in _PARAMS])
        modes = [p[3] for p in _PARAMS]
        prop_used = float(np.mean((steps > 0) & (alphas > 0)))
        print(f"\n=== operator_v2 chosen params (n={len(_PARAMS)} folds) ===")
        print(f"  propagation actively used (steps>0 & alpha>0): {prop_used*100:.1f}% of folds "
              f"(if ~0%, the 'operator' is just distance)")
        print(f"  median steps={np.median(steps):.0f}  median alpha={np.median(alphas):.2f}  "
              f"symmetric={100*np.mean([m=='symmetric' for m in modes]):.0f}%")


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
