"""Directionality decomposition — is the operator genuinely DIRECTED, or locality + symmetric smoothing?

Reviewer concern: the reported operator_v2 symmetrizes the connectivity, and the forward directed
variant only appears as a side analysis, so "directed propagation" may be overclaimed. We test the
question head-on by decomposing the measured connectivity W (row a = response to stimulating a) into
its symmetric and antisymmetric parts and measuring what each contributes, all within the SAME
operator_v2 heat-kernel readout (distance-seeded, amplitude-preserving, spectrally normalized):

    S = (W + W^T)/2   (symmetric / reciprocal)      K = (W - W^T)/2   (antisymmetric / purely directed)

Variants compared (shared sigma, alpha, steps chosen by inner LOO on the symmetric operator;
gamma for sym+skew tuned by inner LOO):
    symmetric   : P = specnorm(S)
    forward     : P = specnorm(W^T)     (efferent: stimulate -> response)
    transpose   : P = specnorm(W)       (afferent: reversed edges)
    sym+skew    : P = specnorm(S + gamma*K)   (does the purely-directed part ADD over symmetric?)

Honest readings this distinguishes:
  - forward >> transpose            => orientation matters; reversing the edges destroys prediction.
  - forward ~ symmetric             => the predictive signal is largely reciprocal.
  - sym+skew > symmetric            => the antisymmetric (purely directed) component adds value.

Run:  python experiments/ccep_directed.py [--fast]
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
import ccep_loso as L  # noqa: E402
from ccep_operator_v2 import _spec_norm, predict_operator_v2  # noqa: E402

SIGMA_GRID = [5, 10, 15, 20, 30, 50]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
STEP_GRID = [0, 1, 2, 3]
GAMMA_GRID = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]   # two-sided weight on antisymmetric (directed) part
REL_MIN = L.REL_MIN


def _build_W(cs, idx):
    n_c = len(cs.contacts)
    W = np.zeros((n_c, n_c)); cnt = np.zeros(n_c)
    for s in idx:
        r = np.nan_to_num(cs.responses[s])
        for a in cs.stim_idx[s]:
            if a >= 0:
                W[a] += r; cnt[a] += 1
    nz = cnt > 0; W[nz] /= cnt[nz, None]
    return W


def _P(W, kind, gamma=0.0, fast=True):
    if kind == "symmetric":
        M = 0.5 * (W + W.T)
        sym = True
    elif kind == "forward":
        M = W.T; sym = False
    elif kind == "transpose":
        M = W; sym = False
    elif kind == "symskew":
        M = 0.5 * (W + W.T) + gamma * 0.5 * (W - W.T)
        sym = abs(gamma) < 1e-9
    else:
        raise ValueError(kind)
    sr = _spec_norm(M, sym=sym) if fast else (
        np.abs(np.linalg.eigvalsh(M)).max() if sym else np.linalg.norm(M, 2))
    return M / sr if sr > 1e-9 else M


def _pick_shared(cs, train_idx):
    """Pick (sigma, alpha, steps) by inner LOO on the SYMMETRIC operator."""
    agg = {}
    maxstep = max(STEP_GRID)
    for j in train_idx:
        inner = [t for t in train_idx if t != j]
        if len(inner) < 3:
            continue
        mask = L._valid_mask(cs, j, inner); tgt = cs.responses[j]
        P = _P(_build_W(cs, inner), "symmetric")
        D = np.linalg.norm(cs.contact_xyz - cs.stim_xyz[j][None], axis=1)
        for sigma in SIGMA_GRID:
            seed = np.exp(-(D ** 2) / (2 * sigma ** 2))
            powers = [seed]; cur = seed.copy()
            for _ in range(maxstep):
                cur = P @ cur; powers.append(cur)
            cum = np.cumsum(np.array(powers[1:]), axis=0)
            for steps in STEP_GRID:
                for alpha in (ALPHA_GRID if steps > 0 else [0.0]):
                    y = powers[0] + (alpha * cum[steps - 1] if steps > 0 else 0.0)
                    agg.setdefault((sigma, alpha, steps), []).append(L.topo_r(y, tgt, mask))
    best, br = (15, 0.5, 1), -2
    for k, v in agg.items():
        m = np.nanmean(v)
        if m > br:
            br, best = m, k
    return best


def _pick_gamma(cs, train_idx, sigma, alpha, steps):
    agg = {g: [] for g in GAMMA_GRID}
    for j in train_idx:
        inner = [t for t in train_idx if t != j]
        if len(inner) < 3:
            continue
        mask = L._valid_mask(cs, j, inner); tgt = cs.responses[j]
        W = _build_W(cs, inner)
        for g in GAMMA_GRID:
            P = _P(W, "symskew", gamma=g)
            y = predict_operator_v2(cs, j, inner, sigma, alpha, steps, mode="symmetric", P=P)
            agg[g].append(L.topo_r(y, tgt, mask))
    return max(GAMMA_GRID, key=lambda g: np.nanmean(agg[g]))


def eval_subject(cs):
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
    if len(keep) < 6:
        return None
    fold = {m: [] for m in ["symmetric", "forward", "transpose", "symskew"]}
    gammas = []
    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        tgt = cs.responses[test_i]; mask = L._valid_mask(cs, test_i, train_idx)
        sigma, alpha, steps = _pick_shared(cs, train_idx)
        W = _build_W(cs, train_idx)
        for kind in ["symmetric", "forward", "transpose"]:
            P = _P(W, kind, fast=False)
            y = predict_operator_v2(cs, test_i, train_idx, sigma, alpha, steps, mode="symmetric", P=P)
            fold[kind].append(L.topo_r(y, tgt, mask))
        g = _pick_gamma(cs, train_idx, sigma, alpha, steps)
        gammas.append(g)
        P = _P(W, "symskew", gamma=g, fast=False)
        y = predict_operator_v2(cs, test_i, train_idx, sigma, alpha, steps, mode="symmetric", P=P)
        fold["symskew"].append(L.topo_r(y, tgt, mask))
    out = {m: float(np.nanmean(v)) for m, v in fold.items()}
    out["_gamma_med"] = float(np.median(gammas))
    return out


def main(fast=False):
    caches = L.all_caches()
    if fast:
        caches = [(d, p) for d, p in caches if d in ("ds004774", "ds004696")]
    rows = {m: [] for m in ["symmetric", "forward", "transpose", "symskew"]}
    gmed = []
    print(f"{'subject':18s} {'sym':>8s} {'fwd':>8s} {'trn':>8s} {'sym+skew':>9s} {'gamma':>6s}")
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        r = eval_subject(cs)
        if r is None:
            continue
        for m in rows:
            rows[m].append(r[m])
        gmed.append(r["_gamma_med"])
        print(f"{ds[-4:]+'/'+cs.subject:18s} {r['symmetric']:8.3f} {r['forward']:8.3f} "
              f"{r['transpose']:8.3f} {r['symskew']:9.3f} {r['_gamma_med']:6.2f}")

    n = len(rows["symmetric"])
    print(f"\n=== subject-level means (n={n}, bootstrap 95% CI) ===")
    for m in ["symmetric", "forward", "transpose", "symskew"]:
        mn, lo, hi = bootstrap_ci(rows[m])
        print(f"  {m:10s} {mn:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== directionality contrasts (paired across subjects) ===")
    def contrast(a, b, name):
        va, vb = rows[a], rows[b]
        d = np.mean(va) - np.mean(vb)
        p = paired_permutation_test(va, vb); dd = cohens_d_paired(va, vb)
        w = sum(1 for x, y in zip(va, vb) if x > y)
        print(f"  {name:28s} Δ={d:+.3f}  p={p:.3g}  d={dd:+.2f}  ({w}/{n})")
    contrast("forward", "transpose", "forward vs transpose")
    contrast("forward", "symmetric", "forward vs symmetric")
    contrast("symskew", "symmetric", "sym+skew vs symmetric (directed add)")
    print(f"\n  median selected gamma (antisymmetric weight) = {np.median(gmed):.2f}  "
          f"(fraction subj with gamma>0: {np.mean(np.array(gmed)>0)*100:.0f}%)")
    print("  Interpretation: forward>>transpose => orientation matters (reversal fails);")
    print("                  sym+skew~symmetric  => purely-directed part adds little beyond reciprocal.")


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
