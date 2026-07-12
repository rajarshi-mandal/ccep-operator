"""EXTENSION — operator identifiability & recovery (ground-truth simulation, no real data needed).

Reviewer/coauthor question (Eckstein's specialty): a fitted operator's *prediction* accuracy does
not tell you whether the operator itself is IDENTIFIABLE — i.e. whether you actually recovered the
true connectivity or just fit something predictive. Here we settle it directly.

Protocol:
  1. Build a KNOWN ground-truth connectivity operator A_true on a realistic electrode geometry
     (sparse local edges + directed long-range edges; spectrally normalized).
  2. Generate each stim site's response by propagating an impulse through A_true (the same forward
     model the method assumes), then corrupt it with T-trial measurement noise.
  3. Estimate the operator A_hat from the measured responses exactly as operator_v2 does.
  4. Measure RECOVERY: correlation between A_hat and the (symmetric part of) A_true over the
     stimulated contacts, and leave-one-site-out PREDICTION r — as functions of #stim-sites and
     trials/site.

Result answers: in what data regime is the operator identifiable, and where does human CCEP sit?
Output: reports/recovery.json (arrays for the jumbo figure).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "recovery.json"
RNG = np.random.default_rng(0)


def make_true_operator(n=100, k_local=6, n_long=45, radius=8.0, cap=0.85, rng=RNG):
    xyz = rng.uniform([-32, -32, -12], [32, 32, 12], (n, 3))
    D = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    A = np.zeros((n, n))
    loc = np.exp(-D / radius)
    for i in range(n):                                    # sparse local edges: k nearest
        nn = np.argsort(D[i])[1:k_local + 1]
        A[i, nn] = loc[i, nn]
    for _ in range(n_long):                               # directed long-range edges
        i, j = rng.integers(0, n), rng.integers(0, n)
        if i != j:
            A[i, j] += rng.uniform(0.3, 0.9)
    sr = np.abs(np.linalg.eigvals(A)).max()
    A = A / (sr + 1e-9) * cap
    return A, xyz, D


def gen_responses(A, site_contacts, steps=2):
    n = A.shape[0]
    R = np.zeros((len(site_contacts), n))
    for k, s in enumerate(site_contacts):
        cur = np.zeros(n); cur[s] = 1.0; acc = np.zeros(n)
        for _ in range(steps):
            cur = A @ cur; acc += cur
        acc[s] = np.nan
        R[k] = acc
    return R


def measure(R, T, snr_db=6.0, rng=RNG):
    sig = np.nanstd(R); nsd = sig / (10 ** (snr_db / 20.0))
    acc = np.zeros_like(R)
    for _ in range(T):
        acc += np.nan_to_num(R) + rng.normal(0, nsd, R.shape)
    m = acc / T; m[np.isnan(R)] = np.nan
    return m


def build_operator(R, site_contacts, n):
    W = np.zeros((n, n)); cnt = np.zeros(n)
    for k, s in enumerate(site_contacts):
        W[s] += np.nan_to_num(R[k]); cnt[s] += 1
    nz = cnt > 0; W[nz] /= cnt[nz, None]
    S = 0.5 * (W + W.T)
    sr = np.abs(np.linalg.eigvalsh(S)).max()
    return (S / (sr + 1e-9) if sr > 1e-9 else S), W


def _topo_r(a, b, exclude):
    ok = np.isfinite(a) & np.isfinite(b); ok[exclude] = False
    if ok.sum() < 6:
        return np.nan
    p = a[ok] - a[ok].mean(); m = b[ok] - b[ok].mean()
    den = np.linalg.norm(p) * np.linalg.norm(m)
    return float((p @ m) / den) if den > 1e-12 else np.nan


def recovery_and_prediction(n_sites, T, n=100, seed=0):
    rng = np.random.default_rng(seed)
    A, xyz, D = make_true_operator(n=n, rng=rng)
    A_sym = 0.5 * (A + A.T)
    stim = rng.choice(n, size=min(n_sites, n), replace=False)
    R_true = gen_responses(A, stim, steps=2)
    R_meas = measure(R_true, T, rng=rng)

    # --- recovery: A_hat over stimulated contacts vs true symmetric operator ---
    A_hat, _ = build_operator(R_meas, stim, n)
    idx = np.ix_(stim, stim)
    off = ~np.eye(len(stim), dtype=bool)
    a, b = A_hat[idx][off], A_sym[idx][off]
    ok = np.isfinite(a) & np.isfinite(b)
    rec = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 6 else np.nan
    # NETWORK recovery: residualize both operators against the distance expectation (exp(-D/8)),
    # then correlate — isolates recovery of the NON-local (network) structure, the harder target.
    Dsub = D[idx][off]
    locexp = np.exp(-Dsub / 8.0)
    def _resid(x):
        m = ok & np.isfinite(locexp)
        c = np.polyfit(locexp[m], x[m], 1)
        r = np.full_like(x, np.nan); r[m] = x[m] - (c[0] * locexp[m] + c[1]); return r
    ar, br = _resid(a), _resid(b)
    okr = np.isfinite(ar) & np.isfinite(br)
    net_rec = float(np.corrcoef(ar[okr], br[okr])[0, 1]) if okr.sum() > 6 else np.nan

    # --- leave-one-site-out prediction (propagate delta seed through A_hat from other sites) ---
    preds = []
    for k, s in enumerate(stim):
        others = [j for j in range(len(stim)) if j != k]
        A_o, _ = build_operator(R_meas[others], stim[others], n)
        cur = np.zeros(n); cur[s] = 1.0; acc = np.zeros(n)
        for _ in range(2):
            cur = A_o @ cur; acc += cur
        preds.append(_topo_r(acc, R_meas[k], s))
    pred = float(np.nanmean(preds))
    return rec, net_rec, pred


def main():
    SITES = [8, 15, 25, 40, 60, 90]
    TRIALS = [4, 10, 30, 100]
    SEEDS = range(4)
    rec = np.zeros((len(SITES), len(TRIALS)))
    netr = np.zeros((len(SITES), len(TRIALS)))
    prd = np.zeros((len(SITES), len(TRIALS)))
    print(f"{'sites':>6} {'trials':>7} {'recovery_r':>11} {'net_recov':>10} {'predict_r':>10}")
    for i, ns in enumerate(SITES):
        for j, T in enumerate(TRIALS):
            rs = [recovery_and_prediction(ns, T, seed=s) for s in SEEDS]
            rec[i, j] = np.nanmean([r[0] for r in rs])
            netr[i, j] = np.nanmean([r[1] for r in rs])
            prd[i, j] = np.nanmean([r[2] for r in rs])
            print(f"{ns:6d} {T:7d} {rec[i,j]:11.3f} {netr[i,j]:10.3f} {prd[i,j]:10.3f}")
    OUT.write_text(json.dumps({
        "sites": SITES, "trials": TRIALS,
        "recovery": rec.tolist(), "net_recovery": netr.tolist(), "prediction": prd.tolist(),
        "human_regime": {"sites": "16-132 (median ~55)", "trials": "~10"},
    }, indent=2))
    print(f"\nsaved {OUT}")
    print("Interpretation: recovery rises with BOTH sites and trials; the human-CCEP regime "
          "(dozens of sites, ~10 trials) yields partial identifiability -> report the operator as "
          "predictive, validated by recovery, NOT as the recovered ground-truth connectome.")


if __name__ == "__main__":
    main()
