"""Phase 2/3/5 — leave-one-site-out-within-subject (LOSO-WS) + the decisive ablation.

The question Phase 1 set up: ds005498 gives each subject their own rest dynamics and
multiple stim sites, so we can hold out one site and predict its evoked topography from the
*subject's own* information. Does subject-specific causal structure beat the population
template that tied the old (subject-blind) model?

This runs the whole decisive comparison in one pass — five predictors of the held-out
site's evoked topography ``y_{s,k}`` (stim parcel ``p``):

  1. pop_mean       — leave-subject-out population-mean template for site k (handoff baseline).
  2. within_mean    — mean of the SAME subject's OTHER sites' topographies. Robust to the
                      route-B cross-subject registration caveat (all in the subject's own space).
  3. fc_corr        — |resting functional-connectivity column at p|. Subject-specific but
                      CORRELATIONAL (zero-lag, undirected, no mechanism) — the Phase-5
                      "causal vs correlational" control.
  4. causal_subj    — do(p) impulse response through the subject's own dynamics A_subj
                      (Route-A: ridge VAR(1) on rest, shrunk to the group A_group). THE MODEL.
  5. causal_group   — do(p) impulse response through the GROUP dynamics A_group (subject
                      channel removed). THE ABLATION: if subject conditioning is what wins,
                      causal_subj must beat this and this must collapse toward pop_mean.

Subject conditioning uses the resting run only (a separate acquisition from the stim runs),
and never the held-out site's own topography — so there is no target leakage (handoff §8).

Scoring: spatial Pearson r(pred, measured) per (subject, site), plus a *deflated* r that
projects out the dominant shared spatial mode (so we score subject×site deviation, not the
common map). Records are nested in subjects, so the headline aggregates to a per-subject mean
r first, then runs paired sign-flip/permutation tests ACROSS the 148 subjects (the honest
independent unit). Report r relative to the 0.75 ceiling (§5.4).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.ds005498_pipeline import DS005498Cache  # noqa: E402
from eval.stats import (bootstrap_ci, cohens_d_paired,  # noqa: E402
                        paired_permutation_test)

CEILING = 0.75


# --------------------------------------------------------------------------------
# Subject-specific dynamics (Route A, closed form) and the do() readout
# --------------------------------------------------------------------------------
def _lag_matrices(rest: np.ndarray):
    """rest [T,d] -> (X0^T X0, X0^T X1) sufficient statistics for VAR(1)."""
    X0, X1 = rest[:-1], rest[1:]
    return X0.T @ X0, X0.T @ X1


def fit_group_A(rests: list[np.ndarray], lam: float) -> np.ndarray:
    """Ridge VAR(1) transition matrix pooled over subjects. A[i,j] = j influences i."""
    d = rests[0].shape[1]
    G00 = np.zeros((d, d)); G01 = np.zeros((d, d))
    for r in rests:
        if r.shape[0] > 1:
            s00, s01 = _lag_matrices(r)
            G00 += s00; G01 += s01
    At = np.linalg.solve(G00 + lam * np.eye(d), G01)   # A^T
    return At.T


def fit_subject_A(rest: np.ndarray, A_group: np.ndarray, lam: float) -> np.ndarray:
    """Ridge-to-group VAR(1) (Route-A hierarchical shrinkage, closed form):
    minimise ||X1 - X0 A^T||^2 + lam ||A - A_group||^2. Weak rest -> shrinks to A_group."""
    d = rest.shape[1]
    if rest.shape[0] <= 1:
        return A_group.copy()
    s00, s01 = _lag_matrices(rest)
    At = np.linalg.solve(s00 + lam * np.eye(d), s01 + lam * A_group.T)
    return At.T


def impulse_topo(A: np.ndarray, p: int, steps: int) -> np.ndarray:
    """do(stim at parcel p): per-parcel response energy of the impulse response of A.

    h0 = e_p; accumulate sum_t (A^t e_p)^2 over ``steps`` -> sqrt = response topography.
    Includes t=0 so the stimulated parcel carries its direct response.
    """
    d = A.shape[0]
    h = np.zeros(d); h[p] = 1.0
    energy = np.zeros(d)
    for _ in range(steps):
        energy += h * h
        h = A @ h
    return np.sqrt(energy)


def fc_column(rest: np.ndarray, p: int) -> np.ndarray:
    """|resting functional-connectivity| of every parcel with the seed p (correlational)."""
    C = np.corrcoef(rest.T)
    C = np.nan_to_num(C)
    return np.abs(C[:, p])


# --------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------
def spatial_r(pred: np.ndarray, meas: np.ndarray) -> float:
    a = pred - pred.mean(); b = meas - meas.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 1e-9 and db > 1e-9 else 0.0


def deflate(x: np.ndarray, mode: np.ndarray) -> np.ndarray:
    return x - (x @ mode) * mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam-group", type=float, default=50.0)
    ap.add_argument("--lam-subj", type=float, default=200.0,
                    help="subject ridge-to-group strength (high = trust the group more)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default="data/processed/ds005498_phase2.json")
    ap.add_argument("--report", default="reports/PHASE2_LOSO_WS.md")
    ap.add_argument("--cache-dir", default="data/processed/ds005498")
    args = ap.parse_args()

    cache = DS005498Cache(cache_dir=args.cache_dir, qc_filter=True)
    recs = cache.records
    by_sub: dict[str, list] = {}
    for r in recs:
        by_sub.setdefault(r.subject, []).append(r)
    subjects = sorted(by_sub)
    d = cache.centroids.shape[0]
    print(f"[phase2] {len(recs)} QC records, {len(subjects)} subjects, d={d}", flush=True)

    # group dynamics + shared spatial mode (first PC of all topographies)
    rests = [by_sub[s][0].subject_rest for s in subjects]
    A_group = fit_group_A(rests, args.lam_group)
    T = np.stack([r.topo for r in recs])
    Tc = T - T.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
    shared_mode = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    print(f"[phase2] A_group spectral radius {max(abs(np.linalg.eigvals(A_group))):.3f}; "
          f"shared mode captures {(Tc @ shared_mode).var()/Tc.var()/d*100:.0f}% var", flush=True)

    A_subj_cache = {s: fit_subject_A(by_sub[s][0].subject_rest, A_group, args.lam_subj)
                    for s in subjects}

    preds = ["pop_mean", "within_mean", "fc_corr", "causal_subj", "causal_group"]
    # per-record raw r and deflated r
    rows = []
    for s in subjects:
        srecs = by_sub[s]
        if len(srecs) < 2:
            continue
        A_subj = A_subj_cache[s]
        rest = srecs[0].subject_rest
        for i, test in enumerate(srecs):
            p, meas = test.stim_parcel, test.topo
            others = [srecs[j].topo for j in range(len(srecs)) if j != i]
            pr = {
                "pop_mean": cache.site_template(test.site_name, exclude_subject=s),
                "within_mean": np.mean(others, axis=0),
                "fc_corr": fc_column(rest, p),
                "causal_subj": impulse_topo(A_subj, p, args.steps),
                "causal_group": impulse_topo(A_group, p, args.steps),
            }
            row = {"subject": s, "site": test.site_name, "rel": test.reliability}
            for k in preds:
                row[k] = spatial_r(pr[k], meas)
                row[k + "_def"] = spatial_r(deflate(pr[k], shared_mode),
                                            deflate(meas, shared_mode))
            rows.append(row)

    # subject-level aggregation (the honest independent unit)
    subj_means = {}
    for s in subjects:
        sr = [r for r in rows if r["subject"] == s]
        if sr:
            subj_means[s] = {k: float(np.mean([r[k] for r in sr]))
                             for k in [c for c in rows[0] if c not in ("subject", "site", "rel")]}
    S = sorted(subj_means)

    def col(metric):
        return np.array([subj_means[s][metric] for s in S])

    def compare(a_key, b_key):
        a, b = col(a_key), col(b_key)
        mean, lo, hi = bootstrap_ci((a - b).tolist())
        return dict(
            a_mean=float(a.mean()), b_mean=float(b.mean()), diff=float((a - b).mean()),
            diff_ci=[lo, hi], p=paired_permutation_test(a.tolist(), b.tolist()),
            d=cohens_d_paired(a.tolist(), b.tolist()),
            frac_a_wins=float((a > b).mean()), n=len(S))

    headline = {m: float(col(m).mean()) for m in preds}
    headline_def = {m: float(col(m + "_def").mean()) for m in preds}
    comparisons = {
        "causal_subj_vs_pop_mean": compare("causal_subj", "pop_mean"),
        "causal_subj_vs_within_mean": compare("causal_subj", "within_mean"),
        "causal_subj_vs_fc_corr": compare("causal_subj", "fc_corr"),
        "ABLATION_causal_subj_vs_causal_group": compare("causal_subj", "causal_group"),
        "causal_group_vs_pop_mean": compare("causal_group", "pop_mean"),
        # deflated (subject×site deviation only)
        "DEFLATED_causal_subj_vs_within_mean": compare("causal_subj_def", "within_mean_def"),
        "DEFLATED_causal_subj_vs_fc_corr": compare("causal_subj_def", "fc_corr_def"),
    }

    out = dict(params=vars(args), n_subjects=len(S), n_records=len(rows),
               ceiling=CEILING, headline_subjlevel=headline,
               headline_deflated=headline_def, comparisons=comparisons)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    _report(out, Path(args.report))
    _print(out)


def _fmt_cmp(c):
    return (f"diff {c['diff']:+.3f} [CI {c['diff_ci'][0]:+.3f},{c['diff_ci'][1]:+.3f}] "
            f"p={c['p']:.3g} d={c['d']:+.2f} wins {c['frac_a_wins']*100:.0f}% (n={c['n']})")


def _print(out):
    print("\n=== Phase 2 LOSO-WS (subject-level mean spatial r) ===")
    for k, v in out["headline_subjlevel"].items():
        print(f"  {k:14s} r={v:+.3f}  ({v/out['ceiling']*100:+.0f}% of {out['ceiling']} ceiling)")
    print("  -- deflated (subject×site deviation) --")
    for k, v in out["headline_deflated"].items():
        print(f"  {k:14s} r={v:+.3f}")
    print("\n=== Decisive comparisons ===")
    for name, c in out["comparisons"].items():
        print(f"  {name}: {_fmt_cmp(c)}")


def _report(out, path: Path):
    L = []; P = L.append
    h, hd, C = out["headline_subjlevel"], out["headline_deflated"], out["comparisons"]
    P("# Phase 2 — LOSO-WS subject-conditioned prediction + ablation (ds005498)\n")
    P(f"- {out['n_subjects']} subjects, {out['n_records']} held-out (subject,site) folds; "
      f"spatial ceiling {out['ceiling']} (§5.4).")
    P(f"- Params: {out['params']}\n")
    P("## Headline — subject-level mean spatial r (raw topography)")
    for k, v in h.items():
        P(f"- `{k}`: **{v:+.3f}** ({v/out['ceiling']*100:+.0f}% of ceiling)")
    P("\n## Deflated (shared mode removed — scores subject×site deviation)")
    for k, v in hd.items():
        P(f"- `{k}_def`: **{v:+.3f}**")
    P("\n## Decisive comparisons (paired across subjects)")
    for name, c in C.items():
        P(f"- **{name}** — {_fmt_cmp(c)}")
    # verdict
    abl = C["ABLATION_causal_subj_vs_causal_group"]
    vs_pop = C["causal_subj_vs_pop_mean"]
    vs_within = C["causal_subj_vs_within_mean"]
    P("\n## Verdict")
    win = (vs_pop["diff"] > 0 and vs_pop["p"] < 0.05 and vs_pop["d"] > 0)
    win_within = (vs_within["diff"] > 0 and vs_within["p"] < 0.05)
    collapse = (abl["diff"] > 0 and abl["p"] < 0.05)
    if win and win_within and collapse:
        P("- **CEILING BROKEN (provisional).** Subject-conditioned causal prediction beats "
          "both the population-mean and within-subject baselines, AND the subject-channel "
          "ablation collapses. Confirm with route-A registration + multi-seed before claiming.")
    elif win and collapse and not win_within:
        P("- **PARTIAL / SUSPECT.** Beats the population template and the ablation collapses, "
          "but does NOT beat the within-subject mean — consistent with the route-B "
          "registration caveat inflating the population baseline. Not a clean win.")
    elif not collapse:
        P("- **NO WIN — subject channel inert.** The ablation does not collapse; subject "
          "dynamics add nothing over group dynamics. Same failure mode as the old NULL. "
          "Revisit registration (route A) / conditioning architecture before tuning.")
    else:
        P("- **NO WIN.** Causal_subj does not clear the win condition. See comparisons above.")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
