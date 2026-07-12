"""Stage 4 (ds002799) — does ANY stim-location-aware predictor beat the subject-mean?

The powered n=9 run found within_mean (the location-AGNOSTIC average of a subject's other es
responses) is the predictor to beat, and the causal do()-readout didn't. Before building the heavy
trained §12 model, settle the prior question with simple, strong location-AWARE predictors: if the
held-out site's known stimulation location helps at all, one of these should beat within_mean —
especially in deflated space (subject-common response removed), which is where site-specificity lives.

Predictors of the held-out site's evoked topography (within-subject LOSO; sites are subject-specific):
  within_mean  — mean of the subject's OTHER sites' topographies (baseline; ignores stim location).
  nearest_site — the other site whose stim coordinate is closest to the held-out one.
  dist_weighted— distance-weighted average of other sites (Gaussian on stim-coordinate distance).
  spatial_gauss— pure location prior: Gaussian bump centred on the held-out stim parcel (Schaefer
                 centroid distance). Uses ONLY where you stimulated, no other-site data.
  causal_subj  — do() impulse response through the subject's rest-fit dynamics (the model).

Scored as spatial Pearson r, raw and DEFLATED (remove the subject's mean-across-sites response =
the subject-common component). Aggregated to subject-level means; each predictor paired vs within_mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ds005498_pipeline import DS005498Cache  # noqa: E402
from eval.stats import bootstrap_ci, cohens_d_paired, paired_permutation_test  # noqa: E402
from phase2_loso_ws import fit_group_A, fit_subject_A, impulse_topo, spatial_r, deflate  # noqa: E402

CEIL = 0.75


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/processed/ds002799")
    ap.add_argument("--sigma-mm", type=float, default=25.0)
    ap.add_argument("--lam-group", type=float, default=50.0)
    ap.add_argument("--lam-subj", type=float, default=200.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default="data/processed/ds002799_phase4.json")
    args = ap.parse_args()

    cache = DS005498Cache(cache_dir=args.cache_dir, qc_filter=True)
    cents = cache.centroids
    Dmm = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)   # [d,d] parcel distances
    by_sub: dict[str, list] = {}
    for r in cache.records:
        by_sub.setdefault(r.subject, []).append(r)
    subjects = [s for s in sorted(by_sub) if len(by_sub[s]) >= 2]
    print(f"[phase4] {sum(len(by_sub[s]) for s in subjects)} records, {len(subjects)} subjects")

    rests = [by_sub[s][0].subject_rest for s in subjects]
    A_group = fit_group_A(rests, args.lam_group)
    A_subj = {s: fit_subject_A(by_sub[s][0].subject_rest, A_group, args.lam_subj) for s in subjects}

    preds = ["within_mean", "nearest_site", "dist_weighted", "spatial_gauss", "causal_subj"]
    loc_preds = ["nearest_site", "dist_weighted", "spatial_gauss", "causal_subj"]
    rows = []
    for s in subjects:
        srecs = by_sub[s]
        for i, test in enumerate(srecs):
            others = [srecs[j] for j in range(len(srecs)) if j != i]
            p = test.stim_parcel
            ck = test.coil_mni
            dists = np.array([np.linalg.norm(o.coil_mni - ck) for o in others])
            w = np.exp(-(dists ** 2) / (2 * args.sigma_mm ** 2)); w = w / (w.sum() + 1e-9)
            wmean = np.mean([o.topo for o in others], axis=0)      # leave-one-out subject-common
            pr = {
                "within_mean": wmean,
                "nearest_site": others[int(np.argmin(dists))].topo,
                "dist_weighted": np.sum([wi * o.topo for wi, o in zip(w, others)], axis=0),
                "spatial_gauss": np.exp(-(Dmm[p] ** 2) / (2 * args.sigma_mm ** 2)),
                "causal_subj": impulse_topo(A_subj[s], p, args.steps),
            }
            row = {"subject": s}
            tgt_dev = test.topo - wmean                            # the site-specific deviation
            for k in preds:
                row[k] = spatial_r(pr[k], test.topo)               # raw
            for k in loc_preds:
                # does the predictor's deviation from the subject-common capture the true deviation?
                row[k + "_dev"] = spatial_r(pr[k] - wmean, tgt_dev)
            rows.append(row)

    metrics = [c for c in rows[0] if c != "subject"]
    subj_means = {s: {m: float(np.mean([r[m] for r in rows if r["subject"] == s])) for m in metrics}
                  for s in subjects}
    col = lambda m: np.array([subj_means[s][m] for s in subjects])

    def cmp(a, b):
        x, y = col(a), col(b)
        mean, lo, hi = bootstrap_ci((x - y).tolist())
        return dict(a=float(x.mean()), b=float(y.mean()), diff=float((x - y).mean()),
                    ci=[lo, hi], p=paired_permutation_test(x.tolist(), y.tolist()),
                    d=cohens_d_paired(x.tolist(), y.tolist()), wins=float((x > y).mean()))

    print("\n=== subject-level mean spatial r (raw, predicting the full topography) ===")
    for m in preds:
        print(f"  {m:14s} {col(m).mean():+.3f}")
    print("\n=== RAW: location-aware predictors vs within_mean (does location beat the subject-mean?) ===")
    out = {"n_subjects": len(subjects), "raw": {m: float(col(m).mean()) for m in preds},
           "vs_within_raw": {}, "deviation_capture": {}}
    for m in loc_preds:
        c = cmp(m, "within_mean")
        print(f"  {m:13s}: diff {c['diff']:+.3f} [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
              f"p={c['p']:.3g} d={c['d']:+.2f} wins {c['wins']*100:.0f}%")
        out["vs_within_raw"][m] = c

    print("\n=== SITE-SPECIFICITY: does each predictor capture the deviation from within_mean? "
          "(r>0 = yes) ===")
    for m in loc_preds:
        v = col(m + "_dev")
        mean, lo, hi = bootstrap_ci(v.tolist())
        p0 = paired_permutation_test(v.tolist(), [0.0] * len(v))
        print(f"  {m:13s} deviation-r = {v.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p(vs0)={p0:.3g} "
              f"frac>0 {float((v>0).mean())*100:.0f}%")
        out["deviation_capture"][m] = dict(mean=float(v.mean()), ci=[lo, hi], p=p0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    best = max(loc_preds, key=lambda m: col(m).mean())
    bc = out["vs_within_raw"][best]
    bestdev = max(loc_preds, key=lambda m: col(m + "_dev").mean())
    dv = out["deviation_capture"][bestdev]
    print(f"\nVERDICT (raw): best location-aware = {best} (r={col(best).mean():+.3f}) vs within_mean "
          f"diff {bc['diff']:+.3f} p={bc['p']:.3g}")
    print(f"VERDICT (site-specificity): best = {bestdev} deviation-r {dv['mean']:+.3f} p={dv['p']:.3g} -> "
          + ("there IS capturable site-specific structure — build the trained readout."
             if dv['mean'] > 0.05 and dv['p'] < 0.1
             else "site-specific deviation is NOT captured — the subject-common response is the ceiling."))


if __name__ == "__main__":
    main()
