"""Learned cross-subject contact-level regressor (lever 1 to push past combo=0.728).

`combo` is one hand-built formula (z(distance) + beta * network-residual) tuned per subject. Here we
instead LEARN the map from a rich per-contact feature vector to the N1 amplitude, with a gradient-
boosted regressor trained ACROSS subjects. Two potential wins over combo:
  (1) nonlinear interactions among locality / network / anatomy features;
  (2) cross-subject pooling — train on 12 patients, predict the 13th (an amortised model).

Leakage control (strict):
  * features for subject S's site s are computed with S's OWN leave-that-site-out predictors
    (within_mean / stim_knn / operator over S's other sites) — never using site s itself;
  * the regressor is trained LEAVE-ONE-SUBJECT-OUT — to score subject S, S contributes NO rows.
So predicting S uses only (a) other patients' learned structure and (b) S's other sites' geometry.

Features per (site s, contact c), then z-scored within each site (topo-r is per-site scale-free):
  d            euclidean stim->contact distance (mm, cross-subject comparable; raw + z)
  d_homotopic  distance from the mirrored (x-flipped) stim coord to c  (homotopic responses)
  same_hemi    contact and stim on same hemisphere (sign of x)
  within_mean  the subject's common response at c
  knn10/25/60  stim-location kNN response prediction at 3 bandwidths (network)
  operator     effective-connectivity propagation at c (network)

Target: N1 amplitude topography (z-scored per site). Metric/eval identical to ccep_loso.
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
    all_caches, topo_r, _valid_mask, predict_distance, predict_stim_knn, predict_operator, REL_MIN,
)
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

# deterministic per-subject baselines from ccep_loso (n=13) for the side-by-side
WITHIN_MEAN = {"4774/sub-MAYO01": 0.022, "4774/sub-MAYO02": 0.095, "4774/sub-MAYO03": 0.134,
               "4774/sub-MAYO04": 0.376, "4774/sub-MAYO05": 0.244, "4696/sub-01": 0.216,
               "4696/sub-02": 0.547, "4696/sub-03": 0.143, "4696/sub-04": 0.303,
               "4696/sub-05": 0.279, "4696/sub-06": 0.316, "4696/sub-07": 0.164, "4696/sub-08": 0.116}
DISTANCE = {"4774/sub-MAYO01": 0.696, "4774/sub-MAYO02": 0.685, "4774/sub-MAYO03": 0.442,
            "4774/sub-MAYO04": 0.636, "4774/sub-MAYO05": 0.469, "4696/sub-01": 0.768,
            "4696/sub-02": 0.500, "4696/sub-03": 0.777, "4696/sub-04": 0.723,
            "4696/sub-05": 0.762, "4696/sub-06": 0.766, "4696/sub-07": 0.564, "4696/sub-08": 0.676}
COMBO = {"4774/sub-MAYO01": 0.781, "4774/sub-MAYO02": 0.715, "4774/sub-MAYO03": 0.495,
         "4774/sub-MAYO04": 0.689, "4774/sub-MAYO05": 0.547, "4696/sub-01": 0.831,
         "4696/sub-02": 0.767, "4696/sub-03": 0.817, "4696/sub-04": 0.803,
         "4696/sub-05": 0.807, "4696/sub-06": 0.833, "4696/sub-07": 0.647, "4696/sub-08": 0.735}

FEATS = ["d", "d_homo", "same_hemi", "z_d", "z_wm", "z_knn10", "z_knn25", "z_knn60", "z_op"]


def _z(x, mask):
    out = np.zeros_like(x, dtype=float)
    ok = mask & np.isfinite(x)
    if ok.sum() >= 2:
        mu, sd = x[ok].mean(), x[ok].std()
        out[ok] = (x[ok] - mu) / (sd + 1e-9)
    return out


def site_features(cs, test_i, train_idx, mask):
    xyz = cs.contact_xyz
    stim = cs.stim_xyz[test_i]
    d = np.linalg.norm(xyz - stim[None], axis=1)
    stim_m = stim * np.array([-1.0, 1.0, 1.0])
    d_homo = np.linalg.norm(xyz - stim_m[None], axis=1)
    same_hemi = (np.sign(xyz[:, 0]) == np.sign(stim[0])).astype(float)
    R = cs.responses[train_idx]
    wm = np.nansum(R, axis=0) / (np.sum(np.isfinite(R), axis=0) + 1e-9)
    knn10 = predict_stim_knn(cs, test_i, train_idx, 10.0)
    knn25 = predict_stim_knn(cs, test_i, train_idx, 25.0)
    knn60 = predict_stim_knn(cs, test_i, train_idx, 60.0)
    op = predict_operator(cs, test_i, train_idx, 3, 2)
    F = np.column_stack([
        d, d_homo, same_hemi,
        _z(d, mask), _z(wm, mask), _z(knn10, mask), _z(knn25, mask), _z(knn60, mask), _z(op, mask),
    ])
    return F


def build_rows(caches):
    """Return per-subject lists of (site_index, X[n_valid,F], y[n_valid], mask, tgt)."""
    data = {}
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        tag = f"{ds[-4:]}/{cs.subject}"
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
        if len(keep) < 6:
            continue
        folds = []
        for test_i in keep:
            train_idx = [t for t in keep if t != test_i]
            mask = _valid_mask(cs, test_i, train_idx)
            F = site_features(cs, test_i, train_idx, mask)
            tgt = cs.responses[test_i]
            vmask = mask & np.isfinite(tgt) & np.all(np.isfinite(F), axis=1)
            y = _z(tgt, vmask)
            folds.append((test_i, F, y, vmask, tgt))
        data[tag] = (cs, folds)
    return data


def main():
    caches = all_caches()
    data = build_rows(caches)
    tags = list(data)

    # pre-stack each subject's training rows (valid contacts only)
    subj_X, subj_y = {}, {}
    for tag, (cs, folds) in data.items():
        Xs, ys = [], []
        for _, F, y, vmask, _ in folds:
            Xs.append(F[vmask]); ys.append(y[vmask])
        subj_X[tag] = np.vstack(Xs); subj_y[tag] = np.concatenate(ys)

    print(f"{'subject':20s} {'within':>7s} {'distance':>9s} {'combo':>7s} {'regress':>8s} {'vs combo':>9s}")
    reg_scores = {}
    for tag in tags:
        cs, folds = data[tag]
        Xtr = np.vstack([subj_X[t] for t in tags if t != tag])
        ytr = np.concatenate([subj_y[t] for t in tags if t != tag])
        gbm = HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=400, l2_regularization=1.0,
            min_samples_leaf=50, random_state=0)
        gbm.fit(Xtr, ytr)
        rs = []
        for test_i, F, y, vmask, tgt in folds:
            pred = np.full(len(tgt), np.nan)
            if vmask.sum() >= 4:
                pred[vmask] = gbm.predict(F[vmask])
            rs.append(topo_r(pred, tgt, vmask))
        score = float(np.nanmean(rs))
        reg_scores[tag] = score
        print(f"{tag:20s} {WITHIN_MEAN[tag]:7.3f} {DISTANCE[tag]:9.3f} {COMBO[tag]:7.3f} "
              f"{score:8.3f} {score-COMBO[tag]:+9.3f}")

    reg = np.array([reg_scores[t] for t in tags])
    cmb = np.array([COMBO[t] for t in tags])
    dist = np.array([DISTANCE[t] for t in tags])
    wm = np.array([WITHIN_MEAN[t] for t in tags])
    print("\n=== subject-level means (bootstrap 95% CI) ===")
    for name, v in [("within_mean", wm), ("distance", dist), ("combo", cmb), ("regressor", reg)]:
        m, lo, hi = bootstrap_ci(v.tolist())
        print(f"  {name:12s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== regressor vs each (paired across 13 subjects) ===")
    for name, v in [("within_mean", wm), ("distance", dist), ("combo", cmb)]:
        diff = reg.mean() - v.mean(); p = paired_permutation_test(reg.tolist(), v.tolist())
        d = cohens_d_paired(reg.tolist(), v.tolist()); win = int((reg > v).sum())
        flag = "  <-- regressor better" if diff > 0 and p < 0.1 else ""
        print(f"  vs {name:12s} Δ={diff:+.3f}  p={p:.3g}  d={d:+.2f}  ({win}/13 win){flag}")


if __name__ == "__main__":
    main()
