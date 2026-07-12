"""CLASS D — hierarchical group+individual model (amortized + personalized).

The pure cross-subject regressor (group/amortized) tied combo; within-subject combo (individual)
is the per-subject best. The hierarchical idea is that group and individual capture COMPLEMENTARY
structure — so stacking them should beat either. We z-combine, per held-out site:
    pred = a * z(combo_within)  +  (1 - a) * z(regressor_cross)
and report a=0.5 (no tuning) and the oracle single best global a (upper bound on complementarity).
If even the oracle ensemble ≈ combo, group and individual are redundant and we're at the model
ceiling for this feature set.
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
    REL_MIN, SIGMA_GRID, TAU_GRID, BETA_GRID, _score_param, _z,
)
import ccep_regressor as R  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402


def main():
    caches = all_caches()
    data = R.build_rows(caches)          # {tag: (cs, folds)}; folds=(test_i,F,y,vmask,tgt)
    tags = list(data)

    # cross-subject regressor predictions per site (leave-one-subject-out)
    subj_X = {t: np.vstack([F[v] for _, F, _, v, _ in data[t][1]]) for t in tags}
    subj_y = {t: np.concatenate([y[v] for _, _, y, v, _ in data[t][1]]) for t in tags}
    reg_pred = {}                        # tag -> {test_i -> pred vector}
    for tag in tags:
        Xtr = np.vstack([subj_X[t] for t in tags if t != tag])
        ytr = np.concatenate([subj_y[t] for t in tags if t != tag])
        gbm = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400,
                                            l2_regularization=1.0, min_samples_leaf=50, random_state=0)
        gbm.fit(Xtr, ytr)
        d = {}
        for test_i, F, y, vmask, tgt in data[tag][1]:
            pr = np.full(len(tgt), np.nan)
            if vmask.sum() >= 4:
                pr[vmask] = gbm.predict(F[vmask])
            d[test_i] = pr
        reg_pred[tag] = d

    # combo predictions per site (within-subject, nested-CV)
    combo_pred = {}
    for tag in tags:
        cs, folds = data[tag]
        d = {}
        for test_i, F, y, vmask, tgt in folds:
            train_idx = [t for t in (f[0] for f in folds) if t != test_i]
            mask = _valid_mask(cs, test_i, train_idx)
            sig = max(SIGMA_GRID, key=lambda s: _score_param(
                cs, train_idx, lambda j, tr, s=s: predict_distance(cs, j, s)))
            tau = max(TAU_GRID, key=lambda tt: _score_param(
                cs, train_idx, lambda j, tr, tt=tt: predict_stim_knn(cs, j, tr, tt)))
            beta = max(BETA_GRID, key=lambda bb: _score_param(
                cs, train_idx,
                lambda j, tr, bb=bb: predict_combo(cs, j, tr, sig, tau, bb, _valid_mask(cs, j, tr))))
            d[test_i] = predict_combo(cs, test_i, train_idx, sig, tau, beta, mask)
        combo_pred[tag] = d

    def ens_score(a):
        out = []
        for tag in tags:
            cs, folds = data[tag]; rs = []
            for test_i, F, y, vmask, tgt in folds:
                mask = vmask
                pe = a * _z(combo_pred[tag][test_i], mask) + (1 - a) * _z(reg_pred[tag][test_i], mask)
                rs.append(topo_r(pe, tgt, mask))
            out.append(float(np.nanmean(rs)))
        return np.array(out)

    combo = ens_score(1.0); reg = ens_score(0.0)
    ens_half = ens_score(0.5)
    grid = np.linspace(0, 1, 21)
    a_star = grid[np.argmax([ens_score(a).mean() for a in grid])]
    ens_best = ens_score(a_star)

    print(f"{'subject':20s} {'combo':>8s} {'regress':>8s} {'ens.5':>8s} {'ens*':>8s}")
    for i, tag in enumerate(tags):
        print(f"{tag:20s} {combo[i]:8.3f} {reg[i]:8.3f} {ens_half[i]:8.3f} {ens_best[i]:8.3f}")

    print("\n=== subject-level means (bootstrap 95% CI) ===")
    for name, v in [("combo", combo), ("regressor", reg),
                    ("ensemble a=.5", ens_half), (f"ensemble a*={a_star:.2f}", ens_best)]:
        m, lo, hi = bootstrap_ci(v.tolist())
        print(f"  {name:18s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== ensemble (oracle a*) vs combo (paired) ===")
    p = paired_permutation_test(ens_best.tolist(), combo.tolist())
    d = cohens_d_paired(ens_best.tolist(), combo.tolist())
    print(f"  Δ={ens_best.mean()-combo.mean():+.3f}  p={p:.3g}  d={d:+.2f}  "
          f"({int((ens_best>combo).sum())}/{len(tags)} subj improve)  at a*={a_star:.2f}")
    if ens_best.mean() - combo.mean() < 0.01:
        print("  -> group and individual are redundant; at the model ceiling for this feature set.")


if __name__ == "__main__":
    main()
