"""Out-of-distribution / prospective-deployment proxy — LEAVE-ONE-DATASET-OUT.

A true prospective clinical trial is out of scope (no new data). The strongest retrospective
analog to "deploy the trained model on a patient/site it has never seen" is leave-one-DATASET-out:
train the cross-subject component on 4 datasets, then evaluate on every subject of the held-out
5th dataset — a cohort recorded on a different scanner, montage, and (for ds004080) a different
file format (BrainVision vs MEF3). If performance survives this shift, the model is deployable
beyond its training distribution.

Two things are tested:
  1. The WITHIN-subject models (distance, operator_v2, combo) have NO cross-subject parameters, so
     they transfer to a new dataset BY CONSTRUCTION — each patient is fit from their own sites.
     This is a deployment strength: nothing to retrain per site/scanner. (We restate, not retest.)
  2. The cross-subject REGRESSOR / ensemble is the only component with learned cross-subject
     parameters. We compare:
        - LOSO regressor  : trained leave-one-SUBJECT-out (in-distribution; the classD setting)
        - OOD  regressor  : trained leave-one-DATASET-out (deployment to a new cohort/format)
     If OOD ≈ LOSO and both beat within_mean, the learned prior transfers across cohorts/formats.

Run:  python experiments/ccep_ood.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa: E402
from ccep_loso import (  # noqa: E402
    all_caches, topo_r, _valid_mask, predict_distance, predict_stim_knn, predict_combo,
    SIGMA_GRID, TAU_GRID, BETA_GRID, _score_param, _z,
)
import ccep_regressor as R  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402


def _gbm():
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400,
                                         l2_regularization=1.0, min_samples_leaf=50, random_state=0)


def _dataset_of(tag):
    return tag.split("/")[0]   # "4774", "4696", ...


def _regressor_preds(data, tags, group_fn):
    """group_fn(tag) -> grouping key; train on all rows whose key != the test tag's key."""
    subj_X = {t: np.vstack([F[v] for _, F, _, v, _ in data[t][1]]) for t in tags}
    subj_y = {t: np.concatenate([y[v] for _, _, y, v, _ in data[t][1]]) for t in tags}
    preds = {}
    for tag in tags:
        key = group_fn(tag)
        tr = [t for t in tags if group_fn(t) != key]
        gbm = _gbm()
        gbm.fit(np.vstack([subj_X[t] for t in tr]), np.concatenate([subj_y[t] for t in tr]))
        d = {}
        for test_i, F, y, vmask, tgt in data[tag][1]:
            pr = np.full(len(tgt), np.nan)
            if vmask.sum() >= 4:
                pr[vmask] = gbm.predict(F[vmask])
            d[test_i] = pr
        preds[tag] = d
    return preds


def _combo_preds(data, tags):
    out = {}
    for tag in tags:
        cs, folds = data[tag]
        idxs = [f[0] for f in folds]
        d = {}
        for test_i, F, y, vmask, tgt in folds:
            train_idx = [t for t in idxs if t != test_i]
            mask = _valid_mask(cs, test_i, train_idx)
            sig = max(SIGMA_GRID, key=lambda s: _score_param(
                cs, train_idx, lambda j, tr, s=s: predict_distance(cs, j, s)))
            tau = max(TAU_GRID, key=lambda tt: _score_param(
                cs, train_idx, lambda j, tr, tt=tt: predict_stim_knn(cs, j, tr, tt)))
            beta = max(BETA_GRID, key=lambda bb: _score_param(
                cs, train_idx,
                lambda j, tr, bb=bb: predict_combo(cs, j, tr, sig, tau, bb, _valid_mask(cs, j, tr))))
            d[test_i] = predict_combo(cs, test_i, train_idx, sig, tau, beta, mask)
        out[tag] = d
    return out


def _score(data, tags, pred_map):
    out = []
    for tag in tags:
        cs, folds = data[tag]
        rs = [topo_r(pred_map[tag][test_i], tgt, vmask) for test_i, F, y, vmask, tgt in folds]
        out.append(float(np.nanmean(rs)))
    return np.array(out)


def _score_within(data, tags):
    wm = []
    for tag in tags:
        cs, folds = data[tag]
        idxs = [f[0] for f in folds]
        rs = []
        for test_i, F, y, vmask, tgt in folds:
            train_idx = [t for t in idxs if t != test_i]
            Rr = cs.responses[train_idx]
            w = np.nansum(Rr, axis=0) / (np.sum(np.isfinite(Rr), axis=0) + 1e-9)
            rs.append(topo_r(w, tgt, vmask))
        wm.append(float(np.nanmean(rs)))
    return np.array(wm)


def main():
    caches = all_caches()
    data = R.build_rows(caches)
    tags = list(data)
    ds_of = {t: _dataset_of(t) for t in tags}
    datasets = sorted(set(ds_of.values()))
    print(f"n={len(tags)} subjects across {len(datasets)} datasets: "
          + ", ".join(f"{d}({sum(v==d for v in ds_of.values())})" for d in datasets))

    print("\ncomputing within_mean, combo, LOSO-regressor, OOD(leave-dataset-out)-regressor ...")
    wm = _score_within(data, tags)
    combo_pred = _combo_preds(data, tags)
    loso_pred = _regressor_preds(data, tags, lambda t: t)            # leave-one-SUBJECT-out
    ood_pred = _regressor_preds(data, tags, lambda t: ds_of[t])      # leave-one-DATASET-out

    combo = _score(data, tags, combo_pred)
    reg_loso = _score(data, tags, loso_pred)
    reg_ood = _score(data, tags, ood_pred)

    def ens(combo_p, reg_p, a=0.5):
        out = []
        for tag in tags:
            cs, folds = data[tag]
            rs = []
            for test_i, F, y, vmask, tgt in folds:
                pe = a * _z(combo_p[tag][test_i], vmask) + (1 - a) * _z(reg_p[tag][test_i], vmask)
                rs.append(topo_r(pe, tgt, vmask))
            out.append(float(np.nanmean(rs)))
        return np.array(out)

    ens_ood = ens(combo_pred, ood_pred, 0.5)

    print("\n=== per-dataset means (deployment: model never saw this dataset's cohort/format) ===")
    print(f"{'dataset':10s} {'n':>3s} {'within':>8s} {'combo':>8s} {'reg_LOSO':>9s} {'reg_OOD':>8s} {'ens_OOD':>8s}")
    for d in datasets:
        m = np.array([ds_of[t] == d for t in tags])
        print(f"{d:10s} {m.sum():3d} {wm[m].mean():8.3f} {combo[m].mean():8.3f} "
              f"{reg_loso[m].mean():9.3f} {reg_ood[m].mean():8.3f} {ens_ood[m].mean():8.3f}")

    print(f"\n=== pooled subject-level means (n={len(tags)}, bootstrap 95% CI) ===")
    for name, v in [("within_mean", wm), ("combo (within-subj)", combo),
                    ("regressor LOSO", reg_loso), ("regressor OOD", reg_ood),
                    ("ensemble OOD", ens_ood)]:
        mn, lo, hi = bootstrap_ci(v.tolist())
        print(f"  {name:20s} {mn:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== deployment-transfer tests (paired across subjects) ===")
    # 1. does the OOD regressor transfer (≈ in-distribution LOSO)?
    diff = reg_ood.mean() - reg_loso.mean(); p = paired_permutation_test(reg_ood.tolist(), reg_loso.tolist())
    print(f"  OOD vs LOSO regressor : Δ={diff:+.3f}  p={p:.3g}  "
          f"-> {'TRANSFERS (no significant drop)' if p > 0.05 or diff > -0.02 else 'degrades OOD'}")
    # 2. does the OOD-deployed model still beat within_mean?
    for name, v in [("regressor OOD", reg_ood), ("ensemble OOD", ens_ood), ("combo within-subj", combo)]:
        diff = v.mean() - wm.mean(); p = paired_permutation_test(v.tolist(), wm.tolist())
        d = cohens_d_paired(v.tolist(), wm.tolist()); win = int((v > wm).sum())
        flag = "  <-- beats within_mean OOD" if diff > 0 and p < 0.05 else ""
        print(f"  {name:20s} vs within_mean: Δ={diff:+.3f} p={p:.3g} d={d:+.2f} ({win}/{len(tags)}){flag}")


if __name__ == "__main__":
    main()
