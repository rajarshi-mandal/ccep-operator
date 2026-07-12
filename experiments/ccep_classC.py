"""CLASS C — richer performance metrics + properly-powered (hierarchical) statistics.

topo-r alone hides a lot. This reports, for `combo` (the best model) vs `within_mean` (the bar),
a panel that characterises performance and uses the ~550 site-level folds for real power:

Metrics per site (then summarised):
  pearson_r   topographic Pearson r (the headline)
  spearman_r  rank correlation (robust to the amplitude distribution)
  prec@5/@10  of the k strongest MEASURED contacts, fraction in the predicted top-k
              (the clinical question: did we find the responders?)
  responder_AUC  ranking AUC for "is this contact a top-tertile responder?"

Statistics:
  * subject-level paired permutation (n=13) — as before;
  * SITE-level win-rate over all folds;
  * HIERARCHICAL bootstrap (resample subjects, then sites within) for a CI/p that respects the
    nesting — far more power than the n=13 sign-flip floor without pseudo-replication.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import paired_permutation_test  # noqa: E402
from ccep_loso import (  # noqa: E402
    all_caches, topo_r, _valid_mask, predict_distance, predict_stim_knn, predict_combo,
    REL_MIN, SIGMA_GRID, TAU_GRID, BETA_GRID, _score_param,
)
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def metrics(pred, meas, mask):
    ok = mask & np.isfinite(pred) & np.isfinite(meas)
    if ok.sum() < 6:
        return None
    p, m = pred[ok], meas[ok]
    pr = float(np.corrcoef(p, m)[0, 1])
    sr = float(spearmanr(p, m).correlation)
    out = {"pearson_r": pr, "spearman_r": sr}
    for k in (5, 10):
        kk = min(k, ok.sum() // 2)
        top_m = set(np.argsort(m)[-kk:]); top_p = set(np.argsort(p)[-kk:])
        out[f"prec@{k}"] = len(top_m & top_p) / kk
    thr = np.quantile(m, 2 / 3)
    lab = (m >= thr).astype(int)
    out["responder_AUC"] = float(roc_auc_score(lab, p)) if 0 < lab.sum() < len(lab) else np.nan
    return out


def main():
    caches = all_caches()
    site_rows = []   # (subject_idx, combo_metrics, within_metrics)
    subj_combo, subj_within = {m: [] for m in
                              ["pearson_r", "spearman_r", "prec@5", "prec@10", "responder_AUC"]}, \
                             {m: [] for m in
                              ["pearson_r", "spearman_r", "prec@5", "prec@10", "responder_AUC"]}
    sidx = 0
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
        if len(keep) < 6:
            continue
        agg_c = {k: [] for k in subj_combo}; agg_w = {k: [] for k in subj_within}
        for test_i in keep:
            train_idx = [t for t in keep if t != test_i]
            mask = _valid_mask(cs, test_i, train_idx)
            tgt = cs.responses[test_i]
            R = cs.responses[train_idx]
            wmean = np.nansum(R, axis=0) / (np.sum(np.isfinite(R), axis=0) + 1e-9)
            sig = max(SIGMA_GRID, key=lambda s: _score_param(
                cs, train_idx, lambda j, tr, s=s: predict_distance(cs, j, s)))
            tau = max(TAU_GRID, key=lambda tt: _score_param(
                cs, train_idx, lambda j, tr, tt=tt: predict_stim_knn(cs, j, tr, tt)))
            beta = max(BETA_GRID, key=lambda bb: _score_param(
                cs, train_idx,
                lambda j, tr, bb=bb: predict_combo(cs, j, tr, sig, tau, bb, _valid_mask(cs, j, tr))))
            pred = predict_combo(cs, test_i, train_idx, sig, tau, beta, mask)
            mc, mw = metrics(pred, tgt, mask), metrics(wmean, tgt, mask)
            if mc is None or mw is None:
                continue
            site_rows.append((sidx, mc["pearson_r"], mw["pearson_r"]))
            for k in agg_c:
                agg_c[k].append(mc[k]); agg_w[k].append(mw[k])
        for k in subj_combo:
            subj_combo[k].append(np.nanmean(agg_c[k])); subj_within[k].append(np.nanmean(agg_w[k]))
        sidx += 1

    print("=== metric panel: combo vs within_mean (subject-level mean over 13) ===")
    print(f"{'metric':14s} {'within':>8s} {'combo':>8s} {'Δ':>8s} {'p(n=13)':>9s}")
    for k in subj_combo:
        w = np.array(subj_within[k]); cb = np.array(subj_combo[k])
        p = paired_permutation_test(cb.tolist(), w.tolist())
        print(f"{k:14s} {np.nanmean(w):8.3f} {np.nanmean(cb):8.3f} {np.nanmean(cb)-np.nanmean(w):+8.3f} {p:9.3g}")

    # hierarchical bootstrap on per-site pearson Δ (combo - within), resampling subjects then sites
    rng = np.random.default_rng(0)
    by_s = {}
    for si, cpr, wpr in site_rows:
        d = cpr - wpr
        if np.isfinite(d):
            by_s.setdefault(si, []).append(d)
    subs = [s for s in by_s if by_s[s]]
    boots = []
    for _ in range(10000):
        ss = rng.choice(subs, size=len(subs), replace=True)
        vals = []
        for s in ss:
            arr = by_s[s]
            idx = rng.integers(0, len(arr), size=len(arr))
            vals.extend(np.array(arr)[idx])
        boots.append(np.mean(vals))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pboot = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    nwin = sum(1 for _, cpr, wpr in site_rows if np.isfinite(cpr - wpr) and cpr > wpr)
    print(f"\n=== site-level power ({len(site_rows)} folds) ===")
    print(f"  combo beats within_mean at {nwin}/{len(site_rows)} sites "
          f"({nwin/len(site_rows)*100:.0f}%)")
    print(f"  hierarchical bootstrap Δpearson = {boots.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  "
          f"p={pboot:.2g}  (subjects→sites; respects nesting)")


if __name__ == "__main__":
    main()
