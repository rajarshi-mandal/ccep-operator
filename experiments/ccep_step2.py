"""STEP 2 — higher-r target reframes on the existing n=93 data (no new acquisition).

(a) RESPONDER DETECTION: instead of full-topography r, score the clinically useful question
    "which contacts respond?" — ROC-AUC and precision@k of combo vs within_mean.
(b) FEW-SHOT / TRANSDUCTIVE: if you may deliver a small pilot sample of pulses at the held-out site
    itself, how high does prediction go? Use the site's own first-half trials (h1) as the pilot,
    score against the independent second half (h2). Compare:
        cross-site only (combo vs h2)   — zero own-trials (the leave-site-out setting)
        own-pilot only  (h1 vs h2)      — a few own pulses, no model
        few-shot blend  (z(combo)+β·z(h1) vs h2, β nested-CV'd)
    This quantifies the value of a pilot sample for the realistic clinical workflow.

Subject-level aggregation over all reliable sites; n=93 across 5 datasets.
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
    _z, REL_MIN, SIGMA_GRID, TAU_GRID, BETA_GRID, _score_param,
)
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def precision_at_k(pred, meas, mask, k):
    ok = mask & np.isfinite(pred) & np.isfinite(meas)
    if ok.sum() < 2 * k:
        return np.nan
    p, m = pred[ok], meas[ok]
    return len(set(np.argsort(m)[-k:]) & set(np.argsort(p)[-k:])) / k


def responder_auc(pred, meas, mask):
    ok = mask & np.isfinite(pred) & np.isfinite(meas)
    if ok.sum() < 6:
        return np.nan
    m = meas[ok]; lab = (m >= np.quantile(m, 2 / 3)).astype(int)
    if lab.sum() == 0 or lab.sum() == len(lab):
        return np.nan
    return roc_auc_score(lab, pred[ok])


def fewshot_r(combo, h1, h2, mask, betas=(0, 0.25, 0.5, 1, 2, 4)):
    """r vs h2 for cross-site-only, own-pilot-only, and the best combo+pilot blend."""
    cross = topo_r(combo, h2, mask)
    pilot = topo_r(h1, h2, mask)
    best = max(betas, key=lambda b: topo_r(_z(combo, mask) + b * _z(h1, mask), h2, mask))
    blend = topo_r(_z(combo, mask) + best * _z(h1, mask), h2, mask)
    return cross, pilot, blend


def main():
    caches = all_caches()
    metr = {k: [] for k in ["auc_wm", "auc_combo", "p5_wm", "p5_combo", "p10_wm", "p10_combo",
                            "fs_cross", "fs_pilot", "fs_blend"]}
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
        if len(keep) < 6:
            continue
        s = {k: [] for k in metr}
        for test_i in keep:
            train = [t for t in keep if t != test_i]
            mask = _valid_mask(cs, test_i, train)
            tgt = cs.responses[test_i]
            R = cs.responses[train]
            wm = np.nansum(R, 0) / (np.sum(np.isfinite(R), 0) + 1e-9)
            sig = max(SIGMA_GRID, key=lambda v: _score_param(cs, train, lambda j, tr, v=v: predict_distance(cs, j, v)))
            tau = max(TAU_GRID, key=lambda v: _score_param(cs, train, lambda j, tr, v=v: predict_stim_knn(cs, j, tr, v)))
            beta = max(BETA_GRID, key=lambda b: _score_param(cs, train, lambda j, tr, b=b: predict_combo(cs, j, tr, sig, tau, b, _valid_mask(cs, j, tr))))
            combo = predict_combo(cs, test_i, train, sig, tau, beta, mask)
            # (a) responder detection
            s["auc_wm"].append(responder_auc(wm, tgt, mask)); s["auc_combo"].append(responder_auc(combo, tgt, mask))
            s["p5_wm"].append(precision_at_k(wm, tgt, mask, 5)); s["p5_combo"].append(precision_at_k(combo, tgt, mask, 5))
            s["p10_wm"].append(precision_at_k(wm, tgt, mask, 10)); s["p10_combo"].append(precision_at_k(combo, tgt, mask, 10))
            # (b) few-shot
            if cs.responses_h1 is not None:
                cr, pl, bl = fewshot_r(combo, cs.responses_h1[test_i], cs.responses_h2[test_i], mask)
                s["fs_cross"].append(cr); s["fs_pilot"].append(pl); s["fs_blend"].append(bl)
        for k in metr:
            if s[k]:
                metr[k].append(np.nanmean(s[k]))

    def line(name, arr):
        m, lo, hi = bootstrap_ci(arr)
        return f"{name:22s} {m:.3f} [{lo:.3f}, {hi:.3f}]"

    print("=== (a) RESPONDER DETECTION: combo vs within_mean (subject-level, n=%d) ===" % len(metr["auc_combo"]))
    for label, wm, cb in [("ROC-AUC", "auc_wm", "auc_combo"), ("precision@5", "p5_wm", "p5_combo"),
                          ("precision@10", "p10_wm", "p10_combo")]:
        p = paired_permutation_test(metr[cb], metr[wm]); d = cohens_d_paired(metr[cb], metr[wm])
        print(f"  {label:14s} within_mean {np.mean(metr[wm]):.3f} -> combo {np.mean(metr[cb]):.3f}  "
              f"(Δ{np.mean(metr[cb])-np.mean(metr[wm]):+.3f}, p={p:.2g}, d={d:+.2f})")

    print("\n=== (b) FEW-SHOT / transductive (r vs held-out half h2, n=%d) ===" % len(metr["fs_blend"]))
    print("  " + line("cross-site only", metr["fs_cross"]) + "   (zero own-trials = leave-site-out)")
    print("  " + line("own-pilot only", metr["fs_pilot"]) + "   (a few own pulses, no model)")
    print("  " + line("few-shot blend", metr["fs_blend"]) + "   (combo + pilot)")
    pj = paired_permutation_test(metr["fs_blend"], metr["fs_cross"])
    print(f"  -> few-shot blend vs cross-site: Δ={np.mean(metr['fs_blend'])-np.mean(metr['fs_cross']):+.3f}, p={pj:.2g}")


if __name__ == "__main__":
    main()
