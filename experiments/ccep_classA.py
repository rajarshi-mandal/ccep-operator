"""CLASS A — anatomy (Destrieux) features for the near/mid-field structure the diagnostic flagged.

The distance diagnostic showed the recoverable headroom is in near/mid contacts (fine spatial
structure an isotropic distance kernel misses), not the far-field (noise floor). Anatomy is the
cheap lever for that: a contact in the SAME gyrus/parcel as the stim site responds differently from
one the same distance away across a sulcus.

ds004696 ships Destrieux parcel labels (the 5 ds004774/MAYO subjects do not), so this isolates the
anatomy gain on those **8 subjects**: base regressor vs +anatomy vs combo, leave-one-subject-out.

Anatomy features per (site, contact): same_parcel (exact Destrieux match to either stim contact),
same_region (match after stripping hemisphere + G/S/G&S prefix), is_white_matter.
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa: E402
from ccep_loso import all_caches, topo_r, _valid_mask, REL_MIN  # noqa: E402
import ccep_regressor as R  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402


def region_root(text):
    if not isinstance(text, str):
        return None
    t = re.sub(r"^(lh|rh)_", "", text)
    t = re.sub(r"^(G&S|G|S)_", "", t)
    return t


def load_anat(dataset, sub, contacts):
    """Destrieux label_text per contact (aligned to `contacts`); None entries if unmapped."""
    hits = glob.glob(str(PROJ / f"Open Neuro {dataset}" / sub / "ses-*" / "ieeg" / "*_electrodes.tsv"))
    if not hits:
        return None
    el = pd.read_csv(sorted(hits)[0], sep="\t")
    if "Destrieux_label_text" not in el.columns:
        return None
    lut = dict(zip(el["name"], el["Destrieux_label_text"]))
    return [lut.get(c) for c in contacts]


def anat_features(cs, test_i, labels):
    """[n_c, 3] anatomy features for the held-out stim site."""
    n_c = len(cs.contacts)
    stim_lab = [labels[a] for a in cs.stim_idx[test_i] if a >= 0 and labels[a] is not None]
    stim_root = {region_root(l) for l in stim_lab}
    same_parcel = np.zeros(n_c); same_region = np.zeros(n_c); is_wm = np.zeros(n_c)
    for c in range(n_c):
        lc = labels[c]
        if lc is None:
            same_parcel[c] = same_region[c] = is_wm[c] = np.nan
            continue
        same_parcel[c] = 1.0 if lc in stim_lab else 0.0
        same_region[c] = 1.0 if region_root(lc) in stim_root else 0.0
        is_wm[c] = 1.0 if "White_Matter" in str(lc) else 0.0
    return np.column_stack([same_parcel, same_region, is_wm])


def main():
    caches = [(ds, c) for ds, c in all_caches() if ds == "ds004696"]
    data, anat = {}, {}
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        tag = f"{ds[-4:]}/{cs.subject}"
        labels = load_anat(ds, cs.subject, cs.contacts)
        if labels is None:
            continue
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
        if len(keep) < 6:
            continue
        folds = []
        for test_i in keep:
            train_idx = [t for t in keep if t != test_i]
            mask = _valid_mask(cs, test_i, train_idx)
            Fbase = R.site_features(cs, test_i, train_idx, mask)
            Fanat = anat_features(cs, test_i, labels)
            tgt = cs.responses[test_i]
            vmask = mask & np.isfinite(tgt) & np.all(np.isfinite(Fbase), axis=1)
            y = R._z(tgt, vmask)
            folds.append((test_i, Fbase, Fanat, y, vmask, tgt))
        data[tag] = (cs, folds)
    tags = list(data)
    print(f"Class A anatomy gain on {len(tags)} ds004696 subjects (leave-one-subject-out)\n")

    def stack(with_anat):
        X, Y = {}, {}
        for tag, (cs, folds) in data.items():
            Xs, ys = [], []
            for _, Fb, Fa, y, vmask, _ in folds:
                F = np.column_stack([Fb, Fa]) if with_anat else Fb
                Xs.append(F[vmask]); ys.append(y[vmask])
            X[tag] = np.vstack(Xs); Y[tag] = np.concatenate(ys)
        return X, Y

    def run(with_anat):
        X, Y = stack(with_anat)
        scores = {}
        for tag in tags:
            Xtr = np.vstack([X[t] for t in tags if t != tag])
            ytr = np.concatenate([Y[t] for t in tags if t != tag])
            gbm = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=400,
                                                l2_regularization=1.0, min_samples_leaf=50,
                                                random_state=0)
            gbm.fit(Xtr, ytr)
            cs, folds = data[tag]; rs = []
            for test_i, Fb, Fa, y, vmask, tgt in folds:
                F = np.column_stack([Fb, Fa]) if with_anat else Fb
                pred = np.full(len(tgt), np.nan)
                if vmask.sum() >= 4:
                    pred[vmask] = gbm.predict(F[vmask])
                rs.append(topo_r(pred, tgt, vmask))
            scores[tag] = float(np.nanmean(rs))
        return scores

    base = run(False); anat_s = run(True)
    print(f"{'subject':14s} {'base':>8s} {'+anatomy':>9s} {'Δ':>7s}")
    for tag in tags:
        print(f"{tag:14s} {base[tag]:8.3f} {anat_s[tag]:9.3f} {anat_s[tag]-base[tag]:+7.3f}")
    b = np.array([base[t] for t in tags]); a = np.array([anat_s[t] for t in tags])
    bm, blo, bhi = bootstrap_ci(b.tolist()); am, alo, ahi = bootstrap_ci(a.tolist())
    p = paired_permutation_test(a.tolist(), b.tolist()); d = cohens_d_paired(a.tolist(), b.tolist())
    print(f"\n  base      {bm:+.3f} [{blo:+.3f}, {bhi:+.3f}]")
    print(f"  +anatomy  {am:+.3f} [{alo:+.3f}, {ahi:+.3f}]")
    print(f"  Δ={am-bm:+.3f}  p={p:.3g}  d={d:+.2f}  ({int((a>b).sum())}/{len(tags)} subj improve)")


if __name__ == "__main__":
    main()
