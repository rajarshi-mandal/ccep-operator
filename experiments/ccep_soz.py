"""TIER-1 EXTENSION (T1.4) — Does the individualized operator localize epileptogenic tissue?

ccepAge (ds004080) ships per-electrode `soz` (seizure-onset zone) and `resected` labels. This tests
whether contact features derived from the fitted effective-connectivity operator discriminate
clinically-defined epileptogenic tissue — turning the stimulation-response model into a CANDIDATE
biomarker. Coverage (audited): 36 subjects with >=1 SOZ contact (353), 40 with >=1 resected (656).

Per-contact operator features (defined for all recording contacts):
  afferent_strength : mean CCEP response of the contact across all stim sites (excitability as a
                      RECEIVER). NOTE: this is essentially "raw response amplitude" — the pre-operator
                      baseline. A biomarker that only reflects this is not an operator contribution.
  efferent_strength : mean total reach evoked when the contact is stimulated (as a DRIVER; stim
                      contacts only, else per-subject median-imputed).
  avg_ctrb          : average controllability (Gu 2015) — network-wide energy delivered by driving it.
  modal_ctrb        : modal controllability — ability to drive hard-to-reach modes.
  asymmetry         : efferent - afferent (directional imbalance).
Geometry nulls: node_density (#contacts within 20 mm), mean_dist to other contacts.

Rigor: (1) univariate within-subject AUC per feature, aggregated across subjects (paired test vs
0.5); (2) MULTIVARIATE leave-one-SUBJECT-out logistic regression (subject-clustered, no leakage),
operator features vs an amplitude+geometry-only baseline, with a within-subject label-permutation
null. The key question is whether OPERATOR-SPECIFIC features (controllability, efferent, asymmetry)
add over amplitude+geometry.

Honest failure mode: SOZ localization from stimulation is hard; a null (operator ~ amplitude/degree)
is a real, reported bound and does not threaten the main result. Never claim clinical outcomes:
"associated with", "candidate biomarker", "retrospective", "prospective validation required".

Output: reports/soz.json.  Run: python experiments/ccep_soz.py
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test  # noqa: E402
import ccep_operator_v2 as V2  # noqa: E402
from ccep_control import controllability  # noqa: E402  (reuse Gramian metrics)

RAW = Path("REDACTED/Open Neuro ds004080")
REL_MIN = V2.REL_MIN
RNG = np.random.default_rng(0)


def load_labels(subject):
    """contact_name -> (soz_bool, resected_bool) from raw electrodes.tsv."""
    matches = list(RAW.glob(f"{subject}/ses-*/ieeg/*electrodes.tsv"))
    if not matches:
        return None
    lab = {}
    with open(matches[0]) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        if "soz" not in hdr:
            return None
        ni = hdr.index("name"); si = hdr.index("soz")
        ri = hdr.index("resected") if "resected" in hdr else None
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= si:
                continue
            soz = p[si].strip().lower() == "yes"
            res = (ri is not None and len(p) > ri and p[ri].strip().lower() == "yes")
            lab[p[ni]] = (soz, res)
    return lab


def contact_features(cs):
    """Per-contact operator + geometry features. Returns dict name->feature-vector and feature names."""
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
    if len(keep) < 6:
        return None
    n_c = len(cs.contacts)
    R = cs.responses[keep]                                    # [n_keep, n_c]
    afferent = np.nanmean(R, axis=0)                          # receiver excitability
    afferent = np.where(np.isfinite(afferent), afferent, np.nanmedian(afferent))
    # efferent: total reach when a contact is a stim contact
    efferent = np.full(n_c, np.nan)
    for i, s in enumerate(keep):
        reach = np.nansum(R[i])
        for a in cs.stim_idx[s]:
            if a >= 0:
                efferent[a] = np.nanmax([efferent[a], reach]) if np.isfinite(efferent[a]) else reach
    eff_med = np.nanmedian(efferent[np.isfinite(efferent)]) if np.isfinite(efferent).any() else 0.0
    efferent = np.where(np.isfinite(efferent), efferent, eff_med)
    # controllability from the symmetric operator over all reliable sites
    A = V2._build_operator(cs, list(keep), "symmetric")
    avg_ctrb, modal_ctrb = controllability(A)
    asym = efferent - afferent
    # geometry nulls
    XYZ = cs.contact_xyz
    D = np.linalg.norm(XYZ[:, None, :] - XYZ[None, :, :], axis=2)
    node_density = (D < 20).sum(axis=1).astype(float)
    mean_dist = np.nanmean(np.where(D > 0, D, np.nan), axis=1)
    feats = {
        "afferent_strength": afferent, "efferent_strength": efferent,
        "avg_ctrb": avg_ctrb, "modal_ctrb": modal_ctrb, "asymmetry": asym,
        "node_density": node_density, "mean_dist": mean_dist,
    }
    return {cs.contacts[i]: {k: float(v[i]) for k, v in feats.items()} for i in range(n_c)}, list(feats.keys())


def auc(scores, labels):
    """Mann-Whitney AUC of scores for binary labels; NaN if a class is empty."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def collect():
    """Gather per-subject feature tables + labels for subjects that have any SOZ or resected."""
    data = []
    fnames = None
    for p in sorted((ROOT / "data" / "processed" / "ds004080").glob("sub-*.npz")):
        cs = CCEPSubject.load(str(p))
        lab = load_labels(cs.subject)
        if lab is None:
            continue
        ff = contact_features(cs)
        if ff is None:
            continue
        feats, fnames = ff
        names = [n for n in cs.contacts if n in feats and n in lab]
        if not names:
            continue
        soz = np.array([lab[n][0] for n in names])
        res = np.array([lab[n][1] for n in names])
        X = np.array([[feats[n][k] for k in fnames] for n in names])
        data.append({"subject": cs.subject, "names": names, "X": X, "soz": soz, "res": res})
    return data, fnames


def univariate(data, fnames, target):
    """Per-subject within-subject AUC for each feature; aggregate across subjects w/ >=1 pos & >=1 neg."""
    out = {}
    for fi, fn in enumerate(fnames):
        aucs = []
        for d in data:
            y = d[target]
            if y.sum() == 0 or (~y).sum() == 0:
                continue
            aucs.append(auc(d["X"][:, fi], y))
        aucs = [a for a in aucs if np.isfinite(a)]
        if len(aucs) < 5:
            continue
        m, lo, hi = bootstrap_ci(aucs)
        p = paired_permutation_test(aucs, [0.5] * len(aucs))
        out[fn] = {"auc": m, "lo": lo, "hi": hi, "p_vs_chance": float(p), "n_subj": len(aucs)}
    return out


def _standardize(X, ref=None):
    mu = ref[0] if ref else np.nanmean(X, axis=0)
    sd = ref[1] if ref else (np.nanstd(X, axis=0) + 1e-9)
    return (X - mu) / sd, (mu, sd)


def multivariate(data, fnames, target, feat_idx):
    """Leave-one-SUBJECT-out logistic regression AUC over the chosen feature columns (subject-clustered).

    Returns pooled held-out AUC and a per-subject-permutation null (shuffle labels within subject).
    """
    from sklearn.linear_model import LogisticRegression
    subj = [d for d in data if d[target].sum() > 0 and (~d[target]).sum() > 0]
    if len(subj) < 6:
        return None
    def run(get_labels):
        preds, labs = [], []
        for i, dtest in enumerate(subj):
            Xtr = np.vstack([subj[j]["X"][:, feat_idx] for j in range(len(subj)) if j != i])
            ytr = np.concatenate([get_labels(subj[j]) for j in range(len(subj)) if j != i])
            if ytr.sum() < 2 or (~ytr.astype(bool)).sum() < 2:
                continue
            Xtr_s, ref = _standardize(Xtr)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(np.nan_to_num(Xtr_s), ytr.astype(int))
            Xte_s, _ = _standardize(dtest["X"][:, feat_idx], ref)
            preds.append(clf.predict_proba(np.nan_to_num(Xte_s))[:, 1])
            labs.append(get_labels(dtest))
        if not preds:
            return np.nan
        return auc(np.concatenate(preds), np.concatenate(labs).astype(bool))
    real = run(lambda d: d[target])
    null = []
    for _ in range(200):
        null.append(run(lambda d: RNG.permutation(d[target])))
    null = np.array([x for x in null if np.isfinite(x)])
    p = float((null >= real).mean()) if len(null) else np.nan
    return {"auc": float(real), "null_mean": float(np.nanmean(null)), "p_perm": p, "n_subj": len(subj)}


def main():
    data, fnames = collect()
    if not data:
        print("no labeled subjects"); return
    print(f"labeled subjects: {len(data)}  (contacts total {sum(len(d['names']) for d in data)})")
    op_feats = ["afferent_strength", "efferent_strength", "avg_ctrb", "modal_ctrb", "asymmetry"]
    geo_feats = ["node_density", "mean_dist"]
    amp_geo = ["afferent_strength", "node_density", "mean_dist"]          # baseline: amplitude+geometry
    all_feats = op_feats + geo_feats
    idx = {f: i for i, f in enumerate(fnames)}

    out = {"n_labeled_subjects": len(data)}
    for target, tname in [("soz", "SEIZURE-ONSET ZONE"), ("res", "RESECTED TISSUE")]:
        npos = sum(int(d[target].sum()) for d in data)
        nsub = sum(1 for d in data if d[target].sum() > 0 and (~d[target]).sum() > 0)
        print(f"\n{'='*70}\n{tname}: {npos} positive contacts, {nsub} usable subjects")
        uni = univariate(data, fnames, target)
        print("  -- univariate within-subject AUC (mean, vs chance 0.5) --")
        for fn, s in sorted(uni.items(), key=lambda kv: -kv[1]["auc"]):
            star = " *" if s["p_vs_chance"] < 0.05 else ""
            tag = " [operator]" if fn in op_feats and fn != "afferent_strength" else \
                  (" [amplitude]" if fn == "afferent_strength" else " [geometry]")
            print(f"     {fn:18s} AUC {s['auc']:.3f} [{s['lo']:.3f},{s['hi']:.3f}] "
                  f"p={s['p_vs_chance']:.3g} (n={s['n_subj']}){tag}{star}")
        mv_all = multivariate(data, fnames, target, [idx[f] for f in all_feats])
        mv_base = multivariate(data, fnames, target, [idx[f] for f in amp_geo])
        mv_oponly = multivariate(data, fnames, target,
                                 [idx[f] for f in ["efferent_strength", "avg_ctrb", "modal_ctrb", "asymmetry"]])
        print("  -- multivariate leave-one-subject-out logistic AUC (subject-clustered) --")
        if mv_base:
            print(f"     amplitude+geometry baseline : AUC {mv_base['auc']:.3f}  (perm p={mv_base['p_perm']:.3g})")
        if mv_oponly:
            print(f"     operator-only (no amplitude): AUC {mv_oponly['auc']:.3f}  (perm p={mv_oponly['p_perm']:.3g})")
        if mv_all:
            gain = (mv_all["auc"] - mv_base["auc"]) if mv_base else float("nan")
            print(f"     full (operator+geometry)    : AUC {mv_all['auc']:.3f}  (perm p={mv_all['p_perm']:.3g})"
                  f"   [+{gain:+.3f} over amp+geom]")
        out[target] = {"n_pos": npos, "n_subj": nsub, "univariate": uni,
                       "mv_full": mv_all, "mv_amp_geom": mv_base, "mv_operator_only": mv_oponly}

    (ROOT / "reports" / "soz.json").write_text(json.dumps(out, indent=2))
    print("\nsaved reports/soz.json")


if __name__ == "__main__":
    main()
