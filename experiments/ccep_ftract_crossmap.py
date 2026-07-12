"""FOLLOW-UP F1 — Cross-map our electrodes into F-TRACT parcels for DIRECT operator validation
and a population-scale cold-start prior.

Our ccepAge electrodes are surface-projected onto fsaverage (verified: contact->pial vertex distance
median 0.00 mm). We map each contact -> nearest fsaverage vertex -> Lausanne2008-250 parcel
(`{hemi}.{name}`, matching F-TRACT's column naming exactly). Then:

  (A) DIRECT OPERATOR VALIDATION — build OUR group parcel x parcel operator (74 ccepAge subjects) in
      Lausanne2008-250 space and correlate it with F-TRACT's 780-patient amplitude/probability matrix
      over shared parcel pairs. Does our small-sample operator recover the population structure, and
      is the DIRECTION (stim->rec) preserved?

  (B) F-TRACT AS A COLD-START PRIOR — for each held-out patient's stim site (parcel p*), predict the
      response at each contact (parcel q) directly from F-TRACT amplitude[p*, q] (external, no leakage).
      Does the 780-patient population prior beat the patient's own geometry — flipping the honest T1.3
      negative (where our 74-subject prior did not)?

Output: reports/ftract_crossmap.json.  Run: python experiments/ccep_ftract_crossmap.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa: E402
import ccep_loso as L  # noqa: E402

FT = Path("REDACTED/data/external/ftract")
FSDIR = FT / "scripts/python/mne_plot_data/MNE-sample-data/subjects"
PARC = "Lausanne2008-250"
ANNOT = "Lausanne_250"
REL_MIN = L.REL_MIN


def build_mapper():
    """Return a function contact_xyz[N,3] -> list of F-TRACT parcel names ('{hemi}.{name}' or None).

    Uses mne.read_labels_from_annot (correctly resolves the annotation color table) to build a
    per-vertex parcel-name lookup for each hemisphere, then assigns each contact to its nearest
    fsaverage pial vertex (contacts sit ON the surface: verified median distance 0.00 mm).
    """
    import mne
    from scipy.spatial import cKDTree
    lh_v, _ = mne.read_surface(str(FSDIR / "fsaverage/surf/lh.pial"))
    rh_v, _ = mne.read_surface(str(FSDIR / "fsaverage/surf/rh.pial"))
    labels = mne.read_labels_from_annot("fsaverage", ANNOT, "both", subjects_dir=str(FSDIR), verbose=False)
    vlab = {"lh": np.full(len(lh_v), None, dtype=object), "rh": np.full(len(rh_v), None, dtype=object)}
    for lab in labels:
        hemi = lab.hemi  # 'lh'/'rh'
        base = lab.name.rsplit("-", 1)[0]  # 'cuneus_1-lh' -> 'cuneus_1'
        if base in ("unknown", "corpuscallosum", "???"):
            continue
        vlab[hemi][lab.vertices] = f"{hemi}.{base}"
    tl, tr = cKDTree(lh_v), cKDTree(rh_v)

    def mapper(X):
        dl, il = tl.query(X); dr, ir = tr.query(X)
        out = []
        for k in range(len(X)):
            if dl[k] <= dr[k]:
                out.append(vlab["lh"][il[k]])
            else:
                out.append(vlab["rh"][ir[k]])
        return out
    return mapper


def load_ft(feature):
    p = FT / "ages_15_100" / "sr_8.40" / "seg_None_None" / "pl_200" / PARC / "export" / feature / f"{feature}.csv"
    rows, header = [], None
    with open(p) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            if header is None and parts[0].strip() == "stimulated parcels":
                header = [c.strip() for c in parts[1:]]; continue
            if header is not None:
                rows.append(parts)
    idx = {p: k for k, p in enumerate(header)}
    M = np.full((len(header), len(header)), np.nan)
    for r in rows:
        sp = r[0].strip()
        if sp in idx:
            for j, v in enumerate(r[1:len(header) + 1]):
                try:
                    M[idx[sp], j] = float(v)
                except ValueError:
                    pass
    return header, idx, M


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return np.nan, int(ok.sum())
    ra = np.argsort(np.argsort(a[ok])).astype(float); rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return (float((ra @ rb) / den) if den > 1e-12 else np.nan), int(ok.sum())


def main():
    header, pidx, ft_amp = load_ft("amplitude")
    _, _, ft_prob = load_ft("probability")
    n_p = len(header)
    mapper = build_mapper()

    # load our subjects, map contacts to F-TRACT parcels
    subs = []
    map_dists = []
    for p in sorted((ROOT / "data" / "processed" / "ds004080").glob("sub-*.npz")):
        cs = CCEPSubject.load(str(p))
        keep = np.arange(len(cs.sites))[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
        if len(keep) < 6:
            continue
        parc = mapper(cs.contact_xyz)
        pcol = np.array([pidx.get(pp, -1) if pp else -1 for pp in parc])
        subs.append({"cs": cs, "keep": keep, "pcol": pcol})
    print(f"mapped subjects: {len(subs)} (parcellation {PARC}, {n_p} parcels)")

    # ---- (A) build OUR group operator in F-TRACT parcel space ----
    Gs = np.zeros((n_p, n_p)); Gc = np.zeros((n_p, n_p))
    for d in subs:
        cs, keep, pcol = d["cs"], d["keep"], d["pcol"]
        R = cs.responses[keep].astype(float)
        mu, sd = np.nanmean(R), np.nanstd(R)
        Rz = (R - mu) / (sd + 1e-9)
        for i, s in enumerate(keep):
            sp = [pcol[a] for a in cs.stim_idx[s] if a >= 0 and pcol[a] >= 0]
            row, ok = Rz[i], np.isfinite(Rz[i])
            for c in range(len(pcol)):
                if not ok[c] or pcol[c] < 0:
                    continue
                for p_ in sp:
                    Gs[p_, pcol[c]] += row[c]; Gc[p_, pcol[c]] += 1
    Gour = np.where(Gc > 0, Gs / np.maximum(Gc, 1), np.nan)
    off = ~np.eye(n_p, dtype=bool)
    r_amp, n1 = _spearman(Gour[off], ft_amp[off])
    r_prob, n2 = _spearman(Gour[off], ft_prob[off])
    # directionality preserved? correlation of our forward vs F-TRACT forward, vs our-forward-to-FT-transpose
    r_fwd, _ = _spearman(Gour[off], ft_amp[off])
    r_rev, _ = _spearman(Gour[off], ft_amp.T[off])
    print(f"\n=== (A) DIRECT operator validation (our 74-subj operator vs F-TRACT 780-pt) ===")
    print(f"  our operator vs F-TRACT amplitude   : Spearman rho={r_amp:+.3f} (n={n1} shared parcel pairs)")
    print(f"  our operator vs F-TRACT probability : rho={r_prob:+.3f} (n={n2})")
    print(f"  direction check: forward rho={r_fwd:+.3f}  vs transpose rho={r_rev:+.3f}"
          + ("  <-- direction preserved" if r_fwd > r_rev else "  (transpose fits better)"))

    # ---- (B) F-TRACT as a cold-start prior ----
    print(f"\n=== (B) F-TRACT 780-pt as a cold-start prior vs geometry (external, no leakage) ===")
    rows = {m: [] for m in ["distance", "ftract_prior", "combo"]}
    for d in subs:
        cs, keep, pcol = d["cs"], d["keep"], d["pcol"]
        for test_i in keep:
            tgt = cs.responses[test_i]
            mask = L._valid_mask(cs, test_i, [t for t in keep if t != test_i])
            dist = L.predict_distance(cs, test_i, 15.0)
            sp = [pcol[a] for a in cs.stim_idx[test_i] if a >= 0 and pcol[a] >= 0]
            pred = np.full(len(pcol), np.nan)
            if sp:
                for c in range(len(pcol)):
                    if pcol[c] >= 0:
                        vals = [ft_amp[s, pcol[c]] for s in sp if np.isfinite(ft_amp[s, pcol[c]])]
                        if vals:
                            pred[c] = np.mean(vals)
            rows["distance"].append(L.topo_r(dist, tgt, mask))
            rows["ftract_prior"].append(L.topo_r(pred, tgt, mask))
            combo = np.nan_to_num(L._z(dist, mask)) + np.nan_to_num(L._z(pred, mask))
            rows["combo"].append(L.topo_r(combo, tgt, mask))
    da = np.array(rows["distance"]); fa = np.array(rows["ftract_prior"]); ca = np.array(rows["combo"])
    ok = np.isfinite(da) & np.isfinite(fa) & np.isfinite(ca)
    for m, v in [("distance", da), ("ftract_prior", fa), ("combo", ca)]:
        vv = v[np.isfinite(v)]; mean, lo, hi = bootstrap_ci(vv.tolist())
        print(f"  {m:14s} {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  (n={len(vv)} folds)")
    d_ft = float(np.mean(fa[ok]) - np.mean(da[ok]))
    d_combo = float(np.mean(ca[ok]) - np.mean(da[ok]))
    p_ft = paired_permutation_test(fa[ok].tolist(), da[ok].tolist())
    p_combo = paired_permutation_test(ca[ok].tolist(), da[ok].tolist())
    print(f"  F-TRACT prior vs distance : delta={d_ft:+.3f}  p={p_ft:.3g}"
          + ("  <-- population prior beats geometry" if d_ft > 0 and p_ft < 0.05 else ""))
    print(f"  combo(F-TRACT+distance) vs distance: delta={d_combo:+.3f}  p={p_combo:.3g}"
          + ("  <-- adds over geometry" if d_combo > 0 and p_combo < 0.05 else ""))

    out = {"parcellation": PARC, "n_parcels": n_p, "n_subjects": len(subs),
           "operator_validation": {"rho_amplitude": r_amp, "rho_probability": r_prob,
                                   "rho_forward": r_fwd, "rho_transpose": r_rev, "n_pairs": n1},
           "coldstart_prior": {"distance": float(np.mean(da[ok])), "ftract_prior": float(np.mean(fa[ok])),
                               "combo": float(np.mean(ca[ok])), "ftract_vs_dist_delta": d_ft,
                               "ftract_vs_dist_p": float(p_ft), "combo_vs_dist_delta": d_combo,
                               "combo_vs_dist_p": float(p_combo), "n_folds": int(ok.sum())}}
    (ROOT / "reports" / "ftract_crossmap.json").write_text(json.dumps(out, indent=2))
    print("\nsaved reports/ftract_crossmap.json")


if __name__ == "__main__":
    main()
