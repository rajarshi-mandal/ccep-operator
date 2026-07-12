"""TIER-2 EXTENSION (T2.B) — Is CCEP effective connectivity grounded in white-matter STRUCTURE?

We compare two independent population references in the SAME Glasser-360 (HCP-MMP1) parcellation —
no fragile electrode-to-parcel crosswalk needed:
  - EFFECTIVE (CCEP): F-TRACT 780-patient connectivity probability / amplitude (Glasser).
  - STRUCTURAL (DWI): ENIGMA/HCP group tractography structural connectome (Glasser).

Both decay with distance, so the decisive test is a DISTANCE-CONTROLLED partial correlation: does CCEP
effective connectivity follow structural connectivity BEYOND what geometry explains? A positive
partial correlation means cortico-cortical evoked propagation is structurally constrained — grounding
the operator's effective connectivity in anatomy (the transitive link: our operator replicates F-TRACT
on conduction/locality/reciprocity (T2.A), and F-TRACT here tracks DWI structure).

Output: reports/struct.json.  Run: python experiments/ccep_struct.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FT = Path("REDACTED/data/external/ftract")
ST = Path("REDACTED/data/external/struct")


def load_ft_hcp(feature, age="ages_15_100"):
    p = FT / age / "sr_8.40" / "seg_None_None" / "pl_200" / "MNI-HCP-MMP1" / "export" / feature / f"{feature}.csv"
    if not p.exists():
        return None, None
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
    return header, M


def load_enigma_sc():
    labels = open(ST / "strucLabels_ctx_glasser_360.csv").read().strip().split(",")
    M = np.loadtxt(ST / "strucMatrix_ctx_glasser_360.csv", delimiter=",")
    return labels, M


def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return np.nan, int(ok.sum())
    ra, rb = _rank(a[ok]), _rank(b[ok])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return (float((ra @ rb) / den) if den > 1e-12 else np.nan), int(ok.sum())


def _partial_spearman(a, b, c):
    """Spearman partial correlation of a,b controlling for c (rank-residualised)."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if ok.sum() < 20:
        return np.nan, int(ok.sum())
    ra, rb, rc = _rank(a[ok]), _rank(b[ok]), _rank(c[ok])
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        beta = np.linalg.lstsq(x1, y, rcond=None)[0]
        return y - x1 @ beta
    er_a, er_b = resid(ra, rc), resid(rb, rc)
    den = np.linalg.norm(er_a - er_a.mean()) * np.linalg.norm(er_b - er_b.mean())
    return (float(((er_a - er_a.mean()) @ (er_b - er_b.mean())) / den) if den > 1e-12 else np.nan), int(ok.sum())


def main():
    en_labels, SC = load_enigma_sc()
    print(f"ENIGMA structural (HCP DWI): {SC.shape}, {len(en_labels)} Glasser parcels")
    out = {"parcellation": "glasser_360", "n_struct_source": "ENIGMA/HCP", "n_ccep_source": "F-TRACT 780pt"}

    for age in ["ages_15_100", "ages_0_15"]:
        hdr, prob = load_ft_hcp("probability", age)
        _, amp = load_ft_hcp("amplitude", age)
        _, dist = load_ft_hcp("euclidian_distance", age)
        if hdr is None:
            continue
        # align F-TRACT parcel order to ENIGMA label order
        pos = {p: i for i, p in enumerate(hdr)}
        order = [pos.get(l) for l in en_labels]
        keep = [i for i, o in enumerate(order) if o is not None]
        oidx = [order[i] for i in keep]
        def reidx(M):
            return M[np.ix_(oidx, oidx)]
        P = reidx(prob); A = reidx(amp); D = reidx(dist); S = SC[np.ix_(keep, keep)]
        # symmetrise effective (DWI structural is undirected); log structural (heavy-tailed)
        Psym = np.nanmean(np.dstack([P, P.T]), axis=2)
        Asym = np.nanmean(np.dstack([A, A.T]), axis=2)
        Slog = np.where(S > 0, np.log10(S + 1), np.nan)
        off = ~np.eye(S.shape[0], dtype=bool)

        r_prob, n1 = _spearman(Psym[off], Slog[off])
        r_amp, n2 = _spearman(Asym[off], Slog[off])
        pr_prob, np1 = _partial_spearman(Psym[off], Slog[off], D[off])
        pr_amp, np2 = _partial_spearman(Asym[off], Slog[off], D[off])
        # reference: how much of each is just distance
        rd_eff, _ = _spearman(Psym[off], D[off])
        rd_str, _ = _spearman(Slog[off], D[off])

        print(f"\n=== {age} (n={n1} parcel pairs) ===")
        print(f"  CCEP prob ~ structural (raw)     : rho={r_prob:+.3f}")
        print(f"  CCEP amp  ~ structural (raw)     : rho={r_amp:+.3f}")
        print(f"  CCEP prob ~ structural | distance: rho={pr_prob:+.3f}  <-- structural beyond geometry")
        print(f"  CCEP amp  ~ structural | distance: rho={pr_amp:+.3f}")
        print(f"  (effective~distance {rd_eff:+.3f}; structural~distance {rd_str:+.3f})")
        out[age] = {"n_pairs": n1,
                    "rho_prob_struct": r_prob, "rho_amp_struct": r_amp,
                    "partial_prob_struct_given_dist": pr_prob, "partial_amp_struct_given_dist": pr_amp,
                    "eff_vs_dist": rd_eff, "struct_vs_dist": rd_str}

    (ROOT / "reports" / "struct.json").write_text(json.dumps(out, indent=2))
    print("\nsaved reports/struct.json")


if __name__ == "__main__":
    main()
