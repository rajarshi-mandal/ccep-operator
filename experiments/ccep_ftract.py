"""TIER-2 EXTENSION (T2.A) — External validation of our headline CCEP findings against the
F-TRACT 780-patient atlas (David/Trebaul/Lemarechal; f-tract.eu, Zenodo 7015415).

Our results rest on n=93 from a single lineage of OpenNeuro datasets. F-TRACT is an independent,
multi-centre CCEP atlas from **780 patients**, age-stratified (0-15 / 15-100), providing parcel x
parcel connectivity probability, amplitude, onset/peak latency, distance, velocity and axonal/
synaptic delays. Data are populated in the Lausanne2008 (here: 250-parcel) and HCP-MMP1 schemes.

Rather than cross-map our fsaverage electrodes into F-TRACT's parcels (surface-projection, lossy),
we replicate our KEY CLAIMS *within F-TRACT's own 780-patient matrices* — a genuine external
replication that breaks the n=93 ceiling for group-level claims:

  (1) CONDUCTION LAW — onset latency grows with distance; recover the apparent conduction speed
      (mm/ms) and compare to our ~3.0 mm/ms; report F-TRACT's own velocity estimate; young vs old.
  (2) AMPLITUDE LOCALITY — CCEP amplitude decays with distance (the basis of the distance baseline).
  (3) RECIPROCAL DOMINANCE + DIRECTIONALITY — connectivity is reciprocal-dominant (corr(M, Mᵀ)),
      yet latency is directionally ASYMMETRIC (validates our 33.8 ms reciprocal-latency asymmetry).
  (4) DEVELOPMENT — do conduction speed / reciprocity differ young vs old (our maturation finding)?

Output: reports/ftract.json.  Run: python experiments/ccep_ftract.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))

FT = Path("REDACTED/data/external/ftract")
PARC = "Lausanne2008-250"


def ft_path(feature, age):
    return FT / age / "sr_8.40" / "seg_None_None" / "pl_200" / PARC / "export" / feature / f"{feature}.csv"


def load_ft(feature, age="ages_15_100"):
    """Parse an F-TRACT matrix CSV -> M[stim, rec] aligned to the header parcel order."""
    p = ft_path(feature, age)
    if not p.exists():
        return None
    rows, header = [], None
    with open(p) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            if header is None and parts[0].strip() == "stimulated parcels":
                header = [c.strip() for c in parts[1:]]
                continue
            if header is not None:
                rows.append(parts)
    if header is None:
        return None
    idx = {p: k for k, p in enumerate(header)}
    M = np.full((len(header), len(header)), np.nan)
    for r in rows:
        sp = r[0].strip()
        if sp not in idx:
            continue
        i = idx[sp]
        for j, v in enumerate(r[1:len(header) + 1]):
            try:
                M[i, j] = float(v)
            except ValueError:
                pass
    return M


def _corr(a, b, spearman=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 20:
        return np.nan, int(ok.sum())
    a, b = a[ok], b[ok]
    if spearman:
        a = np.argsort(np.argsort(a)).astype(float)
        b = np.argsort(np.argsort(b)).astype(float)
    a, b = a - a.mean(), b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return (float((a @ b) / den) if den > 1e-12 else np.nan), int(ok.sum())


def _boot_ci_corr(a, b, spearman=False, nboot=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(nboot):
        idx = rng.integers(0, len(a), len(a))
        r, _ = _corr(a[idx], b[idx], spearman)
        if np.isfinite(r):
            rs.append(r)
    return (np.percentile(rs, 2.5), np.percentile(rs, 97.5)) if rs else (np.nan, np.nan)


def main():
    out = {"parcellation": PARC, "n_patients_atlas": 780}

    # ---- (1) CONDUCTION LAW ----
    print(f"=== (1) CONDUCTION LAW in F-TRACT ({PARC}, 780 patients) ===")
    cond = {}
    for age in ["ages_15_100", "ages_0_15"]:
        lat = load_ft("onset_latency", age); dist = load_ft("euclidian_distance", age)
        vel = load_ft("euclidian_distance_axonal_velocity", age)
        if lat is None or dist is None:
            continue
        r, n = _corr(dist.ravel(), lat.ravel(), spearman=True)
        m = np.isfinite(lat) & np.isfinite(dist) & (dist > 0) & (lat > 0)
        speed = float(np.polyfit(lat[m], dist[m], 1)[0]) if m.sum() > 20 else np.nan  # mm per ms
        vmed = float(np.nanmedian(vel)) if vel is not None and np.isfinite(vel).any() else np.nan
        lo, hi = _boot_ci_corr(dist.ravel(), lat.ravel(), spearman=True)
        cond[age] = {"rho_dist_latency": r, "n_pairs": n, "speed_mm_per_ms": speed,
                     "ftract_median_velocity_mm_per_ms": vmed, "ci": [lo, hi]}
        print(f"  {age:12s}: latency~distance rho={r:+.3f} [{lo:+.3f},{hi:+.3f}] (n={n}); "
              f"apparent speed {speed:.2f} mm/ms; F-TRACT median velocity {vmed:.2f} mm/ms")
    out["conduction"] = cond
    print("  (our within-subject estimate was ~3.0 mm/ms)")

    # ---- (2) AMPLITUDE LOCALITY ----
    print(f"\n=== (2) AMPLITUDE decays with distance (locality basis) ===")
    amp = load_ft("amplitude"); dist = load_ft("euclidian_distance")
    r_ad, n_ad = _corr(dist.ravel(), amp.ravel(), spearman=True)
    print(f"  amplitude~distance Spearman rho={r_ad:+.3f} (n={n_ad})  (expect negative: closer=larger)")
    out["amplitude_locality"] = {"rho_amp_distance": r_ad, "n": n_ad}

    # ---- (3) RECIPROCAL DOMINANCE + DIRECTIONALITY ----
    print(f"\n=== (3) RECIPROCAL DOMINANCE + directional latency asymmetry ===")
    off = ~np.eye(amp.shape[0], dtype=bool)
    r_sym, n_sym = _corr(amp[off], amp.T[off])
    print(f"  amplitude reciprocity corr(M, Mᵀ)={r_sym:+.3f} (n={n_sym})  -> reciprocal-dominant")
    lat = load_ft("onset_latency")
    # reciprocal latency asymmetry: |lat[a,b] - lat[b,a]| over pairs where both exist
    A = lat.copy()
    pair_ok = np.isfinite(A) & np.isfinite(A.T) & off
    iu = np.triu(pair_ok, 1)
    asym = np.abs(A[iu] - A.T[iu])
    med_asym = float(np.nanmedian(asym)) if asym.size else np.nan
    print(f"  reciprocal latency asymmetry: median |lat(a→b)-lat(b→a)| = {med_asym:.1f} ms "
          f"(n={asym.size} recip pairs)  (our finding: 33.8 ms)")
    out["directionality"] = {"amplitude_reciprocity": r_sym, "latency_asymmetry_ms": med_asym,
                             "n_recip_pairs": int(asym.size)}

    # ---- (4) DEVELOPMENT ----
    print(f"\n=== (4) DEVELOPMENT (young vs old) ===")
    dev = {}
    for age in ["ages_0_15", "ages_15_100"]:
        amp_a = load_ft("amplitude", age); lat_a = load_ft("onset_latency", age)
        vel_a = load_ft("euclidian_distance_axonal_velocity", age)
        if amp_a is None:
            continue
        offa = ~np.eye(amp_a.shape[0], dtype=bool)
        rec, _ = _corr(amp_a[offa], amp_a.T[offa])
        vmed = float(np.nanmedian(vel_a)) if vel_a is not None and np.isfinite(vel_a).any() else np.nan
        lmed = float(np.nanmedian(lat_a[np.isfinite(lat_a)])) if lat_a is not None else np.nan
        dev[age] = {"amplitude_reciprocity": rec, "median_velocity": vmed, "median_onset_latency_ms": lmed}
        print(f"  {age:12s}: reciprocity={rec:+.3f}  median velocity={vmed:.2f} mm/ms  "
              f"median onset latency={lmed:.1f} ms")
    out["development"] = dev

    (ROOT / "reports" / "ftract.json").write_text(json.dumps(out, indent=2))
    print("\nsaved reports/ftract.json")


if __name__ == "__main__":
    main()
