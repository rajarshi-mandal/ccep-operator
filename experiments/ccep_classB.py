"""CLASS B — does robust target estimation lift the (far-field) noise floor?

The diagnostic showed the far-field ceiling is only ~0.49 (low-SNR distant responses), capping
overall r. Robust trial handling (drop gross-artifact trials by RMS, median instead of mean
average) should denoise the target -> raise split-half reliability AND the far-field ceiling, which
lifts every model's achievable r for free. Compares standard vs robust caches (data/processed/
ds*  vs  ds*_robust). Reports reliability and distance-stratified half-split ceiling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test  # noqa: E402
from ccep_loso import _valid_mask, REL_MIN  # noqa: E402
from ccep_diagnostic import binned_r, BIN_NAMES  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DATASETS = ["ds004774", "ds004696"]


def subj_stats(cs):
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
    rel = float(np.nanmean(np.clip(cs.reliability[keep], 0, 1)))
    ceil = {b: [] for b in BIN_NAMES}
    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        mask = _valid_mask(cs, test_i, train_idx)
        dist = np.linalg.norm(cs.contact_xyz - cs.stim_xyz[test_i][None], axis=1)
        cv = binned_r(cs.responses_h1[test_i], cs.responses_h2[test_i], mask, dist)
        for k, name in enumerate(BIN_NAMES):
            if np.isfinite(cv[k]):
                ceil[name].append(cv[k])
    return rel, {b: (np.mean(v) if v else np.nan) for b, v in ceil.items()}, len(keep)


def main():
    rows = []
    for ds in DATASETS:
        for p in sorted((PROCESSED / ds).glob("sub-*.npz")):
            rp = PROCESSED / (ds + "_robust") / p.name
            if not rp.exists():
                print(f"missing robust cache for {p.name} (run build_ccep.py --robust)"); return
            std = CCEPSubject.load(str(p)); rob = CCEPSubject.load(str(rp))
            if std.responses_h1 is None or rob.responses_h1 is None:
                print("missing half-split data; rebuild needed"); return
            rs, rc, n = subj_stats(std); os_, oc, _ = subj_stats(rob)
            rows.append((f"{ds[-4:]}/{std.subject}", rs, os_, rc, oc, n))

    print("Class B: robust vs standard target estimation\n")
    print(f"{'subject':14s} {'rel_std':>8s} {'rel_rob':>8s}   "
          f"{'far_std':>8s} {'far_rob':>8s} {'far_Δ':>7s}")
    rstd, rrob, fstd, frob = [], [], [], []
    for tag, rs, os_, rc, oc, n in rows:
        fs_, fo = rc["far(40+)"], oc["far(40+)"]
        rstd.append(rs); rrob.append(os_); fstd.append(fs_); frob.append(fo)
        print(f"{tag:14s} {rs:8.3f} {os_:8.3f}   {fs_:8.3f} {fo:8.3f} {fo-fs_:+7.3f}")

    def summ(name, a, b):
        a, b = np.array(a), np.array(b)
        p = paired_permutation_test(b.tolist(), a.tolist())
        print(f"  {name:18s} std {np.nanmean(a):.3f} -> robust {np.nanmean(b):.3f}  "
              f"Δ={np.nanmean(b)-np.nanmean(a):+.3f}  p={p:.3g}  ({int((b>a).sum())}/{len(a)} up)")

    print("\n=== subject-level (paired) ===")
    summ("reliability", rstd, rrob)
    summ("far-field ceiling", fstd, frob)


if __name__ == "__main__":
    main()
