"""Spatial-block cross-validation + electrode-overlap audit — rules out a trivial reuse/leakage win.

Reviewer concern: CCEP stimulation sites are bipolar electrode PAIRS and contacts may recur across
sites, so leave-one-site-out could reward spatial/electrode reuse rather than genuine prediction
(the model copying an immediately adjacent measured site).

Two safeguards, beyond the existing controls (held-out stim pair excluded from scoring; held-out
site's own response row never enters the operator):

  (1) ELECTRODE-OVERLAP AUDIT. For every held-out site, count how many of its bipolar stim contacts
      are ALSO stim contacts in some training site, and the distance to the nearest training stim
      site. This quantifies how much reuse exists at all.

  (2) SPATIAL-BLOCK CV. Re-run leave-one-site-out but EXCLUDE every training site whose stim
      coordinate lies within a buffer radius B of the held-out site. With B>0 the model cannot lean
      on an adjacent measured site; if the gains survive, they are not a spatial-reuse artifact.
      B=0 reproduces the standard protocol.

Models: within_mean, distance, combo (headline), operator_v2. All hyperparameters by nested inner
LOO on the (buffered) training sites only.

Run:  python experiments/ccep_spatial_cv.py [--fast] [--buffers 0,10,15,20] [--no-op2]
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
import ccep_loso as L  # noqa: E402
import ccep_operator_v2 as O2  # noqa: E402

REL_MIN = L.REL_MIN


def overlap_stats(cs, keep):
    """For each held-out site: shared physical stim contacts with training sites, min stim-distance."""
    shared, mindist = [], []
    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        test_contacts = {int(e) for e in cs.stim_idx[test_i] if e >= 0}
        train_contacts = set()
        for t in train_idx:
            for e in cs.stim_idx[t]:
                if e >= 0:
                    train_contacts.add(int(e))
        shared.append(len(test_contacts & train_contacts))
        d = np.linalg.norm(cs.stim_xyz[train_idx] - cs.stim_xyz[test_i][None], axis=1)
        mindist.append(float(np.nanmin(d)) if len(d) else np.nan)
    return shared, mindist


def eval_subject(cs, buffer, do_op2=True):
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
    if len(keep) < 6:
        return None
    models = ["within_mean", "distance", "combo"] + (["operator_v2"] if do_op2 else [])
    fold = {m: [] for m in models}
    nfolds = 0
    for test_i in keep:
        # training sites: reliable, not the held-out site, and OUTSIDE the buffer radius
        d_all = np.linalg.norm(cs.stim_xyz[keep] - cs.stim_xyz[test_i][None], axis=1)
        train_idx = [int(t) for t, dd in zip(keep, d_all) if t != test_i and dd >= buffer]
        if len(train_idx) < 4:
            continue
        nfolds += 1
        tgt = cs.responses[test_i]; mask = L._valid_mask(cs, test_i, train_idx)

        R = cs.responses[train_idx]
        wmean = np.nansum(R, axis=0) / (np.sum(np.isfinite(R), axis=0) + 1e-9)
        fold["within_mean"].append(L.topo_r(wmean, tgt, mask))

        sig = max(L.SIGMA_GRID, key=lambda s: L._score_param(
            cs, train_idx, lambda j, tr, s=s: L.predict_distance(cs, j, s)))
        fold["distance"].append(L.topo_r(L.predict_distance(cs, test_i, sig), tgt, mask))

        tau = max(L.TAU_GRID, key=lambda tt: L._score_param(
            cs, train_idx, lambda j, tr, tt=tt: L.predict_stim_knn(cs, j, tr, tt)))
        beta = max(L.BETA_GRID, key=lambda b: L._score_param(
            cs, train_idx,
            lambda j, tr, b=b: L.predict_combo(cs, j, tr, sig, tau, b, L._valid_mask(cs, j, tr))))
        fold["combo"].append(L.topo_r(L.predict_combo(cs, test_i, train_idx, sig, tau, beta, mask), tgt, mask))

        if do_op2:
            sg, al, stp, md = O2._best_params(cs, train_idx)
            P = O2._build_operator(cs, train_idx, md)
            op2 = O2.predict_operator_v2(cs, test_i, train_idx, sg, al, stp, md, P=P)
            fold["operator_v2"].append(L.topo_r(op2, tgt, mask))
    return {m: float(np.nanmean(v)) for m, v in fold.items()}, nfolds, len(keep)


def main(argv):
    fast = "--fast" in argv
    do_op2 = "--no-op2" not in argv
    buffers = [0, 10, 15, 20]
    if "--buffers" in argv:
        buffers = [float(x) for x in argv[argv.index("--buffers") + 1].split(",")]
    caches = L.all_caches()
    if fast:
        caches = [(d, p) for d, p in caches if d in ("ds004774", "ds004696")]
    css = [CCEPSubject.load(str(c)) for _, c in caches]

    # ---- (1) electrode-overlap audit (buffer-independent) ----
    all_shared, all_mindist = [], []
    for cs in css:
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
        if len(keep) < 6:
            continue
        sh, md = overlap_stats(cs, keep)
        all_shared += sh; all_mindist += md
    all_shared = np.array(all_shared); all_mindist = np.array(all_mindist)
    print("=== electrode-overlap audit (per held-out site) ===")
    print(f"  held-out sites audited: {len(all_shared)}")
    print(f"  share >=1 physical stim contact with a training site: "
          f"{100*np.mean(all_shared>0):.1f}%  (mean shared contacts={all_shared.mean():.2f})")
    md = all_mindist[np.isfinite(all_mindist)]
    print(f"  distance to NEAREST training stim site (mm): median={np.median(md):.1f}, "
          f"IQR [{np.percentile(md,25):.1f}, {np.percentile(md,75):.1f}], "
          f"<10mm: {100*np.mean(md<10):.0f}%, <20mm: {100*np.mean(md<20):.0f}%")

    # ---- (2) spatial-block CV sweep ----
    models = ["within_mean", "distance", "combo"] + (["operator_v2"] if do_op2 else [])
    print(f"\n=== spatial-block CV: subject-level mean r by buffer radius (n={len(css)} subjects) ===")
    header = f"{'buffer(mm)':>10s} " + " ".join(f"{m:>12s}" for m in models) + f"  {'combo-wm Δ(p,wins)':>22s}"
    print(header)
    results = {}
    for B in buffers:
        rows = {m: [] for m in models}
        folds_kept, folds_tot = 0, 0
        for cs in css:
            r = eval_subject(cs, B, do_op2)
            if r is None:
                continue
            sc, nk, ntot = r
            folds_kept += nk; folds_tot += ntot
            for m in models:
                rows[m].append(sc[m])
        results[B] = rows
        means = {m: np.mean(rows[m]) for m in models}
        wm, cb = rows["within_mean"], rows["combo"]
        p = paired_permutation_test(cb, wm); wins = sum(1 for a, b in zip(cb, wm) if a > b)
        line = f"{B:>10.0f} " + " ".join(f"{means[m]:>12.3f}" for m in models)
        line += f"   Δ={np.mean(cb)-np.mean(wm):+.3f} p={p:.2g} {wins}/{len(wm)}"
        line += f"  [folds kept {folds_kept}/{folds_tot}]"
        print(line)

    # operator_v2 vs distance survival across buffers
    if do_op2:
        print("\n=== operator_v2 vs distance by buffer (does the operator still beat locality?) ===")
        for B in buffers:
            op, di = results[B]["operator_v2"], results[B]["distance"]
            p = paired_permutation_test(op, di); w = sum(1 for a, b in zip(op, di) if a > b)
            print(f"  buffer {B:>4.0f} mm:  operator_v2 {np.mean(op):+.3f} vs distance {np.mean(di):+.3f}  "
                  f"Δ={np.mean(op)-np.mean(di):+.3f}  p={p:.2g}  ({w}/{len(op)})")


if __name__ == "__main__":
    main(sys.argv[1:])
