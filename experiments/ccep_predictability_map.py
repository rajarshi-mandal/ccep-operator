"""EXTENSION — WHERE is the stimulation response predictable? (clinical/anatomical relevance)

For a targeting tool, clinicians care not just about an average r but about *where* the prediction
is trustworthy. We pool per-contact predictions (combo) across all held-out folds and subjects and
compute predicted-vs-measured accuracy stratified by:
  - distance from the stimulation site (near / mid / far),
  - same vs. contralateral hemisphere,
  - homotopic proximity (distance from the mirror-image of the stim site) — CCEP networks are known
    to be strongly homotopic.
Output: reports/predictability_map.json  (strata accuracy + counts for the jumbo figure)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa
import ccep_loso as L  # noqa


def _pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return np.nan, int(ok.sum())
    a = x[ok] - x[ok].mean(); b = y[ok] - y[ok].mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return (float((a @ b) / den) if den > 1e-12 else np.nan), int(ok.sum())


def main():
    caches = L.all_caches()
    P, M, DST, SAME, HOMO = [], [], [], [], []   # pooled per-contact records
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= L.REL_MIN)]
        if len(keep) < 6:
            continue
        xyz = cs.contact_xyz
        for ti in keep:
            tr = [t for t in keep if t != ti]
            mask = L._valid_mask(cs, ti, tr)
            sig = max(L.SIGMA_GRID, key=lambda s: L._score_param(cs, tr, lambda j, t, s=s: L.predict_distance(cs, j, s)))
            tau = max(L.TAU_GRID, key=lambda tt: L._score_param(cs, tr, lambda j, t, tt=tt: L.predict_stim_knn(cs, j, t, tt)))
            beta = max(L.BETA_GRID, key=lambda b: L._score_param(cs, tr, lambda j, t, b=b: L.predict_combo(cs, j, t, sig, tau, b, L._valid_mask(cs, j, t))))
            pred = L.predict_combo(cs, ti, tr, sig, tau, beta, mask)
            meas = cs.responses[ti]
            stim = cs.stim_xyz[ti]
            d = np.linalg.norm(xyz - stim[None], axis=1)
            same = (np.sign(xyz[:, 0]) == np.sign(stim[0]))
            homo = np.linalg.norm(xyz - (stim * np.array([-1., 1., 1.]))[None], axis=1)
            for ci in range(len(cs.contacts)):
                if mask[ci] and np.isfinite(pred[ci]) and np.isfinite(meas[ci]):
                    P.append(pred[ci]); M.append(meas[ci]); DST.append(d[ci]); SAME.append(bool(same[ci])); HOMO.append(homo[ci])
    P, M, DST, HOMO = map(np.array, (P, M, DST, HOMO)); SAME = np.array(SAME)
    print(f"pooled {len(P)} contact-predictions")

    out = {"n_records": int(len(P))}
    # distance x hemisphere strata
    bins = [(0, 20, "near"), (20, 40, "mid"), (40, 999, "far")]
    grid = {}
    print(f"\n{'stratum':22s} {'r(pred,meas)':>12} {'n':>7}")
    for lo, hi, name in bins:
        for hemi, hm in [("same-hemi", SAME), ("cross-hemi", ~SAME)]:
            sel = (DST >= lo) & (DST < hi) & hm
            r, n = _pearson(P[sel], M[sel])
            grid[f"{name}/{hemi}"] = {"r": r, "n": n}
            print(f"  {name+'/'+hemi:20s} {r:12.3f} {n:7d}")
    out["distance_hemi"] = grid
    # homotopic effect: within the far/cross-hemi field, are contacts NEAR the homotopic point more predictable?
    farx = (DST >= 40)
    near_homo = farx & (HOMO < 25)
    far_homo = farx & (HOMO >= 25)
    rh, nh = _pearson(P[near_homo], M[near_homo]); rf, nf = _pearson(P[far_homo], M[far_homo])
    out["homotopic"] = {"near_homotopic": {"r": rh, "n": nh}, "far_from_homotopic": {"r": rf, "n": nf}}
    print(f"\nHomotopic (far field): near-homotopic r={rh:.3f} (n={nh}) vs far-from-homotopic r={rf:.3f} (n={nf})")
    print("  -> if near-homotopic is higher, predictable structure survives at the contralateral mirror.")
    (ROOT / "reports" / "predictability_map.json").write_text(json.dumps(out, indent=2))
    print("saved reports/predictability_map.json")


if __name__ == "__main__":
    main()
