"""EXTENSION — N1 latency: conduction law + is the timing of the response predictable?

Two questions the amplitude model can't answer:
  (A) Conduction: does N1 latency grow with distance from the stim site (finite propagation speed)?
      Pool responsive contacts; regress latency ~ distance -> apparent conduction speed (mm/ms).
  (B) Predictability: can a held-out site's *latency* topography be predicted from the subject's
      other sites? (within-mean-latency baseline vs distance vs stim-kNN vs combo). Restricted to
      RESPONSIVE contacts (amplitude above the site's 60th pct) where latency is meaningful.
Pairs with the developmental finding (ccepAge = transmission speed). Output: reports/latency.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa
import ccep_loso as L  # noqa

SIG = [10, 20, 30, 50, 80]      # mm, distance bandwidth for latency locality
TAU = [10, 20, 40, 70, 1e9]
BETA = [0.0, 0.5, 1.0, 1.5]


def _topo(pred, meas, mask):
    ok = mask & np.isfinite(pred) & np.isfinite(meas)
    if ok.sum() < 5: return np.nan
    p = pred[ok] - pred[ok].mean(); m = meas[ok] - meas[ok].mean()
    d = np.linalg.norm(p) * np.linalg.norm(m)
    return float((p @ m) / d) if d > 1e-12 else np.nan


def _knn_lat(cs, ti, train, tau):
    d = np.linalg.norm(cs.stim_xyz[train] - cs.stim_xyz[ti][None], axis=1)
    w = np.exp(-(d ** 2) / (2 * tau ** 2))
    if w.sum() < 1e-9: w = np.ones_like(w)
    Lm = cs.latency[train]
    return np.nansum(w[:, None] * Lm, axis=0) / (np.nansum(w[:, None] * np.isfinite(Lm), axis=0) + 1e-9)


def main():
    caches = [(d, c) for d, c in L.all_caches()]
    lat_all, dist_all = [], []          # for the conduction law
    rows = {"within": [], "dist": [], "knn": [], "combo": []}
    tags, speeds = [], []
    print(f"{'subject':16s} {'within':>7} {'dist':>7} {'knn':>7} {'combo':>7} {'speed(mm/ms)':>12}")
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        if cs.latency is None or cs.latency.size == 0:
            continue
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= L.REL_MIN)]
        if len(keep) < 6: continue
        xyz = cs.contact_xyz
        # responsive-contact mask per site (amplitude above the site's 60th percentile)
        sub_lat, sub_dist = [], []
        fold = {k: [] for k in rows}
        for ti in keep:
            train = [t for t in keep if t != ti]
            amp = cs.responses[ti]; lat = cs.latency[ti]
            thr = np.nanpercentile(amp[np.isfinite(amp)], 60) if np.isfinite(amp).any() else np.inf
            resp = L._valid_mask(cs, ti, train) & (amp >= thr) & np.isfinite(lat)
            if resp.sum() < 6: continue
            D = np.linalg.norm(xyz - cs.stim_xyz[ti][None], axis=1)
            sub_lat.append(lat[resp]); sub_dist.append(D[resp])
            # predictors of latency
            wm = np.nansum(cs.latency[train], axis=0) / (np.sum(np.isfinite(cs.latency[train]), axis=0) + 1e-9)
            best = {}
            sg = max(SIG, key=lambda s: _topo(np.exp(-(D**2)/(2*s**2)), lat, resp))  # (sign handled by pearson)
            distpred = D  # latency grows with distance -> use raw distance
            tau = max(TAU, key=lambda t: _topo(_knn_lat(cs, ti, train, t), lat, resp))
            knn = _knn_lat(cs, ti, train, tau)
            def z(x, m):
                o = m & np.isfinite(x); out = np.full_like(x, np.nan, float)
                if o.sum() > 2: out[o] = (x[o] - x[o].mean()) / (x[o].std() + 1e-9)
                return out
            resid = z(knn, resp) - z(distpred, resp) * (np.nansum(z(knn, resp) * z(distpred, resp)) / (np.nansum(z(distpred, resp)**2) + 1e-9))
            beta = max(BETA, key=lambda b: _topo(np.nan_to_num(z(distpred, resp)) + b*np.nan_to_num(resid), lat, resp))
            combo = np.nan_to_num(z(distpred, resp)) + beta * np.nan_to_num(resid)
            fold["within"].append(_topo(wm, lat, resp)); fold["dist"].append(_topo(distpred, lat, resp))
            fold["knn"].append(_topo(knn, lat, resp)); fold["combo"].append(_topo(combo, lat, resp))
        if not fold["combo"]: continue
        for k in rows: rows[k].append(float(np.nanmean(fold[k])))
        # conduction speed for this subject
        La = np.concatenate(sub_lat); Da = np.concatenate(sub_dist)
        ok = np.isfinite(La) & np.isfinite(Da)
        lat_all.append(La[ok]); dist_all.append(Da[ok])
        b = np.polyfit(Da[ok], La[ok], 1)[0] if ok.sum() > 10 else np.nan  # ms per mm
        speed = 1.0 / b if b and b > 1e-6 else np.nan                       # mm per ms
        speeds.append(speed); tags.append(f"{ds[-4:]}/{cs.subject}")
        print(f"{tags[-1]:16s} {rows['within'][-1]:7.3f} {rows['dist'][-1]:7.3f} {rows['knn'][-1]:7.3f} {rows['combo'][-1]:7.3f} {speed:12.2f}")

    n = len(tags)
    La = np.concatenate(lat_all); Da = np.concatenate(dist_all)
    r_ld = float(np.corrcoef(La, Da)[0, 1])
    b_all = np.polyfit(Da, La, 1)[0]
    print(f"\n=== CONDUCTION LAW (pooled, {len(La)} responsive contacts) ===")
    print(f"  r(latency, distance) = {r_ld:+.3f}; slope {b_all*1000:.2f} us/mm -> ~{1/b_all:.1f} mm/ms apparent speed")
    print(f"\n=== LATENCY PREDICTABILITY (subject-level, n={n}) ===")
    out = {"n": n, "conduction": {"r_lat_dist": r_ld, "speed_mm_per_ms": float(1/b_all)}, "median_speed": float(np.nanmedian(speeds))}
    for k in rows:
        m, lo, hi = bootstrap_ci(rows[k]); out[k] = {"mean": m, "lo": lo, "hi": hi}
        print(f"  {k:8s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    p = paired_permutation_test(rows["combo"], rows["within"]); d = cohens_d_paired(rows["combo"], rows["within"])
    out["combo_vs_within"] = {"delta": np.mean(rows["combo"]) - np.mean(rows["within"]), "p": p, "d": d}
    print(f"  combo vs within: Δ={np.mean(rows['combo'])-np.mean(rows['within']):+.3f} p={p:.3g} d={d:+.2f}")
    (ROOT / "reports" / "latency.json").write_text(json.dumps(out, indent=2))
    print("saved reports/latency.json")


if __name__ == "__main__":
    main()
