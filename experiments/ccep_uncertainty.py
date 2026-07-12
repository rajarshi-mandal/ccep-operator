"""EXTENSION (methods) — calibrated confidence: does the model know WHERE it is trustworthy?

For a targeting aid, a point prediction is not enough — you need to know which predictions to
trust. We derive a per-contact confidence from cheap, available signals (proximity to the stim
site, response amplitude, and agreement between the locality and network predictors) and test
whether it is CALIBRATED: binning held-out contacts by confidence decile, does actual
predicted-vs-measured accuracy rise monotonically with confidence? A monotone curve = the model
reliably flags its own trustworthy predictions.
Output: reports/uncertainty.json  (per-decile accuracy).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa
import ccep_loso as L  # noqa


def _z(x, m):
    o = m & np.isfinite(x); out = np.zeros_like(x, float)
    if o.sum() > 2: out[o] = (x[o] - x[o].mean()) / (x[o].std() + 1e-9)
    return out


def main():
    caches = L.all_caches()
    P, M, CONF = [], [], []
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= L.REL_MIN)]
        if len(keep) < 6: continue
        xyz = cs.contact_xyz
        for ti in keep:
            tr = [t for t in keep if t != ti]
            mask = L._valid_mask(cs, ti, tr)
            sig = max(L.SIGMA_GRID, key=lambda s: L._score_param(cs, tr, lambda j, t, s=s: L.predict_distance(cs, j, s)))
            tau = max(L.TAU_GRID, key=lambda tt: L._score_param(cs, tr, lambda j, t, tt=tt: L.predict_stim_knn(cs, j, t, tt)))
            beta = max(L.BETA_GRID, key=lambda b: L._score_param(cs, tr, lambda j, t, b=b: L.predict_combo(cs, j, t, sig, tau, b, L._valid_mask(cs, j, t))))
            pred = L.predict_combo(cs, ti, tr, sig, tau, beta, mask)
            meas = cs.responses[ti]
            D = np.linalg.norm(xyz - cs.stim_xyz[ti][None], axis=1)
            loc = L.predict_distance(cs, ti, sig); knn = L.predict_stim_knn(cs, ti, tr, tau)
            # confidence: near (high), strong predicted response (high), locality/network agreement (high)
            agree = -np.abs(_z(loc, mask) - _z(knn, mask))
            conf = _z(-D, mask) + _z(np.nan_to_num(pred), mask) + agree
            # per-fold NORMALIZED absolute error (correct calibration metric; not per-bin r)
            zp = _z(np.nan_to_num(pred), mask); zm = _z(meas, mask)
            err = np.abs(zp - zm)
            for ci in range(len(cs.contacts)):
                if mask[ci] and np.isfinite(pred[ci]) and np.isfinite(meas[ci]):
                    P.append(err[ci]); M.append(meas[ci]); CONF.append(conf[ci])
    P, M, CONF = map(np.array, (P, M, CONF))   # P now holds normalized abs error
    print(f"pooled {len(P)} contacts")
    # decile calibration
    dec = np.quantile(CONF, np.linspace(0, 1, 11))
    out = {"n_records": int(len(P)), "deciles": []}
    print(f"{'conf decile':>12} {'mean |err|':>11} {'n':>7}")
    for i in range(10):
        sel = (CONF >= dec[i]) & (CONF <= dec[i + 1]) if i == 9 else (CONF >= dec[i]) & (CONF < dec[i + 1])
        e = float(np.mean(P[sel])) if sel.sum() > 20 else float("nan")
        out["deciles"].append({"decile": i + 1, "mean_err": e, "n": int(sel.sum())})
        print(f"{i+1:12d} {e:11.3f} {int(sel.sum()):7d}")
    lows = np.nanmean([d["mean_err"] for d in out["deciles"][:3]]); highs = np.nanmean([d["mean_err"] for d in out["deciles"][-3:]])
    out["low_vs_high_err"] = {"low_conf_err": lows, "high_conf_err": highs}
    print(f"\nlow-confidence err={lows:.3f}  vs  high-confidence err={highs:.3f}  "
          f"({'CALIBRATED (err falls with confidence)' if highs < lows else 'not calibrated'})")
    (ROOT / "reports" / "uncertainty.json").write_text(json.dumps(out, indent=2))
    print("saved reports/uncertainty.json")


if __name__ == "__main__":
    main()
