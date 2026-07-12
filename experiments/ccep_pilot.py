"""EXTENSION (methods) — how few sites must you map? Optimal pilot-site selection.

Clinical question: a full CCEP map is many pulses. If you can only stimulate k sites in a new
patient, (a) how well can you predict the unmapped sites' responses, and (b) does choosing those k
sites by spatial COVERAGE (farthest-point sampling) beat random selection? A steep coverage curve
means a short, well-chosen pilot session suffices — directly actionable for the few-shot workflow.
Output: reports/pilot.json  (r vs #mapped-sites, random vs greedy).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa
from eval.stats import bootstrap_ci  # noqa
import ccep_loso as L  # noqa

KS = [3, 5, 8, 12, 20]
REPEATS = 8
RNG = np.random.default_rng(0)


def farthest_point(coords, k, rng):
    """Greedy farthest-point sampling on stim coordinates (maximize spatial coverage)."""
    n = len(coords); start = rng.integers(n); chosen = [start]
    d = np.linalg.norm(coords - coords[start][None], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d)); chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(coords - coords[nxt][None], axis=1))
    return chosen


def predict(cs, test_i, mapped):
    mask = L._valid_mask(cs, test_i, mapped)
    return L.predict_combo(cs, test_i, mapped, sigma=15.0, tau=25.0, beta=1.0, mask=mask), mask


def main():
    caches = L.all_caches()
    rand = {k: [] for k in KS}; greedy = {k: [] for k in KS}
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= L.REL_MIN)]
        if len(keep) < max(KS) + 6:
            continue
        coords = cs.stim_xyz[keep]
        for k in KS:
            rr, gg = [], []
            for rep in range(REPEATS):
                rng = np.random.default_rng(1000 * k + rep)
                # random pilot set
                mapped_r = list(rng.choice(keep, size=k, replace=False))
                test_r = [t for t in keep if t not in mapped_r]
                rs = [L.topo_r(*(lambda p: (p[0], cs.responses[t], p[1]))(predict(cs, t, mapped_r))) for t in test_r]
                rr.append(np.nanmean(rs))
                # greedy coverage pilot set
                idx = farthest_point(coords, k, rng); mapped_g = list(keep[idx])
                test_g = [t for t in keep if t not in mapped_g]
                gs = [L.topo_r(*(lambda p: (p[0], cs.responses[t], p[1]))(predict(cs, t, mapped_g))) for t in test_g]
                gg.append(np.nanmean(gs))
            rand[k].append(float(np.nanmean(rr))); greedy[k].append(float(np.nanmean(gg)))

    n = len(rand[KS[0]])
    out = {"n": n, "ks": KS, "random": {}, "greedy": {}}
    print(f"n={n} subjects (>= {max(KS)+6} reliable sites)")
    print(f"{'#mapped':>8} {'random r':>10} {'greedy r':>10} {'gain':>7}")
    for k in KS:
        mr, lr, hr = bootstrap_ci(rand[k]); mg, lg, hg = bootstrap_ci(greedy[k])
        out["random"][k] = {"mean": mr, "lo": lr, "hi": hr}; out["greedy"][k] = {"mean": mg, "lo": lg, "hi": hg}
        print(f"{k:8d} {mr:10.3f} {mg:10.3f} {mg-mr:+7.3f}")
    (ROOT / "reports" / "pilot.json").write_text(json.dumps(out, indent=2))
    print("saved reports/pilot.json")
    print("Interpretation: if greedy>random and the curve saturates early, a short coverage-guided "
          "pilot session predicts the rest — minimizing pulses.")


if __name__ == "__main__":
    main()
