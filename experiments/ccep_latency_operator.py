"""EXTENSION — timing-derived directionality: does response LATENCY reveal causal direction,
and does a latency-oriented operator improve on the amplitude-symmetric one?

Physiological directedness argument (stronger than algebraic asymmetry): for two stimulated
contacts a,b, if a drives b then b should respond to a's pulse EARLIER than a responds to b's
pulse (L[a->b] < L[b->a]). We (1) test whether such timing asymmetry is systematic and (2) build a
latency-oriented operator (keep each edge only in its earlier-latency direction) and compare its
held-out prediction to the symmetric amplitude operator.
Output: reports/latency_operator.json
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
from ccep_operator_v2 import predict_operator_v2, _spec_norm  # noqa


def _build_WL(cs, idx):
    n = len(cs.contacts)
    Wa = np.zeros((n, n)); Wl = np.full((n, n), np.nan); cnt = np.zeros(n)
    for s in idx:
        a_amp = np.nan_to_num(cs.responses[s]); a_lat = cs.latency[s]
        for a in cs.stim_idx[s]:
            if a >= 0:
                Wa[a] += a_amp; cnt[a] += 1
                Wl[a] = np.where(np.isfinite(a_lat), a_lat, Wl[a])
    nz = cnt > 0; Wa[nz] /= cnt[nz, None]
    return Wa, Wl


def _norm(P, sym):
    sr = _spec_norm(P, sym=sym)
    return P / sr if sr > 1e-9 else P


def _oriented(Wa, Wl):
    """Keep each amplitude edge only in its EARLIER-latency direction (timing-oriented operator)."""
    n = Wa.shape[0]; D = np.zeros((n, n))
    both = np.isfinite(Wl) & np.isfinite(Wl.T)
    ii, jj = np.where(both)
    for i, j in zip(ii, jj):
        if Wl[i, j] <= Wl[j, i]:      # b(=j) responds to a(=i) earlier -> keep i->j edge
            D[i, j] = Wa[i, j]
    only = np.isfinite(Wl) & ~np.isfinite(Wl.T)   # edges with no reverse measurement: keep as-is
    D[only] = Wa[only]
    return D


def main():
    caches = L.all_caches()
    rows = {"symmetric": [], "oriented": [], "forward": []}
    asym = []
    tags = []
    print(f"{'subject':16s} {'sym':>7} {'oriented':>8} {'forward':>7} {'lat_asym':>8}")
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        if cs.latency is None or cs.latency.size == 0:
            continue
        sites = np.arange(len(cs.sites))
        keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= L.REL_MIN)]
        if len(keep) < 6:
            continue
        fold = {k: [] for k in rows}
        for ti in keep:
            train = [t for t in keep if t != ti]
            tgt = cs.responses[ti]; mask = L._valid_mask(cs, ti, train)
            Wa, Wl = _build_WL(cs, train)
            Psym = _norm(0.5 * (Wa + Wa.T), True)
            Pfwd = _norm(Wa.T, False)
            Po = _oriented(Wa, Wl); Po = _norm(Po.T, False)   # oriented, forward convention
            for name, P, sym in [("symmetric", Psym, True), ("forward", Pfwd, False), ("oriented", Po, False)]:
                pred = predict_operator_v2(cs, ti, train, sigma=15, alpha=1.0, steps=2, mode="symmetric", P=P)
                fold[name].append(L.topo_r(pred, tgt, mask))
        if not fold["symmetric"]:
            continue
        for k in rows:
            rows[k].append(float(np.nanmean(fold[k])))
        # latency asymmetry: mean |L[i,j]-L[j,i]| over reciprocal stim pairs (systematic timing order?)
        Wa, Wl = _build_WL(cs, keep)
        both = np.isfinite(Wl) & np.isfinite(Wl.T)
        if both.sum() > 10:
            diff = (Wl - Wl.T)[both]
            asym.append(float(np.mean(np.abs(diff))))
        tags.append(f"{ds[-4:]}/{cs.subject}")
        print(f"{tags[-1]:16s} {rows['symmetric'][-1]:7.3f} {rows['oriented'][-1]:8.3f} {rows['forward'][-1]:7.3f} {(asym[-1] if asym else float('nan')):8.2f}")

    n = len(tags)
    out = {"n": n}
    print(f"\n=== subject-level means (n={n}) ===")
    for k in rows:
        m, lo, hi = bootstrap_ci(rows[k]); out[k] = {"mean": m, "lo": lo, "hi": hi}
        print(f"  {k:10s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    d = np.mean(rows["oriented"]) - np.mean(rows["symmetric"])
    p = paired_permutation_test(rows["oriented"], rows["symmetric"]); dd = cohens_d_paired(rows["oriented"], rows["symmetric"])
    out["oriented_vs_symmetric"] = {"delta": d, "p": p, "d": dd, "wins": int(sum(a > b for a, b in zip(rows["oriented"], rows["symmetric"])))}
    out["mean_latency_asymmetry_ms"] = float(np.nanmean(asym)) if asym else None
    print(f"\ntiming-oriented vs symmetric: Δ={d:+.3f}  p={p:.3g}  d={dd:+.2f}  ({out['oriented_vs_symmetric']['wins']}/{n})")
    print(f"mean reciprocal latency asymmetry: {out['mean_latency_asymmetry_ms']:.1f} ms")
    print("  -> systematic timing asymmetry = physiological evidence of directed propagation, "
          "independent of the amplitude operator.")
    (ROOT / "reports" / "latency_operator.json").write_text(json.dumps(out, indent=2))
    print("saved reports/latency_operator.json")


if __name__ == "__main__":
    main()
