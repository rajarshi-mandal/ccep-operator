"""EXTENSION — does the network mechanism matter MORE for the later N2 than the early N1?

Mechanistic hypothesis for the journal version: the early N1 (10-100 ms) is the direct,
locality-dominated response; the later N2 (100-300 ms) is polysynaptic and should carry MORE
network-propagated (beyond-locality) structure. If so, the operator's network term should contribute
a larger incremental-over-locality for N2 than for N1 — a mechanistic dissociation, not just a
second benchmark.

Reuses the ccep_loso predictors by swapping the target on each subject (N1 = responses, N2 = n2).
Reliability filter uses the N1 split-half (N2 reliability not separately cached) — stated as a caveat.
Output: reports/n2.json
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


def main():
    caches = L.all_caches()
    rows = {"n1_wm": [], "n1_combo": [], "n1_net": [], "n2_wm": [], "n2_combo": [], "n2_net": []}
    tags = []
    print(f"{'subject':16s} {'N1 combo':>9} {'N1 net':>8} {'N2 combo':>9} {'N2 net':>8}")
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        if cs.n2 is None or cs.n2.size == 0:
            continue
        n1 = cs.responses.copy()
        # N1
        cs.responses = n1
        e1 = L.eval_subject(cs); i1 = L.incremental_subject(cs)
        if e1 is None or i1 is None:
            continue
        # N2 (swap target; same reliability mask)
        cs.responses = cs.n2
        e2 = L.eval_subject(cs); i2 = L.incremental_subject(cs)
        cs.responses = n1
        if e2 is None or i2 is None:
            continue
        s1, _ = e1; s2, _ = e2
        rows["n1_wm"].append(s1["within_mean"]); rows["n1_combo"].append(s1["combo"]); rows["n1_net"].append(i1["stim_knn"])
        rows["n2_wm"].append(s2["within_mean"]); rows["n2_combo"].append(s2["combo"]); rows["n2_net"].append(i2["stim_knn"])
        tags.append(f"{ds[-4:]}/{cs.subject}")
        print(f"{tags[-1]:16s} {s1['combo']:9.3f} {i1['stim_knn']:8.3f} {s2['combo']:9.3f} {i2['stim_knn']:8.3f}")

    n = len(tags)
    print(f"\n=== subject-level means (n={n}) ===")
    out = {"n": n}
    for k, v in rows.items():
        m, lo, hi = bootstrap_ci(v); out[k] = {"mean": m, "lo": lo, "hi": hi}
        print(f"  {k:10s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # the headline contrast: network-incremental N2 vs N1 (paired)
    d = np.mean(rows["n2_net"]) - np.mean(rows["n1_net"])
    p = paired_permutation_test(rows["n2_net"], rows["n1_net"])
    dd = cohens_d_paired(rows["n2_net"], rows["n1_net"])
    w = sum(1 for a, b in zip(rows["n2_net"], rows["n1_net"]) if a > b)
    out["net_N2_vs_N1"] = {"delta": d, "p": p, "d": dd, "wins": w, "n": n}
    print(f"\nNETWORK incremental, N2 vs N1: delta={d:+.3f}  p={p:.3g}  d={dd:+.2f}  ({w}/{n} subj N2>N1)")
    print("  -> if positive & significant: the later polysynaptic N2 is MORE network-driven "
          "than the direct N1 — a mechanistic dissociation.")
    (ROOT / "reports" / "n2.json").write_text(json.dumps(out, indent=2))
    print("saved reports/n2.json")


if __name__ == "__main__":
    main()
