"""Phase 2b — group-level pivot: is there ANY real evoked signal once subject noise averages out?

Per-subject personalization is impossible on this data (the evoked target is artifact-dominated;
see reports/PHASE2_SIGNAL_DIAGNOSIS.md). The honest fallback is a group-level question: does the
*population-averaged* evoked response carry real, site-specific structure? Three tests on the
cached topographies (registration route B; a null here under-states a possible route-A signal,
but a positive here would be decisive):

  A. Split-group reliability — per site, correlate the mean topography of two random halves of
     subjects. Averaging N subjects suppresses per-subject noise by ~sqrt(N), so a real but small
     consistent map shows up here even when pairwise cross-subject r ~ 0.
  B. Site decoding — leave-one-subject-out, classify a record's stim site as the best-correlating
     site template (built from the other subjects). Accuracy vs 1/11 = 9.1% chance tells us
     whether the evoked topography carries site information at the group level at all.
  C. Group FIR — population-mean FIR at the coil parcel; does it look like an HRF?

Both raw and shared-mode-deflated (site-specific) variants, since a dominant common map can
inflate A/B without any site specificity.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.ds005498_pipeline import DS005498Cache  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402


def _r(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 1e-9 and db > 1e-9 else 0.0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/processed/ds005498")
    args = ap.parse_args()
    cache = DS005498Cache(cache_dir=args.cache_dir, qc_filter=True)
    recs = cache.records
    d = cache.centroids.shape[0]
    by_site = defaultdict(list)          # site -> list of (subject, topo)
    for r in recs:
        by_site[r.site_name].append((r.subject, r.topo))
    sites = sorted(by_site)
    print(f"[2b] {len(recs)} records, {len(cache.subjects())} subjects, {len(sites)} sites", flush=True)

    # shared mode (first PC of all topographies) for the deflated variant
    T = np.stack([r.topo for r in recs]); Tc = T - T.mean(0)
    _, _, Vt = np.linalg.svd(Tc, full_matrices=False)
    mode = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    defl = lambda x: x - (x @ mode) * mode

    rng = np.random.default_rng(0)

    # ---- A. split-group reliability per site ----
    def split_rel(transform):
        out = []
        for s in sites:
            mats = np.stack([transform(t) for _, t in by_site[s]])
            n = len(mats)
            if n < 6:
                continue
            rs = []
            for _ in range(200):
                idx = rng.permutation(n); h = n // 2
                m1 = mats[idx[:h]].mean(0); m2 = mats[idx[h:2 * h]].mean(0)
                rs.append(_r(m1, m2))
            out.append((s, float(np.mean(rs)), n))
        return out

    print("\n== A. split-group topography reliability (mean of 200 random half-splits) ==")
    print(f"{'site':22s} {'raw':>8s} {'deflated':>10s}  n")
    relA_raw, relA_def = split_rel(lambda x: x), split_rel(defl)
    dmap = {s: r for s, r, _ in relA_def}
    for s, r, n in relA_raw:
        print(f"{s:22s} {r:+8.3f} {dmap.get(s, float('nan')):+10.3f}  {n}")
    print(f"{'MEDIAN':22s} {np.median([r for _,r,_ in relA_raw]):+8.3f} "
          f"{np.median([r for _,r,_ in relA_def]):+10.3f}")

    # ---- B. leave-one-subject-out site decoding ----
    def decode(transform):
        subs = defaultdict(dict)         # subject -> site -> topo
        for r in recs:
            subs[r.subject][r.site_name] = transform(r.topo)
        # precompute per-site sum & count for fast LOSO templates
        ssum = {s: np.zeros(d) for s in sites}; scnt = {s: 0 for s in sites}
        for sub, dd in subs.items():
            for s, t in dd.items():
                ssum[s] += t; scnt[s] += 1
        correct = tot = 0
        per_site_hits = defaultdict(lambda: [0, 0])
        for sub, dd in subs.items():
            for true_s, t in dd.items():
                # leave-this-subject-out templates
                best, bestr = None, -2
                for s in sites:
                    cnt = scnt[s] - (1 if s in dd else 0)
                    if cnt <= 0:
                        continue
                    tmpl = (ssum[s] - (dd[s] if s in dd else 0)) / cnt
                    rr = _r(t, tmpl)
                    if rr > bestr:
                        bestr, best = rr, s
                correct += (best == true_s); tot += 1
                per_site_hits[true_s][0] += (best == true_s); per_site_hits[true_s][1] += 1
        return correct / tot, tot, per_site_hits

    print("\n== B. leave-one-subject-out site decoding (chance = 1/11 = 9.1%) ==")
    for name, tr in [("raw", lambda x: x), ("deflated", defl)]:
        acc, tot, ph = decode(tr)
        # permutation/bootstrap CI on accuracy
        mean, lo, hi = bootstrap_ci([1.0] * int(round(acc * tot)) + [0.0] * (tot - int(round(acc * tot))))
        print(f"  {name:9s} accuracy = {acc*100:5.1f}%  [95% CI {lo*100:.1f}-{hi*100:.1f}]  (n={tot})")

    # ---- C. group-mean FIR at the coil parcel ----
    fir = np.stack([r.fir[r.stim_parcel] for r in recs])
    gm = fir.mean(0)
    lags = [round(2.4 * i, 1) for i in range(fir.shape[1])]
    print(f"\n== C. group-mean FIR @ coil parcel ==\n  lags {lags}\n  FIR  {np.round(gm,3)}  "
          f"(HRF peaks ~4.8-7.2s; argmax at {lags[int(np.argmax(gm))]}s)")

    print("\nVERDICT: site decoding >> 9.1% OR split-group reliability clearly >0 => real "
          "group-level evoked signal exists (worth a route-A group analysis). At chance / ~0 => "
          "the negative extends to the group level on this (route-B) processing.")


if __name__ == "__main__":
    main()
