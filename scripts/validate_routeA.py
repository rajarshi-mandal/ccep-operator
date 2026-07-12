"""Validate route-A registration on a few subjects before committing to the full rebuild.

For each subject: fit T1->MNI once, EPI->T1 once (reused across that subject's stim sites),
carry the Schaefer atlas onto the EPI grid, extract per-site Glover topographies, then run the
two diagnostics that exposed route B:
  * stim-parcel |response| percentile  (route B = 0.41; want -> high, response peaks at coil)
  * same-site cross-subject topo coherence (route B = 0.013; want -> clearly positive)
If both improve, route A works and we rebuild the cache with it.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.ds005498_pipeline import (DS_DEFAULT, N_PARCELS, TR_STIM, coil_to_parcel,  # noqa
                                    evoked_response, find_stim_runs, load_schaefer,
                                    parse_coil_mni)
from data.registration import SubjectRegistration  # noqa: E402


def parcels_from_labels(bold_img, lab_vol, n=N_PARCELS):
    data = np.asarray(bold_img.dataobj, dtype=np.float32)
    T = data.shape[-1]
    flat = data.reshape(-1, T); labf = lab_vol.reshape(-1)
    out = np.zeros((T, n), dtype=np.float32)
    for lab in range(1, n + 1):
        m = labf == lab
        if m.any():
            out[:, lab - 1] = flat[m].mean(0)
    return out


def find_t1(sub_dir):
    t = sorted(sub_dir.glob("ses-*/anat/*_T1w.nii.gz"))
    return t[0] if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subj", type=int, default=8)
    ap.add_argument("--quality", default="fast")
    args = ap.parse_args()

    from nilearn.datasets import load_mni152_template
    atlas_img, centroids, _ = load_schaefer()
    template = load_mni152_template(resolution=2)
    onsets = pd.read_csv(DS_DEFAULT / "task-stim_events.tsv", sep="\t")["onset"].to_numpy(float)

    # richest subjects (most stim sites) that have a T1 — needed for the coherence metric
    scored = []
    for s in sorted(p for p in DS_DEFAULT.glob("sub-*") if p.is_dir()):
        if find_t1(s):
            n = len(find_stim_runs(s))
            if n:
                scored.append((n, s))
    scored.sort(key=lambda t: (-t[0], t[1].name))
    subs = [s for _, s in scored[:args.n_subj]]
    print(f"[routeA] validating on {len(subs)} subjects ({args.quality})", flush=True)

    bysite = defaultdict(list)
    pct = []
    for i, sub in enumerate(subs):
        t0 = time.time()
        try:
            reg = SubjectRegistration(find_t1(sub), atlas_img, template, quality=args.quality)
            sites = find_stim_runs(sub)
            # register one representative stim EPI; reuse labels across sites (shared geometry)
            ref = nib.load(str(next(iter(sites.values()))[0]))
            lab = reg.atlas_in_epi(ref)
            nlab = len(np.unique(lab)) - 1
            for site, paths in sites.items():
                ts = parcels_from_labels(nib.load(str(paths[0])), lab)
                ev = evoked_response(ts, onsets, TR_STIM)
                if ev is None or not np.isfinite(ev["reliability"]) or ev["reliability"] < 0.3:
                    continue
                topo = ev["topo"]
                p = coil_to_parcel(parse_coil_mni(site), centroids)
                t = np.abs(topo)
                rank = np.argsort(np.argsort(t))[p] / (N_PARCELS - 1)
                pct.append(rank)
                bysite[site].append(topo)
            print(f"  [{i+1}/{len(subs)}] {sub.name}: {nlab} labels in EPI, "
                  f"{sum(len(v) for v in bysite.values())} cum topos ({time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(subs)}] {sub.name}: FAIL {type(e).__name__}: {e}", flush=True)

    def meanpair(M):
        M = np.array(M); M = M - M.mean(1, keepdims=True)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        C = M @ M.T; iu = np.triu_indices(len(M), 1)
        return float(C[iu].mean()) if len(iu[0]) else np.nan

    print("\n=== ROUTE A diagnostics ===")
    print(f"stim-parcel |response| percentile: median={np.median(pct):.2f} "
          f"mean={np.mean(pct):.2f}   (route B was 0.41; 1.0=peak at coil)")
    coh = {s: meanpair(v) for s, v in bysite.items() if len(v) >= 3}
    if coh:
        print("same-site cross-subject coherence (route B was 0.013):")
        for s in sorted(coh):
            print(f"  {s:22s} n={len(bysite[s]):2d} r={coh[s]:+.3f}")
        print(f"  -> median across sites: {np.median(list(coh.values())):+.3f}")


if __name__ == "__main__":
    main()
