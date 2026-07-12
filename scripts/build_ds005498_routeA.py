"""Build a route-A (real EPI->T1->MNI registration) topography cache for the group analysis.

Route B is anatomically meaningless, which scrambles site-specific structure across subjects and
makes the chance-level site decoding inconclusive. This rebuilds per-(subject, site) evoked
topographies with proper dipy registration (src/data/registration.py), into a separate cache
(data/processed/ds005498_routeA/), so experiments/phase2b_group_level.py can re-test whether the
group-level evoked response carries site information once parcels actually correspond.

Per subject: fit T1->MNI once; register one EPI per acquisition geometry (rest vs stim, and any
slice-count variants) and reuse its atlas-in-EPI label volume across runs of that geometry.
Incremental/resumable (skips subjects already cached). Compatible with DS005498Cache.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data.ds005498_pipeline import (ARTIFACT_MM, DS_DEFAULT, N_PARCELS, TR_STIM,  # noqa
                                    SubjectRecord, coil_to_parcel, evoked_response,
                                    find_rest_run, find_stim_runs, load_schaefer,
                                    parse_coil_mni, zscore)
from data.registration import SubjectRegistration  # noqa: E402
from build_ds005498 import save_subject, manifest_rows  # noqa: E402
import json

OUT_DIR = Path("data/processed/ds005498_routeA")


def find_t1(sub_dir):
    t = sorted(sub_dir.glob("ses-*/anat/*_T1w.nii.gz"))
    return t[0] if t else None


def parcels_from_labels(bold_img, lab_vol, n=N_PARCELS):
    data = np.asarray(bold_img.dataobj, dtype=np.float32)
    T = data.shape[-1]
    flat = data.reshape(-1, T); labf = lab_vol.reshape(-1)
    out = np.zeros((T, n), dtype=np.float32)
    for L in range(1, n + 1):
        m = labf == L
        if m.any():
            out[:, L - 1] = flat[m].mean(0)
    return out


def build_subject_routeA(sub_dir, reg: SubjectRegistration, centroids, onsets):
    rec = SubjectRecord(subject=sub_dir.name, rest=np.zeros((0, N_PARCELS), np.float32))
    label_cache: dict[tuple, np.ndarray] = {}     # EPI shape -> atlas label volume

    def labels_for(img):
        key = img.shape[:3]
        if key not in label_cache:
            label_cache[key] = reg.atlas_in_epi(img)
        return label_cache[key]

    rest_path = find_rest_run(sub_dir)
    if rest_path is not None:
        rimg = nib.load(str(rest_path))
        rec.rest = zscore(parcels_from_labels(rimg, labels_for(rimg)))

    for site, paths in find_stim_runs(sub_dir).items():
        img = nib.load(str(paths[0]))
        ts = parcels_from_labels(img, labels_for(img))
        ev = evoked_response(ts, onsets, TR_STIM)
        if ev is None:
            continue
        coil = parse_coil_mni(site)
        near = np.argwhere(np.linalg.norm(centroids - coil[None], axis=1) < ARTIFACT_MM)
        rec.sites.append(site); rec.topo.append(ev["topo"]); rec.fir.append(ev["fir"])
        rec.stim_parcel.append(coil_to_parcel(coil, centroids)); rec.coil_mni.append(coil)
        rec.reliability.append(ev["reliability"]); rec.n_pulses.append(ev["n_pulses"])
        rec.artifact_parcels.append([int(x) for x in near.ravel().tolist()])
        rec.run_paths.append([str(p) for p in paths])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subj", type=int, default=40)
    ap.add_argument("--quality", default="fast")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out); (out_dir / "subjects").mkdir(parents=True, exist_ok=True)

    from nilearn.datasets import load_mni152_template
    atlas_img, centroids, labels = load_schaefer()
    template = load_mni152_template(resolution=2)
    np.save(out_dir / "atlas_centroids_mni.npy", centroids)
    onsets = pd.read_csv(DS_DEFAULT / "task-stim_events.tsv", sep="\t")["onset"].to_numpy(float)

    scored = []
    for s in sorted(p for p in DS_DEFAULT.glob("sub-*") if p.is_dir()):
        if find_t1(s) and find_rest_run(s):
            n = len(find_stim_runs(s))
            if n:
                scored.append((n, s))
    scored.sort(key=lambda t: (-t[0], t[1].name))
    subs = [s for _, s in scored[:args.n_subj]]
    print(f"[routeA-build] {len(subs)} subjects ({args.quality})", flush=True)

    all_rows = []
    for i, sub in enumerate(subs):
        npz = out_dir / "subjects" / f"{sub.name}.npz"
        if npz.exists():
            print(f"  [{i+1}/{len(subs)}] {sub.name}: cached", flush=True); continue
        t0 = time.time()
        try:
            reg = SubjectRegistration(find_t1(sub), atlas_img, template, quality=args.quality)
            rec = build_subject_routeA(sub, reg, centroids, onsets)
        except Exception as e:
            print(f"  [{i+1}/{len(subs)}] {sub.name}: FAIL {type(e).__name__}: {e}", flush=True)
            continue
        save_subject(rec, out_dir); all_rows.extend(manifest_rows(rec))
        rels = [r for r in rec.reliability if np.isfinite(r)]
        print(f"  [{i+1}/{len(subs)}] {sub.name}: {len(rec.sites)} sites, "
              f"rest{rec.rest.shape}, med_rel={np.median(rels):.3f} ({time.time()-t0:.0f}s)",
              flush=True)

    man = out_dir / "manifest.json"
    prior = json.loads(man.read_text()).get("rows", []) if man.exists() else []
    done = {r["subject"] for r in all_rows}
    merged = [r for r in prior if r["subject"] not in done] + all_rows
    man.write_text(json.dumps(dict(n_subjects=len({r["subject"] for r in merged}),
                                   n_records=len(merged), reg="epi2mni_affine", rows=merged), indent=2))
    print(f"[routeA-build] {len({r['subject'] for r in merged})} subjects cached -> {out_dir}")


if __name__ == "__main__":
    main()
