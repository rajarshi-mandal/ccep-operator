"""Phase 1 driver — build the ds005498 model-ready cache (handoff §5).

Turns raw ds005498 BOLD into per-subject Schaefer-100 tensors via
``src/data/ds005498_pipeline.py`` and writes an incremental, resumable cache:

  data/processed/ds005498/
    atlas_centroids_mni.npy   [100, 3]  Schaefer parcel centroids (locality prior, §6)
    schaefer_labels.json       parcel names
    subjects/<sub>.npz         per subject: rest[T,100], topo[nsite,100], fir[nsite,100,nb],
                               stim_parcel[nsite], coil_mni[nsite,3], reliability[nsite],
                               n_pulses[nsite], sites[nsite]
    manifest.json              per (subject, site) row: paths, n_pulses, reliability,
                               stim_parcel, artifact_parcels, qc_pass

Resumable like the hardened exp1b run: each subject is written atomically; ``--skip-existing``
(default) skips subjects whose npz already exists. Build ~10 subjects first (``--n-subj 10``),
validate, then scale.

Run from causal-dag-ssm/ with the venv active:
    python scripts/build_ds005498.py --n-subj 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.ds005498_pipeline import (  # noqa: E402
    DS_DEFAULT, N_PARCELS, REL_QC_THRESH, build_subject, find_rest_run,
    find_stim_runs, load_schaefer,
)

OUT_DIR = Path("data/processed/ds005498")


def eligible_subjects(ds: Path) -> list[Path]:
    """Subjects with BOTH a resting run and >=1 stim run, sorted by site coverage desc.

    Sorting by coverage means ``--n-subj 10`` picks the richest subjects first — exactly
    the ones that drive the leave-one-site-out-within-subject design.
    """
    subs = []
    for s in sorted(p for p in ds.glob("sub-*") if p.is_dir()):
        if find_rest_run(s) is None:
            continue
        sites = find_stim_runs(s)
        if sites:
            subs.append((len(sites), s))
    subs.sort(key=lambda t: (-t[0], t[1].name))
    return [s for _, s in subs]


def save_subject(rec, out_dir: Path) -> Path:
    """Write one subject's npz atomically (tmp + replace)."""
    sub_dir = out_dir / "subjects"
    sub_dir.mkdir(parents=True, exist_ok=True)
    path = sub_dir / f"{rec.subject}.npz"
    tmp = sub_dir / f"{rec.subject}.tmp.npz"   # must end in .npz (savez won't append)
    np.savez_compressed(
        tmp,
        subject=rec.subject,
        rest=rec.rest,
        sites=np.array(rec.sites, dtype=object),
        topo=np.array(rec.topo, dtype=np.float32) if rec.topo else np.zeros((0, N_PARCELS), np.float32),
        fir=np.array(rec.fir, dtype=np.float32) if rec.fir else np.zeros((0, N_PARCELS, 0), np.float32),
        stim_parcel=np.array(rec.stim_parcel, dtype=np.int64),
        coil_mni=np.array(rec.coil_mni, dtype=np.float64) if rec.coil_mni else np.zeros((0, 3)),
        reliability=np.array(rec.reliability, dtype=np.float64),
        n_pulses=np.array(rec.n_pulses, dtype=np.int64),
    )
    tmp.replace(path)
    return path


def manifest_rows(rec) -> list[dict]:
    rows = []
    for i, site in enumerate(rec.sites):
        rel = rec.reliability[i]
        rows.append(dict(
            subject=rec.subject, site=site,
            stim_parcel=int(rec.stim_parcel[i]),
            coil_mni=[float(x) for x in rec.coil_mni[i]],
            n_pulses=int(rec.n_pulses[i]),
            reliability=None if rel is None or not np.isfinite(rel) else float(rel),
            artifact_parcels=rec.artifact_parcels[i],
            run_paths=rec.run_paths[i],
            qc_pass=bool(rel is not None and np.isfinite(rel) and rel >= REL_QC_THRESH),
            rest_len=int(rec.rest.shape[0]),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(DS_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--n-subj", type=int, default=None, help="cap number of subjects (richest first)")
    ap.add_argument("--subjects", nargs="*", default=None, help="explicit subject ids (sub-XXXX)")
    ap.add_argument("--start", type=int, default=0, help="slice start into the eligible list")
    ap.add_argument("--end", type=int, default=None, help="slice end into the eligible list")
    ap.add_argument("--reg", default="affine_overlay")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    ds = Path(args.ds)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] loading Schaefer-{N_PARCELS} atlas ...", flush=True)
    atlas_img, centroids, labels = load_schaefer(N_PARCELS)
    np.save(out_dir / "atlas_centroids_mni.npy", centroids)
    (out_dir / "schaefer_labels.json").write_text(json.dumps(labels, indent=2))

    events = pd.read_csv(ds / "task-stim_events.tsv", sep="\t")
    onsets = events["onset"].to_numpy(dtype=float)
    print(f"[build] {len(onsets)} pulse onsets (median ITI "
          f"{np.median(np.diff(np.sort(onsets))):.2f}s)", flush=True)

    if args.subjects:
        subjects = [ds / s for s in args.subjects]
    else:
        subjects = eligible_subjects(ds)
        subjects = subjects[args.start:args.end]
        if args.n_subj is not None:
            subjects = subjects[:args.n_subj]
    print(f"[build] {len(subjects)} subjects to process (reg={args.reg})", flush=True)

    all_rows: list[dict] = []
    for i, sub_dir in enumerate(subjects):
        npz = out_dir / "subjects" / f"{sub_dir.name}.npz"
        if args.skip_existing and npz.exists():
            print(f"  [{i+1}/{len(subjects)}] {sub_dir.name}: skip (cached)", flush=True)
            continue
        t0 = time.time()
        try:
            rec = build_subject(sub_dir, atlas_img, centroids, onsets, reg=args.reg)
        except Exception as e:
            print(f"  [{i+1}/{len(subjects)}] {sub_dir.name}: FAIL {type(e).__name__}: {e}",
                  flush=True)
            continue
        save_subject(rec, out_dir)
        rows = manifest_rows(rec)
        all_rows.extend(rows)
        rels = [r["reliability"] for r in rows if r["reliability"] is not None]
        npass = sum(r["qc_pass"] for r in rows)
        print(f"  [{i+1}/{len(subjects)}] {sub_dir.name}: rest={rec.rest.shape} "
              f"{len(rec.sites)} sites, {npass} pass QC, "
              f"median_rel={np.median(rels):.3f} ({time.time()-t0:.0f}s)" if rels else
              f"  [{i+1}/{len(subjects)}] {sub_dir.name}: {len(rec.sites)} sites, no reliability",
              flush=True)

    # merge manifest with any prior rows (resumable across partial runs)
    man_path = out_dir / "manifest.json"
    prior = []
    if man_path.exists():
        prior = json.loads(man_path.read_text()).get("rows", [])
    done_subs = {r["subject"] for r in all_rows}
    merged = [r for r in prior if r["subject"] not in done_subs] + all_rows
    n_subj = len({r["subject"] for r in merged})
    n_pass = sum(r["qc_pass"] for r in merged)
    man_path.write_text(json.dumps(dict(
        n_subjects=n_subj, n_records=len(merged), n_qc_pass=n_pass,
        rel_qc_thresh=REL_QC_THRESH, n_parcels=N_PARCELS, reg=args.reg,
        rows=merged,
    ), indent=2))
    print(f"\n[build] cache -> {out_dir}")
    print(f"[build] {n_subj} subjects, {len(merged)} (subject,site) records, "
          f"{n_pass} pass QC (rel>={REL_QC_THRESH})")


if __name__ == "__main__":
    main()
