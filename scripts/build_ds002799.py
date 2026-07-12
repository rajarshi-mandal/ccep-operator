"""Build the ds002799 (es-fMRI) cache — schema-identical to ds005498 so the same analysis
harness (phase2_loso_ws.py, phase2b_group_level.py) runs unchanged.

Run after scripts/fetch_ds002799.sh. Processes whatever subjects have derivatives downloaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data.ds002799_pipeline import (DERIV, DS_DEFAULT, build_subject,  # noqa: E402
                                    find_es_runs, find_rest_run, load_schaefer)
from build_ds005498 import save_subject, manifest_rows  # noqa: E402

OUT = Path("data/processed/ds002799")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(DS_DEFAULT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-parcels", type=int, default=100)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()
    ds = Path(args.ds); out = Path(args.out)
    (out / "subjects").mkdir(parents=True, exist_ok=True)

    atlas_img, centroids, labels = load_schaefer(args.n_parcels)
    np.save(out / "atlas_centroids_mni.npy", centroids)

    if args.subjects:
        subs = [f"sub-{s.replace('sub-','')}" for s in args.subjects]
    else:
        subs = sorted(p.name for p in (ds / DERIV).glob("sub-*")
                      if find_es_runs(p) and find_rest_run(p))
    print(f"[build2799] {len(subs)} subjects with es+rest derivatives", flush=True)

    rows = []
    for i, sub in enumerate(subs):
        npz = out / "subjects" / f"{sub}.npz"
        if args.skip_existing and npz.exists():
            print(f"  [{i+1}/{len(subs)}] {sub}: cached", flush=True); continue
        t0 = time.time()
        try:
            rec = build_subject(sub, ds, atlas_img, centroids, n_parcels=args.n_parcels)
        except Exception as e:
            print(f"  [{i+1}/{len(subs)}] {sub}: FAIL {type(e).__name__}: {e}", flush=True); continue
        if not rec.sites:
            print(f"  [{i+1}/{len(subs)}] {sub}: no usable es runs", flush=True); continue
        save_subject(rec, out); rows.extend(manifest_rows(rec))
        rels = [r for r in rec.reliability if np.isfinite(r)]
        print(f"  [{i+1}/{len(subs)}] {sub}: {len(rec.sites)} sites, rest{rec.rest.shape}, "
              f"med_rel={np.median(rels):.3f} ({time.time()-t0:.0f}s)" if rels
              else f"  [{i+1}/{len(subs)}] {sub}: {len(rec.sites)} sites (no rel)", flush=True)

    man = out / "manifest.json"
    prior = json.loads(man.read_text()).get("rows", []) if man.exists() else []
    done = {r["subject"] for r in rows}
    merged = [r for r in prior if r["subject"] not in done] + rows
    man.write_text(json.dumps(dict(dataset="ds002799", n_subjects=len({r["subject"] for r in merged}),
                                   n_records=len(merged), rows=merged), indent=2))
    print(f"[build2799] {len({r['subject'] for r in merged})} subjects, {len(merged)} records -> {out}")


if __name__ == "__main__":
    main()
