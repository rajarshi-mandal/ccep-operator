"""Audit the already-processed data caches (NO new downloads). Reports shapes, dtypes, NaN/Inf,
value ranges, and the interventional record inventory (sites, subjects, datasets, stim parcels).

Purpose: a single source of truth for "what data do we actually have", so claims and splits can
be checked against reality. Writes reports/existing_data_audit.md.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def describe_array(name: str, arr: np.ndarray) -> list[str]:
    finite = np.isfinite(arr)
    return [f"### {name}",
            f"- shape: `{arr.shape}`  dtype: `{arr.dtype}`",
            f"- finite: {int(finite.sum())}/{arr.size}  "
            f"(NaN={int(np.isnan(arr).sum())}, Inf={int(np.isinf(arr).sum())})",
            f"- range: [{np.nanmin(arr):.4g}, {np.nanmax(arr):.4g}]  "
            f"mean={np.nanmean(arr):.4g}  std={np.nanstd(arr):.4g}", ""]


def audit_records(npz_path: Path) -> list[str]:
    recs = list(np.load(npz_path, allow_pickle=True)["records"])
    lines = [f"### {npz_path.name}", f"- records: **{len(recs)}**"]
    if not recs:
        return lines + [""]
    keys = set(recs[0].keys())
    lines.append(f"- record fields: `{sorted(keys)}`")
    dsets = Counter(str(r.get("dataset", "?")) for r in recs)
    subs = Counter(str(r.get("subject", "")) for r in recs)
    lines += [f"- by dataset: `{dict(dsets)}`",
              f"- unique subjects: {len([s for s in subs if s])} "
              f"(blank ids: {subs.get('', 0)})"]
    # region-space schema fields (present only in interventional_region.npz)
    if "site_name" in keys:
        lines.append(f"- by site_name: `{dict(Counter(r['site_name'] for r in recs))}`")
    if "site" in keys:
        lines.append(f"- by site: `{dict(Counter(str(r['site']) for r in recs))}`")
    if "cond" in keys:
        lines.append(f"- by cond: `{dict(Counter(str(r.get('cond')) for r in recs))}`")
    if "stim_parcel" in keys:
        lines.append(f"- stim parcels: `{dict(Counter(int(r['stim_parcel']) for r in recs))}`")
    tep_field = "region_tep" if "region_tep" in keys else ("tep" if "tep" in keys else None)
    if tep_field:
        shapes = Counter(tuple(np.asarray(r[tep_field]).shape) for r in recs)
        lines.append(f"- {tep_field} shapes: `{dict(shapes)}`")
    lines.append("")
    return lines


def main() -> int:
    cfg = load_config()
    proc = Path(cfg.paths.processed_dir)
    md = ["# Existing data cache audit\n",
          f"Processed dir: `{proc}` — NO new data fetched; this audits caches on disk.\n"]

    npy = proc / "observational_fmri.npy"
    if npy.exists():
        md += ["## Observational (fMRI)\n"] + describe_array(npy.name, np.load(npy))
    else:
        md += [f"## Observational (fMRI)\n\n_Missing: {npy.name}_\n"]

    cents = proc / "parcel_centroids_mni.npy"
    if cents.exists():
        md += ["## Parcel centroids (MNI)\n"] + describe_array(cents.name, np.load(cents))

    md += ["## Interventional record caches\n"]
    for nm in ["interventional_region.npz", "interventional_ds004024.npz",
               "interventional_zenodo.npz"]:
        p = proc / nm
        if p.exists():
            md += audit_records(p)
        else:
            md += [f"### {nm}\n\n_Missing_\n"]

    out = REPORTS / "existing_data_audit.md"
    out.write_text("\n".join(md) + "\n")
    print(f"Wrote {out}")
    # also echo the M1 LOSO-relevant headline to console
    reg = proc / "interventional_region.npz"
    if reg.exists():
        recs = list(np.load(reg, allow_pickle=True)["records"])
        m1 = [r for r in recs if r["site_name"] == "M1_L"]
        print(f"  M1_L records: {len(m1)} | unique subjects: "
              f"{len(set(str(r.get('subject','')) for r in m1))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
