"""Build region-space interventional records by applying the spatial bridge.

Combines the cached electrode-space TEPs (ds004024 M1, Zenodo parietal) with the
EEG<->atlas bridge to produce, per stimulation, a record of:

    stim_parcel : int            # which of the d latent regions was stimulated
    region_tep  : [d, T_eeg]     # observed downstream response in region space
    site, dataset, subject

These are the supervision targets for the interventional loss L_int: the model applies
a do-operation at ``stim_parcel`` and its predicted downstream region response is
compared against ``region_tep``.

Subjects do not overlap across the observational (HCP) and interventional (TMS-EEG)
cohorts, so "pairing" is at the population level: the trainer draws observational
batches from HCP and interventional batches from here. This cross-cohort design relies
entirely on the shared 100-region latent space — the bridge is what makes it coherent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_eeg_mapping import CANONICAL_SITES_MNI, EEGAtlasBridge

# Provisional site -> canonical-MNI assignment used to choose the stimulated parcel.
# ds004024 spTMS = left M1 (run-01). Zenodo = parietal set (condition-level site labels
# still to be confirmed from record 4990628; treated as left parietal for now).
_DATASET_SITE = {
    "ds004024": "M1_L",
    "zenodo_parietal": "parietal_L",
}


def _load_records(npz_path: Path) -> list[dict]:
    if not npz_path.exists():
        return []
    return list(np.load(npz_path, allow_pickle=True)["records"])


def build_region_interventional_cache(cfg) -> Path:
    """Map every electrode-space TEP into region space and cache the result."""
    proc = Path(cfg.paths.processed_dir)
    bridge = EEGAtlasBridge(cfg)
    site_parcel = {k: bridge.site_to_parcel(v) for k, v in CANONICAL_SITES_MNI.items()}

    raw_records = []
    raw_records += _load_records(proc / "interventional_ds004024.npz")
    raw_records += _load_records(proc / "interventional_zenodo.npz")

    out = []
    for rec in raw_records:
        region_tep = bridge.tep_to_regions(rec["tep"], rec["ch_names"])  # [d, T]
        site_name = _DATASET_SITE.get(rec["dataset"], None)
        stim_parcel = site_parcel[site_name] if site_name else -1
        out.append({
            "stim_parcel": int(stim_parcel),
            "site_name": site_name,
            "region_tep": region_tep.astype(np.float32),
            "times": rec["times"],
            "dataset": rec["dataset"],
            "subject": rec["subject"],
            "site": rec.get("site"),
            "cond": rec.get("cond"),
        })

    npz_path = proc / "interventional_region.npz"
    np.savez_compressed(npz_path, records=np.array(out, dtype=object))
    manifest = {
        "n_records": len(out),
        "d": bridge.d,
        "tep_T": int(out[0]["region_tep"].shape[1]) if out else None,
        "site_parcels": site_parcel,
        "by_dataset": {
            ds: sum(1 for r in out if r["dataset"] == ds)
            for ds in {r["dataset"] for r in out}
        },
    }
    with open(proc / "interventional_region.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return npz_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config

    cfg = load_config()
    path = build_region_interventional_cache(cfg)
    recs = list(np.load(path, allow_pickle=True)["records"])
    print("cached:", path.name, "| n_records:", len(recs))
    for ds in sorted({r["dataset"] for r in recs}):
        sub = [r for r in recs if r["dataset"] == ds]
        r0 = sub[0]
        print(f"  {ds}: {len(sub)} recs | stim_parcel={r0['stim_parcel']} "
              f"({r0['site_name']}) | region_tep {r0['region_tep'].shape}")
