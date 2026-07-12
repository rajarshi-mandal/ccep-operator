#!/usr/bin/env python
"""Build per-subject CCEP caches for a BIDS-iEEG CCEP dataset (ds004774, ds004696, ...).

Downloads (if missing) each subject's MEF3/BrainVision signal + sidecars from OpenNeuro S3,
then builds the cache via src/data/ccep_pipeline.py.

Usage:
  ../.venv/bin/python scripts/build_ccep.py [ds004774|ds004696] sub-X sub-Y ...
  ../.venv/bin/python scripts/build_ccep.py            # default: ds004774 small set
The first arg, if it matches dsNNNNNN, selects the dataset; remaining args are subjects.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from data.ccep_pipeline import build_subject  # noqa: E402

DEFAULT = ["sub-MAYO01", "sub-MAYO02", "sub-MAYO05", "sub-UMCU59"]


def fetch(dataset: str, dataset_root: Path, sub: str):
    """Selective S3 pull of a subject's ieeg dir (skips if already present)."""
    dest = dataset_root / sub
    if dest.exists() and any(dest.rglob("*_events.tsv")):
        if any(dest.rglob("*_ieeg.mefd")) or any(dest.rglob("*_ieeg.eeg")):
            print(f"  [{sub}] already downloaded")
            return
    print(f"  [{sub}] fetching from S3 ...")
    cmd = ["aws", "s3", "cp", "--no-sign-request", "--recursive",
           f"s3://openneuro.org/{dataset}/{sub}/", str(dest) + "/"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    args = sys.argv[1:]
    dataset = "ds004774"
    robust = False
    if "--robust" in args:
        args.remove("--robust"); robust = True
    if args and re.fullmatch(r"ds\d{6}", args[0]):
        dataset = args.pop(0)
    subs = args or DEFAULT
    dataset_root = PROJ / f"Open Neuro {dataset}"
    suffix = "_robust" if robust else ""
    CACHE_DIR = ROOT / "data" / "processed" / (dataset + suffix)
    os.makedirs(CACHE_DIR, exist_ok=True)
    for sub in subs:
        fetch(dataset, dataset_root, sub)
        print(f"[build] {dataset} {sub}{' (robust)' if robust else ''}")
        cs = build_subject(str(dataset_root), sub, robust=robust)
        # QC summary
        import numpy as np
        rel = cs.reliability[np.isfinite(cs.reliability)]
        print(f"  -> {len(cs.sites)} sites x {len(cs.contacts)} contacts | "
              f"reliability median={np.nanmedian(rel):.2f} "
              f"(>{0.5}: {int((rel>0.5).sum())}/{len(rel)}) | "
              f"trials/site median={int(np.median(cs.n_trials))}")
        out = CACHE_DIR / f"{sub}.npz"
        cs.save(str(out))
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
