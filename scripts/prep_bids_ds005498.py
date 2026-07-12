"""Stage-0 prep — make ds005498 fMRIPrep-ready (dry-run by default; --apply to write).

Gaps found (see reports/FMRIPREP_SETUP.md):
  * no README at the BIDS root (fMRIPrep/BIDS warns)
  * func BOLD jsons lack SliceTiming (only SliceOrder=sequential) -> slice-timing correction needs it
  * func BOLD jsons lack PhaseEncodingDirection (no fieldmaps) -> audited; SyN-SDC used instead

SliceTiming is derived from SliceOrder=sequential (ASCENDING assumed) + TR + slice count:
  slice z acquired at  t_z = z * TR / n_slices   for z = 0..n_slices-1.
Originals are backed up to <file>.json.bak so the change is reversible.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

DS = Path("REDACTED/Open Neuro ds005498")
README = """# Single-pulse TMS-fMRI (ds005498)

Concurrent single-pulse TMS-fMRI, 11 stimulation sites (task label = coil MNI coordinate) plus a
resting run per subject. See dataset_description.json for authors/citation.

Note: SliceTiming sidecars were added by scripts/prep_bids_ds005498.py (derived from
SliceOrder=sequential, ascending). No fieldmaps are present; fMRIPrep uses fieldmap-less SyN-SDC.
"""


def slice_timing(n_slices: int, tr: float) -> list[float]:
    """Sequential-ascending acquisition times within one TR."""
    return [round(z * tr / n_slices, 4) for z in range(n_slices)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(DS))
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()
    ds = Path(args.ds)
    act = "WRITE" if args.apply else "dry-run"
    print(f"[prep] {act} on {ds}")

    # 1. README
    readme = ds / "README"
    if readme.exists():
        print("  README: present")
    else:
        print(f"  README: MISSING -> {act}")
        if args.apply:
            readme.write_text(README)

    # 2 + 3. func BOLD jsons: SliceTiming + PhaseEncodingDirection audit
    bolds = sorted(glob.glob(str(ds / "sub-*/ses-*/func/*_bold.json")))
    n_st_added = n_have_st = n_no_pe = 0
    import nibabel as nib
    for jp in bolds:
        d = json.loads(Path(jp).read_text())
        if "PhaseEncodingDirection" not in d:
            n_no_pe += 1
        if "SliceTiming" in d:
            n_have_st += 1
            continue
        # need slice count from the matching nii
        nii = jp.replace("_bold.json", "_bold.nii.gz")
        if not Path(nii).exists():
            nii = jp.replace("_bold.json", "_bold.nii")
        try:
            n_slices = nib.load(nii).shape[2]
        except Exception:
            continue
        tr = float(d.get("RepetitionTime", 0))
        if tr <= 0:
            continue
        st = slice_timing(n_slices, tr)
        n_st_added += 1
        if args.apply:
            shutil.copyfile(jp, jp + ".bak")
            d["SliceTiming"] = st
            Path(jp).write_text(json.dumps(d, indent=4))

    print(f"  func jsons: {len(bolds)} total | SliceTiming already present {n_have_st} | "
          f"SliceTiming {'written' if args.apply else 'to write'} {n_st_added}")
    print(f"  PhaseEncodingDirection MISSING in {n_no_pe} func jsons -> "
          f"fMRIPrep will use fieldmap-less SyN-SDC (--use-syn-sdc).")
    if not args.apply:
        print("\n  (dry run — re-run with --apply to write README + SliceTiming sidecars)")


if __name__ == "__main__":
    main()
