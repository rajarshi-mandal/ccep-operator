#!/usr/bin/env bash
# Stage-0 — run fMRIPrep on ds005498 via the fmriprep-docker wrapper.
# Usage:  bash scripts/run_fmriprep.sh NTHC1035 NTHC1036 ...   (subject labels, no "sub-" prefix)
# Prereqs: Docker Desktop running, `pip install fmriprep-docker`, FreeSurfer license. See
#          reports/FMRIPREP_SETUP.md. Run scripts/prep_bids_ds005498.py --apply first.
set -euo pipefail

BIDS="REDACTED/Open Neuro ds005498"
OUT="${BIDS}/derivatives/fmriprep"
WORK="/tmp/fmriprep_work"
FS_LICENSE="${FS_LICENSE:-$HOME/freesurfer_license.txt}"

if [ "$#" -lt 1 ]; then echo "usage: $0 <subject-label> [more labels...]"; exit 1; fi
if ! command -v fmriprep-docker >/dev/null 2>&1; then
  echo "ERROR: fmriprep-docker not found. Install with:"
  echo '  "REDACTED/.venv/bin/pip" install fmriprep-docker'; exit 1; fi
if ! docker info >/dev/null 2>&1; then echo "ERROR: Docker is not running. Launch Docker Desktop."; exit 1; fi
if [ ! -f "$FS_LICENSE" ]; then echo "ERROR: FreeSurfer license not at $FS_LICENSE (set FS_LICENSE=...)"; exit 1; fi

mkdir -p "$OUT" "$WORK"
echo "[fmriprep] subjects: $*"
echo "[fmriprep] out: $OUT"

# --output-spaces MNI152NLin6Asym:res-2  == the FSL-MNI space the Schaefer-100 atlas lives in.
# --use-syn-sdc warn  == fieldmap-less distortion correction (dataset has no fieldmaps).
fmriprep-docker \
  "$BIDS" "$OUT" participant \
  --participant-label "$@" \
  --fs-license-file "$FS_LICENSE" \
  --output-spaces MNI152NLin6Asym:res-2 \
  --use-syn-sdc warn \
  --nthreads 8 --omp-nthreads 4 --mem-mb 16000 \
  -w "$WORK" \
  --notrack --stop-on-first-crash

echo "[fmriprep] done -> $OUT"
echo "Next: ../.venv/bin/python scripts/gate0_check.py \"$OUT\""
