#!/usr/bin/env bash
# Selectively fetch ds002799 (es-fMRI) — only the fMRIPrep MNI preproc BOLD + confounds we use,
# plus the tiny es events / ieeg electrode+channel TSVs. Skips raw BOLD (huge) and surface giftis.
# Usage:  bash scripts/fetch_ds002799.sh 357 331 334 335     (bare subject numbers)
set -euo pipefail
DEST="REDACTED/Open Neuro ds002799"
S="s3://openneuro.org/ds002799"
MNI="space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"

for n in "$@"; do
  sub="sub-$n"
  echo "[fetch] $sub ..."
  # fMRIPrep derivatives: es (post-op) + rest (pre-op) preproc BOLD in MNI + confounds
  aws s3 cp --no-sign-request --recursive "$S/derivatives/fmriprep/$sub/" \
      "$DEST/derivatives/fmriprep/$sub/" \
      --exclude "*" \
      --include "*task-es*${MNI}" \
      --include "*task-es*desc-confounds_regressors.tsv" \
      --include "*task-rest*${MNI}" \
      --include "*task-rest*desc-confounds_regressors.tsv" --only-show-errors
  # raw: es events (stim block timing) + ieeg (electrode MNI coords + stimulated channels)
  aws s3 cp --no-sign-request --recursive "$S/$sub/ses-postop/func/" \
      "$DEST/$sub/ses-postop/func/" --exclude "*" --include "*task-es*events.tsv" --only-show-errors
  aws s3 cp --no-sign-request --recursive "$S/$sub/ses-postop/ieeg/" \
      "$DEST/$sub/ses-postop/ieeg/" --only-show-errors
  echo "[fetch] $sub done"
done
echo "[fetch] all done -> $DEST"
