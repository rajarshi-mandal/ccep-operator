#!/usr/bin/env bash
# Storage-smart full data recovery: fetch raw from OpenNeuro -> build .npz caches -> delete raw,
# one dataset at a time. Final footprint = ~tens of MB of caches; peak = one dataset's raw.
set -uo pipefail
cd "REDACTED"
P=".venv/bin/python"
ts() { date +%H:%M:%S; }
echo "[$(ts)] START full recovery"

build_del() {  # <dataset> <subs...>
  local ds="$1"; shift
  echo "[$(ts)] === $ds : fetch+build ${*} ==="
  $P causal-dag-ssm/scripts/build_ccep.py "$ds" "$@"
  local n=$(ls causal-dag-ssm/data/processed/"$ds"/sub-*.npz 2>/dev/null | wc -l | tr -d ' ')
  echo "[$(ts)] $ds built $n caches; deleting raw"
  rm -rf "Open Neuro $ds"
}

build_del ds004774 sub-MAYO01 sub-MAYO02 sub-MAYO03 sub-MAYO04 sub-MAYO05
build_del ds004696 sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08
build_del ds004457 sub-1 sub-2 sub-3 sub-4 sub-5
build_del ds003708 sub-01

# ccepAge: dedicated subset fetcher (download only) -> then build -> delete raw
echo "[$(ts)] === ds004080 : subset fetch (largest run/subject) ==="
$P causal-dag-ssm/scripts/fetch_ds004080_subset.py 74
subs=$(ls "Open Neuro ds004080" 2>/dev/null | grep '^sub-' | tr '\n' ' ')
echo "[$(ts)] ds004080 fetched subjects: $(echo $subs | wc -w | tr -d ' ')"
$P causal-dag-ssm/scripts/build_ccep.py ds004080 $subs
n80=$(ls causal-dag-ssm/data/processed/ds004080/sub-*.npz 2>/dev/null | wc -l | tr -d ' ')
echo "[$(ts)] ds004080 built $n80 caches; deleting raw"
rm -rf "Open Neuro ds004080"

TOTAL=$(find causal-dag-ssm/data/processed -name 'sub-*.npz' 2>/dev/null | wc -l | tr -d ' ')
FOOT=$(du -sh causal-dag-ssm/data/processed 2>/dev/null | awk '{print $1}')
echo "[$(ts)] ALL DONE — $TOTAL caches, footprint $FOOT"
