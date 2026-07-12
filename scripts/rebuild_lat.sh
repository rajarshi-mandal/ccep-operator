#!/usr/bin/env bash
# Rebuild all caches WITH latency (raw already on disk -> no download, just re-epoch).
set -uo pipefail
cd "REDACTED"
P=".venv/bin/python"
ts() { date +%H:%M:%S; }
echo "[$(ts)] REBUILD with latency START"
$P causal-dag-ssm/scripts/build_ccep.py ds004774 sub-MAYO01 sub-MAYO02 sub-MAYO03 sub-MAYO04 sub-MAYO05
$P causal-dag-ssm/scripts/build_ccep.py ds004696 sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08
$P causal-dag-ssm/scripts/build_ccep.py ds004457 sub-1 sub-2 sub-3 sub-4 sub-5
$P causal-dag-ssm/scripts/build_ccep.py ds003708 sub-01
echo "[$(ts)] MEF3 done; ds004080 (74 subj, slow re-epoch) ..."
subs=$(ls "Open Neuro ds004080" 2>/dev/null | grep '^sub-' | tr '\n' ' ')
$P causal-dag-ssm/scripts/build_ccep.py ds004080 $subs
echo "[$(ts)] REBUILD with latency DONE — $(find causal-dag-ssm/data/processed -name 'sub-*.npz' | wc -l | tr -d ' ') caches"
