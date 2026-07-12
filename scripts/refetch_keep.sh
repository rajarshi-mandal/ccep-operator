#!/usr/bin/env bash
# Full raw download, KEEPING raw on disk (for developing raw-dependent extensions: latency,
# full waveform, spectral features, etc.). Unlike refetch_all.sh, does NOT delete raw.
set -uo pipefail
cd "REDACTED"
P=".venv/bin/python"
ts() { date +%H:%M:%S; }
echo "[$(ts)] START full raw download (KEEP raw)"

$P causal-dag-ssm/scripts/build_ccep.py ds004774 sub-MAYO01 sub-MAYO02 sub-MAYO03 sub-MAYO04 sub-MAYO05
echo "[$(ts)] ds004774 done ($(du -sh 'Open Neuro ds004774' 2>/dev/null | awk '{print $1}'))"
$P causal-dag-ssm/scripts/build_ccep.py ds004696 sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08
echo "[$(ts)] ds004696 done ($(du -sh 'Open Neuro ds004696' 2>/dev/null | awk '{print $1}'))"
$P causal-dag-ssm/scripts/build_ccep.py ds004457 sub-1 sub-2 sub-3 sub-4 sub-5
echo "[$(ts)] ds004457 done ($(du -sh 'Open Neuro ds004457' 2>/dev/null | awk '{print $1}'))"
$P causal-dag-ssm/scripts/build_ccep.py ds003708 sub-01
echo "[$(ts)] ds003708 done ($(du -sh 'Open Neuro ds003708' 2>/dev/null | awk '{print $1}'))"
$P causal-dag-ssm/scripts/fetch_ds004080_subset.py 74
echo "[$(ts)] ds004080 done ($(du -sh 'Open Neuro ds004080' 2>/dev/null | awk '{print $1}'))"

echo "[$(ts)] === RAW FOOTPRINT ==="; du -sh "Open Neuro"* 2>/dev/null
echo "[$(ts)] total raw: $(du -ch 'Open Neuro'* 2>/dev/null | tail -1 | awk '{print $1}')"
echo "[$(ts)] ALL RAW DOWNLOADED"
