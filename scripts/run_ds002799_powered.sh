#!/usr/bin/env bash
# Wait for the ds002799 download to finish, build the cache, then run the GATE + within-subject
# LOSO model. Robust wait on the fetch log marker (PID waits were flaky).
set -uo pipefail
cd "REDACTED/causal-dag-ssm"
PY=../.venv/bin/python
LOG=/tmp/fetch_ds002799_b.log

for i in $(seq 1 480); do grep -q "all done" "$LOG" 2>/dev/null && break; sleep 15; done
sleep 5
echo "=== BUILD ==="
$PY scripts/build_ds002799.py 2>&1 | grep -vE "fetch_atlas|NotOpenSSL|warnings.warn"
echo; echo "=== GATE ==="
$PY scripts/validate_ds002799.py --cache-dir data/processed/ds002799 2>&1 | grep -vE "fetch_atlas|NotOpenSSL|warnings.warn" | tail -8
echo; echo "=== WITHIN-SUBJECT LOSO MODEL ==="
$PY experiments/phase2_loso_ws.py --cache-dir data/processed/ds002799 \
    --out data/processed/ds002799_phase2.json --report reports/PHASE2_LOSO_WS_ds002799.md 2>&1 \
    | grep -vE "fetch_atlas|NotOpenSSL|warnings.warn" | tail -24
echo "=== DONE ==="
