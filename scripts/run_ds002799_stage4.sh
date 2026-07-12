#!/usr/bin/env bash
set -uo pipefail
cd "REDACTED/causal-dag-ssm"; PY=../.venv/bin/python
for i in $(seq 1 600); do grep -q "all done" /tmp/fetch_ds002799_c.log 2>/dev/null && break; sleep 15; done
sleep 5
echo "=== BUILD (all 20) ==="; $PY scripts/build_ds002799.py 2>&1 | grep -vE "fetch_atlas|NotOpenSSL|warnings.warn"
echo; echo "=== GATE ==="; $PY scripts/validate_ds002799.py --cache-dir data/processed/ds002799 2>&1 | grep -vE "fetch_atlas|NotOpenSSL|warnings.warn" | tail -7
echo; echo "=== STAGE-4 READOUTS (site-specificity) ==="; $PY experiments/phase4_es_readouts.py --cache-dir data/processed/ds002799 2>&1 | grep -vE "fetch_atlas|NotOpenSSL|warnings.warn" | tail -22
echo "=== DONE ==="
