#!/usr/bin/env bash
set -uo pipefail
cd "REDACTED/causal-dag-ssm"; PY=../.venv/bin/python
for i in $(seq 1 200); do [ -f data/processed/ds002799_d200/manifest.json ] && break; sleep 10; done
sleep 5
echo "=== d200 build summary ==="; grep -vE "fetch_atlas|NotOpenSSL|warnings.warn" /tmp/d200_build.log 2>/dev/null | tail -3
echo; echo "=== PHASE 6 — d=100 (full, 6 seeds) ==="
$PY experiments/phase6_enhanced.py --cache-dir data/processed/ds002799 --seeds 6 --epochs 300 \
   --out data/processed/ds002799_phase6_d100.json 2>&1 | grep -vE "fetch_atlas|NotOpenSSL|warnings.warn" | tail -16
echo; echo "=== PHASE 6 — d=200 (full, 6 seeds, FINER PARCELLATION #4) ==="
$PY experiments/phase6_enhanced.py --cache-dir data/processed/ds002799_d200 --seeds 6 --epochs 300 \
   --out data/processed/ds002799_phase6_d200.json 2>&1 | grep -vE "fetch_atlas|NotOpenSSL|warnings.warn" | tail -16
echo "=== DONE ==="
