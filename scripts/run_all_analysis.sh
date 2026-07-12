#!/usr/bin/env bash
# Regenerate every Exp-1B analysis artifact from whatever folds/checkpoints are on disk.
# Safe to run mid-run (each step degrades gracefully if data is partial).
#   PYBIN=../.venv/bin/python bash scripts/run_all_analysis.sh
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
PYBIN="${PYBIN:-$ROOT/../.venv/bin/python}"
export PYTHONWARNINGS=ignore

echo "== analyze_exp1b (tables + paired stats) =="
"$PYBIN" scripts/analyze_exp1b.py
echo "== baselines on the LOSO split =="
"$PYBIN" experiments/exp3_baselines_exp1b.py
echo "== graph diagnostics (checkpoints) =="
"$PYBIN" scripts/diagnose_graph.py
echo "== anatomy skeleton (Exp-2) =="
"$PYBIN" experiments/exp2_anatomy.py
echo "== figures =="
"$PYBIN" scripts/make_exp1b_figures.py
echo "== data + zenodo audits =="
"$PYBIN" scripts/audit_existing_data_caches.py
"$PYBIN" scripts/zenodo_label_audit.py
echo "All analysis artifacts written to reports/."
