#!/usr/bin/env bash
# Launch the headline Exp-1B 13-fold LOSO run (incremental + resumable).
#   PYBIN=../.venv/bin/python bash scripts/run_full_exp1b.sh
# Re-running with the same args resumes (skips folds already on disk).
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
PYBIN="${PYBIN:-$ROOT/../.venv/bin/python}"
export PYTHONWARNINGS=ignore

INNER="${INNER:-150}"; OUTER="${OUTER:-6}"; LAMBDA="${LAMBDA:-10}"
echo "Exp-1B full run | inner=$INNER outer=$OUTER lambda_int=$LAMBDA | python=$PYBIN"
exec "$PYBIN" -u experiments/exp1b_held_out_subject.py \
  --inner "$INNER" --outer "$OUTER" --lambda-int "$LAMBDA" \
  --save-ckpt --resume --out data/processed/exp1b_results.json
