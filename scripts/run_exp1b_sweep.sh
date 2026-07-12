#!/usr/bin/env bash
# Controlled Exp-1B hyperparameter sweep over the LOSO M1 split.
#
# Grid (small, defensible — not a search, a robustness check):
#   lambda_int : 3 10 30      (weight on the interventional do-loss vs L_obs)
#   outer      : 4 6          (augmented-Lagrangian outer iterations)
#   inner      : 100 150      (inner steps per outer iter)
#   seed       : 0 1 2        (re-init / data-order variation)
#
# Each cell writes its own results JSON + per-fold dir under data/processed/sweep/<tag>/.
# Designed to run AFTER the headline full run completes (it is CPU-heavy); launch with:
#   nohup bash scripts/run_exp1b_sweep.sh > /tmp/exp1b_sweep.log 2>&1 &
# then aggregate with scripts/aggregate_sweep.py.
#
# Honors $PYBIN (path to the venv python). Defaults to the repo .venv.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
PYBIN="${PYBIN:-$ROOT/../.venv/bin/python}"
SWEEP_DIR="data/processed/sweep"
mkdir -p "$SWEEP_DIR"

LAMBDAS="${LAMBDAS:-3 10 30}"
OUTERS="${OUTERS:-4 6}"
INNERS="${INNERS:-100 150}"
SEEDS="${SEEDS:-0 1 2}"
# Optional: restrict folds for a fast sweep, e.g. SWEEP_FOLDS=4 -> folds [0,4)
FOLD_ARGS=""
if [ -n "${SWEEP_FOLDS:-}" ]; then FOLD_ARGS="--end-fold ${SWEEP_FOLDS}"; fi

echo "Exp-1B sweep | python=$PYBIN | lambdas=[$LAMBDAS] outers=[$OUTERS] inners=[$INNERS] seeds=[$SEEDS] $FOLD_ARGS"
n=0
for lam in $LAMBDAS; do
  for outer in $OUTERS; do
    for inner in $INNERS; do
      for seed in $SEEDS; do
        tag="lam${lam}_o${outer}_i${inner}_s${seed}"
        cell="$SWEEP_DIR/$tag"
        mkdir -p "$cell"
        out="$cell/exp1b_results.json"
        if [ -f "$out" ]; then
          echo "[skip] $tag (exists)"; continue
        fi
        n=$((n+1))
        echo "=== [$tag] lambda=$lam outer=$outer inner=$inner seed=$seed ==="
        PYTHONWARNINGS=ignore "$PYBIN" -u experiments/exp1b_held_out_subject.py \
          --inner "$inner" --outer "$outer" --lambda-int "$lam" --seed "$seed" \
          $FOLD_ARGS --out "$out" \
          2>&1 | sed "s/^/[$tag] /"
        # move the incremental fold dir into the cell so cells don't collide
        if [ -d data/processed/exp1b_folds ]; then
          mv data/processed/exp1b_folds "$cell/exp1b_folds" 2>/dev/null || true
        fi
      done
    done
  done
done
echo "Sweep complete. Cells run this invocation: $n. Aggregate: $PYBIN scripts/aggregate_sweep.py"
