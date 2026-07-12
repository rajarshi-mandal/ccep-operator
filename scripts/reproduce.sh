#!/usr/bin/env bash
# One-command reproduction of the CCEP headline results.
#
#   bash scripts/reproduce.sh fetch     # download + build the 93-subject cache from OpenNeuro S3
#   bash scripts/reproduce.sh results   # run every headline analysis on the cache (no raw data)
#   bash scripts/reproduce.sh all       # fetch then results
#   bash scripts/reproduce.sh test      # pytest suite
#
# Requires: pip install -r requirements.txt   (Python 3.9). `fetch` also needs the AWS CLI
# (`aws`, used with --no-sign-request; no credentials required for the public OpenNeuro bucket).
set -euo pipefail
cd "$(dirname "$0")/.."
P="${PYBIN:-.venv/bin/python}"
[ -x "$P" ] || P="../.venv/bin/python"
[ -x "$P" ] || P="python"
echo "using python: $P ($($P --version 2>&1))"

fetch() {
  echo "== fetch + build CCEP caches =="
  $P scripts/build_ccep.py ds004774 sub-MAYO01 sub-MAYO02 sub-MAYO03 sub-MAYO04 sub-MAYO05
  $P scripts/build_ccep.py ds004696 sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08
  $P scripts/build_ccep.py ds004457 sub-1 sub-2 sub-3 sub-4 sub-5
  $P scripts/build_ccep.py ds003708 sub-01
  $P scripts/fetch_ds004080_subset.py 74        # ccepAge, largest-run-per-subject selective fetch
  echo "caches in data/processed/<dataset>/sub-*.npz"
}

results() {
  echo "== reproduce headline results (-> reports/) =="
  mkdir -p reports
  $P experiments/ccep_loso.py            | tee reports/_repro_loso.txt
  $P experiments/ccep_operator_v2.py     | tee reports/_repro_operator_v2.txt
  $P experiments/ccep_classD.py          | tee reports/_repro_ensemble.txt
  $P experiments/ccep_trained.py         | tee reports/_repro_directionality.txt
  $P experiments/ccep_diagnostic.py      | tee reports/_repro_ceiling.txt
  $P experiments/ccep_trials_ablation.py | tee reports/_repro_trials.txt
  $P experiments/ccep_step2.py           | tee reports/_repro_clinical.txt
  [ -f experiments/ccep_ood.py ] && $P experiments/ccep_ood.py | tee reports/_repro_ood.txt || true
  echo "done. compare to reports/RESULTS_DS004774.md"
}

case "${1:-all}" in
  fetch)   fetch ;;
  results) results ;;
  test)    $P -m pytest tests/ -q ;;
  all)     fetch; results ;;
  *) echo "usage: $0 {fetch|results|all|test}"; exit 1 ;;
esac
