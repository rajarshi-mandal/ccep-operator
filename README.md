# An individualized propagation operator predicts cortical responses to stimulation

Predict a **held-out stimulation site's** brain-wide evoked-response topography from the **same
patient's** responses at other sites, using a subject-specific **propagation operator**: a linear
heat-kernel rollout on the patient's measured connectome that cleanly separates a spatial-locality
term (`t=0`) from network propagation (`t>0`). Place a locality seed at the stimulated site,
propagate it through the operator, read out per-contact N1 response energy.

Evaluated on **93 patients across 5 public CCEP/iEEG datasets and 2 recording formats** (MEF3 +
BrainVision) by leave-one-stim-site-out within patient, with every hyperparameter chosen by nested
inner-LOO on training sites only (leakage-aware). The same operator is then reused for network
control, seizure-onset-zone localization, conduction timing, and external validation.

## Headline result (n = 93, leave-one-stim-site-out)

| model | held-out r | note |
|---|---|---|
| `within_mean` (baseline) | 0.235 | the bar to beat |
| `distance` (locality kernel) | 0.641 | strong spatial baseline |
| **`operator`** (this work) | **0.710** | amplitude-preserving propagation; beats `distance` **93/93** |
| `combo` (locality + network residual) | 0.730 | best interpretable blend |
| `ensemble` (group + individual) | 0.743 | best overall |
| `row_norm` (negative control) | 0.622 | below `distance` — amplitude preservation is what helps |

- `operator` beats `distance` by **+0.070 in 93/93 patients** (d = 0.79, p < 1e-4); the propagation
  term (`α, T > 0`) is actively selected in every fold.
- `combo` beats `within_mean` by **+0.495 in 92/93 patients** (d = 3.78).
- Controllability ranking predicts a site's untested network reach (ρ = 0.58, **93/93**).
- Operator features localize the seizure-onset zone beyond amplitude + geometry (AUC 0.61 vs 0.56).
- Responder detection **AUC 0.85**; few-shot calibration **0.90**.
- Conduction law **≈ 3.0 mm/ms**, replicated in the F-TRACT atlas (3.14 mm/ms, 780 patients).
- Convergent negatives: the transferable structure is geometric — individualization needs the
  patient's own stimulation.

## Quick start (full reproduction from raw data)

```bash
# 1. environment (Python 3.9)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. fetch + build the 93-patient CCEP cache from OpenNeuro S3 (selective; ~tens of GB)
bash scripts/reproduce.sh fetch        # downloads + builds data/processed/<ds>/sub-*.npz

# 3. reproduce every headline number (uses the cached .npz; no raw data needed after step 2)
bash scripts/reproduce.sh results      # main LOSO, operator, mechanism, control, clinical, ceiling
```

If you already have the per-subject caches in `data/processed/<dataset>/sub-*.npz`, skip to
step 3. Caches are small (~tens of MB total) and are the only inputs the analysis needs.

## Reproduce a single result

```bash
P=.venv/bin/python
# --- core prediction ---
$P experiments/ccep_loso.py            # main LOSO: within_mean / distance / stim_knn / operator / combo
$P experiments/ccep_operator_v2.py     # the operator beats distance alone (core modeling result)
$P experiments/ccep_classD.py          # group+individual ensemble (best overall)
$P experiments/ccep_directed.py        # operator is genuinely directed, not just symmetric locality
# --- mechanism ---
$P experiments/ccep_n2.py              # network gain is larger for the polysynaptic N2 than the N1
$P experiments/ccep_latency.py         # conduction law (~3 mm/ms) + predictable response timing
$P experiments/ccep_highgamma.py       # high-gamma readout is equally predictable
# --- control & clinical ---
$P experiments/ccep_control.py         # controllability ranks stimulation sites by network reach
$P experiments/ccep_soz.py             # seizure-onset-zone localization from operator features
$P experiments/ccep_step2.py           # responder detection (AUC) + few-shot calibration
# --- generalization & external validation ---
$P experiments/ccep_ood.py             # leave-one-DATASET-out generalization (deployment proxy)
$P experiments/ccep_ftract.py          # F-TRACT atlas (780 pt) + DWI structural connectome
$P experiments/ccep_animal.py          # sites-vs-trials identifiability (DANDI animal microstim)
$P experiments/ccep_lds.py             # per-patient linear dynamical system (dynamic mode decomp.)
# --- ceiling / diagnostics ---
$P experiments/ccep_diagnostic.py      # distance-stratified noise ceiling
$P experiments/ccep_trials_ablation.py # trials -> r saturation
$P experiments/ccep_simulation.py      # ground-truth simulation: regime needed for r > 0.9
```

## Layout

```
src/data/ccep_pipeline.py       # MEF3 (pymef) + BrainVision (mne) -> N1 topography cache
src/model/es_readout.py         # trained operator variants (ESReadout / ESReadout2, torch)
src/eval/stats.py               # bootstrap CI, exact sign-flip permutation, paired Cohen's d
scripts/build_ccep.py           # build one dataset's patients into data/processed/
scripts/reproduce.sh            # one-command fetch + full results
experiments/ccep_*.py           # all analyses (see "Reproduce a single result")
reports/make_fig*.py            # manuscript figure generation (matplotlib)
tests/                          # pytest suite (incl. test_ccep_operator.py)
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## Project arc & legacy code

CCEP is the third modality tried; the first two are documented dead-ends, kept because they
pinpointed **sites-per-patient** (not trials) as the binding constraint — later confirmed by the
animal data — and motivated the pivot to CCEP:

- **ds005498 TMS-fMRI** — the raw-BOLD "evoked" signal was stimulus artifact, not neural. Negative.
- **ds002799 es-fMRI** — real localized HRF signal but ≤ ~7 stim sites/patient, so the operator
  was unestimable.
- **CCEP** — dozens-to-hundreds of sites/patient make the operator learnable. The working path.

The original TMS-EEG causal-DAG-SSM (`experiments/exp1b_*.py`, `src/model/causal_dag_ssm.py`) is
retained for completeness.

## Data availability (all public)

- **CCEP** (OpenNeuro, CC0): ds004774 (ER-Detect / Mayo), ds004696 (HAPwave), ds004457, ds003708,
  ds004080 (ccepAge / van Blooijs).
- **External validation:** F-TRACT axonal-delay atlas (Zenodo 10.5281/zenodo.7015415); a
  group-average DWI structural connectome (Glasser-360); DANDI:000774 (animal microstimulation).
- **Abandoned modality probes:** OpenNeuro ds005498 (TMS-fMRI), ds002799 (es-fMRI).

## Citation

Puli R, Mandal R, Eckstein M. *An individualized propagation operator predicts cortical responses
to stimulation.* Under review (Brain Stimulation); a preliminary subset appears at IEEE EMBS BHI.