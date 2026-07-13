# An individualized propagation operator predicts cortical responses to stimulation

Predict a held-out stimulation site's brainwide evoked response topography from the same
patient's responses at other sites using a subject-specific **propagation operator**, which is a linear
heat-kernel rollout on the patient's measured connectome that separates spatial locality (`t=0`)
from network propagation (`t>0`). Evaluated on 93 patients across 5 public CCEP/iEEG datasets by
leave-one-stim-site-out. The operator reaches r ≈ 0.71 and beats a distance baseline in 93/93 patients.

**Full methods, results, figures, and discussion are in the paper** (Puli, Mandal & Eckstein,
under review at *Brain Stimulation*). This README covers only project structure and reproduction.

## Reproduce

```bash
# environment (Python 3.9)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# fetch and build the CCEP cache with 93 patients from OpenNeuro S3 (tens of GB)
bash scripts/reproduce.sh fetch      # -> data/processed/<ds>/sub-*.npz

# reproduce every headline number from the cache (no raw data needed after fetch)
bash scripts/reproduce.sh results
```

The caches per subject (`data/processed/<dataset>/sub-*.npz`, tens of MB total) are the only
inputs the analysis needs. If you already have them, skip straight to `reproduce.sh results`.

Run any single analysis with `.venv/bin/python experiments/<name>.py`:

| script | what it shows |
|---|---|
| `ccep_loso.py` | main LOSO: within_mean / distance / stim_knn / operator / combo |
| `ccep_operator_v2.py` | operator beats distance alone (core modeling result) |
| `ccep_classD.py` | group and individual ensemble (best overall) |
| `ccep_directed.py` | operator is directed and not symmetric locality |
| `ccep_n2.py` | network gain is larger for the polysynaptic N2 than N1 |
| `ccep_latency.py` | conduction law (~3 mm/ms) and predictable response timing |
| `ccep_highgamma.py` | high-gamma readout is equally predictable |
| `ccep_control.py` | controllability ranks stimulation sites by network reach |
| `ccep_soz.py` | seizure-onset-zone localization from operator features |
| `ccep_step2.py` | responder detection (AUC) and few-shot calibration |
| `ccep_ood.py` | leave-one-dataset-out generalization (deployment proxy) |
| `ccep_ftract.py` | F-TRACT atlas (780 pt) and DWI structural connectome |
| `ccep_animal.py` | sites-vs-trials identifiability (DANDI animal microstim) |
| `ccep_lds.py` | linear dynamical system per patient (dynamic mode decomp.) |
| `ccep_diagnostic.py` | stratified noise ceiling by distance |
| `ccep_trials_ablation.py` | trials → r saturation |
| `ccep_simulation.py` | ground-truth simulation indicates regime needed for r > 0.9 |

## Layout

```
src/data/ccep_pipeline.py   # MEF3 (pymef) and BrainVision (mne) -> N1 topography cache
src/model/es_readout.py     # trained operator variants (ESReadout / ESReadout2, torch)
src/eval/stats.py           # bootstrap CI, exact sign-flip permutation, paired Cohen's d
scripts/build_ccep.py       # build one dataset's patients into data/processed/
scripts/reproduce.sh        # single fetch command and full results
experiments/ccep_*.py       # all analyses (see table above)
reports/make_fig*.py        # manuscript figure generation with matplotlib
tests/                      # pytest suite incl. test_ccep_operator.py
```

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## Data (all public)

- **CCEP** (OpenNeuro, CC0): ds004774, ds004696, ds004457, ds003708, ds004080.
- **External validation:** F-TRACT axonal-delay atlas
  ([Zenodo](https://doi.org/10.5281/zenodo.7015415)); group-average DWI structural connectome
  (Glasser-360); DANDI:000774 (animal microstimulation).
- **Abandoned modality probes:** ds005498 (TMS-fMRI), ds002799 (es-fMRI).

## Citation

Puli R, Mandal R, Eckstein M. *An individualized propagation operator predicts cortical responses
to stimulation.* Under review (*Brain Stimulation*); a preliminary subset appears at IEEE EMBS BHI.
