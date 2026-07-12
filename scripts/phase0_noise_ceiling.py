"""Phase 0 - Noise ceiling for ds005498 concurrent single-pulse TMS-fMRI.

Goal: estimate the maximum achievable correlation (r) for predicting a
subject x site TMS-evoked response. This is the split-half reliability of the
evoked response: if the data only correlates with itself at r_ceiling, no model
can beat r_ceiling on held-out data. This number defines what "maximize r" means
before we build the full model.

Method (per stim run, registration-free, valid for a reliability bound):
  1. Load native-space BOLD; brain-mask via nilearn compute_epi_mask.
  2. Reduce voxels -> 100 spatial parcels via KMeans on voxel coordinates
     (deterministic, matches the model's d=100 latent granularity; odd/even
     share identical parcels so the comparison is fair).
  3. Split the ~68 single pulses into odd / even interleaved sets.
  4. Fit one GLM with two stimulus regressors (odd, even) + polynomial drift.
       - Canonical (Glover) HRF amplitude  -> SPATIAL ceiling (beta topography).
       - FIR (7 post-stim bins)            -> SPATIOTEMPORAL ceiling (shape).
  5. Spatial ceiling   = pearson(beta_odd, beta_even) over 100 parcels.
     Spatiotemporal    = pearson over flattened [100 x 7] FIR responses.
  6. Spearman-Brown correct (split-half halves the data): r_full = 2r/(1+r).

Aggregate the per-run ceilings -> headline noise ceiling.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.masking import compute_epi_mask
from nilearn.glm.first_level import compute_regressor
from sklearn.cluster import KMeans
from scipy.stats import pearsonr

DS = Path("REDACTED/Open Neuro ds005498")
TR = 2.4
PULSE_DUR = 0.3
N_PARCELS = 100
FIR_DELAYS = list(range(0, 7))  # post-stim scans (0..6 -> ~0-14.4 s)


def spearman_brown(r):
    if r is None or not np.isfinite(r) or r <= -1:
        return np.nan
    return 2 * r / (1 + r)


def parcellate(img, n_parcels=N_PARCELS, seed=0):
    """Native-space BOLD -> [n_parcels, T] via spatial KMeans on in-mask voxels."""
    mask = compute_epi_mask(img)
    m = mask.get_fdata().astype(bool)
    data = img.get_fdata(dtype=np.float32)
    ts = data[m]                       # [Nvox, T]
    coords = np.argwhere(m).astype(np.float32)  # [Nvox, 3] (C order, matches ts)
    k = min(n_parcels, ts.shape[0])
    lab = KMeans(n_clusters=k, random_state=seed, n_init=3).fit(coords).labels_
    P = np.zeros((k, ts.shape[1]), dtype=np.float32)
    for j in range(k):
        P[j] = ts[lab == j].mean(0)
    return P                            # [k, T]


def make_reg(onsets, frame_times, hrf_model, fir_delays=None):
    cond = np.vstack([onsets, np.full_like(onsets, PULSE_DUR), np.ones_like(onsets)])
    sig, _ = compute_regressor(cond, hrf_model, frame_times,
                               fir_delays=fir_delays, oversampling=16)
    return sig                          # [T, ncols]


def ceiling_for_run(bold_path, onsets):
    img = nib.load(str(bold_path))
    T = img.shape[-1]
    ft = TR * np.arange(T)
    on = np.sort(onsets[onsets < T * TR])
    odd, even = on[0::2], on[1::2]
    if len(odd) < 3 or len(even) < 3:
        return None

    P = parcellate(img)                 # [k, T]
    Y = (P - P.mean(1, keepdims=True)).T  # [T, k], time-demeaned
    drift = np.column_stack([np.ones(T), np.linspace(-1, 1, T),
                             np.linspace(-1, 1, T) ** 2])

    out = {}
    # --- canonical HRF: spatial (amplitude topography) ceiling ---
    ro = make_reg(odd, ft, "glover")[:, 0]
    re_ = make_reg(even, ft, "glover")[:, 0]
    X = np.column_stack([ro, re_, drift])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)  # [n_reg, k]
    r_sp = pearsonr(beta[0], beta[1])[0]
    out["spatial_r"] = r_sp
    out["spatial_sb"] = spearman_brown(r_sp)

    # --- FIR: spatiotemporal (response shape) ceiling ---
    fo = make_reg(odd, ft, "fir", FIR_DELAYS)     # [T, nbin]
    fe = make_reg(even, ft, "fir", FIR_DELAYS)
    nb = fo.shape[1]
    Xf = np.column_stack([fo, fe, drift])
    bf, *_ = np.linalg.lstsq(Xf, Y, rcond=None)   # [2*nb+3, k]
    Bo = bf[:nb].T.reshape(-1)                     # [k*nb]
    Be = bf[nb:2 * nb].T.reshape(-1)
    r_st = pearsonr(Bo, Be)[0]
    out["spatiotemporal_r"] = r_st
    out["spatiotemporal_sb"] = spearman_brown(r_st)
    out["n_parcels"] = P.shape[0]
    out["n_pulses"] = int(len(on))
    return out


def collect_runs(n_subj, sites_per_subj, seed=0):
    """Pick subjects with the most sites; sample runs across them."""
    rng = np.random.default_rng(seed)
    subs = sorted([p for p in DS.glob("sub-*") if p.is_dir()])
    scored = []
    for s in subs:
        runs = list(s.glob("ses-*/func/*task-stim*_bold.nii.gz"))
        sites = {re.search(r"task-(stim[A-Za-z0-9]+)_", r.name).group(1) for r in runs}
        scored.append((len(sites), s, runs))
    scored.sort(key=lambda t: -t[0])
    chosen = []
    for _, s, runs in scored[: n_subj]:
        rng.shuffle(runs)
        chosen.extend(runs[:sites_per_subj])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subj", type=int, default=6)
    ap.add_argument("--sites-per-subj", type=int, default=5)
    ap.add_argument("--out", default="reports/phase0_noise_ceiling.json")
    args = ap.parse_args()

    events = pd.read_csv(DS / "task-stim_events.tsv", sep="\t")
    onsets = events["onset"].values.astype(float)

    runs = collect_runs(args.n_subj, args.sites_per_subj)
    print(f"[phase0] {len(runs)} runs from {args.n_subj} subjects", flush=True)
    rows = []
    for i, r in enumerate(runs):
        try:
            res = ceiling_for_run(r, onsets)
        except Exception as e:
            print(f"  [{i+1}/{len(runs)}] FAIL {r.name}: {e}", flush=True)
            continue
        if res is None:
            continue
        sub = r.parts[-4]
        site = re.search(r"task-(stim[A-Za-z0-9]+)_", r.name).group(1)
        res.update(subject=sub, site=site)
        rows.append(res)
        print(f"  [{i+1}/{len(runs)}] {sub} {site}: spatial_sb={res['spatial_sb']:.3f} "
              f"spatiotemporal_sb={res['spatiotemporal_sb']:.3f}", flush=True)

    if not rows:
        print("[phase0] no runs processed", file=sys.stderr); sys.exit(1)
    df = pd.DataFrame(rows)

    def summ(col):
        v = df[col].dropna().values
        return dict(median=float(np.median(v)), mean=float(np.mean(v)),
                    q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)),
                    min=float(v.min()), max=float(v.max()), n=int(len(v)))

    summary = {k: summ(k) for k in
               ["spatial_r", "spatial_sb", "spatiotemporal_r", "spatiotemporal_sb"]}
    out = dict(n_runs=len(df), n_subjects=int(df.subject.nunique()),
               params=dict(n_parcels=N_PARCELS, fir_delays=FIR_DELAYS, TR=TR),
               summary=summary, per_run=rows)

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("\n=== NOISE CEILING (Spearman-Brown corrected) ===")
    print(f"  spatial topography : median {summary['spatial_sb']['median']:.3f} "
          f"[IQR {summary['spatial_sb']['q25']:.3f}-{summary['spatial_sb']['q75']:.3f}]")
    print(f"  spatiotemporal     : median {summary['spatiotemporal_sb']['median']:.3f} "
          f"[IQR {summary['spatiotemporal_sb']['q25']:.3f}-{summary['spatiotemporal_sb']['q75']:.3f}]")
    print(f"  -> wrote {outp}")


if __name__ == "__main__":
    main()
