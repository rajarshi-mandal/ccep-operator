"""GATE 0 — does fMRIPrep-preprocessed ds005498 yield a real evoked HRF?

Run after scripts/run_fmriprep.sh. Re-extracts evoked betas on the preprocessed BOLD (already in
MNI152NLin6Asym, so the FSL Schaefer-100 atlas applies cleanly) with fMRIPrep confounds, then tests
the three things raw BOLD failed:
  1. group-mean FIR resembles a canonical HRF (corr > +0.5)        [the decider]
  2. same-site cross-subject topography coherence rises (> ~0.2)
  3. evoked response localizes near the coil (stim-parcel percentile > ~0.7)
PASS on (1) (and ideally 2-3) => the signal is recoverable; scale fMRIPrep + proceed to Stage 1.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.ds005498_pipeline import (DS_DEFAULT, FIR_DELAYS, N_PARCELS, TR_STIM,  # noqa
                                    coil_to_parcel, evoked_response, load_schaefer,
                                    parse_coil_mni)

CONF_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
             "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
             "a_comp_cor_03", "a_comp_cor_04"]
SITE_RE = re.compile(r"task-(stim[A-Za-z0-9]+)_")


def parcellate_mni(bold_img, atlas_img, n=N_PARCELS):
    from nilearn.image import resample_to_img
    ref = nib.Nifti1Image(np.asarray(bold_img.dataobj[..., 0]), bold_img.affine, bold_img.header)
    atl = resample_to_img(atlas_img, ref, interpolation="nearest",
                          force_resample=True, copy_header=True)
    lab = np.rint(np.asarray(atl.dataobj)).astype(np.int32).reshape(-1)
    data = np.asarray(bold_img.dataobj, dtype=np.float32); T = data.shape[-1]
    flat = data.reshape(-1, T)
    out = np.zeros((T, n), np.float32)
    for L in range(1, n + 1):
        m = lab == L
        if m.any():
            out[:, L - 1] = flat[m].mean(0)
    return out


def load_confounds(bold_path):
    conf = re.sub(r"_space-[^_]+_res-[^_]+_desc-preproc_bold\.nii(\.gz)?$",
                  "_desc-confounds_timeseries.tsv", bold_path)
    if not Path(conf).exists():
        return None
    df = pd.read_csv(conf, sep="\t")
    cols = [c for c in CONF_COLS if c in df.columns]
    X = df[cols].to_numpy(dtype=float)
    return np.nan_to_num(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fmriprep_dir")
    ap.add_argument("--space", default="MNI152NLin6Asym")
    args = ap.parse_args()

    atlas_img, centroids, _ = load_schaefer()
    onsets = pd.read_csv(DS_DEFAULT / "task-stim_events.tsv", sep="\t")["onset"].to_numpy(float)
    pat = f"{args.fmriprep_dir}/sub-*/**/func/*task-stim*_space-{args.space}_*desc-preproc_bold.nii.gz"
    bolds = sorted(glob.glob(pat, recursive=True))
    if not bolds:
        print(f"[gate0] no preprocessed stim BOLD found under {args.fmriprep_dir} "
              f"(space {args.space}). Did fMRIPrep finish?"); sys.exit(1)
    print(f"[gate0] {len(bolds)} preprocessed stim runs", flush=True)

    by_site = defaultdict(list); pct = []; topos = []; firs = []
    for i, bp in enumerate(bolds):
        site = SITE_RE.search(Path(bp).name).group(1)
        ts = parcellate_mni(nib.load(bp), atlas_img)
        conf = load_confounds(bp)
        ev = evoked_response(ts, onsets, TR_STIM) if conf is None else \
            evoked_response_with_conf(ts, onsets, TR_STIM, conf)
        if ev is None:
            continue
        p = coil_to_parcel(parse_coil_mni(site), centroids)
        t = np.abs(ev["topo"]); pct.append(np.argsort(np.argsort(t))[p] / (N_PARCELS - 1))
        by_site[site].append(ev["topo"]); firs.append(ev["fir"][p])
        print(f"  [{i+1}/{len(bolds)}] {Path(bp).parts[-1][:40]} rel={ev['reliability']:.3f}",
              flush=True)

    # gate metrics
    from nilearn.glm.first_level import compute_regressor
    lags = [round(TR_STIM * k, 1) for k in FIR_DELAYS]
    ftc = np.arange(0, 20, 0.1)
    g = compute_regressor(np.array([[0.], [0.3], [1.]]), "glover", ftc)[0][:, 0]
    canon = np.interp(lags, ftc, g); canon = canon - canon.mean(); canon /= np.linalg.norm(canon) + 1e-9
    gm = np.mean(firs, 0); gm_c = gm - gm.mean()
    hrf_corr = float(gm_c @ canon / (np.linalg.norm(gm_c) + 1e-9))

    def meanpair(M):
        M = np.array(M); M = M - M.mean(1, keepdims=True)
        M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        C = M @ M.T; iu = np.triu_indices(len(M), 1)
        return float(C[iu].mean()) if len(iu[0]) else np.nan
    coh = np.median([meanpair(v) for v in by_site.values() if len(v) >= 3])

    print("\n=== GATE 0 ===")
    print(f"  (1) group FIR vs canonical HRF : corr = {hrf_corr:+.3f}  (PASS > +0.5; raw was -0.65)")
    print(f"      group FIR @ coil {np.round(gm,3)} (lags {lags})")
    print(f"  (2) same-site cross-subj coherence: {coh:+.3f}  (PASS > ~0.2; raw was 0.01)")
    print(f"  (3) coil-localization percentile : {np.median(pct):.2f}  (PASS > ~0.7; raw was 0.41)")
    verdict = "PASS" if hrf_corr > 0.5 else ("PARTIAL" if hrf_corr > 0.2 else "FAIL")
    print(f"\n  VERDICT: {verdict} — "
          + {"PASS": "signal recovered; scale fMRIPrep to all subjects, proceed to Stage 1.",
             "PARTIAL": "weak signal; try more confounds / denoising before scaling.",
             "FAIL": "no HRF even after fMRIPrep; off-ramp (roadmap 0B/0C)."}[verdict])


def evoked_response_with_conf(parcel_ts, onsets, tr, conf):
    """evoked_response but with fMRIPrep confounds added to the GLM nuisance."""
    from scipy.stats import pearsonr
    d = parcel_ts.shape[1]; T = parcel_ts.shape[0]
    from nilearn.glm.first_level import compute_regressor
    ft = tr * np.arange(T); on = np.sort(onsets[onsets < T * tr])
    if len(on) < 6:
        return None
    Y = parcel_ts - parcel_ts.mean(0, keepdims=True)
    drift = np.column_stack([np.ones(T), np.linspace(-1, 1, T), np.linspace(-1, 1, T) ** 2])
    nuis = np.column_stack([drift, conf[:T]])

    def reg(o, model, fir=None):
        c = np.vstack([o, np.full_like(o, 0.3), np.ones_like(o)])
        return compute_regressor(c, model, ft, fir_delays=fir, oversampling=16)[0]
    g = reg(on, "glover")[:, 0]
    beta, *_ = np.linalg.lstsq(np.column_stack([g, nuis]), Y, rcond=None)
    topo = beta[0].astype(np.float32)
    f = reg(on, "fir", FIR_DELAYS); nb = f.shape[1]
    bf, *_ = np.linalg.lstsq(np.column_stack([f, nuis]), Y, rcond=None)
    fir = bf[:nb].T.astype(np.float32)
    odd, even = on[0::2], on[1::2]; rel = np.nan
    if len(odd) >= 3 and len(even) >= 3:
        bs, *_ = np.linalg.lstsq(np.column_stack([reg(odd, "glover")[:, 0],
                                                  reg(even, "glover")[:, 0], nuis]), Y, rcond=None)
        r = pearsonr(bs[0], bs[1])[0]
        if np.isfinite(r) and r > -1:
            rel = float(2 * r / (1 + r))
    return dict(topo=topo, fir=fir, reliability=rel, n_pulses=int(len(on)))


if __name__ == "__main__":
    main()
