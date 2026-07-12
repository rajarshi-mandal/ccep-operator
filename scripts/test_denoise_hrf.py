"""Decisive gate: can pragmatic in-Python denoising recover an HRF from raw ds005498 stim BOLD?

The raw-BOLD FIR was pulse-synchronous artifact (peak at 0 s lag, no HRF). Before committing to
a full fMRIPrep build, test whether nilearn aCompCor-style nuisance regression + drift recovers a
canonical hemodynamic response. Gate = the group-mean (sign-aligned) FIR at the top-responding
parcels should rise to a peak at ~4.8-7.2 s and return, under denoising — but not (or much less)
without it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.ds005498_pipeline import (DS_DEFAULT, FIR_DELAYS, TR_STIM, load_schaefer,  # noqa
                                    parcel_timeseries, find_stim_runs)


def fir_betas(parcel_ts, onsets, tr, confounds=None):
    from nilearn.glm.first_level import compute_regressor
    T, d = parcel_ts.shape
    ft = tr * np.arange(T)
    on = np.sort(onsets[onsets < T * tr])
    Y = parcel_ts - parcel_ts.mean(0, keepdims=True)
    cond = np.vstack([on, np.full_like(on, 0.3), np.ones_like(on)])
    fir = compute_regressor(cond, "fir", ft, fir_delays=FIR_DELAYS, oversampling=16)[0]
    drift = np.column_stack([np.ones(T), np.linspace(-1, 1, T), np.linspace(-1, 1, T) ** 2])
    cols = [fir, drift] + ([confounds] if confounds is not None else [])
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    nb = fir.shape[1]
    return beta[:nb].T              # [d, nbins]


def sign_aligned_top(fir, k=5):
    """Top-k energy parcels' FIR shapes, each sign-flipped to positive peak, averaged."""
    energy = (fir ** 2).sum(1)
    top = np.argsort(energy)[-k:]
    sh = []
    for p in top:
        f = fir[p] - fir[p].mean()
        if abs(f.min()) > abs(f.max()):
            f = -f
        sh.append(f / (np.linalg.norm(f) + 1e-9))
    return np.mean(sh, 0)


def main():
    atlas_img, _, _ = load_schaefer()
    onsets = pd.read_csv(DS_DEFAULT / "task-stim_events.tsv", sep="\t")["onset"].to_numpy(float)
    from nilearn.image import high_variance_confounds

    subs = []
    for s in sorted(p for p in DS_DEFAULT.glob("sub-*") if p.is_dir()):
        sites = find_stim_runs(s)
        if len(sites) >= 6:
            subs.append((s, sites))
        if len(subs) >= 6:
            break

    raw_sh, den_sh = [], []
    for s, sites in subs:
        for site, paths in list(sites.items())[:6]:
            img = nib.load(str(paths[0]))
            ts = parcel_timeseries(paths[0], atlas_img)
            conf = high_variance_confounds(img, n_confounds=5)        # aCompCor-like [T,5]
            raw_sh.append(sign_aligned_top(fir_betas(ts, onsets, TR_STIM)))
            den_sh.append(sign_aligned_top(fir_betas(ts, onsets, TR_STIM, confounds=conf)))
        print(f"  {s.name}: done ({len(raw_sh)} runs)", flush=True)

    lags = [round(TR_STIM * i, 1) for i in FIR_DELAYS]
    raw_m, den_m = np.mean(raw_sh, 0), np.mean(den_sh, 0)
    print(f"\nFIR lags (s):           {lags}")
    print(f"RAW    group FIR shape: {np.round(raw_m, 3)}")
    print(f"DENOISED group FIR:     {np.round(den_m, 3)}")
    print(f"\nHRF check (peak should be at bin 2-3 = 4.8-7.2 s):")
    print(f"  raw      argmax bin = {int(np.argmax(raw_m))} ({lags[int(np.argmax(raw_m))]}s)")
    print(f"  denoised argmax bin = {int(np.argmax(den_m))} ({lags[int(np.argmax(den_m))]}s)")
    # canonical glover sampled at the FIR lags, for reference correlation
    from nilearn.glm.first_level import compute_regressor
    ft = np.arange(0, 20, 0.1)
    g = compute_regressor(np.array([[0.], [0.3], [1.]]), "glover", ft)[0][:, 0]
    canon = np.interp(lags, ft, g); canon = (canon - canon.mean())
    canon /= np.linalg.norm(canon) + 1e-9
    print(f"  corr(raw, canonical HRF)      = {float(raw_m @ canon / (np.linalg.norm(raw_m)+1e-9)):+.3f}")
    print(f"  corr(denoised, canonical HRF) = {float(den_m @ canon / (np.linalg.norm(den_m)+1e-9)):+.3f}")


if __name__ == "__main__":
    main()
