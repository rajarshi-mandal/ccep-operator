"""Option 1 (tractable subset) — does SLICE-TIMING correction recover an HRF?

Full fMRIPrep is blocked here: no Docker, and the dataset has no fieldmaps /
PhaseEncodingDirection so susceptibility-distortion correction cannot run. But the BOLD json
has SliceOrder=sequential, and slice-timing is the one preprocessing step not yet tried that
affects the FIR *temporal shape* — with TR=2.4 s over 31 sequential slices, within-TR offsets
reach ~2.3 s and would smear a 2.4-s-binned FIR.

This applies per-slice temporal interpolation to the reference time (mid-TR) before
parcellation, then re-extracts the FIR and checks for a canonical HRF. Gate: corr(group FIR,
canonical HRF) should go meaningfully positive if slice-timing was the culprit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data.ds005498_pipeline import (DS_DEFAULT, FIR_DELAYS, TR_STIM, load_schaefer,  # noqa
                                    find_stim_runs)
from test_denoise_hrf import fir_betas, sign_aligned_top  # noqa
from test_motion_hrf import _parcels_from_img  # noqa


def slice_time_correct(img, tr, ref_frac=0.5):
    """Sequential-ascending slice-timing correction to the mid-TR reference (linear interp)."""
    data = np.asarray(img.dataobj, dtype=np.float32)          # [X,Y,Z,T]
    X, Y, Z, T = data.shape
    out = np.empty_like(data)
    k = np.arange(T)
    for z in range(Z):
        off = z / Z                                            # fraction of TR for this slice
        q = k + (ref_frac - off)                               # query positions (frames)
        q = np.clip(q, 0, T - 1)
        lo = np.floor(q).astype(int); hi = np.minimum(lo + 1, T - 1); fr = q - lo
        S = data[:, :, z, :]                                   # [X,Y,T]
        out[:, :, z, :] = (1 - fr) * S[:, :, lo] + fr * S[:, :, hi]
    return nib.Nifti1Image(out, img.affine, img.header)


def main():
    atlas_img, _, _ = load_schaefer()
    onsets = pd.read_csv(DS_DEFAULT / "task-stim_events.tsv", sep="\t")["onset"].to_numpy(float)
    from data.ds005498_pipeline import parcel_timeseries
    from nilearn.image import high_variance_confounds

    subs = []
    for s in sorted(p for p in DS_DEFAULT.glob("sub-*") if p.is_dir()):
        sites = find_stim_runs(s)
        if len(sites) >= 6:
            subs.append((s, sites))
        if len(subs) >= 6:
            break

    raw_sh, st_sh = [], []
    for s, sites in subs:
        for site, paths in list(sites.items())[:6]:
            img = nib.load(str(paths[0]))
            raw_sh.append(sign_aligned_top(fir_betas(parcel_timeseries(paths[0], atlas_img),
                                                     onsets, TR_STIM)))
            st_img = slice_time_correct(img, TR_STIM)
            ts = _parcels_from_img(st_img, atlas_img)
            conf = high_variance_confounds(st_img, n_confounds=5)
            st_sh.append(sign_aligned_top(fir_betas(ts, onsets, TR_STIM, confounds=conf)))
        print(f"  {s.name}: done ({len(raw_sh)} runs)", flush=True)

    lags = [round(TR_STIM * i, 1) for i in FIR_DELAYS]
    from nilearn.glm.first_level import compute_regressor
    ft = np.arange(0, 20, 0.1)
    g = compute_regressor(np.array([[0.], [0.3], [1.]]), "glover", ft)[0][:, 0]
    canon = np.interp(lags, ft, g); canon = canon - canon.mean(); canon /= np.linalg.norm(canon) + 1e-9
    raw_m, st_m = np.mean(raw_sh, 0), np.mean(st_sh, 0)
    print(f"\nFIR lags (s):                  {lags}")
    print(f"RAW          group FIR shape:  {np.round(raw_m,3)}  argmax {lags[int(np.argmax(raw_m))]}s")
    print(f"SLICE-TIMED  group FIR shape:  {np.round(st_m,3)}  argmax {lags[int(np.argmax(st_m))]}s")
    print(f"corr(raw, canonical HRF)         = {float(raw_m@canon/(np.linalg.norm(raw_m)+1e-9)):+.3f}")
    print(f"corr(slice-timed, canonical HRF) = {float(st_m@canon/(np.linalg.norm(st_m)+1e-9)):+.3f}")


if __name__ == "__main__":
    main()
