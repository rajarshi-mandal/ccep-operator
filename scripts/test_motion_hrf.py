"""Final lever — does per-volume motion correction recover an HRF that aCompCor could not?

The 0-s artifact looks like pulse-triggered head motion (aCompCor can't fix that). This does a
fast FFT phase-correlation translation realignment of every volume to the run mean, then
re-extracts the FIR and checks for a canonical HRF. Translation-only is crude (no rotation) but
catches the dominant jerk; if it surfaces an HRF the full fMRIPrep build is justified, if not the
negative is robust. (Full-dataset in-Python MC is impractical — this is a diagnostic gate.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import scipy.ndimage as ndi
from numpy.fft import fftn, ifftn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from data.ds005498_pipeline import (DS_DEFAULT, FIR_DELAYS, TR_STIM, load_schaefer,  # noqa
                                    find_stim_runs)
from test_denoise_hrf import fir_betas, sign_aligned_top  # reuse  # noqa


def _shift_of(ref_f, vol):
    """Translation to APPLY to vol (via ndi.shift) so it aligns to ref. ref_f = fftn(ref-mean).

    Plain normalised cross-correlation (phase-whitening is unstable on edge-zeroed EPI);
    returns the cross-correlation peak offset = the shift to feed ndi.shift(vol, sh)."""
    G = fftn(vol - vol.mean())
    R = ref_f * np.conj(G)
    cc = np.real(ifftn(R))
    peak = np.unravel_index(int(np.argmax(cc)), cc.shape)
    sh = np.array(peak, float)
    # parabolic sub-voxel refinement per axis
    for i, n in enumerate(cc.shape):
        a = list(peak)
        am = a.copy(); am[i] = (peak[i] - 1) % n
        ap = a.copy(); ap[i] = (peak[i] + 1) % n
        ym, y0, yp = cc[tuple(am)], cc[tuple(a)], cc[tuple(ap)]
        denom = (ym - 2 * y0 + yp)
        if abs(denom) > 1e-9:
            sh[i] += 0.5 * (ym - yp) / denom
        if sh[i] > n / 2:
            sh[i] -= n
    return sh


def motion_correct(img):
    data = np.asarray(img.dataobj, dtype=np.float32)        # [X,Y,Z,T]
    ref = data.mean(-1)
    ref_f = fftn(ref - ref.mean())
    out = np.empty_like(data)
    fd = []
    prev = np.zeros(3)
    for t in range(data.shape[-1]):
        sh = _shift_of(ref_f, data[..., t])
        out[..., t] = ndi.shift(data[..., t], sh, order=1, mode="nearest")
        fd.append(np.abs(sh - prev).sum()); prev = sh
    return nib.Nifti1Image(out, img.affine, img.header), float(np.mean(fd))


def parcels(img, atlas_img):
    from data.ds005498_pipeline import parcel_timeseries
    return parcel_timeseries(img, atlas_img)  # accepts path or img? -> needs path; handle below


def main():
    # sign self-test: shift a volume by a known amount, confirm recovery
    rng = np.random.default_rng(0)
    v = rng.standard_normal((24, 24, 20)).astype(np.float32)
    v = ndi.gaussian_filter(v, 2)
    moved = ndi.shift(v, [2.0, -1.0, 0.0], order=1, mode="nearest")
    sh = _shift_of(fftn(v - v.mean()), moved)
    corrected = ndi.shift(moved, sh, order=1, mode="nearest")
    err0 = np.mean((moved - v) ** 2); err1 = np.mean((corrected - v) ** 2)
    ok = "OK" if err1 < 0.25 * err0 else "BAD"
    print(f"[selftest] apply-shift {np.round(sh,2)}; MSE-to-ref {err0:.3f} -> {err1:.3f} ({ok})")

    atlas_img, _, _ = load_schaefer()
    onsets = pd.read_csv(DS_DEFAULT / "task-stim_events.tsv", sep="\t")["onset"].to_numpy(float)
    from data.ds005498_pipeline import parcel_timeseries
    from nilearn.image import high_variance_confounds

    subs = []
    for s in sorted(p for p in DS_DEFAULT.glob("sub-*") if p.is_dir()):
        sites = find_stim_runs(s)
        if len(sites) >= 6:
            subs.append((s, sites))
        if len(subs) >= 4:
            break

    raw_sh, mc_sh = [], []
    fds = []
    for s, sites in subs:
        for site, paths in list(sites.items())[:5]:
            img = nib.load(str(paths[0]))
            ts_raw = parcel_timeseries(paths[0], atlas_img)
            raw_sh.append(sign_aligned_top(fir_betas(ts_raw, onsets, TR_STIM)))
            mc_img, fd = motion_correct(img); fds.append(fd)
            # parcellate the MC image (write nothing; parcel_timeseries needs a path -> pass img)
            ts_mc = _parcels_from_img(mc_img, atlas_img)
            conf = high_variance_confounds(mc_img, n_confounds=5)
            mc_sh.append(sign_aligned_top(fir_betas(ts_mc, onsets, TR_STIM, confounds=conf)))
        print(f"  {s.name}: done, mean FD~{np.mean(fds):.2f} vox ({len(raw_sh)} runs)", flush=True)

    lags = [round(TR_STIM * i, 1) for i in FIR_DELAYS]
    from nilearn.glm.first_level import compute_regressor
    ft = np.arange(0, 20, 0.1)
    g = compute_regressor(np.array([[0.], [0.3], [1.]]), "glover", ft)[0][:, 0]
    canon = np.interp(lags, ft, g); canon = canon - canon.mean(); canon /= np.linalg.norm(canon) + 1e-9
    raw_m, mc_m = np.mean(raw_sh, 0), np.mean(mc_sh, 0)
    print(f"\nFIR lags (s):              {lags}")
    print(f"RAW       group FIR shape: {np.round(raw_m,3)}  argmax {lags[int(np.argmax(raw_m))]}s")
    print(f"MOTION-CORR group FIR:     {np.round(mc_m,3)}  argmax {lags[int(np.argmax(mc_m))]}s")
    print(f"corr(raw, canonical HRF)        = {float(raw_m@canon/(np.linalg.norm(raw_m)+1e-9)):+.3f}")
    print(f"corr(motion-corr, canonical HRF)= {float(mc_m@canon/(np.linalg.norm(mc_m)+1e-9)):+.3f}")


def _parcels_from_img(img, atlas_img, n=100):
    from nilearn.image import resample_to_img
    ref = nib.Nifti1Image(np.asarray(img.dataobj[..., 0]), img.affine, img.header)
    atlas_epi = resample_to_img(atlas_img, ref, interpolation="nearest",
                                force_resample=True, copy_header=True)
    lab = np.rint(np.asarray(atlas_epi.dataobj)).astype(np.int32).reshape(-1)
    data = np.asarray(img.dataobj, dtype=np.float32); T = data.shape[-1]
    flat = data.reshape(-1, T)
    out = np.zeros((T, n), np.float32)
    for L in range(1, n + 1):
        m = lab == L
        if m.any():
            out[:, L - 1] = flat[m].mean(0)
    return out


if __name__ == "__main__":
    main()
