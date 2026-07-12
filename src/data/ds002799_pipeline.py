"""ds002799 (Human es-fMRI) — intracranial electrical-stimulation + BOLD-fMRI extraction.

The no-Docker fMRI testbed for the subject-conditioned causal model. Unlike ds005498 (raw,
single-pulse, artifact-dominated), this dataset ships **fMRIPrep derivatives** (BOLD already in
MNI152NLin2009cAsym + confounds) and uses a **block design** (much better BOLD SNR). Each subject
has multiple es runs at different intracranial stimulation sites + pre-op resting-state — exactly
the structure the model needs (leave-one-site-out within subject, conditioned on rest).

Mapping to the model contract (mirrors ds005498 so the SAME analysis harness runs unchanged):
  * stim site   = the stimulated contact(s) for a run (channels.tsv) -> their MNI coordinates
                  (electrodes.tsv) -> mean = stim coordinate -> nearest Schaefer-100 parcel = do() target.
  * evoked topo = Glover-HRF GLM of the es block events on the Schaefer-parcellated preproc BOLD,
                  with fMRIPrep confounds as nuisance. FIR for the trajectory. Split-half reliability.
  * subject_rest= pre-op resting preproc BOLD -> Schaefer parcels, z-scored.

Caveat: electrodes are in MNI152NLin6Asym, the BOLD in MNI152NLin2009cAsym — the ~1-2 mm template
difference is small vs the 100-parcel scale and the electrode-localization uncertainty; acceptable
for a first pass (resample handles the grid, centroid match handles the coordinate).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.ds005498_pipeline import (FIR_DELAYS, N_PARCELS, SubjectRecord,  # noqa: E402
                                    coil_to_parcel, load_schaefer, zscore)

DS_DEFAULT = Path("REDACTED/Open Neuro ds002799")
DERIV = "derivatives/fmriprep"
MNI = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
PULSE_DUR = 0.5            # model each stim event as a short boxcar (events are 0.05 s pulses in blocks)
CONF_PREFER = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z",
               "csf", "white_matter", "framewise_displacement"]
RUN_RE = re.compile(r"task-es_run-(\d+)")


def find_es_runs(sub_dir_deriv: Path) -> list[Path]:
    return sorted(sub_dir_deriv.glob(f"ses-postop/func/*task-es*{MNI}"))


def find_rest_run(sub_dir_deriv: Path) -> Path | None:
    r = sorted(sub_dir_deriv.glob(f"ses-postop/func/*task-rest*{MNI}")) + \
        sorted(sub_dir_deriv.glob(f"ses-preop/func/*task-rest*{MNI}"))
    return r[0] if r else None


def read_tr(bold_img) -> float:
    tr = float(bold_img.header.get_zooms()[-1])
    return tr if 0.3 < tr < 6 else 2.0


def parse_electrodes(sub_raw: Path) -> dict[str, np.ndarray]:
    f = sorted(sub_raw.glob("ses-postop/ieeg/*electrodes.tsv"))
    if not f:
        return {}
    df = pd.read_csv(f[0], sep="\t")
    return {str(r["name"]): np.array([r["x"], r["y"], r["z"]], float) for _, r in df.iterrows()}


def stim_contacts(run_path: Path, sub_raw: Path) -> list[str]:
    """The stimulated contact names for an es run (from its channels.tsv)."""
    run = RUN_RE.search(run_path.name).group(1)
    ch = sorted(sub_raw.glob(f"ses-postop/ieeg/*task-es_run-{run}_channels.tsv"))
    if not ch:
        return []
    df = pd.read_csv(ch[0], sep="\t")
    return [str(n) for n in df["name"].tolist()]


def parcellate_mni(bold_img, atlas_img, n=N_PARCELS) -> np.ndarray:
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


def load_confounds(run_path: Path, T: int) -> np.ndarray | None:
    conf = Path(str(run_path).replace(f"_{MNI}", "_desc-confounds_regressors.tsv"))
    if not conf.exists():
        conf = Path(re.sub(r"_space-[^_]+_desc-preproc_bold\.nii\.gz$",
                           "_desc-confounds_regressors.tsv", str(run_path)))
    if not conf.exists():
        return None
    df = pd.read_csv(conf, sep="\t")
    cols = [c for c in CONF_PREFER if c in df.columns]
    cols += [c for c in df.columns if c.startswith("a_comp_cor_")][:5]
    X = df[cols].to_numpy(dtype=float)[:T]
    return np.nan_to_num(X)


def _blocks(onsets: np.ndarray, gap: float = 10.0):
    """Group stim pulses into sustained blocks -> (block_onsets, block_durations)."""
    on = np.sort(onsets)
    brk = np.where(np.diff(on) > gap)[0]
    starts = np.concatenate([[on[0]], on[brk + 1]])
    ends = np.concatenate([on[brk], [on[-1]]])
    return starts, np.maximum(ends - starts, 3.0)


def evoked_es(parcel_ts: np.ndarray, onsets: np.ndarray, tr: float, conf=None):
    """Block-design Glover GLM (+fMRIPrep confounds) -> topography, FIR, split-half reliability.

    Models the ~27 s stimulation BLOCKS as sustained boxcars (not individual 0.05 s pulses) for
    SNR; split-half reliability splits the ~10 blocks odd/even.
    """
    from nilearn.glm.first_level import compute_regressor
    from scipy.stats import pearsonr
    T = parcel_ts.shape[0]
    ft = tr * np.arange(T)
    on = np.sort(onsets[onsets < T * tr])
    if len(on) < 6:
        return None
    starts, durs = _blocks(on)
    if len(starts) < 2:
        return None
    Y = parcel_ts - parcel_ts.mean(0, keepdims=True)
    drift = np.column_stack([np.ones(T), np.linspace(-1, 1, T), np.linspace(-1, 1, T) ** 2])
    nuis = drift if conf is None else np.column_stack([drift, conf])

    def reg(s, d, model, fir=None):
        c = np.vstack([s, d, np.ones_like(s)])
        return compute_regressor(c, model, ft, fir_delays=fir, oversampling=16)[0]

    beta, *_ = np.linalg.lstsq(np.column_stack([reg(starts, durs, "glover")[:, 0], nuis]), Y, rcond=None)
    topo = beta[0].astype(np.float32)
    f = reg(starts, durs, "fir", FIR_DELAYS); nb = f.shape[1]
    bf, *_ = np.linalg.lstsq(np.column_stack([f, nuis]), Y, rcond=None)
    fir = bf[:nb].T.astype(np.float32)
    rel = np.nan
    if len(starts) >= 4:
        o, e = slice(0, None, 2), slice(1, None, 2)
        bs, *_ = np.linalg.lstsq(np.column_stack([reg(starts[o], durs[o], "glover")[:, 0],
                                                  reg(starts[e], durs[e], "glover")[:, 0], nuis]),
                                 Y, rcond=None)
        r = pearsonr(bs[0], bs[1])[0]
        if np.isfinite(r) and r > -1:
            rel = float(2 * r / (1 + r))
    return dict(topo=topo, fir=fir, reliability=rel, n_pulses=int(len(starts)))


def build_subject(sub: str, ds: Path, atlas_img, centroids: np.ndarray,
                  n_parcels: int = N_PARCELS) -> SubjectRecord:
    deriv = ds / DERIV / sub
    raw = ds / sub
    elec = parse_electrodes(raw)
    rec = SubjectRecord(subject=sub, rest=np.zeros((0, n_parcels), np.float32))

    rest_path = find_rest_run(deriv)
    if rest_path is not None:
        rimg = nib.load(str(rest_path))
        rec.rest = zscore(parcellate_mni(rimg, atlas_img, n_parcels))

    for run_path in find_es_runs(deriv):
        contacts = stim_contacts(run_path, raw)
        coords = [elec[c] for c in contacts if c in elec]
        if not coords:
            continue
        stim_xyz = np.mean(coords, axis=0)
        img = nib.load(str(run_path))
        tr = read_tr(img)
        ts = parcellate_mni(img, atlas_img, n_parcels)
        ev_dir = ds / sub / "ses-postop" / "func"
        evf = sorted(ev_dir.glob(f"*{RUN_RE.search(run_path.name).group(0)}_events.tsv"))
        if not evf:
            continue
        onsets = pd.read_csv(evf[0], sep="\t")["onset"].to_numpy(float)
        ev = evoked_es(ts, onsets, tr, conf=load_confounds(run_path, ts.shape[0]))
        if ev is None:
            continue
        run_id = RUN_RE.search(run_path.name).group(1)
        rec.sites.append(f"es{run_id}")
        rec.topo.append(ev["topo"]); rec.fir.append(ev["fir"])
        rec.stim_parcel.append(coil_to_parcel(stim_xyz, centroids))
        rec.coil_mni.append(stim_xyz)
        rec.reliability.append(ev["reliability"]); rec.n_pulses.append(ev["n_pulses"])
        rec.artifact_parcels.append([]); rec.run_paths.append([str(run_path)])
    return rec
