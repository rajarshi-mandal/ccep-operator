"""Phase 1 — raw ds005498 BOLD -> model-ready Schaefer-100 tensors.

ds005498 is concurrent single-pulse TMS-fMRI (152 subjects, 11 stim sites, plus a
resting run per subject). BOLD is *native EPI* (64x64x~30, ~3.4 mm), not MNI. This
module turns each raw run into region-space timeseries on the Schaefer-100 atlas so the
downstream Causal DAG-SSM (d=100) can consume it.

Per subject it produces:
  * ``rest``        — [T_rest, 100] z-scored resting parcel timeseries (L_obs / subject
                      conditioning input).
  * per stim site   — Glover-HRF beta **topography** [100] (the primary prediction target),
                      an FIR **trajectory** [100, n_bins] (denoised temporal target), the
                      split-half spatial **reliability** (QC gate), the **stim_parcel** index
                      (coil MNI coord -> nearest Schaefer parcel, for do()), and flags.

Registration (the hard part — see handoff §5.1). Route (B), the pragmatic default here:
resample the MNI Schaefer-100 atlas into each run's EPI grid via the BOLD's own affine
(``nilearn.image.resample_to_img``), then average voxels per label. Both volumes live in a
brain-centred mm frame, so the overlay roughly corresponds; it is NOT subject-specific
registration. Anatomical precision is therefore approximate — this is an accepted Phase-1
limitation, validated by re-estimating the noise ceiling on these parcels (§5.4) and
upgradeable to a full EPI->T1->MNI route later (``reg="epi2mni"`` hook left open).

The evoked-response GLM/FIR + split-half reliability reuse the logic from
``scripts/phase0_noise_ceiling.py`` but operate on Schaefer parcels instead of per-run
KMeans clusters, so parcels are comparable across runs and subjects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import nibabel as nib

DS_DEFAULT = Path("REDACTED/Open Neuro ds005498")

TR_STIM = 2.4
TR_REST = 2.0
PULSE_DUR = 0.3
N_PARCELS = 100
FIR_DELAYS = list(range(0, 7))      # post-stim scans (0..6 -> ~0-14.4 s at TR_stim)
REL_QC_THRESH = 0.3                 # drop runs whose split-half spatial reliability < this
ARTIFACT_MM = 12.0                  # flag parcels within this distance of the coil (TMS artifact)


# --------------------------------------------------------------------------------
# Atlas
# --------------------------------------------------------------------------------
def load_schaefer(n_parcels: int = N_PARCELS):
    """Fetch the Schaefer-2018 atlas (MNI 2 mm) and its parcel centroids.

    Returns ``(atlas_img, centroids_mni [n,3], labels[list[str]])``. Centroids are the
    mean MNI mm coordinate of each label's voxels — used both for the coil->parcel map
    and the observation-matrix locality prior (``obs_matrix_locality_penalty``).
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018
    atl = fetch_atlas_schaefer_2018(n_rois=n_parcels, yeo_networks=7, resolution_mm=2)
    img = nib.load(atl["maps"])
    data = np.asarray(img.dataobj)
    aff = img.affine
    centroids = np.zeros((n_parcels, 3), dtype=np.float64)
    for lab in range(1, n_parcels + 1):
        vox = np.argwhere(data == lab)
        if len(vox):
            world = nib.affines.apply_affine(aff, vox)
            centroids[lab - 1] = world.mean(0)
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atl["labels"]]
    return img, centroids, labels


def parse_coil_mni(site_label: str) -> np.ndarray:
    """``stim34x6x62`` / ``stimMinus38x22x48`` -> MNI mm coordinate ``[3]``.

    The task label *is* the coil MNI coordinate; ``Minus`` encodes a negative axis.
    """
    s = site_label[len("stim"):] if site_label.startswith("stim") else site_label
    s = s.replace("Minus", "-")
    # split on 'x' but keep leading '-' attached to the following number
    nums = re.findall(r"-?\d+", s)
    if len(nums) != 3:
        raise ValueError(f"cannot parse coil coord from {site_label!r} -> {nums}")
    return np.array([float(n) for n in nums], dtype=np.float64)


def coil_to_parcel(coil_mni: np.ndarray, centroids: np.ndarray) -> int:
    """Nearest Schaefer parcel index (0-based) to a coil MNI coordinate."""
    d = np.linalg.norm(centroids - coil_mni[None, :], axis=1)
    return int(np.argmin(d))


# --------------------------------------------------------------------------------
# Parcellation (route B: affine overlay of the MNI atlas onto the native EPI grid)
# --------------------------------------------------------------------------------
def parcel_timeseries(bold_path: Path, atlas_img, n_parcels: int = N_PARCELS,
                      reg: str = "affine_overlay") -> np.ndarray:
    """Native-EPI BOLD -> ``[T, n_parcels]`` mean-per-parcel timeseries.

    ``reg="affine_overlay"``: resample the MNI atlas into the EPI grid by affine and
    average voxels per label (handoff route B). Absent labels (outside the EPI FOV) are
    returned as all-zero columns so the parcel axis is always length ``n_parcels``.
    """
    from nilearn.image import resample_to_img

    img = nib.load(str(bold_path))
    if reg != "affine_overlay":
        raise NotImplementedError(f"reg={reg!r} (EPI->T1->MNI route is a future upgrade)")
    # Resample the MNI atlas onto the EPI's own 3D grid (nearest keeps integer labels).
    ref = nib.Nifti1Image(np.asarray(img.dataobj[..., 0]), img.affine, img.header)
    atlas_epi = resample_to_img(atlas_img, ref, interpolation="nearest",
                                force_resample=True, copy_header=True)
    lab_vol = np.rint(np.asarray(atlas_epi.dataobj)).astype(np.int32)   # [X,Y,Z]
    data = np.asarray(img.dataobj, dtype=np.float32)                     # [X,Y,Z,T]
    T = data.shape[-1]
    flat = data.reshape(-1, T)                                           # [V, T]
    labf = lab_vol.reshape(-1)                                           # [V]
    out = np.zeros((T, n_parcels), dtype=np.float32)
    for lab in range(1, n_parcels + 1):
        m = labf == lab
        if m.any():
            out[:, lab - 1] = flat[m].mean(0)
    return out


def zscore(ts: np.ndarray) -> np.ndarray:
    """Z-score each parcel (column) over time; flat parcels stay zero."""
    mu = ts.mean(0, keepdims=True)
    sd = ts.std(0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((ts - mu) / sd).astype(np.float32)


# --------------------------------------------------------------------------------
# Evoked response (GLM + FIR) and split-half reliability
# --------------------------------------------------------------------------------
def _regressor(onsets, frame_times, hrf_model, fir_delays=None):
    from nilearn.glm.first_level import compute_regressor
    cond = np.vstack([onsets, np.full_like(onsets, PULSE_DUR), np.ones_like(onsets)])
    sig, _ = compute_regressor(cond, hrf_model, frame_times,
                               fir_delays=fir_delays, oversampling=16)
    return sig


def evoked_response(parcel_ts: np.ndarray, onsets: np.ndarray, tr: float):
    """GLM the parcel timeseries against the pulse train.

    Returns a dict with the Glover-HRF beta **topography** ``[d]`` (primary target), the
    **FIR** trajectory ``[d, n_bins]`` (temporal target), the split-half spatial
    **reliability** (Spearman-Brown corrected Pearson r of odd vs even pulse betas), and
    ``n_pulses``. Returns ``None`` if too few pulses fall inside the run.
    """
    from scipy.stats import pearsonr
    d = parcel_ts.shape[1]
    T = parcel_ts.shape[0]
    ft = tr * np.arange(T)
    on = np.sort(onsets[onsets < T * tr])
    if len(on) < 6:
        return None
    Y = parcel_ts - parcel_ts.mean(0, keepdims=True)             # [T, d]
    drift = np.column_stack([np.ones(T), np.linspace(-1, 1, T),
                             np.linspace(-1, 1, T) ** 2])

    # --- full-data Glover beta = the topography target ---
    g_all = _regressor(on, ft, "glover")[:, 0]
    Xa = np.column_stack([g_all, drift])
    beta_all, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    topo = beta_all[0].astype(np.float32)                        # [d]

    # --- full-data FIR = the trajectory target ---
    f_all = _regressor(on, ft, "fir", FIR_DELAYS)                # [T, nbin]
    nb = f_all.shape[1]
    Xf = np.column_stack([f_all, drift])
    bf, *_ = np.linalg.lstsq(Xf, Y, rcond=None)
    fir = bf[:nb].T.astype(np.float32)                           # [d, nbin]

    # --- split-half spatial reliability (odd/even pulse betas, Glover) ---
    odd, even = on[0::2], on[1::2]
    rel = np.nan
    if len(odd) >= 3 and len(even) >= 3:
        ro = _regressor(odd, ft, "glover")[:, 0]
        re_ = _regressor(even, ft, "glover")[:, 0]
        Xs = np.column_stack([ro, re_, drift])
        bs, *_ = np.linalg.lstsq(Xs, Y, rcond=None)
        r = pearsonr(bs[0], bs[1])[0]
        if np.isfinite(r) and r > -1:
            rel = float(2 * r / (1 + r))                          # Spearman-Brown
    return dict(topo=topo, fir=fir, reliability=rel, n_pulses=int(len(on)))


# --------------------------------------------------------------------------------
# Subject assembly
# --------------------------------------------------------------------------------
SITE_RE = re.compile(r"task-(stim[A-Za-z0-9]+)_")


def find_rest_run(sub_dir: Path) -> Path | None:
    runs = sorted(sub_dir.glob("ses-*/func/*task-resting_bold.nii"))
    return runs[0] if runs else None


def find_stim_runs(sub_dir: Path) -> dict[str, list[Path]]:
    """site_label -> list of stim BOLD paths (a site may be repeated across sessions)."""
    out: dict[str, list[Path]] = {}
    for p in sorted(sub_dir.glob("ses-*/func/*task-stim*_bold.nii.gz")):
        m = SITE_RE.search(p.name)
        if m:
            out.setdefault(m.group(1), []).append(p)
    return out


@dataclass
class SubjectRecord:
    subject: str
    rest: np.ndarray                       # [T_rest, d] z-scored, or empty if missing
    sites: list[str] = field(default_factory=list)
    topo: list = field(default_factory=list)          # each [d]
    fir: list = field(default_factory=list)           # each [d, nbin]
    stim_parcel: list = field(default_factory=list)   # int
    coil_mni: list = field(default_factory=list)       # [3]
    reliability: list = field(default_factory=list)    # float
    n_pulses: list = field(default_factory=list)       # int
    artifact_parcels: list = field(default_factory=list)  # list[int] near the coil
    run_paths: list = field(default_factory=list)


def build_subject(sub_dir: Path, atlas_img, centroids: np.ndarray,
                  onsets: np.ndarray, reg: str = "affine_overlay",
                  n_parcels: int = N_PARCELS) -> SubjectRecord:
    """Assemble one subject's rest + per-site evoked records (route B parcellation)."""
    subject = sub_dir.name
    rest_path = find_rest_run(sub_dir)
    if rest_path is not None:
        rest = zscore(parcel_timeseries(rest_path, atlas_img, n_parcels, reg))
    else:
        rest = np.zeros((0, n_parcels), dtype=np.float32)
    rec = SubjectRecord(subject=subject, rest=rest)

    for site, paths in find_stim_runs(sub_dir).items():
        # average across repeated runs of the same site, in parcel space
        tss = [parcel_timeseries(p, atlas_img, n_parcels, reg) for p in paths]
        Tmin = min(t.shape[0] for t in tss)
        ts = np.mean([t[:Tmin] for t in tss], axis=0).astype(np.float32)
        ev = evoked_response(ts, onsets, TR_STIM)
        if ev is None:
            continue
        coil = parse_coil_mni(site)
        sp = coil_to_parcel(coil, centroids)
        near = np.argwhere(np.linalg.norm(centroids - coil[None], axis=1) < ARTIFACT_MM)
        rec.sites.append(site)
        rec.topo.append(ev["topo"])
        rec.fir.append(ev["fir"])
        rec.stim_parcel.append(sp)
        rec.coil_mni.append(coil)
        rec.reliability.append(ev["reliability"])
        rec.n_pulses.append(ev["n_pulses"])
        rec.artifact_parcels.append([int(x) for x in near.ravel().tolist()])
        rec.run_paths.append([str(p) for p in paths])
    return rec


# --------------------------------------------------------------------------------
# Loader (Phase 2 — leave-one-site-out within subject)
# --------------------------------------------------------------------------------
@dataclass
class SiteRecord:
    """One (subject, site) interventional record, mirroring the dataset.py contract
    plus the per-subject rest dynamics needed for subject conditioning (§5.2)."""
    subject: str
    site_name: str
    stim_parcel: int
    region_tep: np.ndarray        # [d, 1] topography (primary target) — model contract shape
    topo: np.ndarray              # [d] Glover beta topography
    fir: np.ndarray               # [d, n_bins] FIR trajectory
    subject_rest: np.ndarray      # [T_rest, d] z-scored rest dynamics
    reliability: float
    coil_mni: np.ndarray          # [3]
    dataset: str = "ds005498"


class DS005498Cache:
    """Loads the Phase-1 cache into per-(subject, site) records for LOSO-WS eval.

    ``qc_filter`` drops records below the split-half reliability threshold (measurement
    noise). ``site_template`` gives the population-mean topography per site — the baseline
    that beat the old (subject-blind) model and which subject conditioning must beat.
    """

    def __init__(self, cache_dir: str | Path = "data/processed/ds005498",
                 qc_filter: bool = True, qc_thresh: float = REL_QC_THRESH):
        self.dir = Path(cache_dir)
        self.centroids = np.load(self.dir / "atlas_centroids_mni.npy")
        self.records: list[SiteRecord] = []
        for npz in sorted((self.dir / "subjects").glob("*.npz")):
            z = np.load(npz, allow_pickle=True)
            rest = z["rest"].astype(np.float32)
            sites = list(z["sites"])
            for i, site in enumerate(sites):
                rel = float(z["reliability"][i])
                if qc_filter and not (np.isfinite(rel) and rel >= qc_thresh):
                    continue
                topo = z["topo"][i].astype(np.float32)
                self.records.append(SiteRecord(
                    subject=str(z["subject"]), site_name=str(site),
                    stim_parcel=int(z["stim_parcel"][i]),
                    region_tep=topo[:, None], topo=topo,
                    fir=z["fir"][i].astype(np.float32),
                    subject_rest=rest, reliability=rel,
                    coil_mni=z["coil_mni"][i],
                ))

    def __len__(self):
        return len(self.records)

    def subjects(self) -> list[str]:
        return sorted({r.subject for r in self.records})

    def sites(self) -> list[str]:
        return sorted({r.site_name for r in self.records})

    def site_template(self, site: str, exclude_subject: str | None = None) -> np.ndarray:
        """Population-mean topography for a site (the baseline). Optionally leave a
        subject out to keep the baseline honest under subject-transfer evaluation."""
        topos = [r.topo for r in self.records
                 if r.site_name == site and r.subject != exclude_subject]
        return np.mean(topos, axis=0) if topos else np.zeros(self.centroids.shape[0], np.float32)

    def loso_ws(self):
        """Yield ``(test_record, train_records)`` for leave-one-site-out within subject:
        hold out one site of a subject, condition on that subject's *other* sites."""
        by_sub: dict[str, list[SiteRecord]] = {}
        for r in self.records:
            by_sub.setdefault(r.subject, []).append(r)
        for sub, recs in by_sub.items():
            if len(recs) < 2:
                continue
            for i, test in enumerate(recs):
                yield test, [r for j, r in enumerate(recs) if j != i]
