"""EEG <-> atlas spatial bridge (spec 3.5 — the flagged failure point).

The model lives in a 100-region latent space (HCP group-ICA d100). The fMRI is already
in that space; the TMS-EEG TEPs are in electrode space. This module bridges them:

  1. Compute each ICA component's MNI centroid from the d100 CIFTI map, using the
     fsLR-32k midthickness surface (cortex) + the CIFTI volume affine (subcortex).
  2. Place EEG electrodes in MNI via an MNE standard montage.
  3. Assign each electrode to its nearest parcel centroid, giving an aggregation
     matrix that maps an electrode-space TEP -> a 100-region response.

It also identifies the stimulation-site parcels (M1, parietal, ...) by nearest centroid
to canonical MNI coordinates, so Experiment 1 can hold out a site.

This uses REAL geometry end-to-end (no placeholder centroids): the surface files ship in
the HCP Structural Connectivity download.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# CIFTI cortical structure names.
_CORTEX_L = "CIFTI_STRUCTURE_CORTEX_LEFT"
_CORTEX_R = "CIFTI_STRUCTURE_CORTEX_RIGHT"

# Canonical MNI coordinates (mm) of common TMS targets, for naming parcels.
CANONICAL_SITES_MNI = {
    "M1_L": (-37.0, -21.0, 58.0),     # left primary motor (hand knob)
    "M1_R": (37.0, -21.0, 58.0),
    "DLPFC_L": (-40.0, 30.0, 30.0),
    "parietal_L": (-30.0, -50.0, 45.0),  # superior parietal / IPS
    "parietal_R": (30.0, -50.0, 45.0),
}


def _cifti_path(cfg) -> Path:
    d = cfg.fmri.ica_dim
    return (
        Path(cfg.paths.processed_dir).parent
        / "raw" / "groupICA"
        / f"groupICA_3T_HCP1200_MSMAll_d{d}.ica" / "melodic_IC.dscalar.nii"
    )


def _ensure_cifti(cfg) -> Path:
    """Extract the d{ica} melodic_IC.dscalar.nii from the groupICA tarball if needed."""
    import tarfile

    path = _cifti_path(cfg)
    if path.exists():
        return path
    d = cfg.fmri.ica_dim
    tar = Path(cfg.paths.hcp_ptn_dir) / "groupICA_3T_HCP1200_MSMAll.tar.gz"
    member = f"groupICA/groupICA_3T_HCP1200_MSMAll_d{d}.ica/melodic_IC.dscalar.nii"
    raw_dir = Path(cfg.paths.processed_dir).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        tf.extract(member, raw_dir)
    return path


def _surface_coords(cfg) -> dict[str, np.ndarray]:
    import nibabel as nib

    base = Path(cfg.paths.hcp_struct_dir)
    out = {}
    for hemi, key in [("L", _CORTEX_L), ("R", _CORTEX_R)]:
        g = nib.load(str(base / f"S1200.{hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii"))
        out[key] = np.asarray(g.darrays[0].data, dtype=np.float64)  # [32492, 3] MNI mm
    return out


def grayordinate_coords(cfg) -> tuple[np.ndarray, "object"]:
    """Build an MNI coordinate for every CIFTI grayordinate of the d{ica} map.

    Returns ``coords [n_grayord, 3]`` (mm) and the loaded CIFTI image.
    """
    import nibabel as nib

    img = nib.load(str(_ensure_cifti(cfg)))
    bm = img.header.get_axis(1)  # BrainModelAxis
    n = bm.size
    coords = np.full((n, 3), np.nan, dtype=np.float64)
    surf = _surface_coords(cfg)
    affine = bm.affine  # voxel IJK (homogeneous) -> MNI mm

    for name, sl, part in bm.iter_structures():
        if name in surf:  # cortical surface vertices
            coords[sl] = surf[name][part.vertex]
        else:  # subcortical volume voxels
            ijk = np.asarray(part.voxel, dtype=np.float64)
            ijk1 = np.column_stack([ijk, np.ones(len(ijk))])
            coords[sl] = (affine @ ijk1.T).T[:, :3]
    return coords, img


def compute_parcel_centroids(cfg, method: str = "peak", top_frac: float = 0.001) -> np.ndarray:
    """MNI location of each ICA component. Cached to ``parcel_centroids_mni.npy``.

    ``method``:
      - ``"peak"``: weighted mean over the component's strongest ``top_frac`` of
        grayordinates (near the |weight| peak). Focal and anatomically meaningful —
        the right choice for localising distributed/bilateral group-ICA components,
        which a global weighted mean would collapse toward brain-center.
      - ``"weighted"``: |weight|-weighted mean over all grayordinates (distributed).

    Returns ``[d, 3]`` (mm).
    """
    proc = Path(cfg.paths.processed_dir)
    cache = proc / "parcel_centroids_mni.npy"
    if cache.exists():
        return np.load(cache)

    coords, img = grayordinate_coords(cfg)
    maps = np.asarray(img.get_fdata(), dtype=np.float64)  # [d, n_grayord]
    valid = ~np.isnan(coords).any(axis=1)
    d, n = maps.shape
    centroids = np.zeros((d, 3), dtype=np.float64)
    for i in range(d):
        w = np.abs(maps[i]).copy()
        w[~valid] = 0.0
        if w.sum() == 0:
            centroids[i] = np.nan
            continue
        if method == "peak":
            k = max(1, int(top_frac * valid.sum()))
            top = np.argpartition(w, -k)[-k:]
            wt = w[top]
            centroids[i] = (wt[:, None] * coords[top]).sum(0) / wt.sum()
        else:
            centroids[i] = (w[:, None] * coords).sum(0) / w.sum()

    proc.mkdir(parents=True, exist_ok=True)
    np.save(cache, centroids)
    return centroids


def electrode_positions(ch_names: list[str], montage_name: str) -> tuple[list[str], np.ndarray]:
    """MNI (head) positions in mm for the named electrodes present in the montage."""
    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    pos = montage.get_positions()["ch_pos"]  # metres, head/MNI-ish frame
    names, xyz = [], []
    for ch in ch_names:
        if ch in pos and np.all(np.isfinite(pos[ch])):
            names.append(ch)
            xyz.append(np.asarray(pos[ch]) * 1000.0)  # m -> mm
    return names, np.asarray(xyz, dtype=np.float64)


class EEGAtlasBridge:
    """Maps an electrode-space TEP into the 100-region parcel space."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.centroids = compute_parcel_centroids(cfg)  # [d, 3] mm
        self.d = self.centroids.shape[0]

    def assign_electrodes(self, ch_names: list[str]):
        """Return (kept_names, parcel_index_per_electrode, aggregation_matrix [d, n_elec])."""
        names, xyz = electrode_positions(ch_names, self.cfg.tms_eeg.montage)
        valid_parcels = ~np.isnan(self.centroids).any(axis=1)
        cent = self.centroids[valid_parcels]
        pidx_compact = np.argmin(
            ((xyz[:, None, :] - cent[None, :, :]) ** 2).sum(-1), axis=1
        )
        parcel_ids = np.where(valid_parcels)[0]
        parcel_of_elec = parcel_ids[pidx_compact]  # [n_elec] -> parcel index in [0, d)

        agg = np.zeros((self.d, len(names)), dtype=np.float32)
        for e, p in enumerate(parcel_of_elec):
            agg[p, e] = 1.0
        # Row-normalise so each parcel is the MEAN of its electrodes.
        rowsum = agg.sum(1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        agg = agg / rowsum
        return names, parcel_of_elec, agg

    def tep_to_regions(self, tep: np.ndarray, ch_names: list[str]) -> np.ndarray:
        """electrode-space TEP ``[n_elec, T]`` -> region-space ``[d, T]``."""
        names, _, agg = self.assign_electrodes(ch_names)
        name_to_row = {n: i for i, n in enumerate(ch_names)}
        idx = [name_to_row[n] for n in names]
        return (agg @ tep[idx]).astype(np.float32)

    def site_to_parcel(self, site_mni: tuple[float, float, float]) -> int:
        """Index of the parcel whose centroid is nearest a given MNI coordinate."""
        valid = ~np.isnan(self.centroids).any(axis=1)
        cent = self.centroids.copy()
        cent[~valid] = 1e9
        return int(np.argmin(((cent - np.asarray(site_mni)) ** 2).sum(1)))

    def named_site_parcels(self) -> dict[str, int]:
        return {k: self.site_to_parcel(v) for k, v in CANONICAL_SITES_MNI.items()}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config

    cfg = load_config()
    bridge = EEGAtlasBridge(cfg)
    cent = bridge.centroids
    valid = ~np.isnan(cent).any(axis=1)
    print(f"parcel centroids: {cent.shape}, valid {valid.sum()}/{len(cent)}")
    print("centroid MNI range mm:", np.nanmin(cent, 0).round(1), np.nanmax(cent, 0).round(1))
    sites = bridge.named_site_parcels()
    print("named-site parcels (by nearest centroid):")
    for k, p in sites.items():
        print(f"  {k:11s} -> parcel {p:3d} @ MNI {cent[p].round(1)}")
    # sanity: electrode assignment on a standard 10-20 set
    demo = ["Fp1", "Fz", "Cz", "C3", "C4", "Pz", "P3", "P4", "O1", "O2"]
    names, pmap, agg = bridge.assign_electrodes(demo)
    print(f"electrode->parcel demo ({len(names)} placed):",
          dict(zip(names, pmap.tolist())))
