"""fMRI (observational) preprocessing — HCP PTN group-ICA node timeseries.

The HCP PTN package ships pre-parcellated resting-state timeseries: for the d100
group-ICA parcellation, each subject is a plain-text file of shape
``[T=4800, d=100]`` (4 runs x 1200 frames, TR = 0.72 s), already cleaned and
registered. There is therefore no voxel-level work to do here — the pipeline is:

    1. extract the ICAd{d} NodeTimeseries tarball once into data/raw/
    2. per subject: load -> (optionally) split into runs -> z-score each node
    3. cache a compact ``[n_subjects, T, d]`` float32 array + a manifest

This is the observational supervision signal ``y(t)`` for the SSM's L_obs term.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np

# Subject .txt is concatenated runs; recon2 HCP PTN = 4 runs x 1200 frames.
FRAMES_PER_RUN = 1200
N_RUNS = 4


def _tarball_path(hcp_ptn_dir: str | Path, ica_dim: int) -> Path:
    return Path(hcp_ptn_dir) / f"NodeTimeseries_3T_HCP1200_MSMAll_ICAd{ica_dim}_ts2.tar.gz"


def _extract_root(raw_dir: str | Path, ica_dim: int) -> Path:
    # The tar lays files under node_timeseries/3T_HCP1200_MSMAll_d{ica_dim}_ts2/
    return Path(raw_dir) / "node_timeseries" / f"3T_HCP1200_MSMAll_d{ica_dim}_ts2"


def extract_timeseries(hcp_ptn_dir: str | Path, raw_dir: str | Path, ica_dim: int) -> Path:
    """Extract the NodeTimeseries tarball into ``raw_dir`` (idempotent)."""
    out_root = _extract_root(raw_dir, ica_dim)
    if out_root.exists() and any(out_root.glob("*.txt")):
        return out_root
    tar = _tarball_path(hcp_ptn_dir, ica_dim)
    if not tar.exists():
        raise FileNotFoundError(f"HCP PTN tarball not found: {tar}")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        tf.extractall(raw_dir)
    return out_root


def list_subjects(hcp_ptn_dir: str | Path, raw_dir: str | Path, ica_dim: int) -> list[str]:
    """Subject IDs available as extracted timeseries files."""
    root = extract_timeseries(hcp_ptn_dir, raw_dir, ica_dim)
    return sorted(p.stem for p in root.glob("*.txt"))


def load_subject(
    subject_id: str,
    hcp_ptn_dir: str | Path,
    raw_dir: str | Path,
    ica_dim: int,
    split_runs: bool = False,
) -> np.ndarray:
    """Load one subject's node timeseries.

    Returns ``[T, d]`` float32, or ``[N_RUNS, frames, d]`` if ``split_runs``.
    """
    root = extract_timeseries(hcp_ptn_dir, raw_dir, ica_dim)
    path = root / f"{subject_id}.txt"
    ts = np.loadtxt(path, dtype=np.float32)  # [T, d]
    if split_runs:
        T = ts.shape[0]
        n_runs = T // FRAMES_PER_RUN
        ts = ts[: n_runs * FRAMES_PER_RUN].reshape(n_runs, FRAMES_PER_RUN, ts.shape[1])
    return ts


def zscore_nodes(ts: np.ndarray, axis: int = -2) -> np.ndarray:
    """Z-score each node's timeseries (mean 0, std 1) along the time axis."""
    mean = ts.mean(axis=axis, keepdims=True)
    std = ts.std(axis=axis, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((ts - mean) / std).astype(np.float32)


def preprocess_subject(
    subject_id: str,
    hcp_ptn_dir: str | Path,
    raw_dir: str | Path,
    ica_dim: int,
    zscore: bool = True,
) -> np.ndarray:
    """Load + (optionally) z-score one subject. Returns ``[T, d]`` float32."""
    ts = load_subject(subject_id, hcp_ptn_dir, raw_dir, ica_dim)
    if zscore:
        # z-score per run so cross-run scanner drift does not leak into the mean
        T = ts.shape[0]
        n_runs = max(1, T // FRAMES_PER_RUN)
        if T % FRAMES_PER_RUN == 0 and n_runs > 1:
            runs = ts.reshape(n_runs, FRAMES_PER_RUN, ts.shape[1])
            runs = zscore_nodes(runs, axis=1)
            ts = runs.reshape(T, ts.shape[1])
        else:
            ts = zscore_nodes(ts, axis=0)
    return ts


def build_observational_cache(cfg, max_subjects: int | None = None) -> Path:
    """Preprocess HCP PTN subjects and cache a stacked array + manifest.

    Writes ``observational_fmri.npy`` (``[n, T, d]``) and
    ``observational_fmri.json`` (subject ids, shape, config) into the processed dir.
    """
    hcp_ptn_dir = cfg.paths.hcp_ptn_dir
    raw_dir = Path(cfg.paths.processed_dir).parent / "raw"
    ica_dim = cfg.fmri.ica_dim
    proc = Path(cfg.paths.processed_dir)
    proc.mkdir(parents=True, exist_ok=True)

    subjects = list_subjects(hcp_ptn_dir, raw_dir, ica_dim)
    if max_subjects is None:
        max_subjects = cfg.fmri.get("max_subjects")
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    arrays = [
        preprocess_subject(s, hcp_ptn_dir, raw_dir, ica_dim, zscore=cfg.fmri.zscore)
        for s in subjects
    ]
    stacked = np.stack(arrays, axis=0)  # [n, T, d]

    npy_path = proc / "observational_fmri.npy"
    np.save(npy_path, stacked)
    manifest = {
        "subjects": subjects,
        "shape": list(stacked.shape),
        "ica_dim": ica_dim,
        "tr": cfg.fmri.tr,
        "zscored": bool(cfg.fmri.zscore),
        "frames_per_run": FRAMES_PER_RUN,
    }
    with open(proc / "observational_fmri.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return npy_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config

    cfg = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"Building observational fMRI cache for {n} subjects (smoke test)...")
    path = build_observational_cache(cfg, max_subjects=n)
    arr = np.load(path)
    print("cached:", path)
    print("shape :", arr.shape, "dtype:", arr.dtype)
    print("per-node mean ~0 / std ~1 check:",
          float(arr.mean()), float(arr.std()))
