"""Torch datasets + collate for the joint observational/interventional training.

Two sample types share the d=100 region latent space:

  * Observational (HCP fMRI): windows of a z-scored ``[T, d]`` BOLD timeseries. Used by
    the SSM likelihood term L_obs.
  * Interventional (TMS-EEG): a ``(stim_parcel, region_tep [d, T_eeg])`` pair. Used by
    the do-operation loss L_int.

The trainer consumes the two streams separately (they have different shapes and losses),
so this module exposes them as two datasets plus simple collate functions rather than
forcing them into one awkward batch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------------
# Observational (fMRI)
# --------------------------------------------------------------------------------
class ObservationalDataset(Dataset):
    """Windows of HCP resting-state region timeseries.

    Each item is a ``[window, d]`` float32 tensor. Windows never cross run
    boundaries (HCP recon2 = 4 runs x 1200 frames), so SSM dynamics stay continuous.
    """

    def __init__(self, processed_dir: str | Path, window: int = 200,
                 stride: int | None = None, frames_per_run: int = 1200):
        arr = np.load(Path(processed_dir) / "observational_fmri.npy")  # [n, T, d]
        self.d = arr.shape[2]
        stride = stride or window
        self.windows = []  # (subject_idx, start)
        self._data = torch.from_numpy(arr)
        n, T, _ = arr.shape
        for s in range(n):
            for run_start in range(0, T, frames_per_run):
                run_end = min(run_start + frames_per_run, T)
                for st in range(run_start, run_end - window + 1, stride):
                    self.windows.append((s, st))
        self.window = window

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        s, st = self.windows[i]
        return self._data[s, st:st + self.window]  # [window, d]


def collate_observational(batch):
    return torch.stack(batch, dim=0)  # [B, window, d]


# --------------------------------------------------------------------------------
# Interventional (TMS-EEG, region space)
# --------------------------------------------------------------------------------
class InterventionalDataset(Dataset):
    """Region-space TEP records: (stim_parcel, region_tep [d, T_eeg]).

    ``site_filter`` keeps only records whose ``site_name`` is in the given set — used
    to build the held-out-site split for Experiment 1 (train M1, test parietal).
    """

    def __init__(self, processed_dir: str | Path, site_filter: set[str] | None = None,
                 normalize: bool = True):
        recs = list(np.load(Path(processed_dir) / "interventional_region.npz",
                            allow_pickle=True)["records"])
        if site_filter is not None:
            recs = [r for r in recs if r["site_name"] in site_filter]
        self.records = recs
        self.normalize = normalize
        self.d = recs[0]["region_tep"].shape[0] if recs else 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        tep = np.asarray(r["region_tep"], dtype=np.float32)  # [d, T]
        if self.normalize:
            # Per-record scale normalisation keeps EEG units from dominating the loss.
            s = np.std(tep)
            if s > 1e-8:
                tep = tep / s
        return {
            "stim_parcel": int(r["stim_parcel"]),
            "region_tep": torch.from_numpy(tep),  # [d, T]
            "site_name": r["site_name"],
            "dataset": r["dataset"],
            "subject": str(r.get("subject", "")),
        }


def collate_interventional(batch):
    return {
        "stim_parcel": torch.tensor([b["stim_parcel"] for b in batch], dtype=torch.long),
        "region_tep": torch.stack([b["region_tep"] for b in batch], dim=0),  # [B, d, T]
        "site_name": [b["site_name"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config

    cfg = load_config()
    proc = cfg.paths.processed_dir

    obs = ObservationalDataset(proc, window=200)
    print(f"Observational: {len(obs)} windows, item shape {tuple(obs[0].shape)}")
    ob = collate_observational([obs[0], obs[1], obs[2]])
    print("  batch:", tuple(ob.shape))

    itv = InterventionalDataset(proc)
    print(f"Interventional: {len(itv)} records, d={itv.d}")
    from collections import Counter
    print("  by site:", Counter(r["site_name"] for r in itv.records))
    ib = collate_interventional([itv[0], itv[1]])
    print("  stim_parcel:", ib["stim_parcel"].tolist(),
          "region_tep:", tuple(ib["region_tep"].shape))
