"""Loader for DANDI:000774 (Denman lab) — mouse intracortical electrical stimulation + spiking.

The dense + high-trial real-data candidate from the dataset hunt. Each NWB session has a units table
(spike_times + Allen-CCF coordinates) and a trials table of single-pulse stimulation events with the
stimulating contact pair and amplitude. We build, per session, the evoked-spiking response
topography over units for each stimulation site, using the strongest amplitude for best SNR.

Per session cache:
    responses   [n_sites, n_units]   trial-averaged evoked spike count (post 2-30ms minus baseline)
    resp_h1/h2  [n_sites, n_units]   half-trial splits (for the noise ceiling)
    unit_xyz    [n_units, 3]         Allen-CCF coordinates
    stim_xyz    [n_sites, 3]         mean coord of the stimulated contact pair (if available)
    reliability [n_sites]            split-half (Spearman-Brown) reliability across trials
    n_trials    [n_sites]
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass

import numpy as np

warnings.filterwarnings("ignore")

EVOKED = (0.002, 0.030)     # post-stim spike-count window (s)
BASELINE = (-0.040, -0.002)


@dataclass
class DandiSession:
    session: str
    unit_xyz: np.ndarray
    sites: list
    responses: np.ndarray
    resp_h1: np.ndarray
    resp_h2: np.ndarray
    stim_xyz: np.ndarray
    reliability: np.ndarray
    n_trials: np.ndarray

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, session=self.session, unit_xyz=self.unit_xyz,
                            sites=np.array(self.sites, dtype=object), responses=self.responses,
                            resp_h1=self.resp_h1, resp_h2=self.resp_h2, stim_xyz=self.stim_xyz,
                            reliability=self.reliability, n_trials=self.n_trials)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        return DandiSession(str(z["session"]), z["unit_xyz"], list(z["sites"]), z["responses"],
                            z["resp_h1"], z["resp_h2"], z["stim_xyz"], z["reliability"], z["n_trials"])


def _evoked(spk, onsets, n_u):
    ew, bw = EVOKED[1] - EVOKED[0], BASELINE[1] - BASELINE[0]
    R = np.zeros((len(onsets), n_u))
    for j, t0 in enumerate(onsets):
        for i, st in enumerate(spk):
            ev = np.searchsorted(st, [t0 + EVOKED[0], t0 + EVOKED[1]])
            bl = np.searchsorted(st, [t0 + BASELINE[0], t0 + BASELINE[1]])
            R[j, i] = (ev[1] - ev[0]) - (bl[1] - bl[0]) * (ew / bw)
    return R


def build_session(nwb_path: str, min_trials: int = 20) -> DandiSession:
    from pynwb import NWBHDF5IO
    io = NWBHDF5IO(nwb_path, "r", load_namespaces=True)
    nwb = io.read()
    tr = nwb.trials.to_dataframe()
    tr["site"] = tr["contact_negative"].astype(str) + "_" + tr["contact_positive"].astype(str)
    u = nwb.units.to_dataframe()
    n_u = len(u)
    unit_xyz = np.vstack(u["ccf_coordinates"].values).astype(float)
    spk = [np.asarray(s) for s in u["spike_times"].values]

    # strongest amplitude only (best SNR); keep its sites with enough trials
    amax = tr["amplitude"].abs().max()
    use = tr[tr["amplitude"].abs() == amax]
    sites = [s for s in sorted(use["site"].unique())
             if (use["site"] == s).sum() >= min_trials]

    resp, h1, h2, sxyz, rel, ntr = [], [], [], [], [], []
    for s in sites:
        rows = use[use["site"] == s]
        on = rows["start_time"].values
        R = _evoked(spk, on, n_u)
        resp.append(R.mean(0))
        a, b = R[: len(R) // 2].mean(0), R[len(R) // 2:].mean(0)
        h1.append(a); h2.append(b)
        rr = np.corrcoef(a, b)[0, 1]
        rel.append((2 * rr) / (1 + rr) if np.isfinite(rr) and rr > -1 else np.nan)
        ntr.append(len(on))
        # stim coord from contact coords if present + finite
        try:
            cp = np.atleast_1d(rows["contact_positive_coords"].iloc[0]).astype(float)
            cn = np.atleast_1d(rows["contact_negative_coords"].iloc[0]).astype(float)
            sxyz.append(np.nanmean([cp[:3], cn[:3]], axis=0))
        except Exception:
            sxyz.append(np.array([np.nan] * 3))
    io.close()
    return DandiSession(os.path.basename(nwb_path).split("_")[0], unit_xyz, sites,
                        np.array(resp), np.array(h1), np.array(h2), np.array(sxyz),
                        np.array(rel), np.array(ntr, dtype=int))


def all_sessions(root="../Open Neuro ds000774"):
    return sorted(glob.glob(os.path.join(root, "*.nwb")))
