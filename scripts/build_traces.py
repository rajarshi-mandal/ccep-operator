"""Extract trial-averaged full evoked TRACES (not just the N1 peak) for the T1.2 dynamical-system
model. Reuses the exact ccep_pipeline epoching (contact selection, baseline, blank, averaging) but
stores the compact post-stim spatiotemporal tensor per subject.

Cache layout (data/traces/<ds>/<sub>.npz):
  traces   [n_sites, n_contacts, T]  trial-averaged evoked voltage, post-stim, downsampled (float32)
  t_ms     [T]                       time axis (ms post-stim)
  contacts, contact_xyz, stim_xyz, stim_idx, reliability, n_trials   (aligned to the scalar cache)

Usage: python scripts/build_traces.py <dataset> <sub1> <sub2> ...
"""
from __future__ import annotations
import sys, os, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data.ccep_pipeline as P  # noqa: E402

TRACE_WIN = (0.011, 0.500)   # post-stim, after the artifact blank
TARGET_FS = 256.0            # downsample target


def build_traces(dataset_root, sub, verbose=True):
    ieeg = P._find_ieeg_dir(dataset_root, sub)
    ev = pd.read_csv(P._sidecar(ieeg, "_events.tsv"), sep="\t")
    ch = pd.read_csv(P._sidecar(ieeg, "_channels.tsv"), sep="\t")
    el = pd.read_csv(P._sidecar(ieeg, "_electrodes.tsv"), sep="\t")
    names, data, fs = P._load_signal(ieeg)
    name2row = {n: i for i, n in enumerate(names)}
    good = set(ch[ch["status"] == "good"]["name"]) if "status" in ch else set(ch["name"])
    if "type" in ch:
        good &= set(ch[~ch["type"].isin({"ECG", "EKG"})]["name"])
    el = el[el["name"].isin(good) & el["name"].isin(names)].dropna(subset=["x", "y", "z"]).copy()
    contacts = list(el["name"])
    contact_xyz = el[["x", "y", "z"]].to_numpy(dtype=float)
    cidx = {n: i for i, n in enumerate(contacts)}
    n_c = len(contacts)
    fs = float(fs)
    n_pre, n_post = int(-P.EPOCH[0] * fs), int(P.EPOCH[1] * fs)
    t = np.arange(-n_pre, n_post) / fs
    base_m = (t >= P.BASE_WIN[0]) & (t <= P.BASE_WIN[1])
    blank_m = (t >= P.BLANK_WIN[0]) & (t <= P.BLANK_WIN[1])
    n1_m = (t >= P.N1_WIN[0]) & (t <= P.N1_WIN[1])
    tr_m = (t >= TRACE_WIN[0]) & (t <= TRACE_WIN[1])
    t_win = t[tr_m]
    # downsample indices
    step = max(1, int(round(fs / TARGET_FS)))
    ds_idx = np.arange(0, t_win.shape[0], step)
    t_ms = t_win[ds_idx] * 1000.0
    ev = ev[ev["status"] == "good"].copy() if "status" in ev else ev
    ev = ev[ev["electrical_stimulation_site"].notna()]
    contact_rows = np.array([name2row[c] for c in contacts])

    sites, traces, stim_xyz, stim_idx, reliab, ntrials = [], [], [], [], [], []
    for site, grp in ev.groupby("electrical_stimulation_site"):
        pair = str(site).split("-")
        onsets = (grp["onset"].to_numpy(dtype=float) * fs).astype(int)
        win = n_pre + n_post
        ep = np.full((len(onsets), n_c, win), np.nan)
        for i, o in enumerate(onsets):
            s0, s1 = o - n_pre, o + n_post
            if 0 <= s0 and s1 <= data.shape[1]:
                ep[i] = data[contact_rows, s0:s1]
        ep = ep - np.nanmean(ep[:, :, base_m], axis=2, keepdims=True)
        ep[:, :, blank_m] = np.nan
        valid = ~np.all(np.isnan(ep[:, 0, :]), axis=1)
        ep = ep[valid]
        if ep.shape[0] < 4:
            continue
        avg = np.nanmean(ep, axis=0)                      # [n_c, win]
        tr = avg[:, tr_m][:, ds_idx]                      # [n_c, T]
        # N1 split-half reliability (Spearman-Brown), same target as the scalar cache
        n = ep.shape[0]
        def _n1(es):
            a = np.nanmean(es, axis=0)[:, n1_m]
            j = np.nanargmax(np.abs(a), axis=1)
            return np.abs(a[np.arange(a.shape[0]), j])
        h1, h2 = _n1(ep[: n // 2]), _n1(ep[n // 2:])
        ok = np.isfinite(h1) & np.isfinite(h2)
        if ok.sum() >= 4:
            rr = np.corrcoef(h1[ok], h2[ok])[0, 1]
            rel = (2 * rr) / (1 + rr) if np.isfinite(rr) else np.nan
        else:
            rel = np.nan
        coords = el[["x", "y", "z"]].to_numpy(dtype=float)
        pcoord = [coords[cidx[p]] for p in pair if p in cidx]
        sites.append(str(site))
        traces.append(tr.astype(np.float32))
        stim_xyz.append(np.mean(pcoord, axis=0) if pcoord else np.array([np.nan] * 3))
        stim_idx.append([cidx.get(p, -1) for p in pair] + [-1] * (2 - len(pair)))
        reliab.append(float(rel))
        ntrials.append(int(n))
    if not sites:
        raise RuntimeError("no sites")
    return {
        "subject": sub, "contacts": np.array(contacts, dtype=object),
        "contact_xyz": contact_xyz, "sites": np.array(sites, dtype=object),
        "traces": np.array(traces, dtype=np.float32), "t_ms": t_ms.astype(np.float32),
        "stim_xyz": np.array(stim_xyz), "stim_idx": np.array(stim_idx, dtype=int),
        "reliability": np.array(reliab), "n_trials": np.array(ntrials), "fs": fs,
    }


def main():
    dataset = sys.argv[1]
    subs = sys.argv[2:]
    root = f"REDACTED/Open Neuro {dataset}"
    outdir = ROOT / "data" / "traces" / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    for sub in subs:
        out = outdir / f"{sub}.npz"
        if out.exists():
            print(f"skip {sub} (exists)"); continue
        try:
            d = build_traces(root, sub)
            np.savez_compressed(out, **d)
            print(f"OK {sub}: sites={len(d['sites'])} contacts={len(d['contacts'])} "
                  f"T={d['traces'].shape[2]} -> {out.name}")
        except Exception as e:
            print(f"FAIL {sub}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
