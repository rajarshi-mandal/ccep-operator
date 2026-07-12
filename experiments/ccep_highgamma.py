"""EXTENSION (raw feature, bounded proof) — is the stimulation-evoked HIGH-GAMMA response
(70-150 Hz induced power) predictable, like the N1 amplitude?

High-gamma broadband power is a more 'activation-like' readout than the N1 evoked peak. We extract
it from the raw signal on the coordinate-rich subset (ds004774 + ds004696) and test whether a
held-out site's high-gamma topography is predictable (LOSO combo) and whether the network term
contributes — a bounded proof of whether this alternative readout carries the same predictable
structure. Output: reports/highgamma.json
"""
from __future__ import annotations
import json, os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import (CCEPSubject, _find_ieeg_dir, _sidecar, _load_signal,  # noqa
                                EPOCH, BASE_WIN, BLANK_WIN)
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa
import ccep_loso as L  # noqa

HG_WIN = (0.010, 0.150)   # early post-stim window for high-gamma power


def hg_subject(dataset_root, sub, cache):
    """Compute a high-gamma-power topography aligned to the cache's sites/contacts."""
    ieeg = _find_ieeg_dir(dataset_root, sub)
    ev = pd.read_csv(_sidecar(ieeg, "_events.tsv"), sep="\t")
    names, data, fs = _load_signal(ieeg)
    name2row = {n: i for i, n in enumerate(names)}
    contacts = cache.contacts
    rows = np.array([name2row[c] for c in contacts if c in name2row])
    cmap = {c: i for i, c in enumerate(contacts)}
    fs = float(fs)
    n_pre, n_post = int(-EPOCH[0] * fs), int(EPOCH[1] * fs)
    t = np.arange(-n_pre, n_post) / fs
    base_m = (t >= BASE_WIN[0]) & (t <= BASE_WIN[1]); hg_m = (t >= HG_WIN[0]) & (t <= HG_WIN[1])
    b, a = butter(4, [70, 150], btype="band", fs=fs)
    ev = ev[ev["status"] == "good"].copy() if "status" in ev else ev
    ev = ev[ev["electrical_stimulation_site"].notna()]
    hg = {}
    for site, grp in ev.groupby("electrical_stimulation_site"):
        onsets = (grp["onset"].to_numpy(float) * fs).astype(int)
        acc = np.zeros(len(contacts)); ntr = 0
        for o in onsets:
            s0, s1 = o - n_pre, o + n_post
            if s0 < 0 or s1 > data.shape[1]:
                continue
            seg = data[rows, s0:s1]
            filt = filtfilt(b, a, seg, axis=1)
            power = np.abs(hilbert(filt, axis=1)) ** 2          # [nc, win]
            base = power[:, base_m].mean(1, keepdims=True) + 1e-9
            norm = power / base
            acc += norm[:, hg_m].mean(1); ntr += 1
        if ntr >= 4:
            hg[str(site)] = acc
    # align to cache.sites order
    topo = np.full((len(cache.sites), len(contacts)), np.nan)
    for i, s in enumerate(cache.sites):
        if s in hg:
            v = hg[s].copy()
            for e in cache.stim_idx[i]:
                if e >= 0: v[e] = np.nan
            topo[i] = v
    return topo


def main():
    caches = [(d, c) for d, c in L.all_caches() if d in ("ds004774", "ds004696")]
    rows = {"within": [], "combo": [], "net": []}
    tags = []
    print(f"{'subject':16s} {'within':>7} {'combo':>7} {'net':>7}")
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        droot = str(PROJ / f"Open Neuro {ds}")
        try:
            hg = hg_subject(droot, cs.subject, cs)
        except Exception as e:
            print(f"  {cs.subject}: skip ({e})"); continue
        cs.responses = hg                       # swap target -> high-gamma
        e = L.eval_subject(cs); i = L.incremental_subject(cs)
        if e is None or i is None:
            continue
        s, _ = e
        rows["within"].append(s["within_mean"]); rows["combo"].append(s["combo"]); rows["net"].append(i["stim_knn"])
        tags.append(f"{ds[-4:]}/{cs.subject}")
        print(f"{tags[-1]:16s} {s['within_mean']:7.3f} {s['combo']:7.3f} {i['stim_knn']:7.3f}")

    n = len(tags); out = {"n": n}
    print(f"\n=== high-gamma predictability (n={n}, coord-rich subset) ===")
    for k in rows:
        m, lo, hi = bootstrap_ci(rows[k]); out[k] = {"mean": m, "lo": lo, "hi": hi}
        print(f"  {k:8s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    p = paired_permutation_test(rows["combo"], rows["within"]); d = cohens_d_paired(rows["combo"], rows["within"])
    out["combo_vs_within"] = {"delta": np.mean(rows["combo"]) - np.mean(rows["within"]), "p": p, "d": d}
    print(f"  combo vs within: Δ={np.mean(rows['combo'])-np.mean(rows['within']):+.3f} p={p:.3g} d={d:+.2f}")
    (ROOT / "reports" / "highgamma.json").write_text(json.dumps(out, indent=2))
    print("saved reports/highgamma.json")


if __name__ == "__main__":
    main()
