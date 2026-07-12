#!/usr/bin/env python
"""CCEP de-risk: load one ds004774 MEF3 subject, epoch around pulses, confirm a clean N1.

Gating question (per HANDOFF §8 / CCEP_ADAPTATION_SCOPE step 1):
  - Does MEF3 load cleanly via pymef?
  - Is there a clear N1 (~10-50 ms) at a near contact after a stim pulse?
If both pass, the CCEP pivot is viable.

Usage: ../.venv/bin/python scripts/ccep_derisk.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUB = "sub-MAYO01"
SES = "ses-ieeg01"
IEEG = os.path.join(
    ROOT, "Open Neuro ds004774", SUB, SES, "ieeg"
)
BASE = f"{SUB}_{SES}"
MEFD = os.path.join(IEEG, f"{BASE}_task-ccep_run-01_ieeg.mefd")
EVENTS = os.path.join(IEEG, f"{BASE}_task-ccep_run-01_events.tsv")
CHANNELS = os.path.join(IEEG, f"{BASE}_task-ccep_run-01_channels.tsv")
ELECTRODES = os.path.join(IEEG, f"{BASE}_electrodes.tsv")


def main():
    from pymef.mef_session import MefSession

    print(f"[1] Opening MEF3 session: {MEFD}")
    # MEF3 sessions in this dataset are unencrypted -> password None
    ms = MefSession(MEFD, None)
    bi = ms.read_ts_channel_basic_info()
    chan_names = [c["name"] for c in bi]
    fs = bi[0]["fsamp"][0] if isinstance(bi[0]["fsamp"], (list, np.ndarray)) else bi[0]["fsamp"]
    n_samp = bi[0]["nsamp"][0] if isinstance(bi[0]["nsamp"], (list, np.ndarray)) else bi[0]["nsamp"]
    print(f"    channels: {len(chan_names)}  fs={fs} Hz  nsamp={n_samp}  dur={n_samp/fs:.1f}s")
    print(f"    first chans: {chan_names[:6]}")

    # --- load events ---
    ev = pd.read_csv(EVENTS, sep="\t")
    ev = ev[ev["status"] == "good"].copy()
    ch = pd.read_csv(CHANNELS, sep="\t")
    el = pd.read_csv(ELECTRODES, sep="\t")
    good_ch = set(ch[ch["status"] == "good"]["name"]) & set(chan_names)
    print(f"[2] events: {len(ev)} good pulses, {ev['electrical_stimulation_site'].nunique()} sites")
    print(f"    good recording channels: {len(good_ch)}")

    fs = float(fs)
    # epoch window
    pre, post = 0.5, 1.0
    n_pre, n_post = int(pre * fs), int(post * fs)
    t = (np.arange(-n_pre, n_post) / fs)

    # Read full continuous data once (faster than per-epoch reads).
    print("[3] Reading continuous data for all channels ...")
    data = ms.read_ts_channels_sample(chan_names, [[0, int(n_samp)]])
    data = np.asarray(data)  # [n_chan, n_samp]
    print(f"    data shape: {data.shape}")
    name2idx = {n: i for i, n in enumerate(chan_names)}

    # pick the stim site with the most pulses
    site_counts = ev["electrical_stimulation_site"].value_counts()
    site = site_counts.index[0]
    site_ev = ev[ev["electrical_stimulation_site"] == site]
    pair = site.split("-")
    print(f"[4] Testing stim site '{site}' with {len(site_ev)} pulses; stim pair={pair}")

    # onsets in samples (onset is seconds from recording start)
    onsets = (site_ev["onset"].values * fs).astype(int)

    # Build epochs [n_trials, n_chan, n_time]
    win = n_pre + n_post
    epochs = np.full((len(onsets), len(chan_names), win), np.nan)
    for i, o in enumerate(onsets):
        s0, s1 = o - n_pre, o + n_post
        if s0 < 0 or s1 > data.shape[1]:
            continue
        epochs[i] = data[:, s0:s1]

    # baseline correct on pre-stim (-0.5..-0.05 s), blank stim artifact (-2..+10 ms)
    base_mask = (t >= -0.5) & (t <= -0.05)
    epochs = epochs - np.nanmean(epochs[:, :, base_mask], axis=2, keepdims=True)
    blank = (t >= -0.002) & (t <= 0.010)
    epochs[:, :, blank] = np.nan

    avg = np.nanmean(epochs, axis=0)  # [n_chan, n_time]

    # N1 window 10-50 ms, exclude stim pair + bad chans
    n1_mask = (t >= 0.010) & (t <= 0.050)
    n1_amp = {}
    for nm in chan_names:
        if nm in pair or nm not in good_ch:
            continue
        seg = avg[name2idx[nm], n1_mask]
        if np.all(np.isnan(seg)):
            continue
        # N1 is typically a sharp deflection; use peak absolute amplitude
        n1_amp[nm] = np.nanmax(np.abs(seg))

    # baseline std per channel to define an SNR threshold
    base_std = {nm: np.nanstd(avg[name2idx[nm], base_mask]) for nm in n1_amp}
    snr = {nm: n1_amp[nm] / (base_std[nm] + 1e-9) for nm in n1_amp}

    ranked = sorted(snr.items(), key=lambda x: -x[1])
    print("[5] Top-10 responding contacts by N1 SNR (peak|10-50ms| / baseline-std):")
    for nm, s in ranked[:10]:
        print(f"    {nm:8s}  N1amp={n1_amp[nm]:8.1f}uV  SNR={s:6.1f}")

    n_sig = sum(1 for nm in snr if snr[nm] > 3 and n1_amp[nm] > 50)
    print(f"[6] contacts with N1 SNR>3 & amp>50uV: {n_sig}/{len(snr)}")

    # also report the peak latency of the strongest contact
    if ranked:
        best = ranked[0][0]
        seg = avg[name2idx[best], n1_mask]
        lat = t[n1_mask][np.nanargmax(np.abs(seg))]
        print(f"    strongest contact {best}: N1 peak latency = {lat*1000:.1f} ms")

    # save a small figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        tw = (t >= -0.1) & (t <= 0.5)
        for nm, _ in ranked[:5]:
            axes[0].plot(t[tw] * 1000, avg[name2idx[nm], tw], label=nm)
        axes[0].axvline(0, color="k", lw=0.5)
        axes[0].axvspan(10, 50, color="orange", alpha=0.15, label="N1 win")
        axes[0].set_xlabel("ms"); axes[0].set_ylabel("uV")
        axes[0].set_title(f"{SUB} stim {site}: top-5 CCEP responses")
        axes[0].legend(fontsize=7)
        # butterfly
        for nm in good_ch:
            if nm not in pair:
                axes[1].plot(t[tw] * 1000, avg[name2idx[nm], tw], lw=0.4, alpha=0.5)
        axes[1].axvline(0, color="k", lw=0.5)
        axes[1].set_xlabel("ms"); axes[1].set_title("butterfly (all good contacts)")
        out = os.path.join(os.path.dirname(__file__), "..", "reports", "ccep_derisk_MAYO01.png")
        fig.tight_layout(); fig.savefig(out, dpi=110)
        print(f"[7] saved figure: {os.path.abspath(out)}")
    except Exception as e:
        print(f"[7] figure skipped: {e}")

    ms.close()
    verdict = "VIABLE" if n_sig >= 3 and ranked and 0.008 <= lat <= 0.060 else "CHECK"
    print(f"\n=== DE-RISK VERDICT: {verdict} ===")
    print(f"    MEF3 loaded: YES | clean N1: {'YES' if n_sig>=3 else 'WEAK'} | n_significant_contacts={n_sig}")


if __name__ == "__main__":
    main()
