"""TMS-EEG (interventional) preprocessing for OpenNeuro ds004024.

ds004024 is BrainVision data sampled at 20 kHz, 64+ EEG channels, single-pulse TMS
(spTMS) to left/right M1. The TMS pulse produces a huge electromagnetic artifact that
must be removed *before* filtering (filtering would otherwise smear it across the
epoch). We follow the standard TESA-style pipeline:

    load -> epoch around pulse -> blank+interpolate the artifact gap -> downsample
    -> bandpass + notch -> average reference -> baseline correct -> average trials

The output per run is a TMS-evoked potential (TEP): ``[n_channels, n_times]`` plus the
time axis and channel MNI positions. The TEP is the interventional supervision signal
``L_int`` regresses against after the model's do-operation graph surgery.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------
# BIDS events
# --------------------------------------------------------------------------------
# Event labels that are never the TMS pulse marker.
_NON_PULSE_LABELS = {"New Segment/", "trial_type", ""}


def read_pulse_samples(vhdr_path: str | Path, pulse_label: str) -> np.ndarray:
    """Return TMS-pulse sample indices from the sibling BIDS ``*_events.tsv``.

    The pulse marker label is not consistent across ds004024 subjects (most use
    ``"Stimulus/A"``, but e.g. sub-CON008 uses ``"SB/A"``; ``"Out/A"`` is the
    simultaneous trigger-out). We prefer the configured label, else fall back to the
    most frequent stimulus marker that is not an "Out/" or segment marker.
    """
    vhdr_path = Path(vhdr_path)
    events_tsv = Path(str(vhdr_path).replace("_eeg.vhdr", "_events.tsv"))
    by_label: dict[str, list[int]] = {}
    with open(events_tsv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            lab = row.get("trial_type", "").strip()
            if lab in _NON_PULSE_LABELS:
                continue
            by_label.setdefault(lab, []).append(int(float(row["sample"])))

    if pulse_label in by_label:
        chosen = pulse_label
    else:
        # Prefer a genuine pulse marker (not the "Out/" trigger), most frequent first.
        non_out = {k: v for k, v in by_label.items() if not k.startswith("Out")}
        pool = non_out or by_label
        chosen = max(pool, key=lambda k: len(pool[k])) if pool else None
    if chosen is None:
        return np.asarray([], dtype=int)
    return np.asarray(sorted(by_label[chosen]), dtype=int)


# --------------------------------------------------------------------------------
# TMS artifact removal
# --------------------------------------------------------------------------------
def interpolate_artifact(
    epochs_data: np.ndarray, times: np.ndarray, blank_tmin: float, blank_tmax: float
) -> np.ndarray:
    """Cubic-spline interpolate across the blanked TMS-artifact window.

    ``epochs_data``: ``[n_epochs, n_ch, n_times]``. The window ``[blank_tmin, blank_tmax]``
    (seconds, relative to the pulse at t=0) is replaced by a cubic spline fit to the
    surrounding good samples, per epoch per channel.
    """
    from scipy.interpolate import CubicSpline

    bad = (times >= blank_tmin) & (times <= blank_tmax)
    good = ~bad
    good_t = times[good]
    out = epochs_data.copy()
    n_ep, n_ch, _ = epochs_data.shape
    for e in range(n_ep):
        for c in range(n_ch):
            cs = CubicSpline(good_t, epochs_data[e, c, good])
            out[e, c, bad] = cs(times[bad])
    return out


# --------------------------------------------------------------------------------
# Single run -> TEP
# --------------------------------------------------------------------------------
def process_run(vhdr_path: str | Path, cfg) -> dict:
    """Preprocess one ds004024 spTMS run into a trial-averaged TEP.

    Returns a dict with ``tep`` ``[n_ch, n_times]``, ``times`` ``[n_times]``,
    ``ch_names``, ``ch_pos`` ``[n_ch, 3]`` (montage MNI, metres), and ``n_trials``.
    """
    import mne

    mne.set_log_level("ERROR")
    e = cfg.tms_eeg

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True)
    # Keep only channels present in the standard montage (drops EOG/EMG/REF helpers).
    montage = mne.channels.make_standard_montage(e.montage)
    keep = [c for c in raw.ch_names if c in montage.ch_names]
    raw.pick(keep)
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    pulse_samples = read_pulse_samples(vhdr_path, e.ds004024_pulse_label)
    if pulse_samples.size == 0:
        raise RuntimeError(f"No TMS pulses found in {vhdr_path}")
    events = np.column_stack(
        [pulse_samples, np.zeros_like(pulse_samples), np.ones_like(pulse_samples)]
    ).astype(int)

    epochs = mne.Epochs(
        raw, events, event_id={"tms": 1},
        tmin=e.epoch_tmin, tmax=e.epoch_tmax,
        baseline=None, preload=True, reject=None,
    )

    # 1) Remove the TMS artifact by interpolation (before any filtering).
    data = epochs.get_data(copy=True)  # [n_ep, n_ch, n_times]
    data = interpolate_artifact(data, epochs.times, e.blank_tmin, e.blank_tmax)
    epochs = mne.EpochsArray(data, epochs.info, tmin=e.epoch_tmin)

    # 2) Downsample (20 kHz -> resample Hz) to make filtering/storage tractable.
    if e.resample:
        epochs.resample(e.resample)

    # 3) Bandpass + notch (operate on the data array: EpochsArray lacks notch_filter).
    sfreq = epochs.info["sfreq"]
    data = epochs.get_data(copy=True)
    data = mne.filter.filter_data(data, sfreq, e.bandpass[0], e.bandpass[1], verbose="ERROR")
    data = mne.filter.notch_filter(data, sfreq, freqs=[e.notch_ds004024], verbose="ERROR")
    epochs = mne.EpochsArray(data, epochs.info, tmin=epochs.tmin)

    # 4) Average reference.
    epochs.set_eeg_reference("average", projection=False)

    # 5) Baseline correct.
    epochs.apply_baseline(tuple(e.baseline))

    tep = epochs.average().get_data()  # [n_ch, n_times]

    pos = epochs.get_montage().get_positions()["ch_pos"]
    ch_pos = np.array([pos[c] for c in epochs.ch_names], dtype=np.float32)

    return {
        "tep": tep.astype(np.float32),
        "times": epochs.times.astype(np.float32),
        "ch_names": list(epochs.ch_names),
        "ch_pos": ch_pos,
        "n_trials": int(len(epochs)),
        "site": "M1",  # ds004024 spTMS targets M1 (left/right hemisphere)
    }


def find_sptms_runs(cfg) -> list[Path]:
    """All spTMS run .vhdr files in the downloaded ds004024 subset."""
    root = Path(cfg.paths.ds004024_dir)
    return sorted(root.glob("sub-*/ses-*/eeg/*task-spTMS*_eeg.vhdr"))


def build_ds004024_cache(cfg, max_runs: int | None = None) -> Path:
    """Preprocess every ds004024 spTMS run and cache a list of TEP records.

    Writes ``interventional_ds004024.npz`` (object array of per-run dicts) into the
    processed dir. Each record holds tep/times/ch_names/ch_pos/site/subject.
    """
    import json

    proc = Path(cfg.paths.processed_dir)
    proc.mkdir(parents=True, exist_ok=True)
    runs = find_sptms_runs(cfg)
    if max_runs is not None:
        runs = runs[:max_runs]

    records = []
    for vhdr in runs:
        subject = vhdr.name.split("_")[0]
        rec = process_run(vhdr, cfg)
        rec["subject"] = subject
        rec["dataset"] = "ds004024"
        records.append(rec)

    npz_path = proc / "interventional_ds004024.npz"
    np.savez_compressed(npz_path, records=np.array(records, dtype=object))
    manifest = {
        "n_runs": len(records),
        "subjects": [r["subject"] for r in records],
        "site": "M1",
        "tep_shape": list(records[0]["tep"].shape) if records else None,
        "sfreq": cfg.tms_eeg.resample,
    }
    with open(proc / "interventional_ds004024.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return npz_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config

    cfg = load_config()
    runs = find_sptms_runs(cfg)
    print(f"Found {len(runs)} spTMS runs.")
    out = process_run(runs[0], cfg)
    print("subject run:", runs[0].name)
    print("  TEP shape :", out["tep"].shape, "(n_ch x n_times)")
    print("  times     : %.3f .. %.3f s" % (out["times"][0], out["times"][-1]))
    print("  n_trials  :", out["n_trials"])
    print("  n_ch w/pos:", out["ch_pos"].shape, "site:", out["site"])
    # Peak GMFP (global mean field power) should occur shortly after the pulse.
    gmfp = out["tep"].std(axis=0)
    t_peak = out["times"][np.argmax(gmfp)]
    print("  GMFP peak at t = %.3f s (expect ~0.02-0.05 s post-pulse)" % t_peak)
