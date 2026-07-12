"""Zenodo parietal TMS-EEG loader (record 4990628).

This dataset is already epoched and baseline-corrected: per subject a ``.mat`` file
holds six conditions, each ``[66 ch, 5160 t, ~110 trials]`` sampled at 2048 Hz over
-510..+2009 ms, with the TMS pulse at sample index 1045 and the device artifact
already linearly interpolated between samples 993..1066. Channel locations (EEGLAB
chanlocs X/Y/Z) ship inside the file.

Because the heavy artifact removal is already done, our job is light: crop to the
analysis window, light filter, downsample to match ds004024, trial-average to a TEP,
and expose channel positions for the spatial bridge.

NOTE on conditions: the six conditions correspond to distinct stimulation
conditions/sites in the original study. The exact cond->site label mapping is NOT in
the per-subject readme; it must be confirmed from the Zenodo record before these are
used as named held-out sites in Experiment 1. For now they are carried as cond1..6.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# EEGLAB chanlocs use an X/Y/Z convention rotated vs. MNE head coords; we only need
# relative geometry for nearest-centroid bridging, so we keep raw chanloc XYZ and
# normalise later in the bridge module.
N_COND = 6
PULSE_IDX_DEFAULT = 1045


def list_subjects(cfg) -> list[str]:
    root = Path(cfg.paths.zenodo_dir)
    return sorted(p.name.split("_")[0] for p in root.glob("*_EEG_data.mat"))


def _load_mat(path: str | Path):
    import scipy.io as sio

    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return m["data"]


def _channel_positions(channel_info) -> tuple[list[str], np.ndarray]:
    names, xyz = [], []
    for ch in channel_info:
        names.append(str(ch.labels))
        # Some channels (EOG) may have empty coords; guard with nan.
        try:
            xyz.append([float(ch.X), float(ch.Y), float(ch.Z)])
        except (ValueError, TypeError):
            xyz.append([np.nan, np.nan, np.nan])
    return names, np.asarray(xyz, dtype=np.float32)


def process_subject(subject_id: str, cfg) -> list[dict]:
    """Load one Zenodo subject and return one TEP record per condition.

    Each record: tep ``[n_ch, n_times]``, times (s), ch_names, ch_pos, n_trials,
    cond index, site label (provisional), subject, dataset.
    """
    import mne

    mne.set_log_level("ERROR")
    e = cfg.tms_eeg
    path = Path(cfg.paths.zenodo_dir) / f"{subject_id}_EEG_data.mat"
    data = _load_mat(path)

    time_ms = np.asarray(data.time_stamp, dtype=np.float64)
    times_s = time_ms / 1000.0
    sfreq = e.zenodo_sfreq
    ch_names, ch_pos = _channel_positions(data.channel_info)

    # Restrict to the same analysis window as ds004024 for consistency.
    win = (times_s >= e.epoch_tmin) & (times_s <= e.epoch_tmax)

    records = []
    for c in range(1, N_COND + 1):
        arr = np.asarray(getattr(data, f"cond{c}"), dtype=np.float32)  # [ch, t, trial]
        n_trials = arr.shape[2]
        # Trial-average -> evoked [ch, t]
        evoked = arr.mean(axis=2)
        evoked = evoked[:, win]
        win_times = times_s[win]

        # Light bandpass to match ds004024 processing (artifact already interpolated).
        evoked = mne.filter.filter_data(
            evoked.astype(np.float64), sfreq, e.bandpass[0], e.bandpass[1], verbose="ERROR"
        )
        evoked = mne.filter.notch_filter(
            evoked, sfreq, freqs=[e.notch_zenodo], verbose="ERROR"
        )

        # Downsample to the target rate (resample along time axis via MNE resample).
        if e.resample and e.resample != sfreq:
            evoked = mne.filter.resample(
                evoked, up=e.resample, down=sfreq, axis=-1, verbose="ERROR"
            )
            n_t = evoked.shape[-1]
            win_times = np.linspace(win_times[0], win_times[-1], n_t)

        records.append({
            "tep": evoked.astype(np.float32),
            "times": win_times.astype(np.float32),
            "ch_names": ch_names,
            "ch_pos": ch_pos,
            "n_trials": int(n_trials),
            "cond": c,
            "site": f"cond{c}",          # provisional; confirm mapping from record 4990628
            "subject": subject_id,
            "dataset": "zenodo_parietal",
        })
    return records


def build_zenodo_cache(cfg, max_subjects: int | None = None) -> Path:
    """Preprocess Zenodo subjects and cache all per-condition TEP records."""
    proc = Path(cfg.paths.processed_dir)
    proc.mkdir(parents=True, exist_ok=True)
    subjects = list_subjects(cfg)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    records = []
    for s in subjects:
        records.extend(process_subject(s, cfg))

    npz_path = proc / "interventional_zenodo.npz"
    np.savez_compressed(npz_path, records=np.array(records, dtype=object))
    manifest = {
        "n_subjects": len(subjects),
        "n_records": len(records),
        "subjects": subjects,
        "n_conditions": N_COND,
        "tep_shape": list(records[0]["tep"].shape) if records else None,
    }
    with open(proc / "interventional_zenodo.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return npz_path


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import load_config

    cfg = load_config()
    subs = list_subjects(cfg)
    print(f"Found {len(subs)} Zenodo subjects.")
    recs = process_subject(subs[0], cfg)
    print(f"subject {subs[0]}: {len(recs)} condition records")
    r = recs[0]
    print("  TEP shape :", r["tep"].shape, "(n_ch x n_times)")
    print("  times     : %.3f .. %.3f s" % (r["times"][0], r["times"][-1]))
    print("  n_trials  :", r["n_trials"], "site:", r["site"])
    print("  ch w/pos  :", r["ch_pos"].shape, "n_named:", len(r["ch_names"]))
    gmfp = r["tep"].std(axis=0)
    print("  GMFP peak at t = %.3f s" % r["times"][np.argmax(gmfp)])
