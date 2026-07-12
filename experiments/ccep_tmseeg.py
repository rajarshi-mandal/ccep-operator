"""FOLLOW-UP F2 — TMS-EEG translational bridge.

Does invasive CCEP connectivity predict NON-INVASIVE TMS-EEG responses? We use ds002094 (single-pulse
TMS to RIGHT motor cortex, 120% RMT, 30-ch EEG). For each subject: epoch around the pulse, clean the
TMS artifact, average -> TEP; source-localize to fsaverage with a template BEM + standard montage;
parcellate to Lausanne2008-250. The falsifiable test: does F-TRACT's 780-patient CCEP connectivity
FROM the right-M1 target parcel predict the source-space TEP topography across parcels, BEYOND
distance from the target?

  measured : parcel-level TEP magnitude in an early cortical window (15-50 ms).
  model    : F-TRACT amplitude[right-M1, parcel]  (invasive CCEP propagation from the target).
  baseline : distance from the right-M1 target parcel (geometry).

Honest failure modes: template head model, 30-ch resolution, source leakage, TMS artifact, and a
genuine modality difference (evoked-potential physiology) all cap the achievable correspondence. A
null is a real, reported bound on invasive->noninvasive transfer.

Output: reports/tmseeg.json.  Run: python experiments/ccep_tmseeg.py
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
TMS = Path("REDACTED/data/external/tmseeg")
FT = Path("REDACTED/data/external/ftract")
FSDIR = FT / "scripts/python/mne_plot_data/MNE-sample-data/subjects"
ANNOT = "Lausanne_250"
PARC = "Lausanne2008-250"
M1_RH = np.array([37.0, -21.0, 58.0])   # right-hand-knob M1 (MNI/fsaverage mm)
TEP_WIN = (0.015, 0.050)                # early cortical spread window (s)
ART_WIN = (-0.002, 0.011)               # TMS artifact blank (s)


def load_ft(feature):
    p = FT / "ages_15_100" / "sr_8.40" / "seg_None_None" / "pl_200" / PARC / "export" / feature / f"{feature}.csv"
    rows, header = [], None
    with open(p) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            if header is None and parts[0].strip() == "stimulated parcels":
                header = [c.strip() for c in parts[1:]]; continue
            if header is not None:
                rows.append(parts)
    idx = {p: k for k, p in enumerate(header)}
    M = np.full((len(header), len(header)), np.nan)
    for r in rows:
        sp = r[0].strip()
        if sp in idx:
            for j, v in enumerate(r[1:len(header) + 1]):
                try:
                    M[idx[sp], j] = float(v)
                except ValueError:
                    pass
    return header, idx, M


def get_labels():
    import mne
    labels = mne.read_labels_from_annot("fsaverage", ANNOT, "both", subjects_dir=str(FSDIR), verbose=False)
    labels = [l for l in labels if l.name.rsplit("-", 1)[0] not in ("unknown", "corpuscallosum", "???")]
    names = [f"{l.hemi}.{l.name.rsplit('-',1)[0]}" for l in labels]
    return labels, names


def parcel_centroids(labels, names):
    import mne
    fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    lh_v, _ = mne.read_surface(str(fs_dir / "surf/lh.pial"))
    rh_v, _ = mne.read_surface(str(fs_dir / "surf/rh.pial"))
    cents = {}
    for l, nm in zip(labels, names):
        v = lh_v if l.hemi == "lh" else rh_v
        cents[nm] = v[l.vertices].mean(0)
    return cents


def process_subject(vhdr, labels, names):
    import mne
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    raw.pick("eeg")
    raw.set_montage("standard_1020", match_case=False, on_missing="ignore")
    raw.set_eeg_reference("average", projection=True)
    # TMS pulse events (R128)
    events, event_id = mne.events_from_annotations(raw)
    pulse_ids = [v for k, v in event_id.items() if "128" in k or "Response" in k]
    if not pulse_ids:
        pulse_ids = list(event_id.values())
    ev = events[np.isin(events[:, 2], pulse_ids)]
    if len(ev) < 20:
        return None
    epochs = mne.Epochs(raw, ev, tmin=-0.1, tmax=0.35, baseline=(-0.1, -0.01),
                        preload=True, reject_by_annotation=False, verbose=False)
    # blank the TMS artifact window (linear interpolate across it)
    data = epochs.get_data()
    t = epochs.times
    art = (t >= ART_WIN[0]) & (t <= ART_WIN[1])
    if art.any():
        i0 = np.where(art)[0][0] - 1; i1 = np.where(art)[0][-1] + 1
        i0 = max(i0, 0); i1 = min(i1, len(t) - 1)
        for e in range(data.shape[0]):
            for c in range(data.shape[1]):
                data[e, c, art] = np.linspace(data[e, c, i0], data[e, c, i1], art.sum())
    epochs._data = data
    evoked = epochs.average()
    noise_cov = mne.compute_covariance(epochs, tmax=-0.01, method="empirical", verbose=False)

    fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    src = mne.read_source_spaces(str(fs_dir / "bem/fsaverage-ico-5-src.fif"), verbose=False)
    bem = str(fs_dir / "bem/fsaverage-5120-5120-5120-bem-sol.fif")
    fwd = mne.make_forward_solution(evoked.info, trans="fsaverage", src=src, bem=bem,
                                    eeg=True, mindist=5.0, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)
    stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2=1.0 / 9.0, method="dSPM", verbose=False)
    # early-window mean magnitude per vertex -> parcel via label mean
    win = (stc.times >= TEP_WIN[0]) & (stc.times <= TEP_WIN[1])
    vmag = np.abs(stc.data[:, win]).mean(1)   # [n_vertices] (lh then rh, ico-5)
    # extract per-label mean using label vertices mapped to src used-vertex ordering
    ltc = mne.extract_label_time_course(stc, labels, src, mode="mean", verbose=False)
    parc_tep = np.abs(ltc[:, win]).mean(1)    # [n_labels]
    return dict(zip(names, parc_tep))


def _rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return np.nan, int(ok.sum())
    ra, rb = _rank(a[ok]), _rank(b[ok])
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return (float((ra @ rb) / den) if den > 1e-12 else np.nan), int(ok.sum())


def _partial_spearman(a, b, c):
    """Spearman partial corr of a,b controlling for c (rank residualisation)."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    if ok.sum() < 8:
        return np.nan
    ra, rb, rc = _rank(a[ok]), _rank(b[ok]), _rank(c[ok])
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]
        return y - x1 @ np.linalg.lstsq(x1, y, rcond=None)[0]
    er_a, er_b = resid(ra, rc), resid(rb, rc)
    den = np.linalg.norm(er_a - er_a.mean()) * np.linalg.norm(er_b - er_b.mean())
    return float(((er_a - er_a.mean()) @ (er_b - er_b.mean())) / den) if den > 1e-12 else np.nan


def main():
    header, pidx, ft_amp = load_ft("amplitude")
    labels, names = get_labels()
    cents = parcel_centroids(labels, names)
    # target parcel = rh.precentral subdivision nearest the right-M1 hand knob
    cand = [(np.linalg.norm(cents[n] - M1_RH), n) for n in names if n.startswith("rh.precentral")]
    if not cand:
        cand = [(np.linalg.norm(cents[n] - M1_RH), n) for n in names if n in pidx]
    target = min(cand)[1]
    print(f"target parcel (right M1): {target}  (in F-TRACT: {target in pidx})")

    vhdrs = sorted(glob.glob(str(TMS / "sub-*/sub-*_task-tmseeg1_eeg.vhdr")))
    print(f"TMS-EEG subjects available: {len(vhdrs)}")
    raw_r, part_r, dist_r, subj = [], [], [], []
    for vhdr in vhdrs:
        sid = Path(vhdr).parent.name
        try:
            tep = process_subject(vhdr, labels, names)
        except Exception as e:
            print(f"  {sid}: FAIL {type(e).__name__}: {e}"); continue
        if tep is None:
            print(f"  {sid}: too few pulses"); continue
        # all parcels present in both TEP and F-TRACT (excluding the target itself)
        shared = [n for n in names if n in pidx and n != target and np.isfinite(ft_amp[pidx[target], pidx[n]])]
        tep_v = np.array([tep[n] for n in shared])
        ftp_v = np.array([ft_amp[pidx[target], pidx[n]] for n in shared])
        dst_v = np.array([-np.linalg.norm(cents[n] - cents[target]) for n in shared])  # nearer = larger
        rm, nm = _spearman(ftp_v, tep_v)          # raw CCEP->TEP
        rp = _partial_spearman(ftp_v, tep_v, dst_v)  # CCEP->TEP controlling for distance
        rd, _ = _spearman(dst_v, tep_v)           # distance->TEP
        raw_r.append(rm); part_r.append(rp); dist_r.append(rd); subj.append(sid)
        print(f"  {sid}: CCEP->TEP raw={rm:+.3f} partial|dist={rp:+.3f}  dist->TEP={rd:+.3f}  (n={nm})")

    if not raw_r:
        print("no subjects processed"); return
    from eval.stats import bootstrap_ci, paired_permutation_test  # noqa
    rm_, rl, rh_ = bootstrap_ci(raw_r); pm, pl, ph = bootstrap_ci([x for x in part_r if np.isfinite(x)])
    dm, dl, dh = bootstrap_ci(dist_r)
    print(f"\n=== TMS-EEG BRIDGE (n={len(raw_r)} subjects) ===")
    print(f"  distance     -> TEP           : rho {dm:+.3f} [{dl:+.3f}, {dh:+.3f}]  (does template localization work?)")
    print(f"  F-TRACT CCEP -> TEP (raw)     : rho {rm_:+.3f} [{rl:+.3f}, {rh_:+.3f}]")
    print(f"  F-TRACT CCEP -> TEP | distance: rho {pm:+.3f} [{pl:+.3f}, {ph:+.3f}]  (beyond geometry)")
    p_raw = paired_permutation_test(raw_r, [0.0] * len(raw_r))
    fp = [x for x in part_r if np.isfinite(x)]
    p_part = paired_permutation_test(fp, [0.0] * len(fp)) if fp else np.nan
    print(f"  raw vs 0: p={p_raw:.3g} ({sum(1 for x in raw_r if x>0)}/{len(raw_r)} >0);  "
          f"partial vs 0: p={p_part:.3g} ({sum(1 for x in fp if x>0)}/{len(fp)} >0)"
          + ("  <-- CCEP predicts TEP beyond geometry" if pm > 0.05 and p_part < 0.1 else "  (weak/null beyond geometry)"))

    out = {"target_parcel": target, "n_subjects": len(raw_r),
           "distance_to_tep_rho": float(dm), "distance_ci": [dl, dh],
           "ftract_raw_rho": float(rm_), "ftract_raw_ci": [rl, rh_],
           "ftract_partial_rho": float(pm), "ftract_partial_ci": [pl, ph],
           "p_raw_vs0": float(p_raw), "p_partial_vs0": float(p_part) if fp else None,
           "n_pos_partial": int(sum(1 for x in fp if x > 0)),
           "per_subject": [{"subject": s, "raw": r, "partial": pp, "distance": dr}
                           for s, r, pp, dr in zip(subj, raw_r, part_r, dist_r)]}
    (ROOT / "reports" / "tmseeg.json").write_text(json.dumps(out, indent=2))
    print("\nsaved reports/tmseeg.json")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    main()
