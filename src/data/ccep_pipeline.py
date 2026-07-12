"""CCEP / iEEG single-pulse pipeline (ds004774 "ER-Detect" and BIDS-iEEG siblings).

Breaks the sites/subject ceiling that capped es-fMRI (~7 sites): CCEP stimulates dozens–100+
electrode pairs per subject, so the subject-specific propagation operator becomes estimable.

Per subject this builds a cache with the same shape the readout expects:

    responses   [n_sites, n_contacts]  trial-averaged CCEP N1 amplitude topography (uV),
                                        NaN where contact == stimulated pair or a bad channel
    responses_signed  [n_sites, n_contacts]  signed N1 (for diagnostics)
    n2          [n_sites, n_contacts]  trial-averaged N2 amplitude (uV)
    reliability [n_sites]              split-half (Spearman-Brown) reliability of the topography
    stim_xyz    [n_sites, 3]           mean coord of the stimulated pair
    stim_idx    [n_sites, 2]           indices (into contacts) of the stimulated pair, -1 if absent
    contact_xyz [n_contacts, 3]        electrode coords (subject native, mm)
    contacts    [n_contacts]           contact names (good recording channels with coords)
    n_trials    [n_sites]              good pulses averaged per site

The signal is MEF3 (read via pymef). BrainVision (.vhdr) subjects are read via MNE if present.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# CCEP component windows (seconds post-onset)
N1_WIN = (0.010, 0.100)   # early direct response
N2_WIN = (0.100, 0.300)   # later/indirect
CRP_WIN = (0.010, 0.300)  # canonical-response window (waveform-shape / matched-filter target)
BASE_WIN = (-0.5, -0.05)  # pre-stim baseline
BLANK_WIN = (-0.002, 0.010)  # stimulus-artifact blank
EPOCH = (-0.5, 1.0)


@dataclass
class CCEPSubject:
    subject: str
    contacts: list
    contact_xyz: np.ndarray      # [n_contacts, 3]
    sites: list                  # stim-site labels e.g. "LTG1-LTG2"
    responses: np.ndarray        # [n_sites, n_contacts] abs N1 (uV), NaN excluded
    responses_signed: np.ndarray
    n2: np.ndarray
    stim_xyz: np.ndarray         # [n_sites, 3]
    stim_idx: np.ndarray         # [n_sites, 2] int
    reliability: np.ndarray      # [n_sites]
    n_trials: np.ndarray         # [n_sites]
    fs: float
    responses_h1: np.ndarray = None   # [n_sites, n_contacts] abs N1 from first half of trials
    responses_h2: np.ndarray = None   # [n_sites, n_contacts] abs N1 from second half of trials
    responses_crp: np.ndarray = None  # [n_sites, n_contacts] canonical-shape (CRP) amplitude
    crp_h1: np.ndarray = None         # CRP amplitude, first half of trials
    crp_h2: np.ndarray = None         # CRP amplitude, second half of trials
    reliability_crp: np.ndarray = None  # [n_sites] split-half reliability of the CRP target
    latency: np.ndarray = None        # [n_sites, n_contacts] N1 peak latency (ms), NaN excluded
    latency_h1: np.ndarray = None     # N1 latency (ms), first half of trials
    latency_h2: np.ndarray = None     # N1 latency (ms), second half of trials

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            subject=self.subject,
            contacts=np.array(self.contacts, dtype=object),
            contact_xyz=self.contact_xyz,
            sites=np.array(self.sites, dtype=object),
            responses=self.responses,
            responses_signed=self.responses_signed,
            n2=self.n2,
            stim_xyz=self.stim_xyz,
            stim_idx=self.stim_idx,
            reliability=self.reliability,
            n_trials=self.n_trials,
            fs=self.fs,
            responses_h1=self.responses_h1 if self.responses_h1 is not None else np.zeros(0),
            responses_h2=self.responses_h2 if self.responses_h2 is not None else np.zeros(0),
            responses_crp=self.responses_crp if self.responses_crp is not None else np.zeros(0),
            crp_h1=self.crp_h1 if self.crp_h1 is not None else np.zeros(0),
            crp_h2=self.crp_h2 if self.crp_h2 is not None else np.zeros(0),
            reliability_crp=self.reliability_crp if self.reliability_crp is not None else np.zeros(0),
            latency=self.latency if self.latency is not None else np.zeros(0),
            latency_h1=self.latency_h1 if self.latency_h1 is not None else np.zeros(0),
            latency_h2=self.latency_h2 if self.latency_h2 is not None else np.zeros(0),
        )

    @staticmethod
    def load(path: str) -> "CCEPSubject":
        z = np.load(path, allow_pickle=True)
        h1 = z["responses_h1"] if "responses_h1" in z.files else None
        h2 = z["responses_h2"] if "responses_h2" in z.files else None
        def _opt(key):
            v = z[key] if key in z.files else None
            return v if (v is not None and v.size) else None
        return CCEPSubject(
            subject=str(z["subject"]),
            contacts=list(z["contacts"]),
            contact_xyz=z["contact_xyz"],
            sites=list(z["sites"]),
            responses=z["responses"],
            responses_signed=z["responses_signed"],
            n2=z["n2"],
            stim_xyz=z["stim_xyz"],
            stim_idx=z["stim_idx"],
            reliability=z["reliability"],
            n_trials=z["n_trials"],
            fs=float(z["fs"]),
            responses_h1=h1 if (h1 is not None and h1.size) else None,
            responses_h2=h2 if (h2 is not None and h2.size) else None,
            responses_crp=_opt("responses_crp"),
            crp_h1=_opt("crp_h1"),
            crp_h2=_opt("crp_h2"),
            reliability_crp=_opt("reliability_crp"),
            latency=_opt("latency"),
            latency_h1=_opt("latency_h1"),
            latency_h2=_opt("latency_h2"),
        )


# ----------------------------------------------------------------------------- IO


def _find_ieeg_dir(dataset_root: str, sub: str) -> str:
    hits = glob.glob(os.path.join(dataset_root, sub, "ses-*", "ieeg"))
    if not hits:
        raise FileNotFoundError(f"no ieeg dir for {sub} under {dataset_root}")
    return sorted(hits)[0]


def _sidecar(ieeg_dir: str, suffix: str) -> str:
    hits = sorted(glob.glob(os.path.join(ieeg_dir, f"*{suffix}")))
    if not hits:
        raise FileNotFoundError(f"missing {suffix} in {ieeg_dir}")
    return hits[0]


def _load_signal(ieeg_dir: str):
    """Return (chan_names, data[n_chan, n_samp], fs). Supports MEF3 (.mefd) and BrainVision."""
    mefd = sorted(glob.glob(os.path.join(ieeg_dir, "*_ieeg.mefd")))
    if mefd:
        from pymef.mef_session import MefSession
        ms = MefSession(mefd[0], None)
        bi = ms.read_ts_channel_basic_info()
        names = [c["name"] for c in bi]

        def _scalar(v):
            return float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)

        fs = _scalar(bi[0]["fsamp"])
        nsamp = int(_scalar(bi[0]["nsamp"]))
        data = np.asarray(ms.read_ts_channels_sample(names, [[0, nsamp]]), dtype=float)
        ms.close()
        return names, data, fs
    vhdr = sorted(glob.glob(os.path.join(ieeg_dir, "*_ieeg.vhdr")))
    if vhdr:
        import mne
        raw = mne.io.read_raw_brainvision(vhdr[0], preload=True, verbose="ERROR")
        return raw.ch_names, raw.get_data() * 1e6, float(raw.info["sfreq"])  # V->uV
    raise FileNotFoundError(f"no MEF3 or BrainVision signal in {ieeg_dir}")


# ----------------------------------------------------------------------- build one


def build_subject(dataset_root: str, sub: str, verbose: bool = True,
                  robust: bool = False, n_trials_cap: int = None) -> CCEPSubject:
    """robust=True: reject gross-artifact trials (RMS > median+4*MAD) and average trials with the
    median instead of the mean (Class B — lifts the low-SNR far-field noise floor).
    n_trials_cap: if set, use only the first N trials per stim site (trials-ablation, Step 1)."""
    ieeg = _find_ieeg_dir(dataset_root, sub)
    ev = pd.read_csv(_sidecar(ieeg, "_events.tsv"), sep="\t")
    ch = pd.read_csv(_sidecar(ieeg, "_channels.tsv"), sep="\t")
    el = pd.read_csv(_sidecar(ieeg, "_electrodes.tsv"), sep="\t")

    names, data, fs = _load_signal(ieeg)
    name2row = {n: i for i, n in enumerate(names)}

    # good recording contacts with coords (intersection of channels<->electrodes<->signal)
    good = set(ch[ch["status"] == "good"]["name"]) if "status" in ch else set(ch["name"])
    # restrict to non-ECG/EKG signal-bearing electrodes
    badtype = {"ECG", "EKG"}
    if "type" in ch:
        good &= set(ch[~ch["type"].isin(badtype)]["name"])
    el = el[el["name"].isin(good) & el["name"].isin(names)].copy()
    el = el.dropna(subset=["x", "y", "z"])
    contacts = list(el["name"])
    contact_xyz = el[["x", "y", "z"]].to_numpy(dtype=float)
    cidx = {n: i for i, n in enumerate(contacts)}
    n_c = len(contacts)
    if verbose:
        print(f"  {sub}: {len(names)} signal chans -> {n_c} good contacts w/ coords")

    fs = float(fs)
    n_pre, n_post = int(-EPOCH[0] * fs), int(EPOCH[1] * fs)
    t = np.arange(-n_pre, n_post) / fs
    base_m = (t >= BASE_WIN[0]) & (t <= BASE_WIN[1])
    blank_m = (t >= BLANK_WIN[0]) & (t <= BLANK_WIN[1])
    n1_m = (t >= N1_WIN[0]) & (t <= N1_WIN[1])
    n2_m = (t >= N2_WIN[0]) & (t <= N2_WIN[1])
    crp_m = (t >= CRP_WIN[0]) & (t <= CRP_WIN[1])

    ev = ev[ev["status"] == "good"].copy() if "status" in ev else ev
    ev = ev[ev["electrical_stimulation_site"].notna()]

    sites, responses, responses_signed, n2v = [], [], [], []
    resp_h1, resp_h2 = [], []
    latv, lat_h1v, lat_h2v = [], [], []
    crp_v, crp_h1v, crp_h2v, reliab_crp = [], [], [], []
    stim_xyz, stim_idx, reliab, ntrials = [], [], [], []

    contact_rows = np.array([name2row[c] for c in contacts])  # signal rows for our contacts

    for site, grp in ev.groupby("electrical_stimulation_site"):
        pair = str(site).split("-")
        onsets = (grp["onset"].to_numpy(dtype=float) * fs).astype(int)
        if n_trials_cap is not None:
            onsets = onsets[:n_trials_cap]
        # epoch [n_trials, n_c, win]
        win = n_pre + n_post
        ep = np.full((len(onsets), n_c, win), np.nan)
        for i, o in enumerate(onsets):
            s0, s1 = o - n_pre, o + n_post
            if s0 < 0 or s1 > data.shape[1]:
                continue
            ep[i] = data[contact_rows, s0:s1]
        # baseline correct + blank artifact
        ep = ep - np.nanmean(ep[:, :, base_m], axis=2, keepdims=True)
        ep[:, :, blank_m] = np.nan
        valid = ~np.all(np.isnan(ep[:, 0, :]), axis=1)
        ep = ep[valid]
        if ep.shape[0] < 4:
            continue
        if robust:
            # drop gross-artifact trials by post-stim RMS (robust z via MAD)
            post = (t >= 0.01) & (t <= 0.3)
            trms = np.sqrt(np.nanmean(ep[:, :, post] ** 2, axis=(1, 2)))
            med = np.nanmedian(trms); mad = np.nanmedian(np.abs(trms - med)) + 1e-9
            gt = trms < med + 4 * 1.4826 * mad
            if gt.sum() >= 4:
                ep = ep[gt]

        _avg = np.nanmedian if robust else np.nanmean

        def _amp(ep_set, mask):
            avg = _avg(ep_set, axis=0)                  # [n_c, win]
            seg = avg[:, mask]
            j = np.nanargmax(np.abs(seg), axis=1)
            signed = seg[np.arange(seg.shape[0]), j]
            return signed

        def _lat(ep_set, mask):
            avg = _avg(ep_set, axis=0)                  # [n_c, win]
            seg = avg[:, mask]
            tmask = t[mask]
            j = np.nanargmax(np.abs(seg), axis=1)
            return tmask[j] * 1000.0                    # peak latency in ms

        n1_signed = _amp(ep, n1_m)
        n2_signed = _amp(ep, n2_m)
        n1_lat = _lat(ep, n1_m)

        # ---- waveform-shape (CRP) amplitude: project each contact's evoked waveform onto the
        # canonical response shape (1st temporal SVD component of responsive contacts). Uses the
        # full post-stim trace -> higher SNR than a single N1 peak. ----
        def _crp(ep_set, canonical=None):
            avg = _avg(ep_set, axis=0)                 # [n_c, win]
            W = np.nan_to_num(avg[:, crp_m])           # [n_c, T]
            if canonical is None:
                amp = np.nanmax(np.abs(W), axis=1)
                thr = np.nanpercentile(amp, 75)
                M = W[amp >= thr]
                if M.shape[0] >= 3:
                    _, _, Vt = np.linalg.svd(M, full_matrices=False)
                    canonical = Vt[0]
                else:
                    k = int(np.nanargmax(amp)); canonical = W[k]
                canonical = canonical / (np.linalg.norm(canonical) + 1e-9)
                if (W @ canonical).sum() < 0:
                    canonical = -canonical
            return W @ canonical, canonical

        crp_signed, canon = _crp(ep)

        # split-half reliability of the N1 topography across trials
        n = ep.shape[0]
        h1 = _amp(ep[: n // 2], n1_m)
        h2 = _amp(ep[n // 2:], n1_m)
        lh1 = _lat(ep[: n // 2], n1_m)
        lh2 = _lat(ep[n // 2:], n1_m)
        # CRP half-splits projected onto the SAME (full-data) canonical shape
        ch1, _ = _crp(ep[: n // 2], canon)
        ch2, _ = _crp(ep[n // 2:], canon)
        excl = [cidx[p] for p in pair if p in cidx]
        keep = np.ones(n_c, bool)
        for e in excl:
            keep[e] = False
        rr = np.corrcoef(np.abs(h1[keep]), np.abs(h2[keep]))[0, 1]
        sb = (2 * rr) / (1 + rr) if np.isfinite(rr) and rr > -1 else np.nan
        rrc = np.corrcoef(np.abs(ch1[keep]), np.abs(ch2[keep]))[0, 1]
        sbc = (2 * rrc) / (1 + rrc) if np.isfinite(rrc) and rrc > -1 else np.nan

        abs_n1 = np.abs(n1_signed)
        abs_n1[~keep] = np.nan
        sn = n1_signed.copy(); sn[~keep] = np.nan
        an2 = np.abs(n2_signed); an2[~keep] = np.nan
        # half-split abs-N1 topographies (for distance-stratified noise-ceiling diagnostics)
        ah1 = np.abs(h1); ah1[~keep] = np.nan
        ah2 = np.abs(h2); ah2[~keep] = np.nan
        # CRP amplitude topographies (abs = response strength along canonical shape)
        acrp = np.abs(crp_signed); acrp[~keep] = np.nan
        ach1 = np.abs(ch1); ach1[~keep] = np.nan
        ach2 = np.abs(ch2); ach2[~keep] = np.nan

        # stim location = mean coord of pair contacts present in electrodes
        pair_xyz = [contact_xyz[cidx[p]] for p in pair if p in cidx]
        sx = np.mean(pair_xyz, axis=0) if pair_xyz else np.array([np.nan] * 3)
        si = [cidx.get(p, -1) for p in pair]
        si = (si + [-1, -1])[:2]

        lat = n1_lat.copy(); lat[~keep] = np.nan
        lath1 = lh1.copy(); lath1[~keep] = np.nan
        lath2 = lh2.copy(); lath2[~keep] = np.nan

        sites.append(str(site))
        responses.append(abs_n1)
        responses_signed.append(sn)
        resp_h1.append(ah1)
        resp_h2.append(ah2)
        latv.append(lat); lat_h1v.append(lath1); lat_h2v.append(lath2)
        crp_v.append(acrp)
        crp_h1v.append(ach1)
        crp_h2v.append(ach2)
        reliab_crp.append(sbc)
        n2v.append(an2)
        stim_xyz.append(sx)
        stim_idx.append(si)
        reliab.append(sb)
        ntrials.append(ep.shape[0])

    return CCEPSubject(
        subject=sub,
        contacts=contacts,
        contact_xyz=contact_xyz,
        sites=sites,
        responses=np.array(responses),
        responses_signed=np.array(responses_signed),
        n2=np.array(n2v),
        stim_xyz=np.array(stim_xyz),
        stim_idx=np.array(stim_idx, dtype=int),
        reliability=np.array(reliab),
        n_trials=np.array(ntrials, dtype=int),
        fs=fs,
        responses_h1=np.array(resp_h1),
        responses_h2=np.array(resp_h2),
        responses_crp=np.array(crp_v),
        crp_h1=np.array(crp_h1v),
        crp_h2=np.array(crp_h2v),
        reliability_crp=np.array(reliab_crp),
        latency=np.array(latv),
        latency_h1=np.array(lat_h1v),
        latency_h2=np.array(lat_h2v),
    )
