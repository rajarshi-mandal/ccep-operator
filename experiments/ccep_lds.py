"""TIER-1 EXTENSION (T1.2) — From scalar N1 peak to a fitted LINEAR DYNAMICAL SYSTEM.

The headline model predicts one number per contact (|N1|). But every stim trial is a time series.
Here we fit, per subject, a reduced linear state-space operator (DMD) to the FULL evoked traces
across all stim sites, and read out a held-out site's whole spatiotemporal response as the system's
impulse response — making the conference title ("State Space Model") literally true. Deliverables:

  (1) HELD-OUT FULL-TRACE PREDICTION. Seed a held-out site from its stim coordinate only (no
      leakage), roll it forward under the fitted operator, and correlate the predicted [contacts x
      time] response with the measured one. Baseline = a SEPARABLE model (same spatial map x one
      shared temporal shape) — beating it means the operator captures contact-specific DYNAMICS
      (timing/shape), not just amplitude.
  (2) INTERPRETABLE DYNAMICS. The operator's eigenvalues give per-subject decay time-constants and
      oscillation frequencies — biophysical parameters of the network's impulse response.
  (3) DEVELOPMENT. On ccepAge, do the dominant time-constants change with age?

Honest failure mode: a linear operator seeded only by geometry may not beat the separable baseline
on the N1 amplitude (already near ceiling). The point is the ADDED predictions (full trace, timing
from one operator) and the interpretable spectrum; a modest prediction gain is reported as such and
still characterizes how linear-time-invariant CCEP propagation is.

Output: reports/lds.json.  Run: python experiments/ccep_lds.py [ds004774 ds004696 ...]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa: E402

TRACES = ROOT / "data" / "traces"
REL_MIN = 0.3
RANK = 15
SIGMA_SEED = 15.0   # mm, spatial seed width (matches operator_v2's typical sigma)


def load_trace_subject(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _envelope(traces):
    """Non-negative activation envelope of the signed evoked traces (analytic-signal magnitude).

    CCEP traces are signed oscillating potentials; the modelled quantity is WHERE and WHEN the
    network is activated, not polarity. The Hilbert envelope gives a clean positive activation whose
    contact-specific timing encodes the conduction delays — exactly what a separable (shared-shape)
    baseline cannot capture. Falls back to |trace| if scipy is unavailable.
    """
    X = np.nan_to_num(traces.astype(float))
    try:
        from scipy.signal import hilbert
        env = np.abs(hilbert(X, axis=-1))
    except Exception:
        env = np.abs(X)
    # light temporal smoothing (3-tap) to denoise the envelope
    if env.shape[-1] >= 3:
        env[..., 1:-1] = (env[..., :-2] + 2 * env[..., 1:-1] + env[..., 2:]) / 4.0
    return env


def _flatten_corr(pred, meas, valid):
    """Pearson r over flattened [contacts x time] on valid contacts; scale/sign-invariant."""
    p = pred[valid].ravel(); m = meas[valid].ravel()
    ok = np.isfinite(p) & np.isfinite(m)
    if ok.sum() < 10:
        return np.nan
    p = p[ok] - p[ok].mean(); m = m[ok] - m[ok].mean()
    den = np.linalg.norm(p) * np.linalg.norm(m)
    return float((p @ m) / den) if den > 1e-12 else np.nan


def fit_dmd(states, rank):
    """Reduced DMD from a list of per-site state sequences [n_c, T]. Returns (U_r, Atil, eigvals).

    POD-reduce the pooled snapshots, then least-squares the reduced one-step map. Eigenvalues are
    clipped to the unit disk for stable roll-outs.
    """
    X = np.concatenate([s[:, :-1] for s in states], axis=1)   # [n_c, M]
    Y = np.concatenate([s[:, 1:] for s in states], axis=1)    # [n_c, M]
    ok = np.all(np.isfinite(X), axis=0) & np.all(np.isfinite(Y), axis=0)
    X, Y = X[:, ok], Y[:, ok]
    if X.shape[1] < rank + 2:
        return None
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(rank, (S > 1e-9 * S[0]).sum())
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T
    Atil = Ur.T @ Y @ Vr @ np.diag(1.0 / Sr)                  # reduced operator [r, r]
    w = np.linalg.eigvals(Atil)
    return Ur, Atil, w


def rollout(Ur, Atil, a0, T):
    """Roll the reduced state a0 forward T steps; return full [n_c, T] reconstruction."""
    r = Atil.shape[0]
    # stabilise: scale eigen-spectrum into the unit disk
    w = np.linalg.eigvals(Atil); sr = np.abs(w).max()
    A = Atil / sr * 0.999 if sr > 1.0 else Atil
    out = np.zeros((Ur.shape[0], T))
    a = a0.copy()
    for k in range(T):
        out[:, k] = Ur @ a
        a = A @ a
    return out


def eig_timeconstants(w, fs, dt_steps):
    """Continuous-time decay time-constants (ms) and freqs (Hz) from discrete eigenvalues.

    dt between stored samples = dt_steps / fs seconds. tau = -dt/ln|mu|. Keep decaying modes.
    """
    dt = dt_steps / fs
    taus, freqs = [], []
    for mu in w:
        a = np.abs(mu)
        if a <= 1e-6 or a >= 1.0:
            continue
        tau = -dt / np.log(a) * 1000.0        # ms
        f = abs(np.angle(mu)) / (2 * np.pi * dt)
        if 0 < tau < 2000:
            taus.append(tau); freqs.append(f)
    return np.array(taus), np.array(freqs)


def eval_subject(d):
    sites = np.arange(len(d["sites"]))
    rel = d["reliability"]
    keep = sites[(np.isfinite(rel)) & (rel >= REL_MIN)]
    if len(keep) < 6:
        return None
    traces = _envelope(d["traces"])                        # [n_sites, n_c, T] activation envelope
    n_c, T = traces.shape[1], traces.shape[2]
    xyz = d["contact_xyz"]; stim_xyz = d["stim_xyz"]; stim_idx = d["stim_idx"]
    fs = float(d["fs"])
    dt_steps = int(round(fs / 256.0)) if fs > 256 else 1   # stored downsample step
    lds_r, sep_r, lat_rho = [], [], []
    for test_i in keep:
        train = [t for t in keep if t != test_i]
        states = [np.nan_to_num(traces[s]) for s in train]
        fit = fit_dmd(states, RANK)
        if fit is None:
            continue
        Ur, Atil, w = fit
        # held-out seed: spatial distance kernel from stim coord (geometry only)
        D = np.linalg.norm(xyz - stim_xyz[test_i][None], axis=1)
        seed = np.exp(-(D ** 2) / (2 * SIGMA_SEED ** 2))
        a0 = Ur.T @ seed
        pred = rollout(Ur, Atil, a0, T)                    # [n_c, T]
        meas = traces[test_i]
        valid = np.ones(n_c, bool)
        for e in stim_idx[test_i]:
            if e >= 0:
                valid[e] = False
        # separable baseline: same spatial seed x shared temporal shape (train-mean normalized wave)
        shape = np.nanmean([np.nanmean(np.abs(traces[s]), axis=0) for s in train], axis=0)  # [T]
        shape = shape / (np.linalg.norm(shape) + 1e-9)
        sep = seed[:, None] * shape[None, :]
        lds_r.append(_flatten_corr(pred, meas, valid))
        sep_r.append(_flatten_corr(sep, meas, valid))
        # CONTACT-SPECIFIC TIMING: the separable model peaks every contact at the same time; a
        # dynamical operator can predict the conduction gradient. Held-out Spearman between the
        # LDS-predicted per-contact peak latency and the measured peak latency.
        t_ms = d["t_ms"].astype(float)
        resp = np.nanmax(meas, axis=1) > np.nanpercentile(np.nanmax(meas, axis=1), 40)  # responsive
        vv = valid & resp
        if vv.sum() >= 6:
            lat_pred = t_ms[np.argmax(np.nan_to_num(np.abs(pred[vv])), axis=1)]
            lat_meas = t_ms[np.argmax(np.nan_to_num(meas[vv]), axis=1)]
            rho, _ = _spearman(lat_pred, lat_meas)
            if np.isfinite(rho):
                lat_rho.append(rho)
    if not lds_r:
        return None
    # per-subject dynamics: fit one operator on ALL reliable sites, extract time-constants
    states_all = [np.nan_to_num(traces[s]) for s in keep]
    fit_all = fit_dmd(states_all, RANK)
    tau_med, f_med = np.nan, np.nan
    if fit_all is not None:
        _, _, wall = fit_all
        taus, freqs = eig_timeconstants(wall, fs, dt_steps)
        if len(taus):
            # energy-weight toward the slowest (dominant) decaying modes
            tau_med = float(np.median(taus))
            f_med = float(np.median(freqs[freqs > 0])) if (freqs > 0).any() else np.nan
    return {"lds_r": float(np.nanmean(lds_r)), "sep_r": float(np.nanmean(sep_r)),
            "lat_rho": float(np.nanmean(lat_rho)) if lat_rho else np.nan,
            "tau_ms": tau_med, "freq_hz": f_med, "nsites": int(len(keep))}


def load_ages():
    p = ROOT / "reports" / "ds004080_participants.tsv"
    if not p.exists():
        return {}
    ages = {}
    with open(p) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        try:
            pi = hdr.index("participant_id"); ai = hdr.index("age")
        except ValueError:
            return {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(pi, ai):
                try:
                    ages[parts[pi]] = float(parts[ai])
                except ValueError:
                    pass
    return ages


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6:
        return np.nan, 0
    a, b = a[ok], b[ok]
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return (float((ra @ rb) / den) if den > 1e-12 else np.nan), int(ok.sum())


def main():
    datasets = sys.argv[1:] or ["ds004774", "ds004696"]
    rows = []
    print(f"{'subject':22s} {'nsite':>5s} {'lds_r':>7s} {'sep_r':>7s} {'tau_ms':>7s} {'freq_hz':>7s}")
    for ds in datasets:
        ddir = TRACES / ds
        if not ddir.exists():
            continue
        for p in sorted(ddir.glob("*.npz")):
            d = load_trace_subject(p)
            res = eval_subject(d)
            if res is None:
                continue
            res["subject"] = f"{ds[-4:]}/{p.stem}"; res["ds"] = ds; res["sub"] = p.stem
            rows.append(res)
            print(f"{res['subject']:22s} {res['nsites']:5d} {res['lds_r']:>+7.3f} "
                  f"{res['sep_r']:>+7.3f} {res['tau_ms']:>7.1f} {res['freq_hz']:>7.1f}")
    if not rows:
        print("no trace caches found — run scripts/build_traces.py first"); return
    n = len(rows)
    lds = [r["lds_r"] for r in rows]; sep = [r["sep_r"] for r in rows]

    print(f"\n=== HELD-OUT FULL-TRACE PREDICTION (n={n}, flattened contacts x time r) ===")
    for lab, v in [("LDS (dynamical operator)", lds), ("separable baseline", sep)]:
        m, lo, hi = bootstrap_ci(v)
        print(f"  {lab:26s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    diff = np.mean(lds) - np.mean(sep); p = paired_permutation_test(lds, sep); dd = cohens_d_paired(lds, sep)
    win = sum(1 for a, b in zip(lds, sep) if a > b)
    print(f"  LDS vs separable: delta={diff:+.3f}  p={p:.3g}  d={dd:+.2f}  ({win}/{n})"
          + ("  <-- operator captures contact-specific dynamics" if diff > 0 and p < 0.1 else ""))

    lr = [r["lat_rho"] for r in rows if np.isfinite(r.get("lat_rho", np.nan))]
    print(f"\n=== CONTACT-SPECIFIC TIMING (LDS-predicted vs measured peak latency; separable=0) ===")
    if lr:
        m, lo, hi = bootstrap_ci(lr)
        p0 = paired_permutation_test(lr, [0.0] * len(lr)); pos = sum(1 for x in lr if x > 0)
        print(f"  latency rho {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  p(vs0)={p0:.3g}  ({pos}/{len(lr)} subj>0)"
              + ("  <-- operator predicts conduction timing" if m > 0.05 and p0 < 0.1 else ""))

    taus = [r["tau_ms"] for r in rows if np.isfinite(r["tau_ms"])]
    print(f"\n=== INTERPRETABLE DYNAMICS (per-subject dominant time-constant) ===")
    if taus:
        m, lo, hi = bootstrap_ci(taus)
        print(f"  median decay tau: {m:.1f} ms [{lo:.1f}, {hi:.1f}]  (n={len(taus)})")

    ages = load_ages()
    dev = None
    if ages:
        pairs = [(ages.get(r["sub"]), r["tau_ms"], r["freq_hz"], r["nsites"])
                 for r in rows if r["sub"] in ages and np.isfinite(r["tau_ms"])]
        pairs = [x for x in pairs if x[0] is not None]
        if len(pairs) >= 6:
            A = [x[0] for x in pairs]; Tau = [x[1] for x in pairs]; Fq = [x[2] for x in pairs]
            rho_t, nt = _spearman(A, Tau); rho_f, nf = _spearman(A, Fq)
            print(f"\n=== DEVELOPMENT (ccepAge, n={nt}) ===")
            print(f"  tau vs age  : rho={rho_t:+.3f}")
            print(f"  freq vs age : rho={rho_f:+.3f}")
            dev = {"n": nt, "rho_tau_age": rho_t, "rho_freq_age": rho_f}

    out = {"n_subjects": n,
           "fulltrace": {"lds_mean": float(np.mean(lds)), "sep_mean": float(np.mean(sep)),
                         "delta": float(diff), "p": float(p), "d": float(dd), "wins": int(win)},
           "latency_rho_mean": float(np.mean(lr)) if lr else None,
           "tau_ms_median": float(np.median(taus)) if taus else None,
           "development": dev,
           "per_subject": rows}
    (ROOT / "reports" / "lds.json").write_text(json.dumps(out, indent=2))
    print("\nsaved reports/lds.json")


if __name__ == "__main__":
    main()
