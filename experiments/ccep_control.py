"""TIER-1 EXTENSION (T1.1) — Stimulation as network CONTROL: inverse targeting + controllability.

The conference/journal model PREDICTS the response to a chosen stimulation site. This turns it
around into a PRESCRIPTIVE tool and asks three falsifiable questions, all held-out (LOSO):

  (A) INVERSE TARGETING / REGRET.  Given a target contact t we wish to modulate, rank every
      candidate stim site by the operator's PREDICTED response at t (built without that site —
      the standard operator_v2 LOSO prediction), pick the top site s*, and score the MEASURED
      response actually evoked at t when s* was stimulated. Compare against the achievable
      oracle (best site in hindsight), a distance chooser (nearest stim site to t), and random.
      Headline metric = normalized capture = (achieved - random) / (oracle - random) in (-inf, 1].

  (B) CONTROLLABILITY PREDICTS REACH.  From the fitted operator compute each node's average and
      modal controllability (Gu et al. 2015 Gramian metrics). Held-out test: does a site's average
      controllability (operator built WITHOUT that site) predict its MEASURED total network reach
      (sum of evoked responses)? A positive within-subject rank correlation means the operator's
      control ranking picks high-impact untested sites — the core targeting claim, mechanism-side.

  (C) CONTROLLABILITY MAP (descriptive).  Average vs modal controllability across contacts and,
      where Destrieux labels exist (ds004080), which regions are the controllable hubs.

Honest failure modes: if the model chooser ties the distance chooser (A), or controllability does
not predict reach (B), those are real, publishable negatives that bound the value of network
information for TARGETING specifically. Reported, not hidden.

Output: reports/control.json.  Run: python experiments/ccep_control.py [--fast]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa: E402
import ccep_loso as L  # noqa: E402
import ccep_operator_v2 as V2  # noqa: E402  (reuse _best_params, _build_operator, predict_operator_v2)

REL_MIN = L.REL_MIN
CTRB_RADIUS = 0.9   # scale the spectrally-normalised (radius-1) operator to this radius so the
                    # controllability Gramian Sum_t A^t (A^t)^T converges (Gu-style stable norm).
CTRB_HORIZON = 60


# ------------------------------------------------------------------ LOSO predictions (operator_v2)

def loso_predictions(cs):
    """Full operator_v2 LOSO predicted topographies + measured, over reliable sites.

    Returns keep (site indices), Yhat [n_keep, n_c], Ymeas [n_keep, n_c] (measured, NaN=excluded).
    Same nested-CV protocol as the headline operator_v2 (no leakage: site's own row never in A).
    """
    sites = np.arange(len(cs.sites))
    rel = cs.reliability
    keep = sites[(np.isfinite(rel)) & (rel >= REL_MIN)]
    if len(keep) < 6:
        return None
    n_c = len(cs.contacts)
    Yhat = np.full((len(keep), n_c), np.nan)
    Ymeas = np.full((len(keep), n_c), np.nan)
    for i, test_i in enumerate(keep):
        train_idx = [t for t in keep if t != test_i]
        sg, al, stp, md = V2._best_params(cs, train_idx)
        P = V2._build_operator(cs, train_idx, md)
        Yhat[i] = V2.predict_operator_v2(cs, test_i, train_idx, sg, al, stp, md, P=P)
        Ymeas[i] = cs.responses[test_i]
    return keep, Yhat, Ymeas


# ------------------------------------------------------------------ (A) inverse targeting / regret

def targeting(cs, keep, Yhat, Ymeas):
    """For each target contact, does the operator pick a stim site that captures its modulation?

    Candidate sites = the reliable sites (rows of Yhat/Ymeas). For target contact t we compare the
    MEASURED evoked response at t across candidate sites; the model/distance/random pick one site
    each and we read its measured value. Normalised capture in (-inf, 1]; 1 = oracle, 0 = random.
    """
    n_keep, n_c = Ymeas.shape
    stim_xyz = cs.stim_xyz[keep]                       # [n_keep, 3]
    caps = {"model": [], "distance": []}
    frac_of_max = {"model": [], "distance": []}
    hit_at1 = {"model": [], "distance": []}            # did the chooser pick the true-best site?
    # "network-dominated" targets: the best site to reach them is NOT the nearest site — geometry
    # is insufficient, so this is where a network operator can add value over distance.
    net_caps = {"model": [], "distance": []}
    net_hit = {"model": [], "distance": []}
    n_net = 0
    for t in range(n_c):
        col = Ymeas[:, t]                              # measured response at t from each candidate
        finite = np.isfinite(col)
        if finite.sum() < 8:
            continue
        vals = col[finite]
        if np.nanstd(vals) < 1e-9 or np.nanmax(vals) <= 0:
            continue
        idx = np.where(finite)[0]
        oracle_local = idx[np.argmax(col[idx])]
        oracle = col[oracle_local]
        rand = np.nanmean(col[idx])
        denom = oracle - rand
        if denom < 1e-9:
            continue
        # model chooser: rank candidate sites by PREDICTED response at t
        ph = Yhat[:, t].copy(); ph[~finite] = -np.inf
        model_local = int(np.argmax(ph))
        # distance chooser: candidate stim site nearest the target contact
        d = np.linalg.norm(stim_xyz - cs.contact_xyz[t][None], axis=1); d[~finite] = np.inf
        dist_local = int(np.argmin(d))
        cm = (col[model_local] - rand) / denom
        cd = (col[dist_local] - rand) / denom
        caps["model"].append(cm); caps["distance"].append(cd)
        frac_of_max["model"].append(col[model_local] / oracle)
        frac_of_max["distance"].append(col[dist_local] / oracle)
        hit_at1["model"].append(float(model_local == oracle_local))
        hit_at1["distance"].append(float(dist_local == oracle_local))
        if oracle_local != dist_local:               # network-dominated target
            n_net += 1
            net_caps["model"].append(cm); net_caps["distance"].append(cd)
            net_hit["model"].append(float(model_local == oracle_local))
            net_hit["distance"].append(float(dist_local == oracle_local))
    if not caps["model"]:
        return None
    out = {
        "capture_model": float(np.mean(caps["model"])),
        "capture_distance": float(np.mean(caps["distance"])),
        "fracmax_model": float(np.mean(frac_of_max["model"])),
        "fracmax_distance": float(np.mean(frac_of_max["distance"])),
        "hit_at1_model": float(np.mean(hit_at1["model"])),
        "hit_at1_distance": float(np.mean(hit_at1["distance"])),
        "n_targets": len(caps["model"]),
        "frac_network_targets": float(n_net / len(caps["model"])),
    }
    if net_caps["model"]:
        out.update({
            "net_capture_model": float(np.mean(net_caps["model"])),
            "net_capture_distance": float(np.mean(net_caps["distance"])),
            "net_hit_model": float(np.mean(net_hit["model"])),
            "net_hit_distance": float(np.mean(net_hit["distance"])),
            "n_net_targets": len(net_caps["model"]),
        })
    return out


# ------------------------------------------------------------------ (B,C) controllability

def controllability(A):
    """Average & modal controllability per node from operator A (Gu et al. 2015).

    A is symmetric, spectrally normalised to radius 1 by _build_operator; rescale to CTRB_RADIUS
    for a convergent Gramian. Average ctrb_i = i-th diagonal of the controllability Gramian with
    single-node input = energy delivered network-wide when driving node i. Modal ctrb_i measures
    ability to drive the low-energy (hard-to-reach) modes.
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros(0), np.zeros(0)
    w = np.linalg.eigvalsh(A)
    sr = np.abs(w).max()
    An = A * (CTRB_RADIUS / sr) if sr > 1e-9 else A
    # average controllability: diag of Sum_t An^t (An^t)^T
    G = np.eye(n)
    M = np.eye(n)
    for _ in range(CTRB_HORIZON):
        M = An @ M
        G = G + M @ M.T
    avg = np.diag(G).astype(float)
    # modal controllability (eigendecomposition of symmetric An)
    lam, U = np.linalg.eigh(An)
    modal = ((1.0 - lam ** 2)[None, :] * (U ** 2)).sum(axis=1)
    return avg, modal


def controllability_reach(cs, keep):
    """Held-out: does a site's avg controllability (operator w/o that site) predict measured reach?

    Reach(s) = nansum of measured responses evoked by site s (total network impact). Controllability
    of s = mean avg-controllability of its stim contacts, from an operator built WITHOUT s.
    Return within-subject Spearman rho between the two across sites.
    """
    ctrb_s, reach_s = [], []
    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        A = V2._build_operator(cs, train_idx, "symmetric")
        avg, _ = controllability(A)
        pair = [a for a in cs.stim_idx[test_i] if a >= 0]
        if not pair:
            continue
        c = float(np.mean(avg[pair]))
        reach = float(np.nansum(cs.responses[test_i]))
        if np.isfinite(c) and np.isfinite(reach):
            ctrb_s.append(c); reach_s.append(reach)
    if len(ctrb_s) < 6:
        return np.nan
    return _spearman(np.array(ctrb_s), np.array(reach_s))


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra @ rb) / den) if den > 1e-12 else np.nan


# ------------------------------------------------------------------ Destrieux hubs (ds004080 only)

def destrieux_hubs(ds, cs, A):
    """If this subject has Destrieux labels (ds004080), return mean avg-controllability per region."""
    if ds != "ds004080":
        return None
    lab = _load_destrieux(cs.subject)
    if lab is None:
        return None
    avg, _ = controllability(A)
    by_region = {}
    for name, a in zip(cs.contacts, avg):
        region = lab.get(str(name))
        if region and region != "n/a":
            by_region.setdefault(region, []).append(float(a))
    return {r: float(np.mean(v)) for r, v in by_region.items() if len(v) >= 2}


_DESTRIEUX_CACHE = {}

def _load_destrieux(subject):
    """Map contact name -> Destrieux_label_text from the raw electrodes.tsv (ds004080)."""
    if subject in _DESTRIEUX_CACHE:
        return _DESTRIEUX_CACHE[subject]
    base = Path("REDACTED/Open Neuro ds004080")
    matches = list(base.glob(f"{subject}/ses-*/ieeg/*electrodes.tsv")) if base.exists() else []
    if not matches:
        _DESTRIEUX_CACHE[subject] = None; return None
    lab = {}
    with open(matches[0]) as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            ni = header.index("name"); li = header.index("Destrieux_label_text")
        except ValueError:
            _DESTRIEUX_CACHE[subject] = None; return None
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(ni, li):
                lab[parts[ni]] = parts[li]
    _DESTRIEUX_CACHE[subject] = lab
    return lab


# ------------------------------------------------------------------ main

def main(fast=False):
    caches = L.all_caches()
    if fast:
        caches = [(d, p) for (d, p) in caches if d in ("ds004774", "ds004696")]
    if not caches:
        print("no caches"); return

    per = []   # per-subject dicts
    region_pool = {}
    print(f"{'subject':20s} {'nsite':>5s} {'capT_mdl':>9s} {'capT_dst':>9s} {'hit@1_m':>8s} "
          f"{'hit@1_d':>8s} {'ctrl->reach':>11s}")
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        lo = loso_predictions(cs)
        if lo is None:
            continue
        keep, Yhat, Ymeas = lo
        tg = targeting(cs, keep, Yhat, Ymeas)
        if tg is None:
            continue
        rho = controllability_reach(cs, keep)
        A_full = V2._build_operator(cs, list(keep), "symmetric")
        hubs = destrieux_hubs(ds, cs, A_full)
        if hubs:
            for r, v in hubs.items():
                region_pool.setdefault(r, []).append(v)
        row = {"subject": f"{ds[-4:]}/{cs.subject}", "nsites": int(len(keep)),
               "ctrl_reach_rho": None if not np.isfinite(rho) else float(rho), **tg}
        per.append(row)
        print(f"{row['subject']:20s} {row['nsites']:5d} {tg['capture_model']:>+9.3f} "
              f"{tg['capture_distance']:>+9.3f} {tg['hit_at1_model']:>8.2f} "
              f"{tg['hit_at1_distance']:>8.2f} "
              f"{('  n/a' if row['ctrl_reach_rho'] is None else f'{rho:>+11.3f}')}")

    if not per:
        print("no evaluable subjects"); return
    n = len(per)

    def col(key):
        return [p[key] for p in per if p.get(key) is not None]

    print(f"\n=== (A) INVERSE TARGETING (n={n} subjects, normalised capture: 1=oracle, 0=random) ===")
    for key, lab in [("capture_model", "model chooser"), ("capture_distance", "distance chooser")]:
        v = col(key); m, loi, hi = bootstrap_ci(v)
        print(f"  {lab:16s} capture {m:+.3f} [{loi:+.3f}, {hi:+.3f}]")
    vm, vd = col("capture_model"), col("capture_distance")
    diff = np.mean(vm) - np.mean(vd)
    p = paired_permutation_test(vm, vd); d = cohens_d_paired(vm, vd)
    win = sum(1 for a, b in zip(vm, vd) if a > b)
    print(f"  model vs distance: delta={diff:+.3f}  p={p:.3g}  d={d:+.2f}  ({win}/{n} subj)"
          + ("  <-- model targets better" if diff > 0 and p < 0.1 else ""))
    fm, fd = col("fracmax_model"), col("fracmax_distance")
    print(f"  fraction-of-oracle: model {np.mean(fm):.3f}  distance {np.mean(fd):.3f}")
    hm, hd = col("hit_at1_model"), col("hit_at1_distance")
    print(f"  exact best-site hit@1: model {np.mean(hm):.3f}  distance {np.mean(hd):.3f}  "
          f"(chance ~ 1/median_nsites)")

    # network-dominated targets: best site != nearest site — where geometry alone is insufficient
    nvm, nvd = col("net_capture_model"), col("net_capture_distance")
    net_stats = None
    if nvm and len(nvm) == len(nvd) and len(nvm) >= 6:
        fnet = col("frac_network_targets")
        ndiff = np.mean(nvm) - np.mean(nvd)
        np_ = paired_permutation_test(nvm, nvd); nd = cohens_d_paired(nvm, nvd)
        nwin = sum(1 for a, b in zip(nvm, nvd) if a > b)
        nhm, nhd = col("net_hit_model"), col("net_hit_distance")
        print(f"\n  -- NETWORK-DOMINATED targets (best site != nearest; {100*np.mean(fnet):.0f}% of targets) --")
        print(f"     capture: model {np.mean(nvm):+.3f}  distance {np.mean(nvd):+.3f}  "
              f"delta={ndiff:+.3f}  p={np_:.3g}  d={nd:+.2f}  ({nwin}/{len(nvm)} subj)"
              + ("  <-- model wins where geometry fails" if ndiff > 0 and np_ < 0.1 else ""))
        print(f"     best-site hit@1: model {np.mean(nhm):.3f}  distance {np.mean(nhd):.3f}")
        net_stats = {"frac_network_targets": float(np.mean(fnet)),
                     "net_capture_model": float(np.mean(nvm)),
                     "net_capture_distance": float(np.mean(nvd)),
                     "net_capture_delta": float(ndiff), "net_capture_p": float(np_),
                     "net_capture_d": float(nd), "net_capture_wins": int(nwin),
                     "net_hit_model": float(np.mean(nhm)), "net_hit_distance": float(np.mean(nhd))}

    print(f"\n=== (B) CONTROLLABILITY PREDICTS REACH (within-subject Spearman rho) ===")
    rr = col("ctrl_reach_rho")
    if rr:
        m, loi, hi = bootstrap_ci(rr)
        p0 = paired_permutation_test(rr, [0.0] * len(rr))
        pos = sum(1 for x in rr if x > 0)
        print(f"  ctrl->reach rho {m:+.3f} [{loi:+.3f}, {hi:+.3f}]  p(vs0)={p0:.3g}  ({pos}/{len(rr)} subj>0)"
              + ("  <-- controllability picks high-impact sites" if m > 0 and p0 < 0.1 else ""))

    if region_pool:
        print(f"\n=== (C) DESTRIEUX CONTROLLABILITY HUBS (ds004080, mean avg-ctrb, top/bottom) ===")
        ranked = sorted(((np.mean(v), r, len(v)) for r, v in region_pool.items() if len(v) >= 3),
                        reverse=True)
        for m, r, k in ranked[:6]:
            print(f"  hub   {r:42s} {m:8.3f}  (n={k})")
        for m, r, k in ranked[-4:]:
            print(f"  low   {r:42s} {m:8.3f}  (n={k})")

    out = {
        "n_subjects": n,
        "targeting": {
            "capture_model_mean": float(np.mean(col("capture_model"))),
            "capture_distance_mean": float(np.mean(col("capture_distance"))),
            "capture_delta": float(diff), "capture_p": float(p), "capture_d": float(d),
            "capture_model_wins": int(win),
            "fracmax_model": float(np.mean(fm)), "fracmax_distance": float(np.mean(fd)),
            "hit_at1_model": float(np.mean(hm)), "hit_at1_distance": float(np.mean(hd)),
            "network_dominated": net_stats,
        },
        "controllability_reach": {
            "rho_mean": float(np.mean(rr)) if rr else None,
            "n_pos": int(sum(1 for x in rr if x > 0)) if rr else 0,
            "n": len(rr),
        },
        "destrieux_hubs": {r: float(np.mean(v)) for r, v in region_pool.items() if len(v) >= 3},
        "per_subject": per,
    }
    (ROOT / "reports" / "control.json").write_text(json.dumps(out, indent=2))
    print("\nsaved reports/control.json")


if __name__ == "__main__":
    main(fast="--fast" in sys.argv)
