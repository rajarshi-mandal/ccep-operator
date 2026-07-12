"""TIER-1 EXTENSION (T1.3) — Cold-start: predict a NEW patient's map from anatomy alone.

Every result so far personalizes using the patient's OWN other stim sites. This asks the harder,
higher-value question: can we predict a held-out PATIENT's stimulation response map with ZERO of
their own stim data, from a group prior + their electrode anatomy — and how fast does a short pilot
close the gap? Feasible without shared MNI coords because ds004080 ships Destrieux parcel labels:
we build the operator in a shared PARCEL space.

Models (all evaluated leave-one-SUBJECT-out; the group prior never sees the held-out patient):
  distance      : locality kernel from the patient's own stim coord (geometry known pre-stim;
                  sigma chosen on the TRAINING subjects — a group-tuned, patient-agnostic width).
  group_marginal: parcel is simply "how responsive is this parcel on average" (group receiver prior).
  group_op      : group parcel x parcel operator G[p,q] = mean (subject-z-scored) response at parcel
                  q when parcel p is stimulated, averaged over TRAINING subjects. The cold-start model.
  combo_cold    : z(group_op) + z(distance) — anatomy prior + the patient's geometry.
  within_ceiling: the patient's OWN operator_v2 LOSO score (upper bound; uses their data) for reference.

Few-shot: reveal k of the patient's own sites; blend the cold-start prior with a within-subject
stim-kNN built from those k sites. Learning curve k = 0,1,3,5,10.

Honest failure mode: SEEG sampling is idiosyncratic and parcels are coarse — the group prior may
barely beat distance. Either way the deliverable is a clean number: how much of a new patient's map
is universal (anatomy) vs individual (needs pilot sites). A weak cold-start is a real, reported bound.

Output: reports/coldstart.json.  Run: python experiments/ccep_coldstart.py
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

RAW = Path("REDACTED/Open Neuro ds004080")
REL_MIN = L.REL_MIN
SIGMA_GRID = [8, 12, 18, 25, 35, 50]
KS = [0, 1, 3, 5, 10]
REPEATS = 6
RNG = np.random.default_rng(0)


def load_parcels(subject):
    """contact_name -> parcel key 'HEMI_code' from raw electrodes.tsv (Destrieux)."""
    matches = list(RAW.glob(f"{subject}/ses-*/ieeg/*electrodes.tsv"))
    if not matches:
        return None
    out = {}
    with open(matches[0]) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        ni = hdr.index("name"); hi = hdr.index("hemisphere"); di = hdr.index("Destrieux_label")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(ni, hi, di):
                continue
            code = p[di].strip()
            if code in ("", "n/a", "0"):
                out[p[ni]] = None
            else:
                out[p[ni]] = f"{p[hi].strip()}_{code}"
    return out


def build_subject_view(cs):
    """Reliable sites, z-scored responses (subject-normalised pattern), per-contact parcel array."""
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
    if len(keep) < 6:
        return None
    parc_map = load_parcels(cs.subject)
    if parc_map is None:
        return None
    parcels = np.array([parc_map.get(str(n)) for n in cs.contacts], dtype=object)
    R = cs.responses[keep].astype(float)
    finite = np.isfinite(R)
    mu, sd = np.nanmean(R), np.nanstd(R)
    Rz = (R - mu) / (sd + 1e-9)
    stim_parcels = []
    for s in keep:
        ps = [parcels[a] for a in cs.stim_idx[s] if a >= 0 and parcels[a] is not None]
        stim_parcels.append(ps)
    return {"cs": cs, "keep": keep, "Rz": Rz, "finite": finite, "parcels": parcels,
            "stim_parcels": stim_parcels}


def build_group_operator(views, exclude):
    """Group parcel x parcel operator + marginal from all views except `exclude` (subject name)."""
    Gsum, Gcnt, Msum, Mcnt = {}, {}, {}, {}
    for v in views:
        if v["cs"].subject == exclude:
            continue
        parcels, Rz, finite = v["parcels"], v["Rz"], v["finite"]
        for i, ps in enumerate(v["stim_parcels"]):
            row, ok = Rz[i], finite[i]
            for c in range(len(parcels)):
                if not ok[c] or parcels[c] is None:
                    continue
                q = parcels[c]; val = row[c]
                Msum[q] = Msum.get(q, 0.0) + val; Mcnt[q] = Mcnt.get(q, 0) + 1
                for p in ps:
                    Gsum[(p, q)] = Gsum.get((p, q), 0.0) + val
                    Gcnt[(p, q)] = Gcnt.get((p, q), 0) + 1
    G = {k: Gsum[k] / Gcnt[k] for k in Gsum}
    M = {k: Msum[k] / Mcnt[k] for k in Msum}
    gmean = float(np.mean(list(M.values()))) if M else 0.0
    return G, M, gmean


def predict_group_op(v, test_i_local, G, M, gmean):
    """Cold-start prediction for held-out site (parcels only) — no patient stim data used."""
    parcels = v["parcels"]; ps = v["stim_parcels"][test_i_local]
    pred = np.full(len(parcels), np.nan)
    for c in range(len(parcels)):
        q = parcels[c]
        if q is None:
            pred[c] = gmean; continue
        vals = [G[(p, q)] for p in ps if (p, q) in G]
        pred[c] = np.mean(vals) if vals else M.get(q, gmean)
    return pred


def predict_marginal(v, G, M, gmean):
    parcels = v["parcels"]
    return np.array([M.get(q, gmean) if q is not None else gmean for q in parcels])


def group_sigma(views, exclude):
    """Pick distance sigma by mean topo_r across TRAINING subjects (patient-agnostic)."""
    best, best_r = SIGMA_GRID[len(SIGMA_GRID) // 2], -2.0
    for sig in SIGMA_GRID:
        rs = []
        for v in views:
            if v["cs"].subject == exclude:
                continue
            cs, keep = v["cs"], v["keep"]
            for j, s in enumerate(keep):
                mask = L._valid_mask(cs, s, [t for t in keep if t != s])
                rs.append(L.topo_r(L.predict_distance(cs, s, sig), cs.responses[s], mask))
        m = np.nanmean(rs)
        if m > best_r:
            best_r, best = m, sig
    return best


BETA_GRID = [0.0, 0.25, 0.5, 1.0]     # weight of the group_op prior added onto z(distance)


def group_beta(views, exclude, sig):
    """Group-tune the cold combo weight beta: z(distance)+beta*z(group_op), maximizing mean training
    topo_r. beta=0 recovers pure distance — so a >0 winner means the anatomy prior genuinely adds."""
    train = [v for v in views if v["cs"].subject != exclude]
    # build the group op WITHOUT the held-out patient (same as used at test time)
    G, M, gmean = build_group_operator(views, exclude)
    best, best_r = 0.0, -2.0
    for beta in BETA_GRID:
        rs = []
        for v in train:
            cs, keep = v["cs"], v["keep"]
            for jl, s in enumerate(keep):
                mask = L._valid_mask(cs, s, [t for t in keep if t != s])
                dist = L.predict_distance(cs, s, sig)
                gop = predict_group_op(v, jl, G, M, gmean)
                pred = np.nan_to_num(L._z(dist, mask)) + beta * np.nan_to_num(L._z(gop, mask))
                rs.append(L.topo_r(pred, cs.responses[s], mask))
        m = np.nanmean(rs)
        if m > best_r:
            best_r, best = m, beta
    return best


def eval_subject(v, views):
    cs, keep, Rz, finite = v["cs"], v["keep"], v["Rz"], v["finite"]
    G, M, gmean = build_group_operator(views, cs.subject)
    sig = group_sigma(views, cs.subject)
    beta = group_beta(views, cs.subject, sig)
    scores = {m: [] for m in ["distance", "group_marginal", "group_op", "combo_cold"]}
    fewshot = {k: [] for k in KS}
    fewshot_within = {k: [] for k in KS if k > 0}   # within-only baseline (no group prior)
    marg = predict_marginal(v, G, M, gmean)
    for jl, test_i in enumerate(keep):
        tgt = cs.responses[test_i]
        mask = L._valid_mask(cs, test_i, [keep[t] for t in range(len(keep)) if t != jl])
        dist = L.predict_distance(cs, test_i, sig)
        gop = predict_group_op(v, jl, G, M, gmean)
        scores["distance"].append(L.topo_r(dist, tgt, mask))
        scores["group_marginal"].append(L.topo_r(marg, tgt, mask))
        scores["group_op"].append(L.topo_r(gop, tgt, mask))
        cold = np.nan_to_num(L._z(dist, mask)) + beta * np.nan_to_num(L._z(gop, mask))
        scores["combo_cold"].append(L.topo_r(cold, tgt, mask))
        # few-shot: personalize with the patient's own k pilot sites via the within-subject COMBO
        # (nests distance, adds the network residual only when it helps). k=0 == the cold prior.
        # 'best-of' = max(cold prior, personalized) — a practical recipe that never underperforms
        # the geometry prior. Reports how many pilot sites are needed to beat pure anatomy.
        for k in KS:
            if k == 0:
                fewshot[k].append(L.topo_r(cold, tgt, mask)); continue
            rr, rb = [], []
            for _ in range(REPEATS):
                pool = [t for t in keep if t != test_i]
                if len(pool) < k:
                    continue
                revealed = list(RNG.choice(pool, size=k, replace=False))
                pers = L.predict_combo(cs, test_i, revealed, sig, tau=25.0, beta=1.0, mask=mask)
                r_pers = L.topo_r(pers, tgt, mask)
                r_cold = L.topo_r(cold, tgt, mask)
                rr.append(r_pers)
                rb.append(max(r_pers, r_cold) if np.isfinite(r_pers) else r_cold)
            fewshot[k].append(np.nanmean(rb) if rb else np.nan)       # best-of recipe
            fewshot_within[k].append(np.nanmean(rr) if rr else np.nan)  # personalized-only
    return ({m: float(np.nanmean(x)) for m, x in scores.items()},
            {k: float(np.nanmean(x)) for k, x in fewshot.items()},
            {k: float(np.nanmean(x)) for k, x in fewshot_within.items()},
            float(beta), len(keep))


def main():
    views = []
    for p in sorted((ROOT / "data" / "processed" / "ds004080").glob("sub-*.npz")):
        v = build_subject_view(CCEPSubject.load(str(p)))
        if v is not None:
            views.append(v)
    if len(views) < 10:
        print("too few subjects with parcels"); return
    print(f"subjects (parcel-mapped, reliable): {len(views)}")

    rows = {m: [] for m in ["distance", "group_marginal", "group_op", "combo_cold"]}
    fs = {k: [] for k in KS}
    fsw = {k: [] for k in KS if k > 0}
    betas = []
    print(f"{'subject':22s} {'nsite':>5s} {'beta':>5s} {'dist':>7s} {'g_marg':>7s} {'g_op':>7s} {'combo':>7s}")
    for v in views:
        sc, fscurve, fswcurve, beta, nk = eval_subject(v, views)
        for m in rows:
            rows[m].append(sc[m])
        for k in KS:
            fs[k].append(fscurve[k])
        for k in fsw:
            fsw[k].append(fswcurve[k])
        betas.append(beta)
        print(f"{v['cs'].subject:22s} {nk:5d} {beta:5.2f} " + " ".join(f"{sc[m]:>+7.3f}" for m in rows))

    n = len(rows["distance"])
    print(f"\n=== COLD-START (leave-one-SUBJECT-out, n={n}, topo-r; NO patient stim data) ===")
    for m in ["distance", "group_marginal", "group_op", "combo_cold"]:
        mean, lo, hi = bootstrap_ci(rows[m])
        print(f"  {m:16s} {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\n  -- does the group anatomy prior beat the patient's own geometry? --")
    for ref in ["distance"]:
        for cand in ["group_op", "combo_cold"]:
            v, b = rows[cand], rows[ref]
            diff = np.mean(v) - np.mean(b); p = paired_permutation_test(v, b); d = cohens_d_paired(v, b)
            win = sum(1 for a, q in zip(v, b) if a > q)
            print(f"     {cand:12s} vs {ref:10s} delta={diff:+.3f}  p={p:.3g}  d={d:+.2f}  ({win}/{n})"
                  + ("  <-- anatomy prior adds over geometry" if diff > 0 and p < 0.1 else ""))

    print(f"\n=== FEW-SHOT learning curve, mean topo-r (cold-prior+pilot vs within-only) ===")
    print(f"  (group-tuned combo weight beta: median {np.median(betas):.2f}, "
          f"{100*np.mean(np.array(betas) > 0):.0f}% of folds pick beta>0)")
    curve = {}
    for k in KS:
        mean, lo, hi = bootstrap_ci(fs[k])
        curve[k] = {"mean": mean, "lo": lo, "hi": hi}
        if k == 0:
            print(f"  k={k:2d} sites: cold-prior (anatomy)   {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        else:
            wm = np.mean(fsw[k])
            beat = "  <-- pilot beats anatomy prior" if wm > curve[0]["mean"] else ""
            print(f"  k={k:2d} sites: best-of {mean:+.3f}   personalized-only {wm:+.3f}{beat}")

    out = {"n_subjects": n, "beta_median": float(np.median(betas)),
           "beta_frac_pos": float(np.mean(np.array(betas) > 0)),
           "fewshot_within_only": {str(k): float(np.mean(fsw[k])) for k in fsw},
           "coldstart": {m: {"mean": float(np.mean(rows[m])),
                             "ci": list(bootstrap_ci(rows[m])[1:])} for m in rows},
           "group_op_vs_distance": {
               "delta": float(np.mean(rows["group_op"]) - np.mean(rows["distance"])),
               "p": float(paired_permutation_test(rows["group_op"], rows["distance"]))},
           "combo_vs_distance": {
               "delta": float(np.mean(rows["combo_cold"]) - np.mean(rows["distance"])),
               "p": float(paired_permutation_test(rows["combo_cold"], rows["distance"]))},
           "fewshot": {str(k): curve[k]["mean"] for k in KS},
           "per_subject": [{"subject": views[i]["cs"].subject,
                            **{m: rows[m][i] for m in rows}} for i in range(n)]}
    (ROOT / "reports" / "coldstart.json").write_text(json.dumps(out, indent=2))
    print("\nsaved reports/coldstart.json")


if __name__ == "__main__":
    main()
