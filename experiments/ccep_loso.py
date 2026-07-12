"""CCEP within-subject leave-one-stim-site-out (ds004774).

The decisive test the es-fMRI path couldn't run for lack of sites: with 30-55 stim sites/subject,
can a structured readout predict a held-out stim site's CCEP N1 topography BETTER than the
within-subject mean (the strong baseline that capped es-fMRI)?

Models (all predict the held-out site's N1 topography over the subject's contacts):
  within_mean : mean N1 topography over the OTHER (training) sites.            <- the bar
  distance    : locality kernel exp(-D^2/2 sigma^2) from the stim coord.        (sigma nested-CV'd)
  stim_knn    : Nadaraya-Watson over stim LOCATION -> weighted avg of training  (tau nested-CV'd)
                site topographies. A strict generalization of within_mean (tau->inf == within_mean).
  operator    : effective-connectivity propagation. Build W from training sites
                (row = response when that contact stimulated), interpolate the held-out stim
                row by stim-distance, propagate a multi-step impulse.            (k,steps nested-CV'd)
  additive    : within_mean + best-deviation-model residual (the §5 fix to beat within_mean).

Rigor (per the audit): all hyperparameters chosen by INNER leave-one-out on the training sites
only (nested CV, no peeking); metric aggregated to SUBJECT level; report per-subject + group with
paired permutation vs within_mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
DATASETS = ["ds004774", "ds004696", "ds004457", "ds003708", "ds004080"]


def all_caches():
    """All CCEP subject caches across datasets, as (dataset, path) sorted pairs."""
    out = []
    for ds in DATASETS:
        for p in sorted((PROCESSED / ds).glob("sub-*.npz")):
            out.append((ds, p))
    return out

SIGMA_GRID = [5, 10, 15, 20, 30, 50]        # mm, distance locality
TAU_GRID = [8, 15, 25, 40, 70, 1e9]         # mm, stim-knn bandwidth (1e9 == within_mean)
STEP_GRID = [1, 2, 3]                        # operator propagation steps
BETA_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]  # weight of network residual added onto locality
REL_MIN = 0.3                               # keep sites with split-half reliability >= this


def _z(x, mask):
    """z-score x over masked finite contacts (NaN elsewhere)."""
    out = np.full_like(x, np.nan, dtype=float)
    ok = mask & np.isfinite(x)
    if ok.sum() < 2:
        return out
    mu, sd = x[ok].mean(), x[ok].std()
    out[ok] = (x[ok] - mu) / (sd + 1e-9)
    return out


def predict_combo(cs, test_i, train_idx, sigma, tau, beta, mask):
    """Locality winner + network-specific residual (the §6 'distance + network-residual' model)."""
    loc = predict_distance(cs, test_i, sigma)
    knn = predict_stim_knn(cs, test_i, train_idx, tau)
    net_resid = _resid(knn, loc, mask)
    return np.nan_to_num(_z(loc, mask)) + beta * np.nan_to_num(_z(net_resid, mask))


def topo_r(pred, meas, mask):
    """Pearson r over valid (mask) contacts; NaN-safe."""
    p, m = pred[mask], meas[mask]
    ok = np.isfinite(p) & np.isfinite(m)
    if ok.sum() < 4:
        return np.nan
    p, m = p[ok] - p[ok].mean(), m[ok] - m[ok].mean()
    den = np.linalg.norm(p) * np.linalg.norm(m)
    return float((p @ m) / den) if den > 1e-12 else np.nan


def _valid_mask(cs, test_i, train_idx):
    """Contacts usable for scoring fold test_i: not the test stim pair, finite in target & wmean."""
    n_c = len(cs.contacts)
    mask = np.ones(n_c, bool)
    for e in cs.stim_idx[test_i]:
        if e >= 0:
            mask[e] = False
    return mask


# -------------------------------------------------------------------- predictors


def predict_distance(cs, test_i, sigma):
    D = np.linalg.norm(cs.contact_xyz - cs.stim_xyz[test_i][None], axis=1)
    return np.exp(-(D ** 2) / (2 * sigma ** 2))


def predict_stim_knn(cs, test_i, train_idx, tau):
    """Weighted avg of training site topographies, weighted by stim-location proximity."""
    d = np.linalg.norm(cs.stim_xyz[train_idx] - cs.stim_xyz[test_i][None], axis=1)
    w = np.exp(-(d ** 2) / (2 * tau ** 2))
    if w.sum() < 1e-9:
        w = np.ones_like(w)
    R = cs.responses[train_idx]                      # [n_train, n_c]
    return np.nansum(w[:, None] * R, axis=0) / (np.nansum(w[:, None] * np.isfinite(R), axis=0) + 1e-9)


def predict_operator(cs, test_i, train_idx, k, steps, mode="forward"):
    """Effective-connectivity propagation.

    W[a,:] = response topography when contact a was a stim contact (from training sites).
    The held-out stim row is interpolated from the k nearest training stim sites (by stim coord).
    Then propagate a multi-step impulse from the held-out stim location through W.

    mode controls how connectivity direction is used (the CCEP directionality test):
      'forward'   : stimulate -> response  (Wn.T @ h), the correct efferent direction.
      'transpose' : use the reversed edges  (Wn @ h), i.e. afferent/incoming connectivity.
      'symmetric' : undirected  (((Wn+Wn.T)/2) @ h).
    """
    n_c = len(cs.contacts)
    W = np.zeros((n_c, n_c))
    has_row = np.zeros(n_c, bool)
    for s in train_idx:
        r = np.nan_to_num(cs.responses[s])
        for a in cs.stim_idx[s]:
            if a >= 0:
                W[a] = r
                has_row[a] = True
    # row-normalise to a propagation operator
    rs = W.sum(1, keepdims=True)
    Wn = W / (rs + 1e-9)

    # held-out impulse: place mass on the k nearest training stim sites' contacts
    d = np.linalg.norm(cs.stim_xyz[train_idx] - cs.stim_xyz[test_i][None], axis=1)
    order = np.argsort(d)[:k]
    h = np.zeros(n_c)
    for rank, s_local in enumerate(order):
        s = train_idx[s_local]
        wgt = np.exp(-(d[s_local] ** 2) / (2 * (d[order].mean() + 1e-6) ** 2))
        for a in cs.stim_idx[s]:
            if a >= 0:
                h[a] += wgt
    if h.sum() < 1e-9:
        return np.full(n_c, np.nan)
    h = h / h.sum()
    if mode == "forward":
        P = Wn.T
    elif mode == "transpose":
        P = Wn
    elif mode == "symmetric":
        P = (Wn + Wn.T) / 2
    else:
        raise ValueError(mode)
    energy = np.zeros(n_c)
    cur = h.copy()
    for _ in range(steps):
        cur = P @ cur
        energy += cur
    return energy


# ---------------------------------------------------------------- nested-CV eval


def _score_param(cs, train_idx, predictor):
    """Inner LOO over training sites; return mean topo_r for a fully-specified predictor closure."""
    rs = []
    for j_local, j in enumerate(train_idx):
        inner_train = [t for t in train_idx if t != j]
        if len(inner_train) < 3:
            continue
        pred = predictor(j, inner_train)
        mask = _valid_mask(cs, j, inner_train)
        rs.append(topo_r(pred, cs.responses[j], mask))
    rs = [r for r in rs if np.isfinite(r)]
    return np.mean(rs) if rs else -1.0


def eval_subject(cs):
    sites = np.arange(len(cs.sites))
    rel = cs.reliability
    keep = sites[(np.isfinite(rel)) & (rel >= REL_MIN)]
    if len(keep) < 6:
        return None
    per_fold = {m: [] for m in ["within_mean", "distance", "stim_knn", "operator", "additive", "combo"]}

    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        tgt = cs.responses[test_i]
        mask = _valid_mask(cs, test_i, train_idx)

        # within_mean
        R = cs.responses[train_idx]
        wmean = np.nansum(R, axis=0) / (np.sum(np.isfinite(R), axis=0) + 1e-9)
        per_fold["within_mean"].append(topo_r(wmean, tgt, mask))

        # distance (nested-CV sigma)
        sig = max(SIGMA_GRID, key=lambda s: _score_param(
            cs, train_idx, lambda j, tr, s=s: predict_distance(cs, j, s)))
        per_fold["distance"].append(topo_r(predict_distance(cs, test_i, sig), tgt, mask))

        # stim_knn (nested-CV tau)
        tau = max(TAU_GRID, key=lambda tt: _score_param(
            cs, train_idx, lambda j, tr, tt=tt: predict_stim_knn(cs, j, tr, tt)))
        knn = predict_stim_knn(cs, test_i, train_idx, tau)
        per_fold["stim_knn"].append(topo_r(knn, tgt, mask))

        # operator (nested-CV steps; k fixed small)
        st = max(STEP_GRID, key=lambda st: _score_param(
            cs, train_idx, lambda j, tr, st=st: predict_operator(cs, j, tr, 3, st)))
        op = predict_operator(cs, test_i, train_idx, 3, st)
        per_fold["operator"].append(topo_r(op, tgt, mask))

        # additive: within_mean + residual deviation from the better of {stim_knn, operator}
        # choose which deviation model on inner CV
        def add_pred(j, tr, which):
            Rj = cs.responses[tr]
            wm = np.nansum(Rj, axis=0) / (np.sum(np.isfinite(Rj), axis=0) + 1e-9)
            base = predict_stim_knn(cs, j, tr, tau) if which == "knn" else predict_operator(cs, j, tr, 3, st)
            dev = np.nan_to_num(base) - np.nan_to_num(wm)
            return wm + dev
        which = max(["knn", "op"], key=lambda w: _score_param(
            cs, train_idx, lambda j, tr, w=w: add_pred(j, tr, w)))
        per_fold["additive"].append(topo_r(add_pred(test_i, train_idx, which), tgt, mask))

        # combo: locality + network residual (nested-CV beta; reuse sig, tau)
        beta = max(BETA_GRID, key=lambda b: _score_param(
            cs, train_idx,
            lambda j, tr, b=b: predict_combo(cs, j, tr, sig, tau, b, _valid_mask(cs, j, tr))))
        per_fold["combo"].append(topo_r(predict_combo(cs, test_i, train_idx, sig, tau, beta, mask), tgt, mask))

    return {m: float(np.nanmean(v)) for m, v in per_fold.items()}, len(keep)


def _resid(x, basis, mask):
    """Residual of x after removing its projection on (centred) basis, over masked contacts."""
    xc = np.where(mask, np.nan_to_num(x), np.nan)
    b = np.where(mask, np.nan_to_num(basis), np.nan)
    ok = np.isfinite(xc) & np.isfinite(b)
    out = np.full_like(x, np.nan, dtype=float)
    bb = b[ok] - b[ok].mean(); nb = np.linalg.norm(bb)
    xx = xc[ok] - xc[ok].mean()
    out[ok] = xx if nb < 1e-9 else xx - (xx @ (bb / nb)) * (bb / nb)
    return out


def incremental_subject(cs):
    """Does the network/operator readout explain target structure BEYOND spatial locality?

    Mirrors phase4b: residualise target & predictor against the (nested-CV) distance prediction,
    then correlate. >0 => mechanism beyond locality. Subject-level mean over folds.
    """
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
    if len(keep) < 6:
        return None
    inc = {"stim_knn": [], "operator": []}
    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        tgt = cs.responses[test_i]
        mask = _valid_mask(cs, test_i, train_idx)
        sig = max(SIGMA_GRID, key=lambda s: _score_param(
            cs, train_idx, lambda j, tr, s=s: predict_distance(cs, j, s)))
        loc = predict_distance(cs, test_i, sig)
        tgt_r = _resid(tgt, loc, mask)
        tau = max(TAU_GRID, key=lambda tt: _score_param(
            cs, train_idx, lambda j, tr, tt=tt: predict_stim_knn(cs, j, tr, tt)))
        st = max(STEP_GRID, key=lambda st: _score_param(
            cs, train_idx, lambda j, tr, st=st: predict_operator(cs, j, tr, 3, st)))
        preds = {"stim_knn": predict_stim_knn(cs, test_i, train_idx, tau),
                 "operator": predict_operator(cs, test_i, train_idx, 3, st)}
        for m, pr in preds.items():
            inc[m].append(topo_r(_resid(pr, loc, mask), tgt_r, mask))
    return {m: float(np.nanmean(v)) for m, v in inc.items()}


def main():
    caches = all_caches()
    if not caches:
        print("no caches; run scripts/build_ccep.py first"); return
    models = ["within_mean", "distance", "stim_knn", "operator", "additive", "combo"]
    rows = {m: [] for m in models}
    subj_names, nsites = [], []
    print(f"{'subject':20s} {'nsites':>6s} " + " ".join(f"{m:>11s}" for m in models))
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        tag = f"{ds[-4:]}/{cs.subject}"
        res = eval_subject(cs)
        if res is None:
            print(f"{tag:20s}  (too few reliable sites)"); continue
        scores, nk = res
        subj_names.append(tag); nsites.append(nk)
        for m in models:
            rows[m].append(scores[m])
        print(f"{tag:20s} {nk:6d} " + " ".join(f"{scores[m]:>+11.3f}" for m in models))

    print("\n=== SUBJECT-LEVEL topography-r (mean over subjects, bootstrap 95% CI) ===")
    for m in models:
        v = np.array(rows[m]); mean, lo, hi = bootstrap_ci(v.tolist())
        print(f"  {m:12s} {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== vs within_mean (the bar) — paired across subjects ===")
    wm = rows["within_mean"]
    for m in ["distance", "stim_knn", "operator", "additive", "combo"]:
        v = rows[m]
        diff = np.mean(v) - np.mean(wm)
        p = paired_permutation_test(v, wm)
        d = cohens_d_paired(v, wm)
        win = sum(1 for a, b in zip(v, wm) if a > b)
        flag = "  <-- beats within_mean" if diff > 0 and p < 0.1 else ""
        print(f"  {m:12s} delta={diff:+.3f}  p={p:.3g}  d={d:+.2f}  ({win}/{len(v)} subj win){flag}")

    # mechanism test: does the network/operator add anything BEYOND spatial locality?
    print("\n=== INCREMENTAL-over-locality (mechanism test: network beyond distance) ===")
    inc = {"stim_knn": [], "operator": []}
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        r = incremental_subject(cs)
        if r is None:
            continue
        for m in inc:
            inc[m].append(r[m])
    for m in ["stim_knn", "operator"]:
        v = np.array(inc[m]); mean, lo, hi = bootstrap_ci(v.tolist())
        p0 = paired_permutation_test(v.tolist(), [0.0] * len(v))
        pos = int((v > 0).sum())
        verdict = "  <-- adds beyond locality" if mean > 0.05 and p0 < 0.1 else "  (no signal beyond locality)"
        print(f"  {m:12s} incremental {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  p(vs0)={p0:.3g}  "
              f"({pos}/{len(v)} subj>0){verdict}")


if __name__ == "__main__":
    main()
