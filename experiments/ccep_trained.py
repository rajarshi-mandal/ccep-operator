"""CCEP directionality + trained operator (follow-up to ccep_loso.py).

Two questions the simple data-operator left open:

A. **Directionality** — CCEP is a directed evoked response (stimulating site i drives site j via
   *efferent* connectivity, i->j). Is that asymmetry real and does it help prediction?
   - descriptive: asymmetry of the site x site effective-connectivity matrix E (corr of E vs E^T).
   - predictive: propagate the data operator FORWARD (Wn.T, efferent) vs TRANSPOSE (Wn, afferent)
     vs SYMMETRIC ((Wn+Wn.T)/2). If forward > transpose, direction carries information.

B. **Trained operator** — does fitting the operator end-to-end (the ESReadout2 form: directed W,
   energy readout, learnable per-step weights), instead of using the raw data operator, improve
   the held-out prediction? Trained per subject by LOSO, initialised at the data operator with an
   L2-to-init prior (the §4 audit flagged d x d operators as over-parameterised on ~40 sites, so we
   regularise toward the well-behaved data prior). Directed-trained vs symmetric-trained tests
   directionality again, now in the learned regime.

All metrics aggregate to SUBJECT level; paired permutation across the 13 subjects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test, cohens_d_paired  # noqa: E402
from ccep_loso import (  # noqa: E402
    all_caches, topo_r, _valid_mask, predict_operator, predict_distance,
    predict_stim_knn, predict_combo, REL_MIN, SIGMA_GRID, TAU_GRID, STEP_GRID,
)

import torch  # noqa: E402

torch.manual_seed(0)


# --------------------------------------------------------------- A. directionality (descriptive)

def site_effconn(cs, sites):
    """E[i,j] = mean N1 amplitude at site j's contacts when stimulating site i (directed)."""
    n = len(sites)
    E = np.full((n, n), np.nan)
    for ii, i in enumerate(sites):
        resp = cs.responses[i]                       # response over contacts when stimulating i
        for jj, j in enumerate(sites):
            if ii == jj:
                continue
            cj = [c for c in cs.stim_idx[j] if c >= 0]
            vals = [resp[c] for c in cj if np.isfinite(resp[c])]
            if vals:
                E[ii, jj] = np.mean(vals)
    return E


def asymmetry(E):
    iu = np.triu_indices_from(E, k=1)
    a, b = E[iu], E.T[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 4:
        return np.nan, np.nan
    r = np.corrcoef(a, b)[0, 1]                       # symmetry: 1 => fully symmetric
    asym = np.linalg.norm(E[ok_mask(E)] - E.T[ok_mask(E)]) / (np.linalg.norm(E[ok_mask(E)]) + 1e-9)
    return r, asym


def ok_mask(E):
    return np.isfinite(E) & np.isfinite(E.T)


# ------------------------------------------------------------------- B. trained operator (torch)

def masked_cos_loss(pred, Y, M):
    """1 - mean over columns of masked, mean-centred cosine. pred,Y,M: [n_c, n_cols]."""
    cnt = M.sum(0).clamp_min(1.0)
    pm = (pred * M).sum(0) / cnt
    ym = (Y * M).sum(0) / cnt
    pc = (pred - pm) * M
    yc = (Y - ym) * M
    num = (pc * yc).sum(0)
    den = torch.sqrt((pc * pc).sum(0) * (yc * yc).sum(0)).clamp_min(1e-8)
    return 1.0 - (num / den).mean()


def rollout(P, H0, log_w):
    """Energy readout: sqrt(sum_t softmax(w)_t * (P^t H0)^2). P:[d,d] H0:[d,n] -> [d,n]."""
    w = torch.softmax(log_w, 0)
    cur = H0
    energy = torch.zeros_like(H0)
    for t in range(log_w.shape[0]):
        energy = energy + w[t] * cur * cur
        cur = P @ cur
    return torch.sqrt(energy + 1e-8)


def train_operator(P_init, H0, Y, M, steps, symmetric, n_iter=250, lr=0.05, l2=3.0):
    d = P_init.shape[0]
    dP = torch.zeros((d, d), requires_grad=True)
    log_w = torch.zeros(steps, requires_grad=True)
    P0 = torch.tensor(P_init, dtype=torch.float32)
    opt = torch.optim.Adam([dP, log_w], lr=lr)
    H0t = torch.tensor(H0, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    Mt = torch.tensor(M, dtype=torch.float32)
    for _ in range(n_iter):
        opt.zero_grad()
        P = P0 + dP
        if symmetric:
            P = 0.5 * (P + P.t())
        pred = rollout(P, H0t, log_w)
        loss = masked_cos_loss(pred, Yt, Mt) + l2 * (dP * dP).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        P = P0 + dP
        if symmetric:
            P = 0.5 * (P + P.t())
    return P, log_w.detach()


def data_P(cs, train_idx, n_c):
    """Forward propagation operator P = Wn.T from training stim->response rows."""
    W = np.zeros((n_c, n_c))
    for s in train_idx:
        r = np.nan_to_num(cs.responses[s])
        for a in cs.stim_idx[s]:
            if a >= 0:
                W[a] = r
    Wn = W / (W.sum(1, keepdims=True) + 1e-9)
    return Wn.T


def impulse_knn(cs, test_i, train_idx, n_c, k=3):
    d = np.linalg.norm(cs.stim_xyz[train_idx] - cs.stim_xyz[test_i][None], axis=1)
    order = np.argsort(d)[:k]
    h = np.zeros(n_c)
    for s_local in order:
        s = train_idx[s_local]
        wgt = np.exp(-(d[s_local] ** 2) / (2 * (d[order].mean() + 1e-6) ** 2))
        for a in cs.stim_idx[s]:
            if a >= 0:
                h[a] += wgt
    return h / (h.sum() + 1e-9)


def impulse_self(cs, site, n_c):
    h = np.zeros(n_c)
    for a in cs.stim_idx[site]:
        if a >= 0:
            h[a] += 1.0
    return h / (h.sum() + 1e-9)


def eval_trained(cs):
    sites = np.arange(len(cs.sites))
    keep = sites[(np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN)]
    if len(keep) < 6:
        return None
    n_c = len(cs.contacts)
    steps = 3
    out = {m: [] for m in ["op_forward", "op_transpose", "op_symmetric",
                           "trained_dir", "trained_sym"]}
    for test_i in keep:
        train_idx = [t for t in keep if t != test_i]
        tgt = cs.responses[test_i]
        mask = _valid_mask(cs, test_i, train_idx)
        # data operator, three directions
        for mode, key in [("forward", "op_forward"), ("transpose", "op_transpose"),
                          ("symmetric", "op_symmetric")]:
            pr = predict_operator(cs, test_i, train_idx, 3, steps, mode=mode)
            out[key].append(topo_r(pr, tgt, mask))
        # trained operator (directed + symmetric), init from data forward operator
        P0 = data_P(cs, train_idx, n_c)
        H0 = np.stack([impulse_self(cs, s, n_c) for s in train_idx], axis=1)   # [n_c, n_train]
        Y = np.stack([np.nan_to_num(cs.responses[s]) for s in train_idx], axis=1)
        M = np.stack([(np.isfinite(cs.responses[s]) &
                       _valid_mask(cs, s, train_idx)).astype(float) for s in train_idx], axis=1)
        h_test = impulse_knn(cs, test_i, train_idx, n_c)
        for sym, key in [(False, "trained_dir"), (True, "trained_sym")]:
            P, log_w = train_operator(P0, H0, Y, M, steps, symmetric=sym)
            with torch.no_grad():
                pred = rollout(P, torch.tensor(h_test[:, None], dtype=torch.float32), log_w)
            out[key].append(topo_r(pred[:, 0].numpy(), tgt, mask))
    return {m: float(np.nanmean(v)) for m, v in out.items()}, len(keep)


def main():
    caches = all_caches()
    if not caches:
        print("no caches; run scripts/build_ccep.py first"); return

    # ---- A. descriptive directionality ----
    print("=== A. effective-connectivity directionality (site x site E vs E^T) ===")
    print(f"{'subject':20s} {'symmetry_r':>11s} {'asym_index':>11s}")
    sym_rs = []
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        keep = np.where((np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN))[0]
        if len(keep) < 6:
            continue
        E = site_effconn(cs, keep)
        r, asym = asymmetry(E)
        sym_rs.append(r)
        print(f"{ds[-4:]+'/'+cs.subject:20s} {r:>+11.3f} {asym:>11.3f}")
    sr = np.array(sym_rs)
    print(f"  mean E-vs-E^T symmetry r = {sr.mean():+.3f} [{np.percentile(sr,2.5):+.3f},"
          f"{np.percentile(sr,97.5):+.3f}]  (1.0 = symmetric; lower = more directed)")

    # ---- B/C. predictive directionality + trained operator ----
    models = ["op_forward", "op_transpose", "op_symmetric", "trained_dir", "trained_sym"]
    rows = {m: [] for m in models}
    print("\n=== predictive: data-operator directions + trained operator (subject-level topo-r) ===")
    print(f"{'subject':20s} " + " ".join(f"{m:>13s}" for m in models))
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        res = eval_trained(cs)
        if res is None:
            continue
        scores, nk = res
        for m in models:
            rows[m].append(scores[m])
        print(f"{ds[-4:]+'/'+cs.subject:20s} " + " ".join(f"{scores[m]:>+13.3f}" for m in models))

    print("\n=== SUBJECT-LEVEL means (bootstrap 95% CI) ===")
    for m in models:
        v = np.array(rows[m]); mean, lo, hi = bootstrap_ci(v.tolist())
        print(f"  {m:14s} {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== directionality: forward vs transpose / symmetric (paired) ===")
    fwd = rows["op_forward"]
    for m in ["op_transpose", "op_symmetric"]:
        v = rows[m]; p = paired_permutation_test(fwd, v); d = cohens_d_paired(fwd, v)
        win = sum(1 for a, b in zip(fwd, v) if a > b)
        print(f"  forward vs {m:14s} Δ={np.mean(fwd)-np.mean(v):+.3f}  p={p:.3g}  d={d:+.2f}  "
              f"({win}/{len(v)} forward wins)")

    print("\n=== trained vs simple data operator (forward), paired ===")
    for m in ["trained_dir", "trained_sym"]:
        v = rows[m]; p = paired_permutation_test(v, fwd); d = cohens_d_paired(v, fwd)
        win = sum(1 for a, b in zip(v, fwd) if a > b)
        flag = "  <-- training helps" if np.mean(v) > np.mean(fwd) and p < 0.1 else ""
        print(f"  {m:14s} vs op_forward  Δ={np.mean(v)-np.mean(fwd):+.3f}  p={p:.3g}  d={d:+.2f}  "
              f"({win}/{len(v)} win){flag}")


if __name__ == "__main__":
    main()
