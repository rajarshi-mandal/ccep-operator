"""Phase 6 — all five model enhancements, integrated (ds002799).

#1 subject-own operator: fit a low-rank ΔA_subj from the subject's OWN train stim→response pairs
   (shrunk to the group), not rest FC, and use it for the held-out site. The real personalization test.
#2 target engineering: predict the top-K group-PC-denoised topography, with reliability-weighted loss.
#3 enhanced readout: directed group operator + NOTEARS acyclicity penalty + learnable per-step weights.
#4 finer parcellation: runs on whatever cache --cache-dir points to (Schaefer-100 or -200).
#5 power/rigor: multi-seed site-holdout CV + bootstrap CIs; formal ablation (drop ΔA_subj).

Reports raw r, site-specific deviation-r, incremental-over-locality, and the ablation
(full − group), for: enhanced full / enhanced group-only / locality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ds005498_pipeline import DS005498Cache  # noqa: E402
from model.es_readout import ESReadout2, topo_loss  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test  # noqa: E402
from phase2_loso_ws import spatial_r  # noqa: E402

SIGMA = 25.0


def _resid(x, basis):
    b = basis - basis.mean(); nb = np.linalg.norm(b)
    if nb < 1e-9:
        return x - x.mean()
    b = b / nb; xc = x - x.mean(); return xc - (xc @ b) * b


def pc_basis(topos, k):
    M = np.stack(topos); mu = M.mean(0); Mc = M - mu
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return mu, Vt[:k]                                  # [d], [k,d]


def pc_project(y, mu, B):
    return mu + (y - mu) @ B.T @ B


def fit_subject_dA(model, sites, d, rank, steps=80, shrink=5.0):
    """#1 — fit low-rank ΔA on the subject's own train (stim, topo) pairs; shrink toward 0."""
    U = torch.zeros(d, rank, requires_grad=True); V = torch.zeros(d, rank, requires_grad=True)
    torch.nn.init.normal_(U, std=0.02); torch.nn.init.normal_(V, std=0.02)
    opt = torch.optim.Adam([U, V], lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(); dA = U @ V.t()
        loss = sum(topo_loss(model.predict(p, dA), y) for p, y in sites) / len(sites)
        loss = loss + shrink * dA.pow(2).mean()
        loss.backward(); opt.step()
    return (U @ V.t()).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/processed/ds002799")
    ap.add_argument("--pcs", type=int, default=15)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--dag-lambda", type=float, default=1e-3)
    ap.add_argument("--out", default="data/processed/ds002799_phase6.json")
    args = ap.parse_args()

    cache = DS005498Cache(cache_dir=args.cache_dir, qc_filter=True)
    cents = cache.centroids; d = cents.shape[0]
    Dmm = np.linalg.norm(cents[:, None] - cents[None], axis=-1)
    by_sub = {}
    for r in cache.records:
        by_sub.setdefault(r.subject, []).append(r)
    subs = [s for s in sorted(by_sub) if len(by_sub[s]) >= 3]
    print(f"[phase6] d={d}, {sum(len(by_sub[s]) for s in subs)} records, {len(subs)} subjects, "
          f"K={args.pcs} PCs, rank={args.rank}", flush=True)

    agg = {k: [] for k in ["full", "group", "locality"]}
    inc = {"full": [], "group": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        train, test = [], []
        for s in subs:
            idx = rng.permutation(len(by_sub[s])); nt = max(1, len(idx) // 3)
            test += [(s, by_sub[s][i]) for i in idx[:nt]]
            train += [(s, by_sub[s][i]) for i in idx[nt:]]
        tr_by_sub = {s: [r for (ss, r) in train if ss == s] for s in subs}
        mu, B = pc_basis([r.topo for (_, r) in train], args.pcs)                 # #2 group PC basis

        # ---- train enhanced group model (#3) on PC-denoised, reliability-weighted targets (#2) ----
        model = ESReadout2(d, steps=args.steps)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        T = [(r.stim_parcel, torch.tensor(pc_project(r.topo, mu, B), dtype=torch.float32),
              max(r.reliability, 0.0) if np.isfinite(r.reliability) else 0.0) for (_, r) in train]
        wsum = sum(w for _, _, w in T) + 1e-9
        for _ in range(args.epochs):
            opt.zero_grad()
            loss = sum(w * topo_loss(model.predict(p, None), y) for p, y, w in T) / wsum
            loss = loss + args.dag_lambda * model.acyclicity()
            loss.backward(); opt.step()
        model.eval()

        # ---- per-subject ΔA from own train sites (#1) ----
        dA = {}
        for s in subs:
            sites = [(r.stim_parcel, torch.tensor(pc_project(r.topo, mu, B), dtype=torch.float32))
                     for r in tr_by_sub[s]]
            dA[s] = fit_subject_dA(model, sites, d, args.rank) if len(sites) >= 2 else None

        with torch.no_grad():
            for s, tr in test:
                if not tr_by_sub[s]:
                    continue
                wmean = np.mean([r.topo for r in tr_by_sub[s]], axis=0)
                meas = tr.topo; p = tr.stim_parcel
                loc = np.exp(-(Dmm[p] ** 2) / (2 * SIGMA ** 2)); loc_dev = loc - wmean
                tgt_r = _resid(meas - wmean, loc_dev)
                pr = {"full": model.predict(p, dA[s]).numpy() if dA[s] is not None
                      else model.predict(p, None).numpy(),
                      "group": model.predict(p, None).numpy(), "locality": loc}
                for k in agg:
                    agg[k].append(spatial_r(pr[k], meas))
                for k in inc:
                    inc[k].append(spatial_r(_resid(pr[k] - wmean, loc_dev), tgt_r))
        print(f"  seed {seed} done", flush=True)

    def S(v):
        m, lo, hi = bootstrap_ci(v); return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]"
    print("\n=== raw spatial r ===")
    for k in ["locality", "group", "full"]:
        print(f"  {k:9s} {S(agg[k])}")
    print("\n=== incremental-r OVER locality ===")
    for k in ["group", "full"]:
        p0 = paired_permutation_test(inc[k], [0.0] * len(inc[k]))
        print(f"  {k:9s} {S(inc[k])}  p(vs0)={p0:.3g}")
    df = np.mean(inc["full"]) - np.mean(inc["group"]); pa = paired_permutation_test(inc["full"], inc["group"])
    print(f"\n=== ABLATION (subject ΔA from own stim data) ===\n  full − group = {df:+.3f}  p={pa:.3g}")
    json.dump({"d": d, "raw": {k: float(np.mean(agg[k])) for k in agg},
               "incremental": {k: float(np.mean(inc[k])) for k in inc},
               "ablation_diff": float(df), "ablation_p": float(pa)}, open(args.out, "w"), indent=2)
    print("\nVERDICT: enhanced model "
          + ("beats locality" if np.mean(inc["full"]) > 0.05 and
             paired_permutation_test(inc["full"], [0.0] * len(inc["full"])) < 0.1 else "ties locality")
          + "; personalization "
          + ("HELPS (full>group)." if df > 0.02 and pa < 0.1 else "does NOT beat the group."))


if __name__ == "__main__":
    main()
