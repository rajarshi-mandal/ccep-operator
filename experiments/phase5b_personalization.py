"""Stage 5b — give subject personalization its best, data-efficient shot (ds002799).

phase5's encoder+U+V conditioning was inert (likely overfit: 11 subjects). Here the subject's
structure enters DIRECTLY as its functional-connectivity deviation from the group, scaled by a
single learnable gain β (A_subj = A_group + β·FC_dev_s). Nothing subject-specific is *learned*, so
personalization cannot overfit — if subject connectivity helps predict held-out sites, β>0 helps
and the ablation (β=0) drops. Uses all subjects with >=2 sites for more data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ds005498_pipeline import DS005498Cache  # noqa: E402
from model.es_readout import ESReadout, topo_loss  # noqa: E402
from eval.stats import bootstrap_ci, paired_permutation_test  # noqa: E402
from phase2_loso_ws import spatial_r  # noqa: E402

SIGMA, STEPS, EPOCHS, SEEDS = 25.0, 6, 400, 6


def _resid(x, basis):
    b = basis - basis.mean(); nb = np.linalg.norm(b)
    if nb < 1e-9:
        return x - x.mean()
    b = b / nb; xc = x - x.mean(); return xc - (xc @ b) * b


def main():
    cache = DS005498Cache(cache_dir="data/processed/ds002799", qc_filter=True)
    cents = cache.centroids; d = cents.shape[0]
    Dmm = np.linalg.norm(cents[:, None] - cents[None], axis=-1)
    by_sub = {}
    for r in cache.records:
        by_sub.setdefault(r.subject, []).append(r)
    subs = [s for s in sorted(by_sub) if len(by_sub[s]) >= 2]
    FC = {s: np.nan_to_num(np.corrcoef(by_sub[s][0].subject_rest.T)) for s in subs}
    FCg = np.mean([FC[s] for s in subs], axis=0)
    FCdev = {s: torch.tensor(FC[s] - FCg, dtype=torch.float32) for s in subs}   # subject deviation
    print(f"[5b] {sum(len(by_sub[s]) for s in subs)} records, {len(subs)} subjects (>=2 sites)")

    agg = {k: [] for k in ["full", "ablate", "locality"]}
    inc = {"full": [], "ablate": []}
    betas = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        train, test = [], []
        for s in subs:
            idx = rng.permutation(len(by_sub[s])); nt = max(1, len(idx) // 3)
            test += [(s, by_sub[s][i]) for i in idx[:nt]]
            train += [(s, by_sub[s][i]) for i in idx[nt:]]
        train_topos = {s: [r.topo for (ss, r) in train if ss == s] for s in subs}
        if any(len(train_topos[s]) == 0 for s, _ in test):
            train_topos = {s: (train_topos[s] or [by_sub[s][0].topo]) for s in subs}

        model = ESReadout(d, steps=STEPS, cond_mode="fc_direct")
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        T = [(r.stim_parcel, FCdev[s], torch.tensor(r.topo)) for (s, r) in train]
        for _ in range(EPOCHS):
            opt.zero_grad(); loss = sum(topo_loss(model.predict(p, c), y) for p, c, y in T) / len(T)
            loss.backward(); opt.step()
        betas.append(float(model.beta.detach()))

        model.eval()
        with torch.no_grad():
            for s, tr in test:
                wmean = np.mean(train_topos[s], axis=0); meas = tr.topo; p = tr.stim_parcel
                loc = np.exp(-(Dmm[p] ** 2) / (2 * SIGMA ** 2)); loc_dev = loc - wmean
                tgt_r = _resid(meas - wmean, loc_dev)
                pr = {"full": model.predict(p, FCdev[s]).numpy(),
                      "ablate": model.predict(p, FCdev[s], ablate=True).numpy(), "locality": loc}
                for k in agg:
                    agg[k].append(spatial_r(pr[k], meas))
                for k in inc:
                    inc[k].append(spatial_r(_resid(pr[k] - wmean, loc_dev), tgt_r))
        print(f"  seed {seed}: beta={model.beta.item():+.3f}", flush=True)

    def summ(v):
        m, lo, hi = bootstrap_ci(v); return f"{m:+.3f} [{lo:+.3f},{hi:+.3f}]"
    print(f"\nlearned beta (subject-FC gain): mean {np.mean(betas):+.3f} "
          f"(>0 => model wants subject connectivity)")
    print("\n=== incremental-r OVER locality ===")
    for k in ["ablate", "full"]:
        p0 = paired_permutation_test(inc[k], [0.0] * len(inc[k]))
        print(f"  {k:7s} {summ(inc[k])}  p(vs0)={p0:.3g}")
    df = np.mean(inc["full"]) - np.mean(inc["ablate"])
    pa = paired_permutation_test(inc["full"], inc["ablate"])
    print(f"\n=== ABLATION (subject FC channel) ===\n  full − ablate = {df:+.3f}  p={pa:.3g}")
    json.dump({"beta_mean": float(np.mean(betas)), "incremental": {k: float(np.mean(inc[k])) for k in inc},
               "ablation_diff": float(df), "ablation_p": float(pa)},
              open("data/processed/ds002799_phase5b.json", "w"), indent=2)
    print("\nVERDICT: " + ("SUBJECT PERSONALIZATION HELPS — full beats ablation (β>0)."
                           if df > 0.02 and pa < 0.1
                           else "personalization still does NOT beat the group — subject FC adds no gain."))


if __name__ == "__main__":
    main()
