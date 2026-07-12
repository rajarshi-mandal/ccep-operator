"""Stage 2-3-5 — train the subject-conditioned do()-readout and run the formal ablation (ds002799).

Trains src/model/es_readout.ESReadout end-to-end to predict held-out es sites' evoked topography,
and answers the two decisive questions:
  * Does the TRAINED model capture site-specific structure BEYOND spatial locality? (incremental-r)
  * Does the SUBJECT CHANNEL matter? — ablation: drop ΔA_subj (A=A_group). If the full model beats
    the ablation, subject conditioning is real; if not, it is inert (posterior collapse).

Eval: leave-out a fraction of each subject's sites as test, train on the rest (pooled across
subjects, amortized), repeat over seeds. Compares trained_full / trained_ablation / locality /
within_mean on raw spatial r, deviation-r (vs within_mean), and incremental-r over locality.
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

SIGMA, STEPS, EPOCHS, SEEDS = 25.0, 6, 400, 5


def _resid(x, basis):
    b = basis - basis.mean(); nb = np.linalg.norm(b)
    if nb < 1e-9:
        return x - x.mean()
    b = b / nb; xc = x - x.mean()
    return xc - (xc @ b) * b


def main():
    cache = DS005498Cache(cache_dir="data/processed/ds002799", qc_filter=True)
    cents = cache.centroids; d = cents.shape[0]
    Dmm = np.linalg.norm(cents[:, None] - cents[None], axis=-1)
    by_sub = {}
    for r in cache.records:
        by_sub.setdefault(r.subject, []).append(r)
    subs = [s for s in sorted(by_sub) if len(by_sub[s]) >= 3]   # need train+test sites
    fc_feat = {s: torch.tensor(np.nan_to_num(np.corrcoef(by_sub[s][0].subject_rest.T)).mean(0),
                               dtype=torch.float32) for s in subs}
    print(f"[phase5] {sum(len(by_sub[s]) for s in subs)} records, {len(subs)} subjects (>=3 sites)")

    agg = {k: [] for k in ["full", "ablate", "locality", "within"]}
    agg_dev = {k: [] for k in ["full", "ablate", "locality"]}
    agg_inc = {"full": [], "ablate": []}

    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        train, test = [], []
        for s in subs:
            idx = rng.permutation(len(by_sub[s]))
            ntest = max(1, len(idx) // 3)
            test += [(s, by_sub[s][i]) for i in idx[:ntest]]
            train += [(s, by_sub[s][i]) for i in idx[ntest:]]
        train_topos = {s: [r.topo for (ss, r) in train if ss == s] for s in subs}

        model = ESReadout(d, rank=8, steps=STEPS)
        opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
        T = [(r.stim_parcel, fc_feat[s], torch.tensor(r.topo)) for (s, r) in train]
        for ep in range(EPOCHS):
            opt.zero_grad(); loss = 0.0
            for p, ff, y in T:
                loss = loss + topo_loss(model.predict(p, ff), y)
            (loss / len(T)).backward(); opt.step()

        model.eval()
        with torch.no_grad():
            for s, test_r in test:
                tr = train_topos[s]
                if not tr:
                    continue
                wmean = np.mean(tr, axis=0)
                meas = test_r.topo; p = test_r.stim_parcel
                loc = np.exp(-(Dmm[p] ** 2) / (2 * SIGMA ** 2))
                pred_full = model.predict(p, fc_feat[s], ablate=False).numpy()
                pred_abl = model.predict(p, fc_feat[s], ablate=True).numpy()
                preds = {"full": pred_full, "ablate": pred_abl, "locality": loc, "within": wmean}
                for k, pr in preds.items():
                    agg[k].append(spatial_r(pr, meas))
                tgt_dev = meas - wmean; loc_dev = loc - wmean
                tgt_r = _resid(tgt_dev, loc_dev)
                for k in ["full", "ablate", "locality"]:
                    agg_dev[k].append(spatial_r(preds[k] - wmean, tgt_dev))
                for k in ["full", "ablate"]:
                    agg_inc[k].append(spatial_r(_resid(preds[k] - wmean, loc_dev), tgt_r))
        print(f"  seed {seed}: trained ({len(T)} train, {len(test)} test pairs)", flush=True)

    def summ(v):
        v = np.array(v); m, lo, hi = bootstrap_ci(v.tolist()); return m, lo, hi

    print("\n=== raw spatial r (full topography) ===")
    for k in ["within", "locality", "ablate", "full"]:
        m, lo, hi = summ(agg[k]); print(f"  {k:9s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("\n=== deviation-r (site-specific) ===")
    for k in ["locality", "ablate", "full"]:
        m, lo, hi = summ(agg_dev[k]); print(f"  {k:9s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print("\n=== incremental-r OVER locality (the causal-value test) ===")
    for k in ["ablate", "full"]:
        v = agg_inc[k]; m, lo, hi = summ(v)
        p0 = paired_permutation_test(v, [0.0] * len(v))
        print(f"  {k:9s} {m:+.3f} [{lo:+.3f},{hi:+.3f}] p(vs0)={p0:.3g}")
    # ABLATION: full vs ablate on incremental
    pa = paired_permutation_test(agg_inc["full"], agg_inc["ablate"])
    df = np.mean(agg_inc["full"]) - np.mean(agg_inc["ablate"])
    print(f"\n=== ABLATION (subject channel) ===\n  full − ablate incremental-r = {df:+.3f} p={pa:.3g}")
    json.dump({"raw": {k: float(np.mean(agg[k])) for k in agg},
               "deviation": {k: float(np.mean(agg_dev[k])) for k in agg_dev},
               "incremental": {k: float(np.mean(agg_inc[k])) for k in agg_inc},
               "ablation_diff": float(df), "ablation_p": float(pa)},
              open("data/processed/ds002799_phase5.json", "w"), indent=2)
    print("\nVERDICT: "
          + ("trained model beats locality "
             + ("AND subject conditioning helps (ablation drops)."
                if df > 0.02 and pa < 0.1 else "but subject channel is ~inert (ablation flat).")
             if np.mean(agg_inc["full"]) > 0.05 and
             paired_permutation_test(agg_inc["full"], [0.0] * len(agg_inc["full"])) < 0.1
             else "trained model does NOT beat locality."))


if __name__ == "__main__":
    main()
