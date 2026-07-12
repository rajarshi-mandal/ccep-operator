"""Exp-5 (Arm B) — latency re-analysis of the *already-trained* Exp-1B fold models.

The Exp-1B headline collapses each region's response to an energy scalar; on that static
topography the causal model ties the group-mean template (H2 null) by construction. This arm
asks a question the energy metric cannot: does the same trained causal graph predict **when**
each region responds — the propagation *order* after a do(M1) pulse?

Pure re-analysis: we reload the 13 Exp-1B fold checkpoints (identical models, no retraining) and
score per-region temporal centre-of-mass (a robust latency proxy) by Spearman rank correlation
against the held-out subject's measured latency. Predicted "steps" and measured "samples" live on
different, unidentified scales, so only the *rank order* of activation is claimed.

Fair baseline: the group-mean latency template (mean measured latency over training subjects).
If the causal graph's predicted latency-order beats that template where the energy metric tied,
the causal structure carries information the template lacks — on the *same* published models.
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
from config import load_config  # noqa: E402
from data.dataset import InterventionalDataset  # noqa: E402
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from eval.readouts import temporal_com, spearmanr, downstream_mask  # noqa: E402


@torch.no_grad()
def measured_latency(record) -> torch.Tensor:
    """Per-region temporal centre-of-mass of a measured TEP ``[d, T]`` -> ``[d]``."""
    return temporal_com(record["region_tep"].unsqueeze(0), time_dim=-1)[0]


@torch.no_grad()
def predicted_latency(model, stim_parcel, steps) -> torch.Tensor:
    """Per-region temporal COM of the do(stim) rollout -> ``[d]`` (abstract-step units)."""
    pred = model.predict_intervention(torch.tensor([stim_parcel]), 1.0, steps)  # [1, steps, d]
    return temporal_com(pred, time_dim=1)[0]


def run(cfg, proc, args):
    d = cfg.parcellation.d
    m1 = InterventionalDataset(proc, site_filter={"M1_L"})
    records = [m1[i] for i in range(len(m1))]
    ckpt_dir = Path(proc) / "checkpoints"
    meas_lat = [measured_latency(r) for r in records]                       # per subject [d]

    folds = []
    for k in range(len(records)):
        ck_path = ckpt_dir / f"exp1b_fold{k:02d}.pt"
        if not ck_path.exists():
            print(f"  fold {k:02d}: checkpoint missing, skipping")
            continue
        ck = torch.load(ck_path, weights_only=False)
        held_idx = int(ck.get("held_idx", k))
        held = records[held_idx]
        stim = int(held["stim_parcel"])
        model = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                             init_scale=cfg.model.init_state_scale)
        model.load_state_dict(ck["state_dict"])

        keep = downstream_mask(d, stim)
        lat_pred = predicted_latency(model, stim, args.steps)
        lat_meas = meas_lat[held_idx]
        # group-mean latency template (leave-one-out over training subjects)
        tmpl = torch.stack([meas_lat[j] for j in range(len(records)) if j != held_idx]).mean(0)

        folds.append({
            "fold": k, "held_idx": held_idx, "subject": str(held["subject"]),
            "causal_latency_rho": spearmanr(lat_pred[keep], lat_meas[keep]),
            "template_latency_rho": spearmanr(tmpl[keep], lat_meas[keep]),
        })
        print(f"  fold {k:02d} {folds[-1]['subject']:>12}: "
              f"causal rho={folds[-1]['causal_latency_rho']:+.3f}  "
              f"template rho={folds[-1]['template_latency_rho']:+.3f}", flush=True)

    c = np.array([f["causal_latency_rho"] for f in folds])
    t = np.array([f["template_latency_rho"] for f in folds])
    print("\n===== Exp-5 latency re-analysis (Arm B) =====")
    print(f"folds={len(folds)} | causal {c.mean():+.3f}±{c.std():.3f} | "
          f"template {t.mean():+.3f}±{t.std():.3f} | diff {np.mean(c - t):+.3f}")

    out = {"experiment": "5_latency_reanalysis", "metric": "spearman_latency_downstream",
           "steps": args.steps, "n_folds": len(folds),
           "causal_latency_rho_mean": float(c.mean()),
           "template_latency_rho_mean": float(t.mean()),
           "folds": folds}
    out_path = Path(args.out) if args.out else Path(proc) / "exp5_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("results ->", out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64, help="do-rollout length for latency resolution")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config()
    return run(cfg, cfg.paths.processed_dir, args)


if __name__ == "__main__":
    sys.exit(main())
