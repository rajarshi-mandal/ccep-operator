"""Exp-6 (Arm C) — early->late within-subject causal forecast on the trained Exp-1B models.

A group-level model cannot beat a group-mean template on a per-subject static metric because the
mean is the Bayes-optimal group predictor (the H2 ceiling). This arm sidesteps that ceiling
*without new data*: it conditions on information the template can never use — the held-out
subject's **own** early cortical response — and asks the learned causal graph to forecast the
**late** downstream propagation.

Mechanism (no model change): seed the SSM's deterministic rollout with the held subject's measured
early-window state ``h0`` and roll forward through the trained graph ``A`` (model.ssm.rollout).
The predicted late-window energy topography is scored against the measured late-window topography.

Control for trivial autocorrelation: the causal forecast must beat **persistence** (the measured
early topography used as-is to predict the late topography). Persistence is the graph-free null —
beating it proves the *graph* routes early activity to the correct late regions, not that early
and late are merely correlated. We also report an untrained-graph forecast as the floor.
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
from eval.readouts import pearsonr, windowed_energy, downstream_mask  # noqa: E402


def find_t0(proc):
    """Sample index of the TMS pulse (times >= 0) for the M1 records; fallback to 100."""
    raw = list(np.load(Path(proc) / "interventional_region.npz", allow_pickle=True)["records"])
    for r in raw:
        if r["site_name"] == "M1_L" and "times" in r:
            times = np.asarray(r["times"], dtype=float)
            idx = np.where(times >= 0)[0]
            return int(idx[0]) if idx.size else 100
    return 100


@torch.no_grad()
def causal_forecast(model, h0, roll_steps):
    """Roll the trained graph forward from a measured early state -> late energy topography ``[d]``."""
    H = model.ssm.rollout(h0.unsqueeze(0), roll_steps, model.A)   # [1, roll_steps, d]
    return H[0].pow(2).mean(dim=0).clamp_min(1e-12).sqrt()        # [d]


def run(cfg, proc, args):
    d = cfg.parcellation.d
    m1 = InterventionalDataset(proc, site_filter={"M1_L"})
    records = [m1[i] for i in range(len(m1))]
    ckpt_dir = Path(proc) / "checkpoints"

    t0 = find_t0(proc)
    e0, e1 = t0, t0 + args.early_ms                              # early window (immediate response)
    l0, l1 = t0 + args.early_ms, t0 + args.early_ms + args.late_ms  # late window (propagation)
    print(f"Exp-6 early->late forecast | t0={t0} early=[{e0},{e1}) late=[{l0},{l1}) "
          f"roll_steps={args.roll_steps}")

    untrained = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                             init_scale=cfg.model.init_state_scale)

    folds = []
    for k in range(len(records)):
        ck_path = ckpt_dir / f"exp1b_fold{k:02d}.pt"
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, weights_only=False)
        held_idx = int(ck.get("held_idx", k))
        held = records[held_idx]
        stim = int(held["stim_parcel"])
        tep = held["region_tep"]                                  # [d, T]
        model = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                             init_scale=cfg.model.init_state_scale)
        model.load_state_dict(ck["state_dict"])

        keep = downstream_mask(d, stim)
        h0 = tep[:, e0:e1].mean(dim=1)                           # measured early state [d]
        late_meas = windowed_energy(tep.unsqueeze(0), l0, l1, time_dim=-1)[0]  # [d]
        early_topo = windowed_energy(tep.unsqueeze(0), e0, e1, time_dim=-1)[0]  # [d] (persistence)

        causal = causal_forecast(model, h0, args.roll_steps)
        floor = causal_forecast(untrained, h0, args.roll_steps)

        folds.append({
            "fold": k, "held_idx": held_idx, "subject": str(held["subject"]),
            "causal_r": pearsonr(causal[keep], late_meas[keep]),
            "persistence_r": pearsonr(early_topo[keep], late_meas[keep]),
            "untrained_r": pearsonr(floor[keep], late_meas[keep]),
        })
        print(f"  fold {k:02d} {folds[-1]['subject']:>12}: causal r={folds[-1]['causal_r']:+.3f}  "
              f"persistence r={folds[-1]['persistence_r']:+.3f}  "
              f"untrained r={folds[-1]['untrained_r']:+.3f}", flush=True)

    cz = np.array([f["causal_r"] for f in folds])
    pz = np.array([f["persistence_r"] for f in folds])
    print("\n===== Exp-6 early->late forecast (Arm C) =====")
    print(f"folds={len(folds)} | causal {cz.mean():+.3f}±{cz.std():.3f} | "
          f"persistence {pz.mean():+.3f}±{pz.std():.3f} | diff {np.mean(cz - pz):+.3f}")

    out = {"experiment": "6_earlylate_forecast", "metric": "late_energy_downstream_r",
           "t0": t0, "early_ms": args.early_ms, "late_ms": args.late_ms,
           "roll_steps": args.roll_steps, "n_folds": len(folds),
           "causal_r_mean": float(cz.mean()), "persistence_r_mean": float(pz.mean()),
           "folds": folds}
    out_path = Path(args.out) if args.out else Path(proc) / "exp6_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("results ->", out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--early-ms", type=int, default=30, help="early window length (samples @1kHz)")
    ap.add_argument("--late-ms", type=int, default=170, help="late window length (samples @1kHz)")
    ap.add_argument("--roll-steps", type=int, default=32, help="abstract rollout steps for the late forecast")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config()
    return run(cfg, cfg.paths.processed_dir, args)


if __name__ == "__main__":
    sys.exit(main())
