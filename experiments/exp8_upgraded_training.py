"""Exp-8 — train the FULL upgraded objective end-to-end and test whether better model quality
flips the H2 ceiling. Diagnostic (small fold count) — the Bayes argument predicts it cannot.

Upgrades stacked vs the stock Exp-1B objective (all from the audit):
  * GAP 1: learned observation matrix ``C`` (``learn_C=True``) + anatomical locality penalty.
  * GAP 4: temporal ``waveform_loss`` (calibrated time axis) instead of energy-only ``L_int``.
  * GAP 5: ``inject=True`` so the dead input map ``B`` becomes a trainable stimulus-spread.
  * GAP 2: denser graph allowed (lower threshold) + higher ``lambda_int``.

Metric: held-out downstream energy Pearson r vs the leave-one-out mean-topography baseline
(the exact H2 test). If the upgraded model still only ties the mean, that confirms model quality
is orthogonal to the ceiling: the limit is the missing subject input, not the objective.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402
from data.dataset import (ObservationalDataset, InterventionalDataset,  # noqa: E402
                          collate_observational)
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from training.losses import (obs_nll, waveform_loss, acyclicity_penalty,  # noqa: E402
                             response_energy, obs_matrix_locality_penalty)
from baselines.topo_baselines import mean_topography  # noqa: E402
from eval.readouts import pearsonr, downstream_mask  # noqa: E402


@torch.no_grad()
def held_out_r(model, held, steps, inject):
    stim = int(held["stim_parcel"])
    pred = model.predict_intervention(torch.tensor([stim]), 1.0, steps, inject=inject)
    pred_topo = response_energy(pred, dim=1)[0]
    meas_topo = response_energy(held["region_tep"].unsqueeze(0), dim=-1)[0]
    keep = downstream_mask(pred_topo.shape[0], stim)
    return pearsonr(pred_topo[keep], meas_topo[keep])


def train_fold(cfg, proc, records, held_idx, centroids, args):
    d = cfg.parcellation.d
    train_records = [r for i, r in enumerate(records) if i != held_idx]
    obs_ds = ObservationalDataset(proc, window=args.window)
    obs_loader = DataLoader(obs_ds, batch_size=cfg.train.batch_size, shuffle=True,
                            collate_fn=collate_observational)
    obs_iter = iter(obs_loader)

    torch.manual_seed(args.seed)
    model = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                         init_scale=cfg.model.init_state_scale, learn_C=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    stim = torch.tensor([int(r["stim_parcel"]) for r in train_records])
    tep = torch.stack([r["region_tep"] for r in train_records])
    rho, alpha = cfg.dag.rho_init, cfg.dag.alpha_init
    h_prev = float("inf")

    l_int0 = None
    for outer in range(args.outer):
        for step in range(args.inner):
            try:
                y = next(obs_iter)
            except StopIteration:
                obs_iter = iter(obs_loader); y = next(obs_iter)
            opt.zero_grad()
            l_obs = obs_nll(model, y)
            l_int = waveform_loss(model, stim, tep, steps=args.steps, inject=True,
                                  energy_weight=args.energy_weight)
            if l_int0 is None:
                l_int0 = float(l_int.detach())
            h = model.acyclicity()
            pen = acyclicity_penalty(h, alpha, rho)
            loc = obs_matrix_locality_penalty(model.ssm.C, centroids)
            loss = l_obs + args.lambda_int * l_int + pen + args.lambda_local * loc
            loss.backward()
            opt.step()
        h_new = float(model.acyclicity().detach())
        if h_new > cfg.dag.progress_rate * h_prev:
            rho = min(rho * cfg.dag.rho_mult, cfg.dag.rho_max)
        alpha = alpha + rho * h_new
        h_prev = h_new
    return model, l_int0, float(l_int.detach()), h_new


def run(cfg, proc, args):
    centroids = torch.from_numpy(
        np.load(Path(proc) / "parcel_centroids_mni.npy")).float()
    m1 = InterventionalDataset(proc, site_filter={"M1_L"})
    records = [m1[i] for i in range(len(m1))]
    n = len(records)
    folds_to_run = list(range(min(args.folds, n)))

    rows = []
    for k in folds_to_run:
        model, li0, li1, h = train_fold(cfg, proc, records, k, centroids, args)
        held = records[k]
        causal_r = held_out_r(model, held, args.steps, inject=True)
        tmpl = mean_topography([records[j] for j in range(n) if j != k])
        keep = downstream_mask(tmpl.shape[0], int(held["stim_parcel"]))
        meas = response_energy(held["region_tep"].unsqueeze(0), dim=-1)[0]
        tmpl_r = pearsonr(tmpl[keep], meas[keep])
        rows.append({"fold": k, "subject": str(held["subject"]),
                     "L_int_start": li0, "L_int_end": li1, "h": h,
                     "causal_r": causal_r, "template_r": tmpl_r,
                     "diff": causal_r - tmpl_r})
        print(f"  fold {k:02d} {rows[-1]['subject']:>12}: L_int {li0:.3f}->{li1:.3f} "
              f"h={h:.1e} | causal r={causal_r:+.3f} template r={tmpl_r:+.3f} "
              f"diff={causal_r - tmpl_r:+.3f}", flush=True)

    c = np.array([r["causal_r"] for r in rows])
    t = np.array([r["template_r"] for r in rows])
    print("\n===== Exp-8 upgraded-objective H2 test =====")
    print(f"folds={len(rows)} | upgraded causal {c.mean():+.3f} | mean-topo {t.mean():+.3f} | "
          f"diff {np.mean(c - t):+.3f}")
    print("L_int decreased on every fold:", all(r["L_int_end"] < r["L_int_start"] for r in rows),
          "(upgraded objective trains)")

    out = {"experiment": "8_upgraded_training", "folds": rows,
           "causal_r_mean": float(c.mean()), "template_r_mean": float(t.mean()),
           "diff_mean": float(np.mean(c - t))}
    out_path = Path(args.out) if args.out else Path(proc) / "exp8_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print("results ->", out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--inner", type=int, default=120)
    ap.add_argument("--outer", type=int, default=4)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--lambda-int", type=float, default=20.0)
    ap.add_argument("--lambda-local", type=float, default=1.0)
    ap.add_argument("--energy-weight", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config()
    cfg["train"]["lambda_int"] = args.lambda_int
    return run(cfg, cfg.paths.processed_dir, args)


if __name__ == "__main__":
    sys.exit(main())
