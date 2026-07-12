"""Experiment 1 — held-out-site TMS prediction (the go/no-go gate; spec 6.1, 9.6).

The central claim: a causal graph learned from observational fMRI + interventional TMS at
*some* sites can predict the downstream response to stimulating a *held-out* site it never
saw stimulated. This is the test a non-causal correlational model cannot pass.

Protocol (leave-one-site-out over the available stimulation sites):
  * Hold out one site's TEPs entirely (default: parietal).
  * Train CausalDAGSSM on ALL observational fMRI + the remaining sites' interventional TEPs.
  * Evaluate on the held-out site: run ``do(stim=held_out_parcel)``, reduce to a per-region
    response topography, and correlate (Pearson r) with each measured held-out TEP topography.

We report the honest causal number too: ``downstream r`` excludes the stimulated parcel
itself (which is clamped in the prediction and dominates the measured TEP), so the score
reflects predicting *where the signal propagates*, not the trivial "the stimulated site
responds". Two baselines contextualise it: an untrained model and the mean training-site
topography (a non-causal "average response" predictor).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import load_config  # noqa: E402
from data.dataset import (ObservationalDataset, InterventionalDataset,  # noqa: E402
                          collate_observational, collate_interventional)
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from training.losses import response_energy  # noqa: E402
from training.trainer import Trainer  # noqa: E402


def pearsonr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


@torch.no_grad()
def evaluate_site(model, records, steps, exclude_stim=True):
    """Mean Pearson r between predicted and measured response topographies on a site.

    ``records``: list of held-out InterventionalDataset items. Returns dict with full and
    downstream-only (stim parcel removed) mean/median r, plus per-record r list.
    """
    full, down = [], []
    for r in records:
        stim = torch.tensor([r["stim_parcel"]])
        tep = r["region_tep"].unsqueeze(0)                       # [1, d, T]
        pred = model.predict_intervention(stim, amplitude=1.0, steps=steps)  # [1, steps, d]
        p = response_energy(pred, dim=1)[0]                      # [d]
        m = response_energy(tep, dim=-1)[0]                      # [d]
        full.append(pearsonr(p, m))
        if exclude_stim:
            keep = torch.ones(p.shape[0], dtype=torch.bool)
            keep[r["stim_parcel"]] = False
            down.append(pearsonr(p[keep], m[keep]))
    out = {"full_mean_r": float(np.mean(full)), "full_median_r": float(np.median(full)),
           "per_record_full_r": full}
    if exclude_stim:
        out.update({"downstream_mean_r": float(np.mean(down)),
                    "downstream_median_r": float(np.median(down)),
                    "per_record_downstream_r": down})
    return out


@torch.no_grad()
def site_mean_topographies(model, records, steps):
    """Mean measured and mean predicted response topography over a set of records."""
    meas = torch.stack([response_energy(r["region_tep"].unsqueeze(0), dim=-1)[0]
                        for r in records]).mean(0)               # [d]
    preds = []
    for r in records:
        pred = model.predict_intervention(torch.tensor([r["stim_parcel"]]), 1.0, steps)
        preds.append(response_energy(pred, dim=1)[0])
    return meas, torch.stack(preds).mean(0)


@torch.no_grad()
def evaluate_contrast(model, train_records, held_records, steps, stim_parcels):
    """Headline causal metric: correlate the predicted vs measured SITE CONTRAST.

    Δ_meas = topo(held) − topo(train);  Δ_pred = pred(held) − pred(train), over regions.
    This cancels the ~87% site-invariant component (see memory: Exp-1 topography confound),
    so only causal structure that distinguishes stimulation sites can score. The two
    stimulated parcels are excluded so the score reflects downstream propagation only.
    """
    meas_tr, pred_tr = site_mean_topographies(model, train_records, steps)
    meas_he, pred_he = site_mean_topographies(model, held_records, steps)
    d_meas = meas_he - meas_tr
    d_pred = pred_he - pred_tr
    keep = torch.ones(d_meas.shape[0], dtype=torch.bool)
    for p in stim_parcels:
        keep[p] = False
    return {"contrast_r": pearsonr(d_pred[keep], d_meas[keep]),
            "contrast_r_with_stim": pearsonr(d_pred, d_meas)}


@torch.no_grad()
def baseline_mean_topography(train_records, held_records, exclude_stim=True):
    """Non-causal baseline: predict every held-out TEP with the mean TRAINING topography."""
    train_topo = torch.stack([response_energy(r["region_tep"].unsqueeze(0), dim=-1)[0]
                              for r in train_records]).mean(0)   # [d]
    full, down = [], []
    for r in held_records:
        m = response_energy(r["region_tep"].unsqueeze(0), dim=-1)[0]
        full.append(pearsonr(train_topo, m))
        if exclude_stim:
            keep = torch.ones(m.shape[0], dtype=torch.bool)
            keep[r["stim_parcel"]] = False
            down.append(pearsonr(train_topo[keep], m[keep]))
    return {"full_mean_r": float(np.mean(full)),
            "downstream_mean_r": float(np.mean(down)) if exclude_stim else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=None,
                    help="site_name to hold out (default: auto = the parietal site)")
    ap.add_argument("--inner", type=int, default=200, help="inner steps per outer iter")
    ap.add_argument("--outer", type=int, default=8, help="augmented-Lagrangian outer iters")
    ap.add_argument("--window", type=int, default=60, help="fMRI window length")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda-int", type=float, default=None,
                    help="override cfg.train.lambda_int (weight on interventional loss)")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = load_config()
    if args.lambda_int is not None:
        # NB: cfg.train returns a *copy*; mutate the underlying nested dict in place.
        cfg["train"]["lambda_int"] = args.lambda_int
    proc = cfg.paths.processed_dir
    d = cfg.parcellation.d

    # --- discover sites; pick the held-out one ---
    all_itv = InterventionalDataset(proc)
    sites = Counter(r["site_name"] for r in all_itv.records)
    print("available interventional sites:", dict(sites))
    heldout = args.heldout
    if heldout is None:
        parietal = [s for s in sites if "parietal" in s.lower()]
        heldout = parietal[0] if parietal else sorted(sites, key=lambda s: sites[s])[0]
    train_sites = {s for s in sites if s != heldout}
    print(f"HELD-OUT site: {heldout}  |  TRAIN sites: {sorted(train_sites)}")
    if not train_sites:
        raise SystemExit("Need >=2 sites for leave-one-site-out; only one site has data.")

    # --- data loaders ---
    obs_ds = ObservationalDataset(proc, window=args.window)
    if args.max_subjects is not None:
        # subsample subjects by trimming windows (cheap dev path)
        obs_ds.windows = [w for w in obs_ds.windows if w[0] < args.max_subjects]
    train_itv = InterventionalDataset(proc, site_filter=train_sites)
    held_itv = InterventionalDataset(proc, site_filter={heldout})
    held_records = [held_itv[i] for i in range(len(held_itv))]
    train_records = [train_itv[i] for i in range(len(train_itv))]
    print(f"obs windows={len(obs_ds)}  train TEPs={len(train_itv)}  held TEPs={len(held_itv)}")

    obs_loader = DataLoader(obs_ds, batch_size=cfg.train.batch_size, shuffle=True,
                            collate_fn=collate_observational)
    itv_loader = DataLoader(train_itv, batch_size=min(8, len(train_itv)), shuffle=True,
                            collate_fn=collate_interventional)

    eval_steps = min(held_records[0]["region_tep"].shape[-1], 32)

    # stim parcels for each side (excluded from the contrast — they're clamped/dominant)
    held_stim = sorted({r["stim_parcel"] for r in held_records})
    train_stim = sorted({r["stim_parcel"] for r in train_records})
    contrast_excl = held_stim + train_stim

    # --- untrained baseline (random graph) ---
    model = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                         init_scale=cfg.model.init_state_scale)
    pre = evaluate_site(model, held_records, eval_steps)
    pre_contrast = evaluate_contrast(model, train_records, held_records, eval_steps,
                                     contrast_excl)
    print(f"\n[untrained] held-out full r={pre['full_mean_r']:.3f} "
          f"downstream r={pre['downstream_mean_r']:.3f} "
          f"contrast r={pre_contrast['contrast_r']:.3f}")

    # --- train ---
    trainer = Trainer(model, cfg)
    trainer.fit(obs_loader, itv_loader, inner_steps=args.inner, outer_steps=args.outer)

    # --- evaluate held-out site ---
    post = evaluate_site(model, held_records, eval_steps)
    post_contrast = evaluate_contrast(model, train_records, held_records, eval_steps,
                                      contrast_excl)
    base = baseline_mean_topography(train_records, held_records)

    success_r = cfg.exp1.success_r
    # Headline causal metric is the site CONTRAST r (raw topography is confounded; the
    # mean-topo baseline below shows why). Pass/fail is judged on the contrast.
    passed = post_contrast["contrast_r"] > success_r

    print("\n================ Experiment 1 results ================")
    print(f"held-out site          : {heldout}")
    print(f"--- raw topography (confounded — for transparency only) ---")
    print(f"mean-topo baseline    r : {base['downstream_mean_r']:.3f}  "
          f"(non-causal; high => confounded)")
    print(f"trained downstream    r : {post['downstream_mean_r']:.3f}")
    print(f"--- site contrast (HEADLINE causal metric) ---")
    print(f"untrained contrast    r : {pre_contrast['contrast_r']:.3f}")
    print(f"TRAINED  contrast     r : {post_contrast['contrast_r']:.3f}")
    print(f"success bar (r>{success_r})     : {'PASS ✓' if passed else 'below bar'}")
    print("======================================================")

    results = {
        "heldout_site": heldout, "train_sites": sorted(train_sites),
        "n_held": len(held_records), "n_train_teps": len(train_records),
        "eval_steps": eval_steps, "success_r": success_r, "passed": bool(passed),
        "headline_contrast_r": post_contrast["contrast_r"],
        "untrained": {k: v for k, v in pre.items() if not k.startswith("per_")},
        "untrained_contrast": pre_contrast,
        "trained": {k: v for k, v in post.items() if not k.startswith("per_")},
        "trained_contrast": post_contrast,
        "baseline_mean_topography": base,
        "final_h": trainer.state.h, "final_rho": trainer.state.rho,
        "n_edges": int((model.extract_dag(cfg.dag.threshold).abs() > 0).sum()),
    }
    out_path = Path(args.out) if args.out else Path(proc) / "exp1_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("results ->", out_path)


if __name__ == "__main__":
    main()
