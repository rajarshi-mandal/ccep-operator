"""Exp-4 (Arm A) — counterfactual cross-site do-transfer.  THE causal flagship.

Question only a causal model can answer: train the graph on stimulation at site A (never having
seen site B stimulated), then predict the response to stimulating site B via ``do(parcel_B)``.

Why this can win where Exp-1B tied: a group-mean template is built from the *training* site and
is structurally blind to a different stimulation site. Only a causal graph re-routes a new
intervention. So the fair baseline (a wrong-site template) cannot transfer, and a group-level
model is *allowed* to beat it here.

The cross-dataset confound (M1 = ds004024, parietal = Zenodo; different scanner/montage) is
controlled by the **intervention-specificity crossover**, computed entirely *within one trained
model* so scanner/montage differences cancel:

    r_correct = corr( energy(do(stim_B)),  measured_B )      # the right intervention
    r_wrong   = corr( energy(do(stim_A)),  measured_B )      # the wrong intervention, same model

If r_correct > r_wrong (paired over held records), the model's prediction is *specific to the
intervention applied*, i.e. genuinely causal routing — not a generic response pattern or a
cross-dataset artifact. We run both directions (train M1 -> predict parietal, and the reverse).

Reuses the hardened Trainer + datasets unchanged; the model core is untouched.
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
sys.path.insert(0, str(ROOT / "experiments"))
from config import load_config  # noqa: E402
from data.dataset import (ObservationalDataset, InterventionalDataset,  # noqa: E402
                          collate_observational, collate_interventional)
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from training.losses import response_energy  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from baselines.topo_baselines import mean_topography  # noqa: E402
from eval.readouts import pearsonr, downstream_mask  # noqa: E402

CLEAN_CONDS = {"1", "2", "3"}


def load_site_records(proc, site_name):
    """All region-space records for a site; for parietal, restrict to clean cond1-3."""
    ds = InterventionalDataset(proc, site_filter={site_name})
    records = [ds[i] for i in range(len(ds))]
    if site_name == "parietal_L":
        raw = list(np.load(Path(proc) / "interventional_region.npz",
                           allow_pickle=True)["records"])
        raw_site = [r for r in raw if r["site_name"] == site_name]
        keep = [i for i, r in enumerate(raw_site) if str(r.get("cond")) in CLEAN_CONDS]
        records = [records[i] for i in keep]
        for i, gi in enumerate(keep):
            records[i]["subject"] = f"{records[i]['subject']}_c{raw_site[gi].get('cond')}"
    return records


@torch.no_grad()
def transfer_eval(model, held_records, correct_stim, wrong_stim, steps):
    """Per-held-record do-transfer scores + the within-model intervention-specificity crossover."""
    pred_correct = response_energy(
        model.predict_intervention(torch.tensor([correct_stim]), 1.0, steps), dim=1)[0]  # [d]
    pred_wrong = response_energy(
        model.predict_intervention(torch.tensor([wrong_stim]), 1.0, steps), dim=1)[0]    # [d]
    d = pred_correct.shape[0]
    keep = downstream_mask(d, correct_stim)
    rows = []
    for r in held_records:
        meas = response_energy(r["region_tep"].unsqueeze(0), dim=-1)[0]                   # [d]
        rows.append({
            "subject": r["subject"],
            "r_correct": pearsonr(pred_correct[keep], meas[keep]),
            "r_wrong": pearsonr(pred_wrong[keep], meas[keep]),
        })
    return rows


@torch.no_grad()
def wrong_site_template_eval(train_records, held_records, correct_stim):
    """Fair non-causal baseline: the TRAINING-site mean topography used to predict the held site.

    A template learned on site A is structurally unable to transfer to site B — this is exactly
    the predictor the causal model must beat to justify the do-operator.
    """
    tmpl = mean_topography(train_records)                                                 # [d]
    keep = downstream_mask(tmpl.shape[0], correct_stim)
    out = []
    for r in held_records:
        meas = response_energy(r["region_tep"].unsqueeze(0), dim=-1)[0]
        out.append(pearsonr(tmpl[keep], meas[keep]))
    return out


def train_on_site(cfg, proc, train_site, train_records, args):
    """Train a fresh CausalDAGSSM on one site's TEPs + all observational fMRI."""
    obs_ds = ObservationalDataset(proc, window=args.window)
    obs_loader = DataLoader(obs_ds, batch_size=cfg.train.batch_size, shuffle=True,
                            collate_fn=collate_observational)

    class _Recs(torch.utils.data.Dataset):
        def __len__(self): return len(train_records)
        def __getitem__(self, i): return train_records[i]
    itv_loader = DataLoader(_Recs(), batch_size=min(8, len(train_records)), shuffle=True,
                            collate_fn=collate_interventional)

    torch.manual_seed(args.seed)
    model = CausalDAGSSM(cfg.parcellation.d, variant=cfg.dag.variant,
                         input_dim=cfg.model.input_dim, init_scale=cfg.model.init_state_scale)
    ckpt = Path(proc) / "checkpoints" / f"exp4_{train_site}.train.pt"
    Trainer(model, cfg).fit(obs_loader, itv_loader, inner_steps=args.inner,
                            outer_steps=args.outer, verbose=False,
                            ckpt_path=(ckpt if args.save_ckpt else None), resume=args.resume)
    return model


def run_direction(cfg, proc, train_site, held_site, args):
    """Train on ``train_site``, predict ``held_site`` via do(); report transfer + crossover."""
    train_records = load_site_records(proc, train_site)
    held_records = load_site_records(proc, held_site)
    correct_stim = int(held_records[0]["stim_parcel"])
    wrong_stim = int(train_records[0]["stim_parcel"])
    steps = min(held_records[0]["region_tep"].shape[-1], 32)

    model_ckpt = Path(proc) / "checkpoints" / f"exp4_{train_site}.model.pt"
    model = CausalDAGSSM(cfg.parcellation.d, variant=cfg.dag.variant,
                         input_dim=cfg.model.input_dim, init_scale=cfg.model.init_state_scale)
    if args.resume and model_ckpt.exists():
        model.load_state_dict(torch.load(model_ckpt, weights_only=False)["state_dict"])
        print(f"  [{train_site}->{held_site}] loaded trained model {model_ckpt.name}")
    else:
        print(f"  [{train_site}->{held_site}] training on {len(train_records)} {train_site} TEPs ...",
              flush=True)
        model = train_on_site(cfg, proc, train_site, train_records, args)
        model_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "train_site": train_site}, model_ckpt)

    rows = transfer_eval(model, held_records, correct_stim, wrong_stim, steps)
    tmpl_r = wrong_site_template_eval(train_records, held_records, correct_stim)
    for row, t in zip(rows, tmpl_r):
        row["r_template"] = t

    r_correct = [x["r_correct"] for x in rows]
    r_wrong = [x["r_wrong"] for x in rows]
    return {
        "train_site": train_site, "held_site": held_site,
        "correct_stim": correct_stim, "wrong_stim": wrong_stim,
        "n_held": len(held_records), "eval_steps": steps,
        "r_correct_mean": float(np.mean(r_correct)),
        "r_wrong_mean": float(np.mean(r_wrong)),
        "r_template_mean": float(np.mean(tmpl_r)),
        "specificity_mean": float(np.mean(np.array(r_correct) - np.array(r_wrong))),
        "n_edges": int((model.extract_dag(cfg.dag.threshold).abs() > 0).sum()),
        "per_record": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=int, default=150)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--lambda-int", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    cfg["train"]["lambda_int"] = args.lambda_int
    proc = cfg.paths.processed_dir

    print(f"Exp-4 cross-site do-transfer | lambda_int={args.lambda_int} "
          f"inner={args.inner} outer={args.outer} resume={args.resume}")
    directions = [("M1_L", "parietal_L"), ("parietal_L", "M1_L")]
    results = []
    for train_site, held_site in directions:
        res = run_direction(cfg, proc, train_site, held_site, args)
        results.append(res)
        print(f"\n== {train_site} -> {held_site} ==")
        print(f"  do(correct={res['correct_stim']}) transfer r : {res['r_correct_mean']:+.3f}")
        print(f"  do(wrong  ={res['wrong_stim']}) transfer r   : {res['r_wrong_mean']:+.3f}")
        print(f"  wrong-site template r              : {res['r_template_mean']:+.3f}")
        print(f"  intervention specificity (corr-wrong): {res['specificity_mean']:+.3f}", flush=True)

    out = {"experiment": "4_cross_site_do_transfer", "directions": results}
    out_path = Path(args.out) if args.out else Path(proc) / "exp4_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nresults ->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
