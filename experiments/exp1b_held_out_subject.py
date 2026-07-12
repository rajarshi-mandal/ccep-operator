"""Experiment 1B — leave-one-subject-out held-out TMS prediction (re-scoped go/no-go).

Re-scope of Exp-1 after the leave-one-SITE-out version proved infeasible (only 2 genuine
sites, from 2 different datasets). Here we stay within a SINGLE dataset+site — ds004024 M1
(13 subjects) — and ask the cleaner question:

    Trained on group fMRI + the M1 TEPs of N-1 subjects, can the causal DAG-SSM's do(M1)
    operation predict the TEP of the held-out subject?

Honesty notes baked into the reporting:
  * The model has NO subject-specific input (HCP fMRI subjects != ds004024 TMS subjects), so
    it emits ONE group-level do(M1) prediction per fold. LOSO therefore tests group->subject
    generalization, and cannot beat a group-mean baseline by construction. We report the
    mean-topo baseline alongside so this ceiling is explicit.
  * Raw response topography is confounded by a site-invariant spatial pattern (see memory:
    Exp-1 topography confound), so the headline number is the DOWNSTREAM r — correlation of
    predicted vs measured response topography with the stimulated parcel (40) excluded. That
    isolates "does the learned graph route M1 stimulation to the right downstream regions",
    which is the causal claim. We also report the untrained-graph downstream r as the floor:
    the model must move the held-out prediction up from there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config  # noqa: E402
from data.dataset import (ObservationalDataset, InterventionalDataset,  # noqa: E402
                          collate_observational, collate_interventional)
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from training.losses import response_energy  # noqa: E402
from training.trainer import Trainer  # noqa: E402
from exp1_held_out_tms import pearsonr, evaluate_site  # noqa: E402


@torch.no_grad()
def baseline_mean_topo_r(train_recs, held_rec, stim_parcel):
    """Group-mean-topography baseline: predict held subject with mean of training subjects."""
    train_topo = torch.stack([response_energy(r["region_tep"].unsqueeze(0), dim=-1)[0]
                              for r in train_recs]).mean(0)
    m = response_energy(held_rec["region_tep"].unsqueeze(0), dim=-1)[0]
    keep = torch.ones(m.shape[0], dtype=torch.bool)
    keep[stim_parcel] = False
    return {"full": pearsonr(train_topo, m),
            "downstream": pearsonr(train_topo[keep], m[keep])}


def run_fold(cfg, obs_ds, m1_records, held_idx, args, eval_steps):
    """Train on all-but-one M1 subject (+fMRI); evaluate the held-out subject."""
    held = m1_records[held_idx]
    train_recs = [r for j, r in enumerate(m1_records) if j != held_idx]

    obs_loader = DataLoader(obs_ds, batch_size=cfg.train.batch_size, shuffle=True,
                            collate_fn=collate_observational)

    class _Recs(torch.utils.data.Dataset):
        def __len__(self): return len(train_recs)
        def __getitem__(self, i): return train_recs[i]
    itv_loader = DataLoader(_Recs(), batch_size=min(8, len(train_recs)), shuffle=True,
                            collate_fn=collate_interventional)

    torch.manual_seed(args.seed + held_idx)
    model = CausalDAGSSM(cfg.parcellation.d, variant=cfg.dag.variant,
                         input_dim=cfg.model.input_dim,
                         init_scale=cfg.model.init_state_scale)
    # The untrained floor is computed on the deterministically re-seeded fresh model, so it is
    # identical whether the fold runs start-to-finish or resumes from a mid-fold checkpoint.
    pre = evaluate_site(model, [held], eval_steps)            # untrained floor

    # Intra-fold "freeze": checkpoint after every outer iteration so a power-off / crash mid-fold
    # only loses the in-progress outer step, not the whole fold. --resume picks it back up.
    ckpt_dir = Path(cfg.paths.processed_dir) / "checkpoints"
    save_ckpt = getattr(args, "save_ckpt", False)
    train_ckpt = ckpt_dir / f"exp1b_fold{held_idx:02d}.train.pt" if save_ckpt else None
    if train_ckpt is not None and getattr(args, "resume", False) and train_ckpt.exists():
        print(f"  fold {held_idx:02d}: found mid-fold freeze {train_ckpt.name} -> resuming",
              flush=True)
    state = Trainer(model, cfg).fit(
        obs_loader, itv_loader, inner_steps=args.inner, outer_steps=args.outer, verbose=False,
        ckpt_path=train_ckpt, resume=getattr(args, "resume", False))
    post = evaluate_site(model, [held], eval_steps)
    base = baseline_mean_topo_r(train_recs, held, int(held["stim_parcel"]))

    if save_ckpt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "held_idx": held_idx,
                    "subject": str(held["subject"]), "stim_parcel": int(held["stim_parcel"]),
                    "final_h": state.h, "final_rho": state.rho,
                    "lambda_int": args.lambda_int, "inner": args.inner, "outer": args.outer},
                   ckpt_dir / f"exp1b_fold{held_idx:02d}.pt")
        # fold finished cleanly — drop the mid-fold freeze so the next --resume re-trains nothing
        if train_ckpt is not None and train_ckpt.exists():
            train_ckpt.unlink()
    return {
        "subject": str(held["subject"]),
        "untrained_downstream_r": pre["downstream_mean_r"],
        "trained_full_r": post["full_mean_r"],
        "trained_downstream_r": post["downstream_mean_r"],
        "baseline_full_r": base["full"],
        "baseline_downstream_r": base["downstream"],
        "final_h": state.h,
        "final_rho": state.rho,
        "n_edges": int((model.extract_dag(cfg.dag.threshold).abs() > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=int, default=150)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--lambda-int", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--folds", type=int, default=None, help="limit #folds (default all)")
    ap.add_argument("--start-fold", type=int, default=0)
    ap.add_argument("--end-fold", type=int, default=None, help="exclusive; default n_subjects")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ckpt", action="store_true", help="save per-fold model checkpoint")
    ap.add_argument("--resume", action="store_true",
                    help="skip folds whose per-fold JSON already exists")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    cfg["train"]["lambda_int"] = args.lambda_int
    proc = cfg.paths.processed_dir

    m1 = InterventionalDataset(proc, site_filter={"M1_L"})
    m1_records = [m1[i] for i in range(len(m1))]
    n_sub = len(m1_records)
    start = args.start_fold
    end = args.end_fold if args.end_fold is not None else (args.folds or n_sub)
    end = min(end, n_sub)
    obs_ds = ObservationalDataset(proc, window=args.window)
    eval_steps = min(m1_records[0]["region_tep"].shape[-1], 32)
    fold_dir = Path(proc) / "exp1b_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exp-1B leave-one-subject-out | M1 subjects={n_sub} folds=[{start},{end}) "
          f"obs_windows={len(obs_ds)} lambda_int={args.lambda_int} "
          f"resume={args.resume} save_ckpt={args.save_ckpt}")

    fold_results = []
    for k in range(start, end):
        fpath = fold_dir / f"fold{k:02d}.json"
        if args.resume and fpath.exists():
            res = json.load(open(fpath))
            print(f"  fold {k:02d} {res['subject']:>12}: [resumed] "
                  f"trained downstream r={res['trained_downstream_r']:+.3f}")
            fold_results.append(res)
            continue
        res = run_fold(cfg, obs_ds, m1_records, k, args, eval_steps)
        res["fold"] = k
        with open(fpath, "w") as fh:          # incremental save: a crash never loses a fold
            json.dump(res, fh, indent=2)
        fold_results.append(res)
        print(f"  fold {k:02d} {res['subject']:>12}: "
              f"trained downstream r={res['trained_downstream_r']:+.3f} "
              f"(untrained {res['untrained_downstream_r']:+.3f}, "
              f"baseline {res['baseline_downstream_r']:+.3f}) | "
              f"trained full r={res['trained_full_r']:+.3f} | h={res['final_h']:.1e}",
              flush=True)
    n_folds = len(fold_results)

    def agg(key):
        v = [r[key] for r in fold_results]
        return float(np.mean(v)), float(np.std(v))

    md, sd = agg("trained_downstream_r")
    mu, su = agg("untrained_downstream_r")
    mb, sb = agg("baseline_downstream_r")
    mf, sf = agg("trained_full_r")
    success_r = cfg.exp1.success_r
    passed = md > success_r

    print("\n============= Experiment 1B results (LOSO) =============")
    print(f"folds                  : {n_folds}")
    print(f"untrained downstream r : {mu:+.3f} ± {su:.3f}   (floor)")
    print(f"mean-topo baseline   r : {mb:+.3f} ± {sb:.3f}   (group-mean ceiling, non-causal)")
    print(f"TRAINED downstream   r : {md:+.3f} ± {sd:.3f}   (HEADLINE causal metric)")
    print(f"TRAINED full         r : {mf:+.3f} ± {sf:.3f}   (confounded; context only)")
    print(f"success bar (r>{success_r})    : {'PASS' if passed else 'below bar'}")
    print("=======================================================")

    results = {"experiment": "1B_leave_one_subject_out", "site": "M1_L",
               "n_folds": n_folds, "success_r": success_r, "passed": bool(passed),
               "lambda_int": args.lambda_int, "inner": args.inner, "outer": args.outer,
               "headline_trained_downstream_r_mean": md,
               "headline_trained_downstream_r_std": sd,
               "untrained_downstream_r_mean": mu, "baseline_downstream_r_mean": mb,
               "trained_full_r_mean": mf, "folds": fold_results}
    out_path = Path(args.out) if args.out else Path(proc) / "exp1b_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("results ->", out_path)


if __name__ == "__main__":
    main()
