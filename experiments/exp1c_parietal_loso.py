"""Exp-1C (EXPLORATORY) — leave-one-subject-out parietal TMS prediction, clean Zenodo conditions.

Mirror of Exp-1B but on the Zenodo parietal arm, restricted to conditions 1-3 (genuine P3
parietal stimulation; cond4-6 are excluded as likely sham — see reports/zenodo_label_audit.md).
Each parietal "subject" here is a (subject, condition) record at stim parcel 13.

EXPLORATORY status: the parietal data is a different dataset/montage than the fMRI and has the
known artifact caveat, so this is a secondary check, not a headline. Reuses Exp-1B's hardened
``run_fold`` so the training/eval path is identical. Per-fold JSONs -> data/processed/exp1c_folds.
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
from config import load_config  # noqa: E402
from data.dataset import ObservationalDataset, InterventionalDataset  # noqa: E402
from exp1b_held_out_subject import run_fold  # noqa: E402

CLEAN_CONDS = {"1", "2", "3"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", type=int, default=150)
    ap.add_argument("--outer", type=int, default=6)
    ap.add_argument("--lambda-int", type=float, default=10.0)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--start-fold", type=int, default=0)
    ap.add_argument("--end-fold", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    cfg["train"]["lambda_int"] = args.lambda_int
    proc = cfg.paths.processed_dir

    par = InterventionalDataset(proc, site_filter={"parietal_L"})
    records = [par[i] for i in range(len(par))]
    # restrict to clean conditions; cond lives on the raw record, surfaced via the dataset item?
    # InterventionalDataset item omits 'cond', so re-read raw records to filter.
    raw = list(np.load(Path(proc) / "interventional_region.npz", allow_pickle=True)["records"])
    raw_par = [r for r in raw if r["site_name"] == "parietal_L"]
    keep_idx = [i for i, r in enumerate(raw_par) if str(r.get("cond")) in CLEAN_CONDS]
    records = [records[i] for i in keep_idx]
    # tag a unique label per record so folds are identifiable
    for i, (gi, rec) in enumerate(zip(keep_idx, records)):
        rec["subject"] = f"{rec['subject']}_c{raw_par[gi].get('cond')}"
    n = len(records)
    if n == 0:
        print("No clean parietal records (cond1-3) found.")
        return 0

    obs_ds = ObservationalDataset(proc, window=args.window)
    eval_steps = min(records[0]["region_tep"].shape[-1], 32)
    start = args.start_fold
    end = args.end_fold if args.end_fold is not None else n
    end = min(end, n)
    fold_dir = Path(proc) / "exp1c_folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exp-1C EXPLORATORY parietal LOSO | clean records={n} folds=[{start},{end}) "
          f"obs_windows={len(obs_ds)} lambda_int={args.lambda_int}")

    results = []
    for k in range(start, end):
        fpath = fold_dir / f"fold{k:02d}.json"
        if args.resume and fpath.exists():
            res = json.load(open(fpath)); results.append(res)
            print(f"  fold {k:02d} [resumed] trained downstream r={res['trained_downstream_r']:+.3f}")
            continue
        res = run_fold(cfg, obs_ds, records, k, args, eval_steps)
        res["fold"] = k
        with open(fpath, "w") as fh:
            json.dump(res, fh, indent=2)
        results.append(res)
        print(f"  fold {k:02d} {res['subject']:>16}: trained downstream r="
              f"{res['trained_downstream_r']:+.3f} (untrained {res['untrained_downstream_r']:+.3f}, "
              f"baseline {res['baseline_downstream_r']:+.3f})", flush=True)

    def agg(key):
        v = [r[key] for r in results]
        return float(np.mean(v)), float(np.std(v))

    md, sd = agg("trained_downstream_r")
    mu, _ = agg("untrained_downstream_r")
    mb, _ = agg("baseline_downstream_r")
    print("\n===== Exp-1C (EXPLORATORY parietal) =====")
    print(f"folds={len(results)} | trained {md:+.3f}±{sd:.3f} | untrained {mu:+.3f} | baseline {mb:+.3f}")

    out = {"experiment": "1C_parietal_loso_exploratory", "site": "parietal_L",
           "clean_conditions": sorted(CLEAN_CONDS), "n_folds": len(results),
           "trained_downstream_r_mean": md, "untrained_downstream_r_mean": mu,
           "baseline_downstream_r_mean": mb, "folds": results}
    out_path = Path(args.out) if args.out else Path(proc) / "exp1c_results.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("results ->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
