"""Exp-3 — run the non-causal baselines on the SAME Exp-1B LOSO M1 split.

For each held-out M1 subject we score (downstream Pearson r, stim parcel excluded):
  * mean_topography  — mean of the OTHER subjects' measured topographies (strong ceiling)
  * fc_propagation   — |resting-state FC row| of the stimulated parcel
  * distance_decay   — exp(-dist) geometry null from the stimulated centroid
  * untrained_model  — do(stim) energy of a random-W CausalDAGSSM (the trained-model floor)

Output:
  * reports/exp1b_baselines.csv  — per-fold downstream r for every baseline
  * reports/exp1b_baselines.md   — mean ± SD table; if Exp-1B trained folds exist, a paired
    trained-vs-best-baseline comparison (sign-flip permutation + Cohen's d).

This is the "is the causal model actually adding anything over cheap correlational predictors"
gate for H2.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from config import load_config  # noqa: E402
from data.dataset import InterventionalDataset  # noqa: E402
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from baselines.topo_baselines import (mean_topography, measured_topography,  # noqa: E402
                                      functional_connectivity, fc_propagation,
                                      distance_decay, untrained_model_topography)
from eval.stats import (bootstrap_ci, exact_sign_flip_test,  # noqa: E402
                        cohens_d_paired)
from exp1_held_out_tms import pearsonr  # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

BASELINES = ["mean_topography", "fc_propagation", "distance_decay", "untrained_model"]


def downstream_r(pred_topo: torch.Tensor, meas_topo: torch.Tensor, stim: int) -> float:
    keep = torch.ones(meas_topo.shape[0], dtype=torch.bool)
    keep[stim] = False
    return pearsonr(pred_topo[keep], meas_topo[keep])


def main() -> int:
    cfg = load_config()
    proc = cfg.paths.processed_dir
    d = cfg.parcellation.d

    m1 = InterventionalDataset(proc, site_filter={"M1_L"})
    records = [m1[i] for i in range(len(m1))]
    n = len(records)
    if n == 0:
        print("No M1_L records found; nothing to baseline.")
        return 0

    obs = np.load(Path(proc) / "observational_fmri.npy")          # [n_sub, T, d]
    fc = functional_connectivity(obs)
    cents_path = Path(proc) / "parcel_centroids_mni.npy"
    centroids = np.load(cents_path) if cents_path.exists() else None
    steps = min(records[0]["region_tep"].shape[-1], 32)

    # one untrained model, fixed seed → deterministic floor
    torch.manual_seed(0)
    untrained = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                             init_scale=cfg.model.init_state_scale)

    rows = []
    for k in range(n):
        held = records[k]
        stim = int(held["stim_parcel"])
        meas = measured_topography(held)
        train_recs = [r for j, r in enumerate(records) if j != k]
        preds = {
            "mean_topography": mean_topography(train_recs),
            "fc_propagation": fc_propagation(fc, stim),
            "untrained_model": untrained_model_topography(untrained, stim, steps),
        }
        if centroids is not None:
            preds["distance_decay"] = distance_decay(centroids, stim)
        row = {"fold": k, "subject": held["subject"], "stim_parcel": stim}
        for name, p in preds.items():
            row[name] = downstream_r(p, meas, stim)
        rows.append(row)
        print(f"  fold {k:02d} {held['subject']:>12}: "
              + "  ".join(f"{nm}={row.get(nm, float('nan')):+.3f}" for nm in BASELINES))

    # per-fold CSV
    cols = ["fold", "subject", "stim_parcel"] + [b for b in BASELINES if b in rows[0]]
    with open(REPORTS / "exp1b_baselines.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # summary table
    md = ["# Exp-1B baselines (LOSO over ds004024 M1)\n",
          f"Folds: **{n}**  | downstream Pearson r (stim parcel excluded)\n",
          "| baseline | mean | sd | 95% CI |", "|---|---|---|---|"]
    means = {}
    for b in BASELINES:
        vals = np.array([r[b] for r in rows if b in r], dtype=float)
        if vals.size == 0:
            continue
        mean, lo, hi = bootstrap_ci(vals)
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        means[b] = vals
        md.append(f"| {b} | {mean:+.3f} | {sd:.3f} | [{lo:+.3f}, {hi:+.3f}] |")

    # paired comparison vs trained Exp-1B folds, if available
    trained = load_trained_downstream(proc)
    if trained:
        md.append("\n## Trained causal model vs baselines (paired)\n")
        tr = np.array([trained[k] for k in sorted(trained)], dtype=float)
        md.append(f"Trained downstream r (n={tr.size}): "
                  f"mean {tr.mean():+.3f} ± {tr.std(ddof=1) if tr.size > 1 else 0:.3f}\n")
        md.append("| baseline | trained−baseline | sign-flip p | Cohen's d |")
        md.append("|---|---|---|---|")
        for b, vals in means.items():
            m = min(tr.size, vals.size)
            if m < 2:
                continue
            a, bb = tr[:m], vals[:m]
            p = exact_sign_flip_test(a, bb) if m <= 20 else float("nan")
            dd = cohens_d_paired(a, bb)
            md.append(f"| {b} | {(a - bb).mean():+.3f} | {p:.4f} | {dd:+.3f} |")

    (REPORTS / "exp1b_baselines.md").write_text("\n".join(md) + "\n")
    print("\nWrote reports/exp1b_baselines.csv, reports/exp1b_baselines.md")
    return 0


def load_trained_downstream(proc) -> dict[int, float]:
    """Collect trained downstream r per fold from incremental fold JSONs or results.json."""
    out: dict[int, float] = {}
    agg = Path(proc) / "exp1b_results.json"
    if agg.exists():
        try:
            for r in json.load(open(agg)).get("folds", []):
                if "fold" in r and "trained_downstream_r" in r:
                    out[int(r["fold"])] = float(r["trained_downstream_r"])
        except (json.JSONDecodeError, OSError):
            pass
    for fp in sorted(glob.glob(str(Path(proc) / "exp1b_folds" / "fold*.json"))):
        try:
            r = json.load(open(fp))
            if "trained_downstream_r" in r:
                k = int(r.get("fold", Path(fp).stem.replace("fold", "")))
                out[k] = float(r["trained_downstream_r"])
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return out


if __name__ == "__main__":
    sys.exit(main())
