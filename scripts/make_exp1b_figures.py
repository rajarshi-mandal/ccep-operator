"""Figures for Exp-1B (LOSO held-out-subject TMS prediction).

Reads the incremental per-fold JSONs (data/processed/exp1b_folds/fold*.json) or the aggregate
results.json, and renders to reports/figures/:
  * exp1b_per_fold.png  — grouped bars: trained / untrained / mean-topo baseline downstream r,
    one group per held-out subject, with the success bar (r=0.55) drawn.
  * exp1b_summary.png   — mean ± SD across folds for each method (the headline figure).

Uses a non-interactive Matplotlib backend; safe to run head-less and mid-run.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
SUCCESS_R = 0.55


def load_folds() -> list[dict]:
    agg = PROC / "exp1b_results.json"
    by_fold: dict[int, dict] = {}
    if agg.exists():
        try:
            for r in json.load(open(agg)).get("folds", []):
                if "fold" in r:
                    by_fold[int(r["fold"])] = r
        except (json.JSONDecodeError, OSError):
            pass
    for fp in sorted(glob.glob(str(PROC / "exp1b_folds" / "fold*.json"))):
        try:
            r = json.load(open(fp))
            k = int(r.get("fold", Path(fp).stem.replace("fold", "")))
            by_fold[k] = r
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return [by_fold[k] for k in sorted(by_fold)]


def per_fold_figure(folds: list[dict]) -> None:
    subs = [f.get("subject", str(f.get("fold", i))) for i, f in enumerate(folds)]
    tr = [f["trained_downstream_r"] for f in folds]
    un = [f["untrained_downstream_r"] for f in folds]
    ba = [f["baseline_downstream_r"] for f in folds]
    x = np.arange(len(folds))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(folds) * 0.9), 4.5))
    ax.bar(x - w, tr, w, label="trained (causal)", color="#2b7bba")
    ax.bar(x, ba, w, label="mean-topo baseline", color="#9bbb59")
    ax.bar(x + w, un, w, label="untrained (floor)", color="#c0504d")
    ax.axhline(SUCCESS_R, ls="--", color="k", lw=1, label=f"success bar r={SUCCESS_R}")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("downstream Pearson r (stim parcel excluded)")
    ax.set_title("Exp-1B LOSO: held-out-subject TMS prediction (per fold)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "exp1b_per_fold.png", dpi=150)
    plt.close(fig)


def summary_figure(folds: list[dict]) -> None:
    keys = [("trained_downstream_r", "trained\n(causal)", "#2b7bba"),
            ("baseline_downstream_r", "mean-topo\nbaseline", "#9bbb59"),
            ("untrained_downstream_r", "untrained\n(floor)", "#c0504d")]
    means, sds, labels, colors = [], [], [], []
    for k, lab, c in keys:
        v = np.array([f[k] for f in folds if k in f], dtype=float)
        means.append(v.mean()); sds.append(v.std(ddof=1) if v.size > 1 else 0.0)
        labels.append(lab); colors.append(c)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.bar(x, means, yerr=sds, capsize=5, color=colors)
    ax.axhline(SUCCESS_R, ls="--", color="k", lw=1, label=f"success bar r={SUCCESS_R}")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("mean downstream Pearson r")
    ax.set_title(f"Exp-1B summary (n={len(folds)} folds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "exp1b_summary.png", dpi=150)
    plt.close(fig)


def main() -> int:
    folds = load_folds()
    if not folds:
        print("No fold data yet; no figures rendered.")
        return 0
    per_fold_figure(folds)
    summary_figure(folds)
    print(f"Rendered reports/figures/exp1b_per_fold.png and exp1b_summary.png "
          f"({len(folds)} folds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
