"""Analyze Exp-1B leave-one-subject-out results: CI, sign-flip permutation, paired Cohen's d.

Reads either the aggregate ``data/processed/exp1b_results.json`` or, if absent/partial, the
incremental per-fold files ``data/processed/exp1b_folds/fold*.json`` (so it works while the
background LOSO run is still in progress). Writes:
  * reports/exp1b_folds.csv   — one row per fold (all per-fold metrics)
  * reports/exp1b_table.csv   — one row per metric (mean, sd, bootstrap 95% CI)
  * reports/exp1b_summary.md   — human-readable summary with paired stats

Exits gracefully (non-zero only on truly empty input) if expected keys are missing.
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.stats import (bootstrap_ci, exact_sign_flip_test,  # noqa: E402
                        paired_permutation_test, cohens_d_paired)

PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# metric key -> human label
METRICS = {
    "untrained_downstream_r": "untrained downstream r (floor)",
    "trained_downstream_r": "TRAINED downstream r (headline)",
    "trained_full_r": "trained full r (confounded)",
    "baseline_downstream_r": "mean-topo baseline downstream r",
    "baseline_full_r": "mean-topo baseline full r",
}


def load_folds() -> list[dict]:
    """Prefer aggregate results.json folds; fall back to incremental per-fold JSONs."""
    agg = PROC / "exp1b_results.json"
    folds: list[dict] = []
    if agg.exists():
        try:
            data = json.load(open(agg))
            folds = data.get("folds", [])
        except (json.JSONDecodeError, OSError):
            folds = []
    # If aggregate missing/short, union-in the incremental per-fold files.
    by_fold: dict[int, dict] = {}
    for r in folds:
        if "fold" in r:
            by_fold[int(r["fold"])] = r
    for fp in sorted(glob.glob(str(PROC / "exp1b_folds" / "fold*.json"))):
        try:
            r = json.load(open(fp))
        except (json.JSONDecodeError, OSError):
            continue
        k = r.get("fold")
        if k is None:
            # derive from filename foldNN.json
            stem = Path(fp).stem
            k = int(stem.replace("fold", "")) if stem.replace("fold", "").isdigit() else len(by_fold)
        by_fold[int(k)] = r
    return [by_fold[k] for k in sorted(by_fold)]


def col(folds: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in folds if key in r], dtype=float)


def write_folds_csv(folds: list[dict]) -> None:
    if not folds:
        return
    # stable column order: fold, subject, then sorted numeric keys
    keys = ["fold", "subject"]
    extra = sorted(k for k in folds[0] if k not in keys)
    keys += extra
    with open(REPORTS / "exp1b_folds.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in folds:
            w.writerow(r)


def write_table_csv(folds: list[dict]) -> list[dict]:
    rows = []
    with open(REPORTS / "exp1b_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "n", "mean", "sd", "boot_lo95", "boot_hi95"])
        for key, label in METRICS.items():
            v = col(folds, key)
            if v.size == 0:
                continue
            mean, lo, hi = bootstrap_ci(v)
            sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
            w.writerow([key, v.size, f"{mean:.4f}", f"{sd:.4f}",
                        f"{lo:.4f}", f"{hi:.4f}"])
            rows.append({"key": key, "label": label, "n": int(v.size),
                         "mean": mean, "sd": sd, "lo": lo, "hi": hi})
    return rows


def paired_block(folds: list[dict], a_key: str, b_key: str, name: str) -> str:
    a = col(folds, a_key)
    b = col(folds, b_key)
    n = min(a.size, b.size)
    if n < 2:
        return f"### {name}\nInsufficient paired folds (n={n}).\n"
    a, b = a[:n], b[:n]
    diff = a - b
    p_exact = exact_sign_flip_test(a, b) if n <= 20 else float("nan")
    p_perm = paired_permutation_test(a, b)
    d = cohens_d_paired(a, b)
    md, lo, hi = bootstrap_ci(diff)
    wins = int((diff > 0).sum())
    lines = [f"### {name}",
             f"- n folds: {n}",
             f"- mean paired diff (A−B): {md:+.4f}  (bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}])",
             f"- folds where A>B: {wins}/{n}",
             f"- exact sign-flip permutation p: {p_exact:.4f}" if n <= 20
             else f"- permutation p (Monte-Carlo): {p_perm:.4f}",
             f"- paired Cohen's d: {d:+.3f}",
             ""]
    return "\n".join(lines)


def main() -> int:
    folds = load_folds()
    if not folds:
        print("No Exp-1B fold data found yet (data/processed/exp1b_folds/ empty). "
              "Nothing to analyze.")
        return 0

    write_folds_csv(folds)
    table = write_table_csv(folds)

    # console summary
    print(f"Exp-1B analysis | folds available: {len(folds)}")
    for row in table:
        print(f"  {row['label']:<40} {row['mean']:+.3f} ± {row['sd']:.3f}  "
              f"CI[{row['lo']:+.3f}, {row['hi']:+.3f}]")

    md = ["# Exp-1B (LOSO) analysis\n",
          f"Folds analyzed: **{len(folds)}** "
          f"(subjects: {', '.join(str(r.get('subject', '?')) for r in folds)})\n",
          "## Per-metric mean ± SD with bootstrap 95% CI\n",
          "| metric | n | mean | sd | 95% CI |",
          "|---|---|---|---|---|"]
    for row in table:
        md.append(f"| {row['label']} | {row['n']} | {row['mean']:+.3f} | "
                  f"{row['sd']:.3f} | [{row['lo']:+.3f}, {row['hi']:+.3f}] |")
    md.append("\n## Paired comparisons (headline metric)\n")
    md.append(paired_block(folds, "trained_downstream_r", "untrained_downstream_r",
                           "Trained vs Untrained (downstream r)"))
    md.append(paired_block(folds, "trained_downstream_r", "baseline_downstream_r",
                           "Trained vs Mean-topo baseline (downstream r)"))

    (REPORTS / "exp1b_summary.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote reports/exp1b_folds.csv, reports/exp1b_table.csv, "
          f"reports/exp1b_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
