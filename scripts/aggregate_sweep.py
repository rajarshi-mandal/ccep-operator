"""Aggregate the Exp-1B sweep cells into one ranked table.

Scans ``data/processed/sweep/<tag>/exp1b_results.json`` (written by run_exp1b_sweep.sh) and,
for each cell, records the headline trained downstream r (mean ± SD over folds) alongside the
untrained and mean-topo baseline means. Ranks cells by headline r and flags the complexity-
ladder decision: an escalation is only "worth it" if it improves headline r by >= 0.03 over the
simplest comparable cell.

Writes reports/exp1b_sweep.csv and reports/exp1b_sweep.md. Safe to run mid-sweep.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "data" / "processed" / "sweep"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TAG_RE = re.compile(r"lam(?P<lam>[\d.]+)_o(?P<outer>\d+)_i(?P<inner>\d+)_s(?P<seed>\d+)")


def parse_tag(tag: str) -> dict:
    m = TAG_RE.match(tag)
    if not m:
        return {"lambda_int": None, "outer": None, "inner": None, "seed": None}
    return {"lambda_int": float(m["lam"]), "outer": int(m["outer"]),
            "inner": int(m["inner"]), "seed": int(m["seed"])}


def load_cell(path: Path) -> dict | None:
    try:
        data = json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None
    folds = data.get("folds", [])
    if not folds:
        return None
    tr = np.array([f["trained_downstream_r"] for f in folds if "trained_downstream_r" in f])
    un = np.array([f["untrained_downstream_r"] for f in folds if "untrained_downstream_r" in f])
    ba = np.array([f["baseline_downstream_r"] for f in folds if "baseline_downstream_r" in f])
    row = parse_tag(path.parent.name)
    row.update({
        "tag": path.parent.name, "n_folds": int(tr.size),
        "trained_mean": float(tr.mean()) if tr.size else float("nan"),
        "trained_sd": float(tr.std(ddof=1)) if tr.size > 1 else 0.0,
        "untrained_mean": float(un.mean()) if un.size else float("nan"),
        "baseline_mean": float(ba.mean()) if ba.size else float("nan"),
    })
    return row


def main() -> int:
    cells = sorted(glob.glob(str(SWEEP / "*" / "exp1b_results.json")))
    rows = [r for r in (load_cell(Path(p)) for p in cells) if r]
    if not rows:
        print(f"No sweep cells found under {SWEEP}. Run scripts/run_exp1b_sweep.sh first.")
        return 0
    rows.sort(key=lambda r: r["trained_mean"], reverse=True)

    cols = ["tag", "lambda_int", "outer", "inner", "seed", "n_folds",
            "trained_mean", "trained_sd", "untrained_mean", "baseline_mean"]
    with open(REPORTS / "exp1b_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    best = rows[0]
    simplest = min(rows, key=lambda r: (r["lambda_int"] or 0, r["outer"] or 0,
                                        r["inner"] or 0))
    gain = best["trained_mean"] - simplest["trained_mean"]
    md = ["# Exp-1B sweep aggregate\n",
          f"Cells: **{len(rows)}** | ranked by trained downstream r\n",
          "| tag | λ | outer | inner | seed | folds | trained r | untrained | baseline |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['tag']} | {r['lambda_int']} | {r['outer']} | {r['inner']} | "
                  f"{r['seed']} | {r['n_folds']} | {r['trained_mean']:+.3f}±{r['trained_sd']:.3f} "
                  f"| {r['untrained_mean']:+.3f} | {r['baseline_mean']:+.3f} |")
    md.append("\n## Complexity-ladder decision\n")
    md.append(f"- Best cell: **{best['tag']}** trained r = {best['trained_mean']:+.3f}")
    md.append(f"- Simplest cell: **{simplest['tag']}** trained r = {simplest['trained_mean']:+.3f}")
    md.append(f"- Gain (best − simplest) = **{gain:+.3f}** "
              f"({'ESCALATE — exceeds +0.03 bar' if gain >= 0.03 else 'STAY SIMPLE — below +0.03 bar'})")
    (REPORTS / "exp1b_sweep.md").write_text("\n".join(md) + "\n")
    print(f"Aggregated {len(rows)} cells -> reports/exp1b_sweep.csv, reports/exp1b_sweep.md")
    print(f"  best={best['tag']} ({best['trained_mean']:+.3f}) "
          f"gain-over-simplest={gain:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
