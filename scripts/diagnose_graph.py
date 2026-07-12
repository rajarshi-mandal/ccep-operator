"""Graph diagnostics for the Exp-1B per-fold CausalDAGSSM checkpoints.

For every ``data/processed/checkpoints/exp1b_fold*.pt`` we reload the learned weighted
adjacency A (= masked W) and report:
  * acyclicity h(A) recorded at train end, and edge counts at thresholds {0.1, 0.2, 0.3}
  * whether the thresholded graph is a DAG (NetworkX) and its #weakly-connected components
  * in-/out-degree of the stimulated parcel (M1) — does the learned graph actually wire M1
    to downstream regions, which is the mechanism the do(M1) prediction relies on?
  * the top-K strongest directed edges (i<-j, "j influences i")

Edge convention (see model): ``A[i, j]`` = "j influences i", so column j = OUT-edges of j,
row i = IN-edges of i. Writes reports/graph_diagnostics.md and reports/graph_edges.csv.
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402

try:
    import networkx as nx
    HAVE_NX = True
except ImportError:
    HAVE_NX = False

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
THRESHOLDS = (0.1, 0.2, 0.3)
TOP_K = 50


def adjacency_from_ckpt(ckpt: dict, cfg) -> torch.Tensor:
    """Rebuild the model from a checkpoint and return its masked adjacency A ``[d, d]``."""
    d = cfg.parcellation.d
    model = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                         init_scale=cfg.model.init_state_scale)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        return model.A.detach().cpu()


def is_dag(adj: np.ndarray):
    """Return (is_dag, n_components) for binarized adjacency, NetworkX if available."""
    b = (np.abs(adj) > 0).astype(int)
    if HAVE_NX:
        g = nx.DiGraph(b.T)   # A[i,j]=j->i; nx edge u->v means row u col v, so transpose
        return nx.is_directed_acyclic_graph(g), nx.number_weakly_connected_components(g)
    # fallback: nilpotency check — a binary DAG's adjacency is nilpotent (A^d == 0)
    p = b.copy()
    for _ in range(b.shape[0]):
        p = (p @ b > 0).astype(int)
        if p.sum() == 0:
            return True, -1
    return (np.trace(p) == 0), -1


def top_edges(adj: np.ndarray, k: int):
    """Top-k strongest directed edges as (weight, src j, dst i)."""
    a = adj.copy()
    np.fill_diagonal(a, 0.0)
    idx = np.argsort(np.abs(a), axis=None)[::-1][:k]
    out = []
    for flat in idx:
        i, j = np.unravel_index(flat, a.shape)   # row i (dst), col j (src)
        w = a[i, j]
        if abs(w) < 1e-9:
            break
        out.append((float(w), int(j), int(i)))
    return out


def main() -> int:
    cfg = load_config()
    ckpts = sorted(glob.glob(str(ROOT / "data" / "processed" / "checkpoints" / "exp1b_fold*.pt")))
    if not ckpts:
        print("No checkpoints found (data/processed/checkpoints/exp1b_fold*.pt). "
              "Run exp1b with --save-ckpt first.")
        return 0

    md = ["# Exp-1B graph diagnostics\n",
          f"Checkpoints: **{len(ckpts)}** | thresholds {THRESHOLDS} | "
          f"edge convention A[i,j]='j influences i'\n",
          "| fold | subject | stim(M1) | h(A) | "
          + " | ".join(f"|E|@{t}" for t in THRESHOLDS)
          + " | DAG@0.3 | M1 out-deg@0.2 | M1 in-deg@0.2 |",
          "|---|---|---|---|" + "---|" * (len(THRESHOLDS) + 3)]
    edge_rows = []

    for fp in ckpts:
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        adj = adjacency_from_ckpt(ck, cfg).numpy()
        stim = int(ck.get("stim_parcel", -1))
        hval = float(ck.get("final_h", float("nan")))
        ecounts = [int((np.abs(adj) > t).sum()) for t in THRESHOLDS]
        bin03 = np.abs(adj) > 0.3
        dag03, _ = is_dag(bin03.astype(int))
        # degrees at 0.2: column stim = out-edges of stim, row stim = in-edges
        bin02 = np.abs(adj) > 0.2
        out_deg = int(bin02[:, stim].sum()) if stim >= 0 else -1   # stim influences others
        in_deg = int(bin02[stim, :].sum()) if stim >= 0 else -1
        fold = int(ck.get("held_idx", -1))
        subj = ck.get("subject", "?")
        md.append(f"| {fold} | {subj} | {stim} | {hval:.1e} | "
                  + " | ".join(str(e) for e in ecounts)
                  + f" | {'yes' if dag03 else 'NO'} | {out_deg} | {in_deg} |")
        for w, j, i in top_edges(adj, TOP_K):
            edge_rows.append({"fold": fold, "weight": f"{w:+.4f}",
                              "src_j": j, "dst_i": i,
                              "src_is_M1": int(j == stim), "dst_is_M1": int(i == stim)})

    with open(REPORTS / "graph_edges.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fold", "weight", "src_j", "dst_i",
                                           "src_is_M1", "dst_is_M1"])
        w.writeheader()
        w.writerows(edge_rows)

    # cross-fold edge consistency: which directed edges recur across folds?
    from collections import Counter
    pair_counts = Counter((r["src_j"], r["dst_i"]) for r in edge_rows)
    md.append("\n## Most consistent top-50 edges across folds\n")
    md.append("| src j (influences) | dst i | folds present |")
    md.append("|---|---|---|")
    for (j, i), c in pair_counts.most_common(20):
        md.append(f"| {j} | {i} | {c} |")

    md.append(f"\n_Top-{TOP_K} edges per fold written to reports/graph_edges.csv "
              f"({len(edge_rows)} rows). NetworkX available: {HAVE_NX}._\n")
    (REPORTS / "graph_diagnostics.md").write_text("\n".join(md) + "\n")
    print(f"Diagnosed {len(ckpts)} checkpoints -> reports/graph_diagnostics.md, "
          f"reports/graph_edges.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
