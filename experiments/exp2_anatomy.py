"""Exp-2 (SKELETON) — does the learned causal graph agree with an anatomical/structural prior?

This is a scaffold, not a validated result. It loads the Exp-1B fold checkpoints, extracts each
learned adjacency, and — IF a ground-truth structural prior is supplied — reports SHD, directed
edge precision/recall/F1, and top-k edge overlap (src/eval/graph_metrics.py). Without a prior it
still reports cross-fold edge stability (the consensus graph), which is informative on its own.

No structural connectome is bundled (the run sheet forbids new datasets), so the anatomical
comparison is gated behind ``--prior path.npy``. Provide a ``[d, d]`` binary/weighted prior to
activate it. This file exists so the validation is one small step away, with the metrics already
tested.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from eval.graph_metrics import (threshold_adjacency, is_dag_binary,  # noqa: E402
                                structural_hamming_distance, edge_precision_recall,
                                top_k_edges)

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def load_adjacencies(cfg) -> list[tuple[int, np.ndarray]]:
    d = cfg.parcellation.d
    out = []
    for fp in sorted(glob.glob(str(ROOT / "data" / "processed" / "checkpoints"
                                   / "exp1b_fold*.pt"))):
        ck = torch.load(fp, map_location="cpu", weights_only=False)
        model = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                             init_scale=cfg.model.init_state_scale)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        with torch.no_grad():
            out.append((int(ck.get("held_idx", -1)), model.A.detach().cpu().numpy()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default=None, help="path to [d,d] structural prior .npy")
    ap.add_argument("--tau", type=float, default=0.2, help="binarisation threshold")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    cfg = load_config()
    adjs = load_adjacencies(cfg)
    if not adjs:
        print("No Exp-1B checkpoints found; run exp1b with --save-ckpt first.")
        return 0

    prior = None
    if args.prior and Path(args.prior).exists():
        prior = np.load(args.prior)
        prior_bin = threshold_adjacency(prior, 0.0)

    md = ["# Exp-2 anatomy validation (skeleton)\n",
          f"Checkpoints: **{len(adjs)}** | τ={args.tau} | "
          f"prior: {'supplied' if prior is not None else 'NONE (stability-only)'}\n",
          "| fold | edges@τ | is_dag | "
          + ("SHD | precision | recall | f1 |" if prior is not None else "") ,
          "|---|---|---|" + ("---|---|---|---|" if prior is not None else "")]

    consensus = Counter()
    for fold, adj in adjs:
        b = threshold_adjacency(adj, args.tau)
        ne = int(b.sum())
        dag = is_dag_binary(b)
        row = f"| {fold} | {ne} | {'yes' if dag else 'NO'} |"
        if prior is not None:
            shd = structural_hamming_distance(b, prior_bin)
            pr = edge_precision_recall(b, prior_bin)
            row += f" {shd} | {pr['precision']:.3f} | {pr['recall']:.3f} | {pr['f1']:.3f} |"
        md.append(row)
        for j, i, _w in top_k_edges(adj, args.topk):
            consensus[(j, i)] += 1

    md.append("\n## Cross-fold consensus edges (top-50 per fold)\n")
    md.append("| src j | dst i | folds |")
    md.append("|---|---|---|")
    for (j, i), c in consensus.most_common(25):
        md.append(f"| {j} | {i} | {c} |")

    if prior is None:
        md.append("\n_No structural prior supplied — anatomical agreement not computed. "
                  "Pass `--prior path.npy` (a [d,d] connectome) to activate SHD / precision / "
                  "recall. Metrics are unit-tested in tests/test_graph_metrics.py._\n")

    out = REPORTS / "exp2_anatomy.md"
    out.write_text("\n".join(md) + "\n")
    print(f"Wrote {out} ({len(adjs)} folds, prior={'yes' if prior is not None else 'no'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
