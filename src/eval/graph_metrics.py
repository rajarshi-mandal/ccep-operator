"""Graph-structure metrics for validating a learned adjacency against a ground-truth prior.

Pure, dependency-light functions (numpy only) used by the Exp-2 anatomy-validation skeleton.
Convention matches the model: ``A[i, j]`` = "j influences i". All functions operate on dense
``[d, d]`` arrays. ``threshold_adjacency`` binarises; the rest consume binary or weighted A.
"""
from __future__ import annotations

import numpy as np


def threshold_adjacency(adj: np.ndarray, tau: float) -> np.ndarray:
    """Binary adjacency: 1 where ``|adj| > tau`` (diagonal forced to 0)."""
    b = (np.abs(adj) > tau).astype(int)
    np.fill_diagonal(b, 0)
    return b


def is_dag_binary(adj_bin: np.ndarray) -> bool:
    """True iff the binary adjacency is acyclic (nilpotency: A^d == 0)."""
    b = (np.asarray(adj_bin) != 0).astype(int)
    np.fill_diagonal(b, 0)
    p = b.copy()
    for _ in range(b.shape[0]):
        if p.sum() == 0:
            return True
        p = (p @ b > 0).astype(int)
    return p.sum() == 0


def structural_hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """SHD between two binary DAGs: # edge insertions/deletions/reversals to match.

    Counts a reversed edge once (not twice). Both inputs are binarised on entry.
    """
    A = (np.asarray(a) != 0).astype(int)
    B = (np.asarray(b) != 0).astype(int)
    np.fill_diagonal(A, 0); np.fill_diagonal(B, 0)
    d = A.shape[0]
    shd = 0
    for i in range(d):
        for j in range(i + 1, d):
            # consider the unordered pair {i,j}: each can have an edge in either direction
            a_ij, a_ji = A[i, j], A[j, i]
            b_ij, b_ji = B[i, j], B[j, i]
            if (a_ij, a_ji) == (b_ij, b_ji):
                continue
            # reversal: exactly one directed edge each way, opposite directions
            if a_ij + a_ji == 1 and b_ij + b_ji == 1:
                shd += 1
            else:
                shd += abs(a_ij - b_ij) + abs(a_ji - b_ji)
    return int(shd)


def edge_precision_recall(pred: np.ndarray, true: np.ndarray) -> dict:
    """Directed-edge precision/recall/F1 of ``pred`` against ``true`` (both binarised)."""
    P = (np.asarray(pred) != 0).astype(int)
    T = (np.asarray(true) != 0).astype(int)
    np.fill_diagonal(P, 0); np.fill_diagonal(T, 0)
    tp = int(((P == 1) & (T == 1)).sum())
    fp = int(((P == 1) & (T == 0)).sum())
    fn = int(((P == 0) & (T == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def top_k_edges(adj: np.ndarray, k: int) -> list[tuple[int, int, float]]:
    """Top-k strongest directed edges as ``(src j, dst i, weight)`` by ``|weight|``."""
    a = np.asarray(adj, dtype=float).copy()
    np.fill_diagonal(a, 0.0)
    order = np.argsort(np.abs(a), axis=None)[::-1][:k]
    out = []
    for flat in order:
        i, j = np.unravel_index(flat, a.shape)
        if abs(a[i, j]) < 1e-12:
            break
        out.append((int(j), int(i), float(a[i, j])))
    return out
