"""Tests for src/eval/graph_metrics.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.graph_metrics import (threshold_adjacency, is_dag_binary,  # noqa: E402
                                structural_hamming_distance, edge_precision_recall,
                                top_k_edges)


def test_threshold_and_diagonal():
    a = np.array([[5.0, 0.4], [0.2, 5.0]])
    b = threshold_adjacency(a, 0.3)
    assert b.tolist() == [[0, 1], [0, 0]]  # diagonal zeroed, 0.2 dropped, 0.4 kept


def test_is_dag():
    dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    cyc = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    assert is_dag_binary(dag)
    assert not is_dag_binary(cyc)


def test_shd_identical_zero():
    a = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert structural_hamming_distance(a, a) == 0


def test_shd_reversal_is_one():
    a = np.array([[0, 1], [0, 0]])   # 1->0 (i=0 from j=1)
    b = np.array([[0, 0], [1, 0]])   # reversed
    assert structural_hamming_distance(a, b) == 1


def test_shd_single_insertion():
    a = np.zeros((3, 3))
    b = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    assert structural_hamming_distance(a, b) == 1


def test_precision_recall_perfect():
    a = np.array([[0, 1], [0, 0]])
    m = edge_precision_recall(a, a)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_precision_recall_partial():
    pred = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])  # 2 edges, 1 correct
    true = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])  # 2 edges
    m = edge_precision_recall(pred, true)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_top_k_edges_orders_by_magnitude():
    a = np.array([[0.0, 0.9, 0.1], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    edges = top_k_edges(a, 2)
    assert edges[0] == (1, 0, 0.9)   # src j=1 -> dst i=0, weight 0.9
    assert edges[1][2] == 0.5


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
