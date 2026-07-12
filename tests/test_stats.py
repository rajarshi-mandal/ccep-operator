"""Tests for src/eval/stats.py — bootstrap CI, sign-flip permutation, paired Cohen's d."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eval.stats import (bootstrap_ci, exact_sign_flip_test,  # noqa: E402
                        paired_permutation_test, cohens_d_paired)


def test_bootstrap_ci_ordered():
    mean, lo, hi = bootstrap_ci([0.5, 0.6, 0.7, 0.55, 0.65], seed=0)
    assert lo <= mean <= hi
    assert abs(mean - 0.6) < 0.05


def test_bootstrap_ci_constant():
    mean, lo, hi = bootstrap_ci([0.3, 0.3, 0.3])
    assert mean == 0.3 and lo == 0.3 and hi == 0.3


def test_sign_flip_identical_is_one():
    # a == b => mean diff 0 => every sign-flip ties => p == 1.
    a = [0.4, 0.5, 0.6]
    assert exact_sign_flip_test(a, a) == 1.0


def test_sign_flip_large_separation_small_p():
    a = [0.7, 0.6, 0.65, 0.62, 0.68, 0.71, 0.59]
    b = [-0.1, 0.0, 0.05, -0.05, 0.1, -0.02, 0.03]
    p = exact_sign_flip_test(a, b)
    # All diffs positive => only the all-+ assignment reaches |obs| => p = 2/2^n (two-sided).
    assert p == 2 / (2 ** len(a))
    assert p < 0.05


def test_permutation_matches_exact_for_small_n():
    a = [0.7, 0.6, 0.65, 0.62, 0.68]
    b = [0.1, 0.2, 0.0, 0.15, 0.05]
    assert paired_permutation_test(a, b) == exact_sign_flip_test(a, b)


def test_cohens_d_zero_for_identical():
    a = [0.4, 0.5, 0.6]
    assert cohens_d_paired(a, a) == 0.0


def test_cohens_d_positive_for_consistent_gain():
    a = [0.7, 0.6, 0.65, 0.62, 0.68]
    b = [0.1, 0.05, 0.0, 0.08, 0.12]
    assert cohens_d_paired(a, b) > 1.0  # large, consistent effect


def test_cohens_d_sign():
    a = [0.1, 0.25, 0.15]
    b = [0.5, 0.58, 0.56]
    assert cohens_d_paired(a, b) < 0  # b consistently larger => negative d


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
