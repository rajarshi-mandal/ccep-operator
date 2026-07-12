"""Small, dependency-light statistics for paired LOSO fold comparisons.

All functions take plain python lists or 1-D numpy arrays. They are deliberately simple and
exact where feasible (sign-flip permutation is exact for n<=~20 folds), so the Exp-1B
trained-vs-untrained and trained-vs-baseline comparisons are defensible for a paper.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], n_boot: int = 10000, ci: float = 95,
                 seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap mean and (lower, upper) percentile CI of a sample.

    Returns ``(mean, lo, hi)``. Resamples the values with replacement ``n_boot`` times.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return float(v.mean()), lo, hi


def exact_sign_flip_test(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact two-sided paired sign-flip permutation p-value on the mean difference.

    Enumerates all 2^n sign assignments of the paired differences d = a - b. Valid (exact)
    for small n (folds). Returns the two-sided p-value: fraction of sign-flips whose
    |mean| >= |observed mean|.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    count = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        if abs((d * np.asarray(signs)).mean()) >= obs - 1e-12:
            count += 1
    return count / total


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000,
                            seed: int = 0) -> float:
    """Two-sided paired permutation (sign-flip) p-value.

    Uses the exact enumeration for n<=18, otherwise a Monte-Carlo sign-flip approximation.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n == 0:
        return float("nan")
    if n <= 18:
        return exact_sign_flip_test(a, b)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice((1.0, -1.0), size=(n_perm, n))
    stats = np.abs((signs * d).mean(axis=1))
    return float((stats >= obs - 1e-12).mean())


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Paired Cohen's d = mean(d) / std(d) (sample std, ddof=1)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return float("inf") if abs(d.mean()) > 0 else 0.0
    return float(d.mean() / sd)


if __name__ == "__main__":
    a = [0.74, 0.59, 0.66, 0.61, 0.70]
    b = [-0.17, 0.03, 0.10, -0.05, 0.12]
    print("bootstrap_ci(a):", bootstrap_ci(a))
    print("sign-flip p(a vs b):", exact_sign_flip_test(a, b))
    print("cohens_d_paired(a,b):", cohens_d_paired(a, b))
