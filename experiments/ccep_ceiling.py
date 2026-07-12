"""Noise ceiling for the CCEP LOSO prediction (how much headroom is left above combo=0.728?).

The split-half reliability R of a site's N1 topography is the hard upper bound on prediction: no
predictor can correlate with the measured topography better than an oracle holding the *true*
(noise-free) topography, which correlates sqrt(R) with the measurement. So:

    ceiling(topo-r) = sqrt(reliability)   per site, averaged to subject level.

This is the ABSOLUTE ceiling (trial noise only). The practical "predictable-from-other-sites"
ceiling is lower — but if combo sits well below sqrt(R), the target is clean enough that real
improvement is possible (the gap is signal, not noise).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa: E402
from eval.stats import bootstrap_ci  # noqa: E402
from ccep_loso import all_caches, REL_MIN  # noqa: E402

# combo subject-level scores from ccep_loso (n=13), for the side-by-side
COMBO = {
    "4774/sub-MAYO01": 0.781, "4774/sub-MAYO02": 0.715, "4774/sub-MAYO03": 0.495,
    "4774/sub-MAYO04": 0.689, "4774/sub-MAYO05": 0.547, "4696/sub-01": 0.831,
    "4696/sub-02": 0.767, "4696/sub-03": 0.817, "4696/sub-04": 0.803,
    "4696/sub-05": 0.807, "4696/sub-06": 0.833, "4696/sub-07": 0.647, "4696/sub-08": 0.735,
}


def main():
    caches = all_caches()
    print(f"{'subject':20s} {'nsites':>6s} {'reliab':>7s} {'ceiling':>8s} {'combo':>7s} {'headroom':>9s}")
    ceil, combo, head = [], [], []
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        tag = f"{ds[-4:]}/{cs.subject}"
        rel = cs.reliability
        keep = (np.isfinite(rel)) & (rel >= REL_MIN)
        R = np.clip(rel[keep], 0, 1)
        ceiling = float(np.mean(np.sqrt(R)))
        cmb = COMBO.get(tag, np.nan)
        ceil.append(ceiling); combo.append(cmb); head.append(ceiling - cmb)
        print(f"{tag:20s} {int(keep.sum()):6d} {float(np.mean(R)):7.3f} {ceiling:8.3f} "
              f"{cmb:7.3f} {ceiling-cmb:+9.3f}")

    cm, clo, chi = bootstrap_ci(ceil)
    mm, mlo, mhi = bootstrap_ci(combo)
    hm, hlo, hhi = bootstrap_ci(head)
    print("\n=== subject-level (mean, 95% CI) ===")
    print(f"  ceiling  {cm:.3f} [{clo:.3f}, {chi:.3f}]")
    print(f"  combo    {mm:.3f} [{mlo:.3f}, {mhi:.3f}]")
    print(f"  headroom {hm:.3f} [{hlo:.3f}, {hhi:.3f}]   (recoverable r above combo)")
    print(f"\n  combo realises {mm/cm*100:.0f}% of the absolute ceiling; "
          f"{hm:.2f} of topo-r is still on the table (upper bound).")


if __name__ == "__main__":
    main()
