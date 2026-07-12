"""r>0.9 attempt on DANDI:000774 (mouse e-stim + dense spiking).

Tests whether dense readout + high trials lets prediction of a held-out stimulation site's evoked
topography cross r=0.9. Reports, per session: noise ceiling (half-split), leave-stim-site-out
within_mean and a combo (within_mean + responsiveness residual), and leave-unit-out super-resolution
(CCF interpolation). Honest read on whether r>0.9 is reached and via what (common response vs
site-specific structure).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data.dandi774_pipeline import build_session, all_sessions  # noqa: E402

CACHE = ROOT / "data" / "processed" / "ds000774"
REL_MIN = 0.5


def pear(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else np.nan


def superres(mean, xyz, rng):
    valid = np.where(np.isfinite(mean))[0]
    if len(valid) < 12:
        return np.nan
    perm = rng.permutation(valid); nho = int(0.3 * len(valid))
    ho, obs = perm[:nho], perm[nho:]
    best, br = 200.0, -2
    for bw in [100, 200, 400, 800, 1600]:
        pr = []
        for i in range(len(obs)):
            m = np.ones(len(obs), bool); m[i] = False
            d = np.linalg.norm(xyz[obs[m]] - xyz[obs[i]][None], axis=1); w = np.exp(-(d ** 2) / (2 * bw ** 2))
            pr.append((w * mean[obs[m]]).sum() / (w.sum() + 1e-9))
        r = pear(np.array(pr), mean[obs])
        if np.isfinite(r) and r > br:
            br, best = r, bw
    d = np.linalg.norm(xyz[ho][:, None] - xyz[obs][None], axis=2); w = np.exp(-(d ** 2) / (2 * best ** 2))
    pred = (w * mean[obs][None]).sum(1) / (w.sum(1) + 1e-9)
    return pear(pred, mean[ho])


def main():
    rng = np.random.default_rng(0)
    paths = all_sessions()
    print(f"{'session':10s} {'sites':>5s} {'units':>6s} {'tr/site':>8s} {'ceiling':>8s} "
          f"{'wmean LSO':>10s} {'wmean max':>10s} {'>0.9':>6s} {'superres':>9s}")
    all_lso, all_max = [], []
    for p in paths:
        try:
            cs = build_session(p)
        except Exception as e:
            print(f"{Path(p).name.split('_')[0]:10s}  (unreadable: {type(e).__name__})")
            continue
        keep = np.where((np.isfinite(cs.reliability)) & (cs.reliability >= REL_MIN))[0]
        if len(keep) < 3:
            print(f"{cs.session:10s} {len(cs.sites):5d}  (only {len(keep)} reliable sites)")
            continue
        M = cs.responses[keep]
        lso = [pear(np.nanmean(M[[j for j in range(len(keep)) if j != k]], 0), M[k])
               for k in range(len(keep))]
        sr = [superres(cs.responses[s], cs.unit_xyz, rng) for s in keep]
        ceil = np.nanmedian(cs.reliability[keep])
        lso = np.array(lso)
        n9 = int((lso > 0.9).sum())
        all_lso.append(np.nanmean(lso)); all_max.append(np.nanmax(lso))
        print(f"{cs.session:10s} {len(keep):5d} {cs.unit_xyz.shape[0]:6d} "
              f"{int(np.median(cs.n_trials[keep])):8d} {ceil:8.3f} {np.nanmean(lso):10.3f} "
              f"{np.nanmax(lso):10.3f} {n9:4d}/{len(keep)} {np.nanmean(sr):9.3f}")
    if all_lso:
        print(f"\n  across sessions: within_mean LSO mean={np.mean(all_lso):.3f}, "
              f"best-site mean={np.mean(all_max):.3f}")
        print(f"  sessions reaching a site with r>0.9: "
              f"{sum(1 for m in all_max if m > 0.9)}/{len(all_max)}")


if __name__ == "__main__":
    main()
