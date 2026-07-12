"""r>0.9 test on the Daie/Svoboda all-optical influence-mapping data (the dense + many-site +
multi-trial regime the simulation said reaches r>0.9).

Operates on per-session caches (data/processed/daie/sess*.npz) built by src/data/daie_pipeline.py:
    responses   [n_groups, n_neurons]   trial-averaged influence (post-stim minus baseline)
    resp_h1/h2  [n_groups, n_neurons]   half-trial splits (noise ceiling)
    neuron_xy   [n_neurons, 2]          imaged neuron locations (pixels)
    stim_xy     [n_groups, 2]           mean photostim target location per group
    target_mask [n_groups, n_neurons]   True where neuron is a photostim target of that group (exclude)

Task = leave-photostim-group-out: predict a held-out group's influence topography over neurons from
the other groups. Models: within_mean, distance (locality from stim_xy), combo (locality + network
residual). Plus super-resolution (leave-neuron-out within a group). Metric = Pearson r, ceiling =
half-split. Subject(session)-level aggregation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval.stats import bootstrap_ci, paired_permutation_test  # noqa: E402

CACHE = ROOT / "data" / "processed" / "daie"
REL_MIN = 0.5
SIG = [20, 40, 80, 160, 320]      # px locality bandwidths
TAU = [40, 80, 160, 320, 1e9]     # px stim-knn bandwidths


def pear(a, b, mask):
    ok = mask & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else np.nan


def z(x, mask):
    out = np.zeros_like(x, float); ok = mask & np.isfinite(x)
    if ok.sum() >= 2:
        out[ok] = (x[ok] - x[ok].mean()) / (x[ok].std() + 1e-9)
    return out


def evalu_session(d):
    R = d["responses"]; nxy = d["neuron_xy"]; sxy = d["stim_xy"]
    tmask = d["target_mask"]; rel = d["reliability"]
    h1, h2 = d["resp_h1"], d["resp_h2"]
    keep = np.where((np.isfinite(rel)) & (rel >= REL_MIN))[0]
    if len(keep) < 6:
        return None
    out = {m: [] for m in ["within_mean", "distance", "combo", "ceiling", "superres"]}
    for gi in keep:
        train = [g for g in keep if g != gi]
        valid = ~tmask[gi] & np.isfinite(R[gi])
        tgt = R[gi]
        # within_mean
        Rt = R[train]
        wm = np.nansum(Rt, 0) / (np.sum(np.isfinite(Rt), 0) + 1e-9)
        out["within_mean"].append(pear(wm, tgt, valid))
        # distance (nested-CV sigma on train groups)
        def dist_pred(g, sig):
            dd = np.linalg.norm(nxy - sxy[g][None], axis=1)
            return np.exp(-(dd ** 2) / (2 * sig ** 2))
        def score_sig(sig):
            rs = [pear(dist_pred(g, sig), R[g], ~tmask[g] & np.isfinite(R[g])) for g in train]
            return np.nanmean(rs)
        sig = max(SIG, key=score_sig)
        loc = dist_pred(gi, sig)
        out["distance"].append(pear(loc, tgt, valid))
        # combo: locality + stim-knn residual (nested-CV tau, beta=1)
        def knn_pred(g, tau):
            dd = np.linalg.norm(sxy[train] - sxy[g][None], axis=1); w = np.exp(-(dd ** 2) / (2 * tau ** 2))
            Rg = R[train]
            return np.nansum(w[:, None] * Rg, 0) / (np.nansum(w[:, None] * np.isfinite(Rg), 0) + 1e-9)
        def score_tau(tau):
            rs = []
            for g in train:
                tr2 = [t for t in train if t != g]
                dd = np.linalg.norm(sxy[tr2] - sxy[g][None], axis=1); w = np.exp(-(dd ** 2) / (2 * tau ** 2))
                Rg = R[tr2]; kp = np.nansum(w[:, None] * Rg, 0) / (np.nansum(w[:, None] * np.isfinite(Rg), 0) + 1e-9)
                vg = ~tmask[g] & np.isfinite(R[g])
                comb = z(dist_pred(g, sig), vg) + z(kp - dist_pred(g, sig), vg)
                rs.append(pear(comb, R[g], vg))
            return np.nanmean(rs)
        tau = max(TAU, key=score_tau)
        knn = knn_pred(gi, tau)
        comb = z(loc, valid) + z(np.nan_to_num(knn) - loc, valid)
        out["combo"].append(pear(comb, tgt, valid))
        # ceiling
        out["ceiling"].append(pear(h1[gi], h2[gi], valid))
        # super-resolution (leave-neuron-out interpolation of this group's own field)
        vidx = np.where(valid)[0]
        if len(vidx) >= 12:
            rng = np.random.default_rng(gi)
            perm = rng.permutation(vidx); nho = int(0.3 * len(vidx))
            ho, obs = perm[:nho], perm[nho:]
            bw = 80.0
            dd = np.linalg.norm(nxy[ho][:, None] - nxy[obs][None], axis=2); w = np.exp(-(dd ** 2) / (2 * bw ** 2))
            pred = (w * tgt[obs][None]).sum(1) / (w.sum(1) + 1e-9)
            m = np.zeros(len(tgt), bool); m[ho] = True
            out["superres"].append(pear(pred, tgt, m))
    return {m: float(np.nanmean(v)) for m, v in out.items()}, len(keep)


def main():
    caches = sorted(CACHE.glob("*.npz"))
    if not caches:
        print("no caches; build with src/data/daie_pipeline.py first"); return
    rows = {m: [] for m in ["within_mean", "distance", "combo", "ceiling", "superres"]}
    print(f"{'session':10s} {'groups':>6s} {'ceiling':>8s} {'wmean':>7s} {'dist':>7s} {'combo':>7s} {'superres':>9s}")
    for c in caches:
        d = dict(np.load(c, allow_pickle=True))
        res = evalu_session(d)
        if res is None:
            continue
        s, nk = res
        for m in rows:
            rows[m].append(s[m])
        print(f"{c.stem:10s} {nk:6d} {s['ceiling']:8.3f} {s['within_mean']:7.3f} "
              f"{s['distance']:7.3f} {s['combo']:7.3f} {s['superres']:9.3f}")
    print("\n=== session-level means (bootstrap 95% CI) ===")
    for m in ["ceiling", "within_mean", "distance", "combo", "superres"]:
        v = np.array(rows[m]); mn, lo, hi = bootstrap_ci(v.tolist())
        n9 = int((v > 0.9).sum())
        print(f"  {m:12s} {mn:+.3f} [{lo:+.3f}, {hi:+.3f}]   sessions r>0.9: {n9}/{len(v)}")
    best = max(["within_mean", "distance", "combo"], key=lambda m: np.nanmean(rows[m]))
    bv = np.array(rows[best])
    print(f"\n  best model = {best}: {np.nanmean(bv):.3f}, {int((bv>0.9).sum())}/{len(bv)} sessions >0.9")


if __name__ == "__main__":
    main()
