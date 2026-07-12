"""Quick r>0.9 test on Daie/Svoboda 2020 all-optical data (fixed (ti,N,tr) axis order)."""
import h5py, numpy as np

MAT = "../Open Neuro daie2020/Daie_et_al_2020_targeted_photostim.mat"


def pear(a, b, m):
    ok = m & np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 6:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    de = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / de) if de > 1e-12 else np.nan


def main():
    f = h5py.File(MAT, "r"); d = f["data"]; ns = d["R"].shape[0]
    rng = np.random.default_rng(0)
    ceil, sr, lso, dst = [], [], [], []
    nsess = 0
    skipped = 0
    for s in range(ns):
      try:
        R = f[d["R"][s, 0]]; pg = R.shape[0] - 1
        if pg < 2:
            continue
        XY = np.array(f[d["XY"][s, 0]])
        if XY.shape[0] != 2:
            XY = XY.T
        N = XY.shape[1]; xy = XY.T
        dist = f[d["distance"][s, 0]]
        topo, h1, h2, tgt, dgs = [], [], [], [], []
        good = True
        for g in range(1, pg + 1):
            a = np.array(f[R[g, 0]])
            if a.ndim != 3 or a.shape[1] != N:
                good = False; break
            tr = a.shape[2]; ti = a.shape[0]
            nb = max(2, ti // 6)                       # baseline = first ~1/6 of frames
            inf = a[nb:].mean(0) - a[:nb].mean(0)      # (N, tr) post-stim minus baseline, per trial
            topo.append(inf.mean(1))
            h1.append(inf[:, :tr // 2].mean(1)); h2.append(inf[:, tr // 2:].mean(1))
            dg = np.array(f[dist[g - 1, 0]]).ravel()
            if dg.size != N:
                dg = np.full(N, np.nan)
            dgs.append(dg); tgt.append(dg < 15)
        if not good or len(topo) < 2:
            continue
        topo = np.array(topo); h1 = np.array(h1); h2 = np.array(h2)
        tgt = np.array(tgt); dgs = np.array(dgs); nsess += 1
        for gi in range(len(topo)):
            valid = (~tgt[gi]) & np.isfinite(topo[gi])
            ceil.append(pear(h1[gi], h2[gi], valid))
            dg = dgs[gi]
            cand = [pear(np.exp(-dg / L), topo[gi], valid) for L in [20, 40, 80, 160, 320]]
            dst.append(np.nanmax(cand) if np.any(np.isfinite(cand)) else np.nan)
            vidx = np.where(valid)[0]
            if len(vidx) >= 15:
                perm = rng.permutation(vidx); nho = int(0.3 * len(vidx))
                ho, obs = perm[:nho], perm[nho:]
                dd = np.linalg.norm(xy[ho][:, None] - xy[obs][None], axis=2)
                w = np.exp(-(dd ** 2) / (2 * 50.0 ** 2))
                pred = (w * topo[gi][obs][None]).sum(1) / (w.sum(1) + 1e-9)
                sr.append(pear(pred, topo[gi][ho], np.ones(len(ho), bool)))
            others = [j for j in range(len(topo)) if j != gi]
            lso.append(pear(np.nanmean(topo[others], 0), topo[gi], valid))
      except Exception:
        skipped += 1
        continue
    print("skipped sessions:", skipped)
    for nm, arr in [("CEILING (split-half)", ceil), ("DISTANCE model", dst),
                    ("SUPER-RES (leave-neuron-out)", sr), ("LEAVE-GROUP-OUT within_mean", lso)]:
        a = np.array(arr)
        print(f"{nm:30s} n={len(a):4d} mean={np.nanmean(a):.3f} median={np.nanmedian(a):.3f} "
              f"frac>0.9={(a > 0.9).mean() * 100:.0f}%")
    print("sessions used:", nsess)


if __name__ == "__main__":
    main()
