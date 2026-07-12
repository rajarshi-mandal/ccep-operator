"""Audit the Zenodo parietal conditions in region space, to confirm the cond1-3 vs cond4-6 split.

Memory (zenodo-conditions): cond1-3 are genuine P3 parietal stimulation; cond4-6 peak on an
ocular (VEOG) artifact and are likely sham. This script checks that hypothesis quantitatively in
the projected region space:
  * per-condition mean response-energy topography and its peak parcel
  * correlation of each condition's mean topography to the cond1-3 consensus topography
  * the peak parcel's MNI centroid (anterior+superior ⇒ consistent with frontal/ocular origin)

Writes reports/zenodo_label_audit.md. Read-only; no training.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def energy(tep: np.ndarray) -> np.ndarray:
    """RMS-over-time region topography from [d, T]."""
    return np.sqrt((tep.astype(float) ** 2).mean(axis=-1))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(); b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / den) if den > 1e-12 else 0.0


def main() -> int:
    cfg = load_config()
    proc = Path(cfg.paths.processed_dir)
    recs = list(np.load(proc / "interventional_region.npz", allow_pickle=True)["records"])
    par = [r for r in recs if r["site_name"] == "parietal_L"]
    if not par:
        print("No parietal_L records found.")
        return 0
    cents = np.load(proc / "parcel_centroids_mni.npy")

    by_cond = defaultdict(list)
    for r in par:
        by_cond[str(r.get("cond"))].append(energy(np.asarray(r["region_tep"])))
    cond_topo = {c: np.stack(v).mean(0) for c, v in by_cond.items()}

    # consensus of the "clean" conditions 1-3
    clean = [cond_topo[c] for c in ("1", "2", "3") if c in cond_topo]
    consensus = np.stack(clean).mean(0) if clean else None

    md = ["# Zenodo parietal label audit (region space)\n",
          f"Parietal records: **{len(par)}** across conditions "
          f"{sorted(by_cond, key=lambda x: x)}; stim parcel = "
          f"{par[0]['stim_parcel']}.\n",
          "| cond | n | peak parcel | peak MNI (x,y,z) | r to cond1-3 consensus |",
          "|---|---|---|---|---|"]
    for c in sorted(cond_topo, key=lambda x: (len(x), x)):
        topo = cond_topo[c]
        peak = int(np.argmax(topo))
        xyz = cents[peak]
        r = pearson(topo, consensus) if consensus is not None else float("nan")
        md.append(f"| {c} | {len(by_cond[c])} | {peak} | "
                  f"({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f}) | {r:+.3f} |")

    md.append("\n## Interpretation\n")
    if consensus is not None:
        peak_clean = int(np.argmax(consensus))
        md.append(f"- cond1-3 consensus peaks at parcel **{peak_clean}** "
                  f"(MNI {cents[peak_clean].round(0).tolist()}).")
        for c in ("4", "5", "6"):
            if c in cond_topo:
                r = pearson(cond_topo[c], consensus)
                pk = int(np.argmax(cond_topo[c]))
                flag = "LOW — supports sham/artifact" if r < 0.5 else "high — unexpectedly similar"
                md.append(f"- cond{c}: peak parcel {pk}, r to clean consensus = {r:+.3f} ({flag}).")
    md.append("\n_Conclusion fed to memory:zenodo-conditions — only cond1-3 used for the "
              "exploratory parietal arm (Exp-1C); cond4-6 excluded as likely sham._\n")

    out = REPORTS / "zenodo_label_audit.md"
    out.write_text("\n".join(md) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
