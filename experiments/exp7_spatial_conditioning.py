"""Exp-7 (GAP-3 attempt) — spatial near->far subject conditioning on the trained Exp-1B graphs.

The proven Bayes ceiling: with no subject input the model emits one group prediction, and the
group mean is optimal, so every group-level spatial readout ties the mean (H2, Arms A/B/C). The
ONLY way to legitimately beat the mean without new data is to condition on the held-out subject's
*own* measured response — information the group mean cannot use.

Arm C tried temporal conditioning (early->late) and lost to persistence. This arm tries the
orthogonal *spatial* axis: seed the trained causal graph with the held subject's measured energy
in the regions NEAR the stimulation site, propagate through ``A``, and predict the DISTAL regions.
A subject whose near-field response is atypical should, IF the graph routes subject-consistently,
get a personalized distal prediction the group mean cannot make.

Fair baseline: the leave-one-out group-mean distal topography (the same Bayes-optimal predictor
that ties every other arm). Pearson is scale-invariant, so amplitude-rescaling the mean by the
observed near amplitude does not change it — the causal graph must beat the mean's *pattern* using
the subject's own near pattern. Eval-only on the 13 published checkpoints; no retraining.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import load_config  # noqa: E402
from data.dataset import InterventionalDataset  # noqa: E402
from model.causal_dag_ssm import CausalDAGSSM  # noqa: E402
from eval.readouts import pearsonr  # noqa: E402
from eval.stats import bootstrap_ci, exact_sign_flip_test, cohens_d_paired  # noqa: E402


def near_far_split(centroids, stim, radius_mm):
    """Near = parcels within ``radius_mm`` of the stim centroid (excl. stim); far = the rest."""
    d = np.linalg.norm(centroids - centroids[stim], axis=1)
    near = (d <= radius_mm); near[stim] = False
    far = (d > radius_mm)
    return near, far


@torch.no_grad()
def causal_near_to_far(model, near_energy, near_mask, steps):
    """Seed the latent with the measured NEAR energy, roll the trained graph, read FAR energy."""
    d = near_energy.shape[0]
    h0 = torch.zeros(d)
    h0[near_mask] = near_energy[near_mask]
    H = model.ssm.rollout(h0.unsqueeze(0), steps, model.A)[0]      # [steps, d]
    return H.pow(2).mean(dim=0).clamp_min(1e-12).sqrt()           # [d] far energy proxy


def run(cfg, proc, args):
    d = cfg.parcellation.d
    centroids = np.load(Path(proc) / "parcel_centroids_mni.npy")
    m1 = InterventionalDataset(proc, site_filter={"M1_L"})
    records = [m1[i] for i in range(len(m1))]
    ckpt_dir = Path(proc) / "checkpoints"

    # measured per-region energy topography for every subject
    meas_energy = [r["region_tep"].pow(2).mean(dim=-1).clamp_min(1e-12).sqrt() for r in records]

    folds = []
    for k in range(len(records)):
        ck_path = ckpt_dir / f"exp1b_fold{k:02d}.pt"
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, weights_only=False)
        held_idx = int(ck.get("held_idx", k))
        held = records[held_idx]
        stim = int(held["stim_parcel"])
        model = CausalDAGSSM(d, variant=cfg.dag.variant, input_dim=cfg.model.input_dim,
                             init_scale=cfg.model.init_state_scale)
        model.load_state_dict(ck["state_dict"])

        near, far = near_far_split(centroids, stim, args.radius_mm)
        far_t = torch.from_numpy(far)
        if far_t.sum() < 3:
            continue
        far_meas = meas_energy[held_idx][far_t]

        causal_far = causal_near_to_far(model, meas_energy[held_idx],
                                        torch.from_numpy(near), args.steps)[far_t]
        # leave-one-out group-mean distal topography (the fair Bayes baseline)
        tmpl = torch.stack([meas_energy[j] for j in range(len(records))
                            if j != held_idx]).mean(0)[far_t]

        folds.append({
            "fold": k, "held_idx": held_idx, "subject": str(held["subject"]),
            "n_far": int(far_t.sum()),
            "causal_r": pearsonr(causal_far, far_meas),
            "template_r": pearsonr(tmpl, far_meas),
        })
        print(f"  fold {k:02d} {folds[-1]['subject']:>12}: "
              f"causal r={folds[-1]['causal_r']:+.3f}  "
              f"template r={folds[-1]['template_r']:+.3f}  (n_far={folds[-1]['n_far']})",
              flush=True)

    c = [f["causal_r"] for f in folds]
    t = [f["template_r"] for f in folds]
    mean_d, lo, hi = bootstrap_ci([ci - ti for ci, ti in zip(c, t)])
    p = exact_sign_flip_test(c, t)
    dd = cohens_d_paired(c, t)
    print("\n===== Exp-7 spatial near->far conditioning (GAP-3 attempt) =====")
    print(f"folds={len(folds)} radius={args.radius_mm}mm | causal {np.mean(c):+.3f} | "
          f"template {np.mean(t):+.3f} | diff {mean_d:+.3f} [{lo:+.3f},{hi:+.3f}] "
          f"p={p:.3f} d={dd:+.2f}")
    win = (p < 0.05) and (dd > 0) and ((lo > 0) or (hi < 0)) and (np.mean(np.array(c) > np.array(t)) > 0.5)
    print("VERDICT:", "BEATS the group mean (subject conditioning works)" if win
          else "ties/loses the group mean (GAP-3 unsolved with this data)")

    out = {"experiment": "7_spatial_conditioning", "metric": "far_energy_r",
           "radius_mm": args.radius_mm, "steps": args.steps, "n_folds": len(folds),
           "causal_r_mean": float(np.mean(c)), "template_r_mean": float(np.mean(t)),
           "diff_mean": float(mean_d), "ci": [lo, hi], "sign_flip_p": p,
           "cohens_d": dd, "win": bool(win), "folds": folds}
    out_path = Path(args.out) if args.out else Path(proc) / "exp7_results.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print("results ->", out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-mm", type=float, default=40.0,
                    help="near-field radius around the stim parcel (MNI mm)")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config()
    return run(cfg, cfg.paths.processed_dir, args)


if __name__ == "__main__":
    sys.exit(main())
