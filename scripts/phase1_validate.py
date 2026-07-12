"""Phase 1 validation + §5.4 ceiling re-estimation on the real Schaefer-100 derivatives.

Three things the cache must tell us before any modeling:
  1. QC / spatial ceiling — median split-half reliability of the Schaefer-100 evoked
     topographies (the §5.4 re-estimate of the 0.76 native-KMeans proxy). This bounds r.
  2. Baseline strength — how well the population-mean site template already predicts each
     subject's topography. This is the thing the old subject-blind model tied and which
     subject conditioning must beat.
  3. The decisive ratio — reliable signal that is *subject-specific deviation* from the
     template vs. the template itself. If essentially all reliable variance IS the
     template (deviation ~ noise), the ceiling is not practically breakable on this target;
     if reliability clearly exceeds template-correlation, there is exploitable subject
     structure. This is the cheapest honest read on whether Phase 2+ can win.

Writes reports/PHASE1_CEILING_DS005498.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.ds005498_pipeline import DS005498Cache  # noqa: E402


def _r(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 1e-9 and db > 1e-9 else np.nan


def summ(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return dict(n=len(v), median=float(np.median(v)), mean=float(np.mean(v)),
                q25=float(np.percentile(v, 25)), q75=float(np.percentile(v, 75)))


def main():
    cache = DS005498Cache(qc_filter=True)
    recs = cache.records
    subs, sites = cache.subjects(), cache.sites()
    rel = [r.reliability for r in recs]

    # baseline: r(subject topo, leave-this-subject-out site template)
    base_r, dev_resid = [], []
    for r in recs:
        tmpl = cache.site_template(r.site_name, exclude_subject=r.subject)
        br = _r(r.topo, tmpl)
        base_r.append(br)
        # subject deviation = topo minus its projection on the template
        t = tmpl - tmpl.mean(); t = t / (np.linalg.norm(t) + 1e-9)
        x = r.topo - r.topo.mean()
        resid = x - (x @ t) * t
        dev_resid.append(np.linalg.norm(resid) / (np.linalg.norm(x) + 1e-9))  # frac energy off-template

    rel_s, base_s, dev_s = summ(rel), summ(base_r), summ(dev_resid)

    # decisive read: reliable signal vs template-explained signal
    # reliability ~ ceiling correlation; base_r ~ template correlation. Gap => subject room.
    gap = rel_s["median"] - base_s["median"]

    lines = []
    P = lines.append
    P("# Phase 1 — ceiling re-estimation on real Schaefer-100 derivatives (ds005498)\n")
    P(f"- Subjects: **{len(subs)}**, QC-passing (subject,site) records: **{len(recs)}**, "
      f"sites: {len(sites)}")
    P(f"- Registration: route B (affine overlay of MNI Schaefer-100 onto native EPI)\n")
    P("## 1. Spatial-topography ceiling (§5.4 re-estimate)")
    P(f"- Split-half reliability (Spearman-Brown), Schaefer-100: "
      f"**median {rel_s['median']:.3f}** [IQR {rel_s['q25']:.3f}–{rel_s['q75']:.3f}]")
    P(f"- Phase-0 native-KMeans proxy was 0.76 → real-derivative ceiling is "
      f"{'consistent' if abs(rel_s['median']-0.76)<0.12 else 'shifted'} "
      f"({rel_s['median']:.2f} vs 0.76). **Use {rel_s['median']:.2f} as the spatial target ceiling.**\n")
    P("## 2. Population-mean baseline (the thing to beat)")
    P(f"- r(subject topography, leave-subject-out site template): "
      f"**median {base_s['median']:.3f}** [IQR {base_s['q25']:.3f}–{base_s['q75']:.3f}]")
    P(f"- Off-template energy fraction (subject deviation magnitude): "
      f"median {dev_s['median']:.3f}\n")
    P("## 3. Decisive read — is the ceiling practically breakable?")
    P(f"- Ceiling (reliable signal) median r ≈ **{rel_s['median']:.2f}**; "
      f"template already explains median r ≈ **{base_s['median']:.2f}**.")
    P(f"- Headroom for subject-specific structure ≈ **{gap:+.2f} r**.")
    if gap > 0.15:
        P("- **Interpretation: there is reliable subject-specific structure beyond the "
          "template.** Subject conditioning (Phase 3/§12) has room to beat the baseline. "
          "Proceed to LOSO-WS.")
    elif gap > 0.05:
        P("- **Interpretation: modest headroom.** A win is possible but will be small; "
          "target the *deflated* topography (project out the shared mode) and report r "
          "relative to ceiling.")
    else:
        P("- **Interpretation: little headroom — most reliable signal IS the template.** "
          "Same risk as the old NULL; the deviation target is near noise. Treat a Phase-2 "
          "win as unlikely on raw topography; pivot to deflated/PC targets early.")
    P("")
    P("## 4. CAVEAT — route-B registration confounds the cross-subject baseline")
    P("The site template averages topographies *across* subjects, so it needs accurate "
      "cross-subject anatomical correspondence. Route B (affine overlay, no subject-specific "
      "registration) degrades that correspondence, which **deflates the template** and can "
      "masquerade as subject-specificity. Therefore the +headroom above is an UPPER bound, "
      "not a confirmed win.")
    P("- Mitigations baked into the eval: LOSO-**within-subject** conditions on the same "
      "subject's other sites + rest in that subject's own parcel space, so it does not rely "
      "on cross-subject correspondence; the Phase-5 ablation (zero the subject channel → r "
      "must collapse) is the real test that the signal is subject-specific and not "
      "registration noise.")
    P("- Before trusting a baseline-beating claim, either (a) confirm the win survives a "
      "*within-subject* mean baseline, or (b) upgrade to route A (EPI→T1→MNI) and re-check "
      "the template strength.")
    P("")

    out = Path("reports/PHASE1_CEILING_DS005498.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"-> wrote {out}")


if __name__ == "__main__":
    main()
