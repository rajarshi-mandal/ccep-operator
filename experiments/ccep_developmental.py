"""EXTENSION — developmental effects: does the stimulation-response operator change with age?

ds004080 (ccepAge) spans ages ~4-51 and was collected to study developmental changes in cortico-
cortical transmission. We ask whether our operator-level quantities vary with age across its 74
subjects: (a) individualized predictability (combo r), (b) the network-beyond-locality contribution,
(c) operator symmetry corr(W, Wᵀ). A developmental trend would be a genuinely new, mechanistic
finding for the journal version (none of it is in the conference paper).

Output: reports/developmental.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from data.ccep_pipeline import CCEPSubject  # noqa
import ccep_loso as L  # noqa


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan, np.nan
    rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok]))
    r = np.corrcoef(rx, ry)[0, 1]
    n = ok.sum()
    t = r * np.sqrt((n - 2) / max(1e-9, 1 - r * r))
    # two-sided p via normal approx
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    return float(r), float(p)


def operator_symmetry(cs, keep):
    n_c = len(cs.contacts); W = np.zeros((n_c, n_c)); cnt = np.zeros(n_c)
    for s in keep:
        r = np.nan_to_num(cs.responses[s])
        for a in cs.stim_idx[s]:
            if a >= 0:
                W[a] += r; cnt[a] += 1
    nz = cnt > 0; W[nz] /= cnt[nz, None]
    m = (W != 0) & (W.T != 0)
    if m.sum() < 20:
        return np.nan
    return float(np.corrcoef(W[m], W.T[m])[0, 1])


def main():
    ages = {}
    with open(ROOT / "reports" / "ds004080_participants.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                ages[row["participant_id"]] = float(row["age"])
            except (ValueError, KeyError):
                pass
    caches = [(d, c) for d, c in L.all_caches() if d == "ds004080"]
    rec = {"age": [], "combo": [], "within": [], "net": [], "sym": [], "nsites": [], "trials": []}
    print(f"{'subject':22s} {'age':>4} {'combo':>7} {'net':>7} {'sym':>6}")
    for ds, c in caches:
        cs = CCEPSubject.load(str(c))
        age = ages.get(cs.subject)
        if age is None:
            continue
        e = L.eval_subject(cs); i = L.incremental_subject(cs)
        if e is None or i is None:
            continue
        s, nk = e
        keep = [k for k in range(len(cs.sites)) if np.isfinite(cs.reliability[k]) and cs.reliability[k] >= L.REL_MIN]
        rec["age"].append(age); rec["combo"].append(s["combo"]); rec["within"].append(s["within_mean"])
        rec["net"].append(i["stim_knn"]); rec["sym"].append(operator_symmetry(cs, keep))
        rec["nsites"].append(nk); rec["trials"].append(float(np.median(cs.n_trials)))
        print(f"{cs.subject:22s} {age:4.0f} {s['combo']:7.3f} {i['stim_knn']:7.3f} {rec['sym'][-1]:6.3f}")

    n = len(rec["age"])
    out = {"n": n, "records": rec, "age_corr": {}}
    print(f"\n=== correlation with AGE (Spearman, n={n}) ===")
    for k in ["combo", "within", "net", "sym"]:
        # partial out #sites and trials confounds by correlating on residuals? report raw + note.
        r, p = _spearman(rec["age"], rec[k])
        out["age_corr"][k] = {"rho": r, "p": p}
        flag = "  <-- age effect" if np.isfinite(p) and p < 0.05 else ""
        print(f"  age vs {k:8s}: rho={r:+.3f}  p={p:.3g}{flag}")
    # control: age also correlates with #sites/trials? report to flag confounds
    for k in ["nsites", "trials"]:
        r, p = _spearman(rec["age"], rec[k]); out["age_corr"][k] = {"rho": r, "p": p}
        print(f"  (control) age vs {k:6s}: rho={r:+.3f} p={p:.3g}")
    (ROOT / "reports" / "developmental.json").write_text(json.dumps(out, indent=2))
    print("saved reports/developmental.json")


if __name__ == "__main__":
    main()
