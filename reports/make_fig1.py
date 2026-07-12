"""Figure 1 — The individualized propagation operator: concept, data, and core result."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fig_lib import (ROOT, REP, INK, TEAL, AMBER, NAVY, CORAL, GREY, LGREY, VIOLET, GREEN,
                     CMAP_AMP, CMAP_TRACE, panel, load_subject, per_subject_models, glass_markers)
import sys; sys.path.insert(0, str(ROOT/"experiments"))
import ccep_operator_v2 as V2
import ccep_loso as L

SUB = "sub-ccepAgeUMCU48"     # coordinate-rich developmental subject
cs, tr = load_subject(SUB)
keep = np.arange(len(cs.sites))[(np.isfinite(cs.reliability)) & (cs.reliability >= 0.3)]

# choose an example stim site with a clear, reliable, spatially-spread response
sc = [(int(np.isfinite(cs.responses[s]).sum()), s) for s in keep]
site = sorted(sc)[-8][1]
resp = cs.responses[site]
xyz = cs.contact_xyz

fig = plt.figure(figsize=(11, 6.6))
gs = GridSpec(2, 12, figure=fig, hspace=0.55, wspace=1.6, left=0.045, right=0.985, top=0.93, bottom=0.09)

# ---------- A: electrode montage + stim site ----------
axA = fig.add_subplot(gs[0, 0:3]); panel(axA, "A")
stim_c = [a for a in cs.stim_idx[site] if a >= 0]
glass_markers(axA, xyz, np.log1p(np.nan_to_num(resp)), CMAP_AMP, display="z",
              hl=stim_c[0] if stim_c else None, size=18)
axA.set_title("Stimulate one site,\nrecord the whole array", fontsize=8.5)

# ---------- B: CCEP traces (butterfly) + heatmap ----------
gsB = gs[0, 3:6].subgridspec(2, 1, hspace=0.15)
axB1 = fig.add_subplot(gsB[0]); panel(axB1, "B")
if tr is not None:
    tsite = list(tr["sites"]).index(cs.sites[site]) if cs.sites[site] in list(tr["sites"]) else 0
    T = tr["t_ms"]; W = tr["traces"][tsite]        # [n_c, T]
    order = np.argsort(-np.nan_to_num(resp))
    for c in order[:60]:
        axB1.plot(T, W[c], color=CORAL if resp[c] > np.nanpercentile(resp,80) else GREY,
                  lw=0.35, alpha=0.7)
    axB1.axvline(0, color=INK, lw=0.8, ls="--"); axB1.set_xlim(-5, 300)
    axB1.set_xticklabels([]); axB1.set_ylabel("µV", fontsize=7)
    axB1.set_title("Cortico-cortical evoked potentials", fontsize=8.5)
    axB2 = fig.add_subplot(gsB[1])
    vlim = np.nanpercentile(np.abs(W), 98)
    axB2.imshow(W[order], aspect="auto", cmap=CMAP_TRACE, vmin=-vlim, vmax=vlim,
                extent=[T[0], T[-1], len(order), 0])
    axB2.axvline(0, color=INK, lw=0.8, ls="--"); axB2.set_xlim(-5, 300)
    axB2.set_xlabel("time (ms)", fontsize=7); axB2.set_ylabel("contacts", fontsize=7)

# ---------- C: the operator (connectome) ----------
axC = fig.add_subplot(gs[0, 6:9]); panel(axC, "C")
A = V2._build_operator(cs, list(keep), "symmetric")
o = np.argsort(-A.sum(0))
im = axC.imshow(A[np.ix_(o, o)], cmap=CMAP_AMP, vmin=0, vmax=np.nanpercentile(A, 99))
axC.set_title("Effective-connectivity\noperator A", fontsize=8.5)
axC.set_xlabel("contact"); axC.set_ylabel("contact"); axC.set_xticks([]); axC.set_yticks([])
cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)

# ---------- D: predicted vs measured (held-out site) ----------
gsD = gs[0, 9:12].subgridspec(2, 2, height_ratios=[1.35,1], hspace=0.1, wspace=0.1)
train = [t for t in keep if t != site]
sg, al, stp, md = V2._best_params(cs, train)
P = V2._build_operator(cs, train, md)
pred = V2.predict_operator_v2(cs, site, train, sg, al, stp, md, P=P)
mask = L._valid_mask(cs, site, train)
axDm = fig.add_subplot(gsD[0,0]); panel(axDm, "D")
glass_markers(axDm, xyz, np.log1p(np.nan_to_num(resp)), CMAP_AMP, display="z", size=12)
axDm.set_title("measured", fontsize=7)
axDp = fig.add_subplot(gsD[0,1])
glass_markers(axDp, xyz, np.log1p(np.clip(pred,0,None)), CMAP_AMP, display="z", size=12)
axDp.set_title("predicted", fontsize=7)
axDs = fig.add_subplot(gsD[1,:])
mm = mask & np.isfinite(resp) & np.isfinite(pred)
r = L.topo_r(pred, resp, mask)
axDs.scatter(pred[mm], resp[mm], s=6, color=TEAL, alpha=0.6, edgecolor="none")
axDs.set_xlabel("predicted", fontsize=7); axDs.set_ylabel("measured", fontsize=7)
axDs.text(0.05,0.86,f"held-out site\nr = {r:.2f}", transform=axDs.transAxes, fontsize=7)
axDs.set_xticks([]); axDs.set_yticks([])

# ---------- E: model comparison ----------
axE = fig.add_subplot(gs[1, 0:5]); panel(axE, "E")
labels=["within-\nmean","operator\nv1","distance","stim-\nkNN","operator","combo","ensemble"]
vals=[0.235,0.622,0.641,0.688,0.710,0.730,0.743]
err=[0.02]*7
cols=[GREY,LGREY,NAVY,"#5B7FA6",TEAL,AMBER,GREEN]
axE.axhspan(0.78,0.85,color=LGREY,alpha=0.5,zorder=0)
axE.text(6.4,0.815,"recoverable\nceiling",fontsize=6,color=INK,va="center")
axE.bar(range(7),vals,yerr=err,color=cols,width=0.7,zorder=3,error_kw=dict(lw=0.8,capsize=2))
for i,v in enumerate(vals): axE.text(i,v+0.03,f"{v:.3f}",ha="center",fontsize=6.5,fontweight="bold")
axE.axhline(0.235,color=GREY,ls=":",lw=0.8)
axE.set_ylim(0,0.9); axE.set_xticks(range(7)); axE.set_xticklabels(labels,fontsize=6.5)
axE.set_ylabel("held-out topography r (n=93)")
axE.set_title("The operator beats proximity", fontsize=8.5, loc="left")

# ---------- F: per-subject strip, operator vs distance ----------
axF = fig.add_subplot(gs[1, 5:12]); panel(axF, "F")
M = per_subject_models()
d, v2 = M["distance"], M["op_v2"]
order = np.argsort(v2)
x = np.arange(len(v2))
for i,xi in enumerate(order):
    axF.plot([i,i],[d[xi],v2[xi]], color=LGREY, lw=0.5, zorder=1)
axF.scatter(x, d[order], s=8, color=NAVY, label="distance", zorder=2)
axF.scatter(x, v2[order], s=8, color=TEAL, label="operator", zorder=3)
wins=int((v2>d).sum())
axF.set_xlim(-1,len(v2)); axF.set_ylim(0.2,0.95)
axF.set_xlabel("patients (sorted by operator r)"); axF.set_ylabel("held-out r")
axF.legend(loc="lower right", ncol=2)
axF.set_title(f"Operator > distance in {wins}/93 patients", fontsize=8.5, loc="left")

fig.savefig(REP/"figs"/"Figure_1.png", bbox_inches="tight", facecolor="white")
print("saved Figure_1.png  (example site r=%.2f, wins=%d/93)"%(r,wins))
