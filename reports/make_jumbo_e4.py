#!/usr/bin/env python3
"""JUMBO figure E4 — Tier-1 next-level extensions: control, cold-start, SOZ biomarker, dynamics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

R = Path(__file__).resolve().parent
OUT = R / "jumbo"; OUT.mkdir(exist_ok=True)
INK="#0E2233"; TEAL="#17B2A3"; AMBER="#F4A300"; VIOLET="#7C6BD8"; CORAL="#EF6F6C"; BLUE="#3E7CB1"; GREY="#9AA7B2"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica Neue","Arial","DejaVu Sans"],
    "font.size":9.5,"axes.titlesize":11,"axes.labelsize":9.5,"axes.edgecolor":INK,"axes.linewidth":0.9,
    "xtick.color":INK,"ytick.color":INK,"text.color":INK,"axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":300,"savefig.dpi":300})

ctl = json.loads((R/"control.json").read_text())
soz = json.loads((R/"soz.json").read_text())
cs  = json.loads((R/"coldstart.json").read_text())
lds = json.loads((R/"lds.json").read_text())

def plabel(ax,l): ax.text(-0.15,1.09,l,transform=ax.transAxes,fontsize=13,fontweight="bold",color=INK,va="top")

fig=plt.figure(figsize=(12,12)); gs=GridSpec(3,2,figure=fig,hspace=0.46,wspace=0.30,left=0.08,right=0.97,top=0.93,bottom=0.05)
fig.suptitle("Figure E4 | Next-level extensions: from prediction to control, generalization, and clinic",
    x=0.08,ha="left",fontsize=13,fontweight="bold")

# (a) T1.1 controllability -> reach (per-subject rho, 93/93 positive)
ax=fig.add_subplot(gs[0,0]); plabel(ax,"a")
rho=[p["ctrl_reach_rho"] for p in ctl["per_subject"] if p.get("ctrl_reach_rho") is not None]
ax.hist(rho,bins=18,color=TEAL,edgecolor=INK,linewidth=0.5,zorder=3)
ax.axvline(0,color=CORAL,ls="--",lw=1.2)
m=ctl["controllability_reach"]["rho_mean"]
ax.axvline(m,color=AMBER,lw=2)
ax.text(0.03,0.93,f"controllability predicts\na site's untested REACH\nmean rho={m:+.2f}, {ctl['controllability_reach']['n_pos']}/{ctl['controllability_reach']['n']} subj >0, p<0.001",
        transform=ax.transAxes,va="top",fontsize=8.6)
ax.set_xlabel("within-subject Spearman ρ (controllability to reach)"); ax.set_ylabel("subjects")
ax.set_title("Stimulation as network control",loc="left")

# (b) T1.3 cold-start bars + few-shot
ax=fig.add_subplot(gs[0,1]); plabel(ax,"b")
names=["group\nmarginal","group\noperator","own\ngeometry"]; vals=[cs["coldstart"]["group_marginal"]["mean"],cs["coldstart"]["group_op"]["mean"],cs["coldstart"]["distance"]["mean"]]
ax.bar(range(3),vals,color=[GREY,TEAL,BLUE],width=0.6,zorder=3)
for i,v in enumerate(vals): ax.text(i,v+0.01,f"{v:.2f}",ha="center",fontsize=8.5,fontweight="bold")
ax.set_ylim(0,0.75); ax.set_xticks(range(3)); ax.set_xticklabels(names,fontsize=8.5)
ax.set_ylabel("predict UNSEEN patient, r (n=74)")
ax.set_title("Cold-start from anatomy alone",loc="left")
ax.text(0.03,0.97,"group operator: r=0.42 with ZERO own\ndata — but geometry alone wins",transform=ax.transAxes,ha="left",va="top",fontsize=8,color=INK)
# few-shot inset (upper-left white space, above the short bars)
axi=ax.inset_axes([0.11,0.50,0.33,0.32])
ks=[0,1,3,5,10]; fs=[cs["fewshot"][str(k)] for k in ks]
axi.plot(ks,fs,"-o",color=AMBER,ms=3,lw=1.5); axi.set_title("few-shot: +pilot sites",fontsize=6.5)
axi.tick_params(labelsize=6); axi.set_xlabel("pilot sites",fontsize=6); axi.set_ylabel("r",fontsize=6)

# (c) T1.4 SOZ multivariate AUC
ax=fig.add_subplot(gs[1,0]); plabel(ax,"c")
b=soz["soz"]["mv_amp_geom"]["auc"]; o=soz["soz"]["mv_operator_only"]["auc"]; f=soz["soz"]["mv_full"]["auc"]
ax.bar([0,1,2],[b,o,f],color=[GREY,TEAL,AMBER],width=0.6,zorder=3)
for i,v in enumerate([b,o,f]): ax.text(i,v+0.006,f"{v:.3f}",ha="center",fontsize=8.5,fontweight="bold")
ax.axhline(0.5,color=CORAL,ls="--",lw=1); ax.text(2.4,0.505,"chance",fontsize=7,color=CORAL)
ax.set_ylim(0.5,0.66); ax.set_xticks([0,1,2]); ax.set_xticklabels(["amplitude\n+geometry","operator\nonly","full"],fontsize=8.5)
ax.set_ylabel("seizure-onset-zone AUC"); ax.set_title("Operator localizes epileptogenic tissue",loc="left")
ax.text(0.5,0.90,f"operator-only beats amplitude+geometry\n(subject-clustered LOSO, perm p≈0)",transform=ax.transAxes,ha="center",va="top",fontsize=8)

# (d) T1.4 SOZ univariate feature AUCs
ax=fig.add_subplot(gs[1,1]); plabel(ax,"d")
uni=soz["soz"]["univariate"]
order=["afferent_strength","efferent_strength","asymmetry","avg_ctrb","modal_ctrb"]
labs=["afferent\n(amplitude)","efferent","asymmetry","avg\nctrb","modal\nctrb"]
au=[uni[k]["auc"] for k in order]; cols=[GREY,TEAL,TEAL,TEAL,VIOLET]
ax.bar(range(5),au,color=cols,width=0.66,zorder=3)
for i,v in enumerate(au): ax.text(i,v+0.006 if v>0.5 else v-0.02,f"{v:.2f}",ha="center",fontsize=8,fontweight="bold")
ax.axhline(0.5,color=CORAL,ls="--",lw=1)
ax.set_ylim(0.38,0.68); ax.set_xticks(range(5)); ax.set_xticklabels(labs,fontsize=8)
ax.set_ylabel("within-subject AUC"); ax.set_title("SOZ = high avg / LOW modal controllability",loc="left")

# (e) T1.2 LDS full-trace vs separable
ax=fig.add_subplot(gs[2,0]); plabel(ax,"e")
l=lds["fulltrace"]["lds_mean"]; s=lds["fulltrace"]["sep_mean"]
ax.bar([0,1],[l,s],color=[VIOLET,GREY],width=0.55,zorder=3)
for i,v in enumerate([l,s]): ax.text(i,v+0.008,f"{v:.2f}",ha="center",fontsize=8.5,fontweight="bold")
ax.set_ylim(0,0.6); ax.set_xticks([0,1]); ax.set_xticklabels(["dynamical\noperator","separable\nbaseline"],fontsize=8.5)
ax.set_ylabel("full-trace prediction r (n=37)"); ax.set_title("CCEP is largely separable",loc="left")
ax.text(0.5,0.9,f"but the operator predicts contact-specific\nCONDUCTION TIMING (ρ={lds['latency_rho_mean']:.2f}, 34/37, p≈0)",
        transform=ax.transAxes,ha="center",va="top",fontsize=8)

# (f) T1.1 targeting honest negative
ax=fig.add_subplot(gs[2,1]); plabel(ax,"f")
mm=ctl["targeting"]["capture_model_mean"]; dd=ctl["targeting"]["capture_distance_mean"]
ax.bar([0,1],[mm,dd],color=[TEAL,BLUE],width=0.5,zorder=3)
for i,v in enumerate([mm,dd]): ax.text(i,v+0.008,f"{v:.2f}",ha="center",fontsize=8.5,fontweight="bold")
ax.set_ylim(0,0.55); ax.set_xticks([0,1]); ax.set_xticklabels(["operator\nchooser","distance\nchooser"],fontsize=8.5)
ax.set_ylabel("target-capture (1=oracle, 0=random)"); ax.set_title("Single-site targeting: honest tie",loc="left")
ax.text(0.5,0.9,"operator nests distance, so geometry already\ncaptures point-targeting (delta n.s.)",transform=ax.transAxes,ha="center",va="top",fontsize=8)

fig.savefig(OUT/"jumbo_E4.png",bbox_inches="tight"); plt.close(fig)
print("saved", OUT/"jumbo_E4.png")
