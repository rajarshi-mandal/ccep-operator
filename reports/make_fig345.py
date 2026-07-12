"""Figures 3 (control+clinical), 4 (generalization+dynamics), 5 (external validation)."""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fig_lib import (ROOT, REP, INK, TEAL, AMBER, NAVY, CORAL, GREY, LGREY, VIOLET, GREEN,
                     CMAP_AMP, CMAP_DIV, panel, load_subject, glass_markers)
import sys; sys.path.insert(0, str(ROOT/"experiments"))
import ccep_operator_v2 as V2
from ccep_control import controllability
J=lambda n: json.loads((REP/f"{n}.json").read_text())

def spearman_line(ax,x,y,color):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y)
    x,y=x[ok],y[ok]
    b,a=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),20)
    ax.plot(xs,b*xs+a,color=color,lw=1.5)
    r=np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1]
    return r

# ============================ FIGURE 3 — control + clinical ============================
ctl=J("control"); soz=J("soz")
fig=plt.figure(figsize=(11,3.2)); gs=GridSpec(1,5,figure=fig,wspace=0.5,left=0.05,right=0.99,top=0.86,bottom=0.2)

# A controllability->reach strip
axA=fig.add_subplot(gs[0,0]); panel(axA,"A")
rho=np.array([p["ctrl_reach_rho"] for p in ctl["per_subject"] if p.get("ctrl_reach_rho") is not None])
o=np.argsort(rho)
axA.vlines(np.arange(len(rho)),0,rho[o],color=LGREY,lw=0.5)
axA.scatter(np.arange(len(rho)),rho[o],s=8,color=TEAL,zorder=3)
axA.axhline(0,color=CORAL,ls="--",lw=0.8)
axA.set_ylim(-0.05,0.9); axA.set_xlabel("patients"); axA.set_ylabel(r"controllability$\to$reach $\rho$")
axA.set_title(f"Controllability predicts\nuntested reach ({int((rho>0).sum())}/93)",fontsize=8,loc="left")

# B controllability on brain
axB=fig.add_subplot(gs[0,1]); panel(axB,"B")
cs,_=load_subject("sub-ccepAgeUMCU48")
keep=np.arange(len(cs.sites))[(np.isfinite(cs.reliability))&(cs.reliability>=0.3)]
A=V2._build_operator(cs,list(keep),"symmetric"); avg,_=controllability(A)
glass_markers(axB,cs.contact_xyz,avg,CMAP_AMP,display="z",size=16)
axB.set_title("Average controllability\n(example patient)",fontsize=8)

# C SOZ AUC
axC=fig.add_subplot(gs[0,2]); panel(axC,"C")
b=soz["soz"]["mv_amp_geom"]["auc"]; op=soz["soz"]["mv_operator_only"]["auc"]; f=soz["soz"]["mv_full"]["auc"]
axC.bar([0,1,2],[b,op,f],color=[GREY,TEAL,AMBER],width=0.62,zorder=3)
for i,v in enumerate([b,op,f]): axC.text(i,v+0.004,f"{v:.3f}",ha="center",fontsize=6.5,fontweight="bold")
axC.axhline(0.5,color=CORAL,ls="--",lw=0.8); axC.set_ylim(0.5,0.66)
axC.set_xticks([0,1,2]); axC.set_xticklabels(["amp+\ngeom","operator\nonly","full"],fontsize=6.5)
axC.set_ylabel("seizure-onset-zone AUC")
axC.set_title("Operator localizes\nepileptogenic tissue",fontsize=8,loc="left")

# D SOZ univariate features
axD=fig.add_subplot(gs[0,3]); panel(axD,"D")
uni=soz["soz"]["univariate"]; order=["afferent_strength","efferent_strength","asymmetry","avg_ctrb","modal_ctrb"]
labs=["affer.","effer.","asym.","avg\nctrb","modal\nctrb"]; au=[uni[k]["auc"] for k in order]
cols=[GREY,TEAL,TEAL,TEAL,VIOLET]
axD.bar(range(5),au,color=cols,width=0.7,zorder=3)
for i,v in enumerate(au): axD.text(i,v+0.005 if v>.5 else v-0.03,f"{v:.2f}",ha="center",fontsize=6,fontweight="bold")
axD.axhline(0.5,color=CORAL,ls="--",lw=0.8); axD.set_ylim(0.38,0.68)
axD.set_xticks(range(5)); axD.set_xticklabels(labs,fontsize=6.5); axD.set_ylabel("within-subject AUC")
axD.set_title("SOZ = high avg /\nlow modal ctrb",fontsize=8,loc="left")

# E targeting honest tie
axE=fig.add_subplot(gs[0,4]); panel(axE,"E")
tg=ctl["targeting"]; mm=tg["capture_model_mean"]; dd=tg["capture_distance_mean"]
axE.bar([0,1],[mm,dd],color=[TEAL,NAVY],width=0.55,zorder=3)
for i,v in enumerate([mm,dd]): axE.text(i,v+0.006,f"{v:.2f}",ha="center",fontsize=7,fontweight="bold")
axE.set_ylim(0,0.5); axE.set_xticks([0,1]); axE.set_xticklabels(["operator","distance"],fontsize=7)
axE.set_ylabel("target-capture (1=oracle)")
axE.set_title("Single-site targeting:\nhonest tie",fontsize=8,loc="left")
fig.savefig(REP/"figs"/"Figure_3.png",bbox_inches="tight",facecolor="white"); plt.close(fig); print("Figure_3 ok")

# ============================ FIGURE 4 — generalization + dynamics ============================
dev=J("developmental"); cs4=J("coldstart"); lds=J("lds")
rec=dev["records"]
fig=plt.figure(figsize=(11,3.2)); gs=GridSpec(1,5,figure=fig,wspace=0.5,left=0.05,right=0.99,top=0.86,bottom=0.2)
age=np.array(rec["age"],float); pred=np.array(rec["combo"],float); sym=np.array(rec["sym"],float)

axA=fig.add_subplot(gs[0,0]); panel(axA,"A")
axA.scatter(age,pred,s=10,color=TEAL,alpha=0.7,edgecolor="none")
r=spearman_line(axA,age,pred,CORAL)
axA.set_xlabel("age (years)"); axA.set_ylabel("predictability r")
axA.set_title(f"Predictability rises\nwith age ($\\rho$={r:+.2f})",fontsize=8,loc="left")

axB=fig.add_subplot(gs[0,1]); panel(axB,"B")
axB.scatter(age,sym,s=10,color=VIOLET,alpha=0.7,edgecolor="none")
r=spearman_line(axB,age,sym,CORAL)
axB.set_xlabel("age (years)"); axB.set_ylabel("operator reciprocity")
axB.set_title(f"Reciprocity rises\nwith age ($\\rho$={r:+.2f})",fontsize=8,loc="left")

axC=fig.add_subplot(gs[0,2]); panel(axC,"C")
names=["group\nmarginal","group\noperator","own\ngeometry"]
vals=[cs4["coldstart"]["group_marginal"]["mean"],cs4["coldstart"]["group_op"]["mean"],cs4["coldstart"]["distance"]["mean"]]
axC.bar(range(3),vals,color=[GREY,TEAL,NAVY],width=0.62,zorder=3)
for i,v in enumerate(vals): axC.text(i,v+0.01,f"{v:.2f}",ha="center",fontsize=6.5,fontweight="bold")
axC.set_ylim(0,0.75); axC.set_xticks(range(3)); axC.set_xticklabels(names,fontsize=6.5)
axC.set_ylabel("predict unseen patient, r")
axC.set_title("Cold-start: transferable\npart is geometric",fontsize=8,loc="left")
axi=axC.inset_axes([0.55,0.14,0.4,0.34]); ks=[0,1,3,5,10]
axi.plot(ks,[cs4["fewshot"][str(k)] for k in ks],"-o",color=AMBER,ms=2.5,lw=1.2)
axi.set_title("few-shot",fontsize=6); axi.tick_params(labelsize=5); axi.set_xlabel("pilot",fontsize=5)

axD=fig.add_subplot(gs[0,3]); panel(axD,"D")
an=J("animal"); ks=sorted(int(k) for k in an["reliability_vs_trials"]); rv=[an["reliability_vs_trials"][str(k)] for k in ks]
axD.plot(ks,rv,"-o",color=TEAL,lw=1.8,ms=5,zorder=3)
axD.set_ylim(0,0.85); axD.set_xlabel("trials per site"); axD.set_ylabel("split-half reliability")
axD.set_title("Dense SITES, not trials,\nlimit identifiability",fontsize=8,loc="left")
axD.text(0.5,0.28,f"but {int(an['sites_per_session']['frac_under_6']*100)}% of sessions <6 sites\n(median {an['sites_per_session']['median']:.0f})",
         transform=axD.transAxes,ha="center",fontsize=6)

axE=fig.add_subplot(gs[0,4]); panel(axE,"E")
l=lds["fulltrace"]["lds_mean"]; s=lds["fulltrace"]["sep_mean"]
axE.bar([0,1],[l,s],color=[VIOLET,GREY],width=0.55,zorder=3)
for i,v in enumerate([l,s]): axE.text(i,v+0.008,f"{v:.2f}",ha="center",fontsize=7,fontweight="bold")
axE.set_ylim(0,0.6); axE.set_xticks([0,1]); axE.set_xticklabels(["dynamical","separable"],fontsize=7)
axE.set_ylabel("full-trace r (n=37)")
axE.set_title("CCEP largely separable;\noperator adds timing",fontsize=8,loc="left")
axE.text(0.5,0.9,f"timing $\\rho$={lds['latency_rho_mean']:.2f}, 34/37",transform=axE.transAxes,ha="center",fontsize=6)
fig.savefig(REP/"figs"/"Figure_4.png",bbox_inches="tight",facecolor="white"); plt.close(fig); print("Figure_4 ok")

# ============================ FIGURE 5 — external validation ============================
ft=J("ftract"); cm=J("ftract_crossmap"); st=J("struct"); tm=J("tmseeg")
fig=plt.figure(figsize=(11,3.2)); gs=GridSpec(1,4,figure=fig,wspace=0.42,left=0.055,right=0.99,top=0.86,bottom=0.2)

axA=fig.add_subplot(gs[0,0]); panel(axA,"A")
ov=cm["operator_validation"]
axA.bar([0,1],[ov["rho_amplitude"],ov["rho_probability"]],color=[TEAL,NAVY],width=0.55,zorder=3)
for i,v in enumerate([ov["rho_amplitude"],ov["rho_probability"]]): axA.text(i,v+0.005,f"{v:.2f}",ha="center",fontsize=7,fontweight="bold")
axA.set_ylim(0,0.32); axA.set_xticks([0,1]); axA.set_xticklabels(["vs\namplitude","vs\nprobability"],fontsize=6.5)
axA.set_ylabel(r"our operator vs F-TRACT $\rho$")
axA.set_title(f"Recovers 780-patient\nstructure (n$\\sim$10$^4$ pairs)",fontsize=8,loc="left")

axB=fig.add_subplot(gs[0,1]); panel(axB,"B")
loc=-ft["amplitude_locality"]["rho_amp_distance"]; rc=ft["directionality"]["amplitude_reciprocity"]
axB.bar([0,1],[loc,rc],color=[TEAL,VIOLET],width=0.5,zorder=3)
for i,v in enumerate([loc,rc]): axB.text(i,v+0.006,f"{v:.2f}",ha="center",fontsize=7,fontweight="bold")
axB.set_ylim(0,0.42); axB.set_xticks([0,1]); axB.set_xticklabels(["locality\n$|\\rho|$","reciprocal\ndominance"],fontsize=6.5)
axB.set_ylabel("F-TRACT effect (780 pt)")
axB.set_title("Locality & reciprocity\nreplicate",fontsize=8,loc="left")

axC=fig.add_subplot(gs[0,2]); panel(axC,"C")
raw=st["ages_15_100"]["rho_prob_struct"]; part=st["ages_15_100"]["partial_prob_struct_given_dist"]
axC.bar([0,1],[raw,part],color=[NAVY,GREY],width=0.5,zorder=3)
for i,v in enumerate([raw,part]): axC.text(i,v+0.005,f"{v:.2f}",ha="center",fontsize=7,fontweight="bold")
axC.set_ylim(0,0.26); axC.set_xticks([0,1]); axC.set_xticklabels(["raw","| distance"],fontsize=6.5)
axC.set_ylabel(r"CCEP $\sim$ DWI structural $\rho$")
axC.set_title("Follows structure\nvia geometry",fontsize=8,loc="left")

axD=fig.add_subplot(gs[0,3]); panel(axD,"D")
per=tm.get("per_subject",[])
if per:
    vals=[p.get("partial",p.get("raw",0)) for p in per]
    axD.axhline(0,color=CORAL,ls="--",lw=0.8)
    axD.bar(range(len(vals)),sorted(vals),color=[TEAL if v>0 else GREY for v in sorted(vals)],width=0.7,zorder=3)
axD.set_ylim(-0.45,0.45); axD.set_xlabel("TMS-EEG subjects")
axD.set_ylabel(r"CCEP$\to$TEP $\rho$ | distance")
axD.set_title("TMS-EEG bridge:\ninconclusive (1/6)",fontsize=8,loc="left")
fig.savefig(REP/"figs"/"Figure_5.png",bbox_inches="tight",facecolor="white"); plt.close(fig); print("Figure_5 ok")
