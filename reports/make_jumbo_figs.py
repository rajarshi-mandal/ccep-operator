#!/usr/bin/env python3
"""Two JUMBO multi-panel figures for the Brain Stimulation journal extension, from real results."""
import json, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

R = Path(__file__).resolve().parent
OUT = R / "jumbo"; OUT.mkdir(exist_ok=True)
INK="#0E2233"; TEAL="#17B2A3"; AMBER="#F4A300"; VIOLET="#7C6BD8"; CORAL="#EF6F6C"
BLUE="#3E7CB1"; GREY="#9AA7B2"; LIGHT="#EAEEF2"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica Neue","Arial","DejaVu Sans"],
    "font.size":9,"axes.titlesize":10.5,"axes.labelsize":9,"axes.edgecolor":INK,"axes.linewidth":0.9,
    "xtick.color":INK,"ytick.color":INK,"text.color":INK,"axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":300,"savefig.dpi":300})

def loadjson(n):
    p=R/n
    return json.loads(p.read_text()) if p.exists() else None

def parse_opv2():
    within,op2=[],[]
    p=R/"_operator_v2_n93.txt"
    if not p.exists(): return np.array([]),np.array([])
    for l in p.read_text().splitlines():
        m=re.match(r'\s*\d{4}/\S+\s+\d+\s+([+\-0-9.]+)\s+([+\-0-9.]+)\s+([+\-0-9.]+)\s+([+\-0-9.]+)',l)
        if m: within.append(float(m.group(1))); op2.append(float(m.group(4)))
    return np.array(within),np.array(op2)

def panel_label(ax,l):
    ax.text(-0.13,1.06,l,transform=ax.transAxes,fontsize=13,fontweight="bold",color=INK,va="top")

# ============================================================ JUMBO A
def jumbo_A():
    fig=plt.figure(figsize=(13,8.2)); gs=GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.32,
        left=0.06,right=0.985,top=0.9,bottom=0.08)
    fig.suptitle("Figure E1 | Individualized prediction, mechanism, and identifiability of the network operator",
        x=0.06,ha="left",fontsize=13,fontweight="bold")

    # (a) headline bars
    ax=fig.add_subplot(gs[0,0]); panel_label(ax,"a")
    names=["within\nmean","operator\nv1","distance","operator\nv2","combo","ensemble"]
    vals=[0.235,0.622,0.641,0.710,0.730,0.743]; cols=[GREY,BLUE,BLUE,TEAL,AMBER,"#0E9E86"]
    ax.bar(range(6),vals,color=cols,width=0.72,zorder=3)
    for i,v in enumerate(vals): ax.text(i,v+0.015,f"{v:.3f}",ha="center",fontsize=7.5,fontweight="bold")
    ax.axhline(0.235,color=GREY,ls=":",lw=0.8); ax.set_ylim(0,0.9); ax.set_xticks(range(6))
    ax.set_xticklabels(names,fontsize=7); ax.set_ylabel("held-out r (LOSO, n=93)")
    ax.set_title("Operator beats locality on its own",loc="left")

    # (b) per-subject win scatter
    ax=fig.add_subplot(gs[0,1]); panel_label(ax,"b")
    wm,op2=parse_opv2()
    if len(wm):
        ax.plot([-.1,1],[-.1,1],"--",color=GREY,lw=1); ax.fill_between([-.1,1],[-.1,1],[1,1],color=TEAL,alpha=0.06)
        ax.scatter(wm,op2,s=26,c=TEAL,edgecolor="white",lw=0.4,alpha=0.9,zorder=3)
        ax.text(0.05,0.9,f"{int((op2>wm).sum())}/{len(op2)} above",fontsize=9,fontweight="bold")
    ax.set_xlim(-.1,1); ax.set_ylim(-.1,1); ax.set_aspect("equal")
    ax.set_xlabel("within-mean r"); ax.set_ylabel("operator r"); ax.set_title("Every patient",loc="left")

    # (c) directionality decomposition
    ax=fig.add_subplot(gs[0,2]); panel_label(ax,"c")
    dn=["afferent\n(rev.)","symmetric","forward","+directed"]; dv=[0.642,0.695,0.703,0.706]
    ax.bar(range(4),dv,color=[GREY,BLUE,TEAL,AMBER],width=0.66,zorder=3)
    for i,v in enumerate(dv): ax.text(i,v+0.004,f"{v:.3f}",ha="center",fontsize=7.5,fontweight="bold")
    ax.set_ylim(0.6,0.74); ax.set_xticks(range(4)); ax.set_xticklabels(dn,fontsize=7)
    ax.set_ylabel("held-out r"); ax.set_title("Directed, but reciprocal-dominant",loc="left")

    # (d) N1 vs N2 network dissociation
    ax=fig.add_subplot(gs[1,0]); panel_label(ax,"d")
    n2=loadjson("n2.json")
    if n2:
        combo=[n2["n1_combo"]["mean"],n2["n2_combo"]["mean"]]; net=[n2["n1_net"]["mean"],n2["n2_net"]["mean"]]
        x=np.arange(2); w=0.36
        ax.bar(x-w/2,combo,w,color=AMBER,label="combo (predictability)",zorder=3)
        ax.bar(x+w/2,net,w,color=TEAL,label="network increment",zorder=3)
        for xi,(a,b) in enumerate(zip(combo,net)):
            ax.text(xi-w/2,a+0.01,f"{a:.2f}",ha="center",fontsize=7); ax.text(xi+w/2,b+0.01,f"{b:.2f}",ha="center",fontsize=7)
        d=n2["net_N2_vs_N1"]
        ax.text(0.5,0.05,f"network N2>N1: +{d['delta']:.3f}, {d['wins']}/{d['n']}, p={d['p']:.0e}",
                transform=ax.transAxes,ha="center",fontsize=7.5,color=INK)
        ax.legend(frameon=False,fontsize=7,loc="upper right")
    ax.set_ylim(0,0.85); ax.set_xticks([0,1]); ax.set_xticklabels(["N1 (early)","N2 (late)"])
    ax.set_ylabel("r"); ax.set_title("The later N2 is more network-driven",loc="left")

    # (e) ceiling by distance
    ax=fig.add_subplot(gs[1,1]); panel_label(ax,"e")
    bins=["near\n0-20","mid\n20-40","far\n40+"]; ceil=[0.905,0.778,0.539]; combo=[0.499,0.573,0.550]
    x=np.arange(3); w=0.38
    ax.bar(x-w/2,ceil,w,color=GREY,label="noise ceiling",zorder=3); ax.bar(x+w/2,combo,w,color=AMBER,label="combo",zorder=3)
    for xi,(a,b) in enumerate(zip(ceil,combo)):
        ax.text(xi-w/2,a+0.01,f"{a:.2f}",ha="center",fontsize=7); ax.text(xi+w/2,b+0.01,f"{b:.2f}",ha="center",fontsize=7)
    ax.set_ylim(0,1.02); ax.set_xticks(x); ax.set_xticklabels(bins,fontsize=8); ax.set_ylabel("r")
    ax.legend(frameon=False,fontsize=7); ax.set_title("Far field is at its noise floor",loc="left")

    # (f) recovery/identifiability
    ax=fig.add_subplot(gs[1,2]); panel_label(ax,"f")
    rec=loadjson("recovery.json")
    if rec:
        sites=rec["sites"]; trials=rec["trials"]; recov=np.array(rec["recovery"]); netr=np.array(rec["net_recovery"]); pred=np.array(rec["prediction"])
        ti=trials.index(10) if 10 in trials else 1
        ax.plot(sites,netr[:,ti],"-o",color=TEAL,ms=4,label="operator recovery (network)")
        ax.plot(sites,pred[:,ti],"-s",color=AMBER,ms=4,label="held-out prediction")
        ax.axvspan(40,90,color=TEAL,alpha=0.06); ax.text(62,0.15,"human\nregime",fontsize=6.5,ha="center",color=GREY)
    ax.set_ylim(0,1.0); ax.set_xlabel("# stimulation sites"); ax.set_ylabel("r")
    ax.legend(frameon=False,fontsize=6.8,loc="lower right"); ax.set_title("Identifiability (simulation)",loc="left")
    fig.savefig(OUT/"jumbo_A.png",bbox_inches="tight"); plt.close(fig); print("jumbo_A")

# ============================================================ JUMBO B
def jumbo_B():
    fig=plt.figure(figsize=(13,8.2)); gs=GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.32,
        left=0.06,right=0.985,top=0.9,bottom=0.08)
    fig.suptitle("Figure E2 | Robustness, developmental effects, and clinical translation",
        x=0.06,ha="left",fontsize=13,fontweight="bold")

    # (a) spatial-block robustness
    ax=fig.add_subplot(gs[0,0]); panel_label(ax,"a")
    B=[0,10,15,20]; within=[0.236,0.186,0.125,0.071]; dist=[0.641,0.640,0.639,0.638]; combo=[0.730,0.682,0.649,0.631]
    ax.plot(B,within,"-o",color=GREY,lw=2,ms=6,label="within-mean")
    ax.plot(B,dist,"-o",color=BLUE,lw=2,ms=6,label="distance")
    ax.plot(B,combo,"-o",color=AMBER,lw=2.2,ms=7,label="full model")
    ax.fill_between(B,within,combo,color=AMBER,alpha=0.08)
    ax.set_ylim(0,0.8); ax.set_xticks(B); ax.set_xlabel("exclude sites within X mm"); ax.set_ylabel("held-out r")
    ax.legend(frameon=False,fontsize=7); ax.set_title("Not a reuse artifact",loc="left")

    dev=loadjson("developmental.json"); rec=dev["records"] if dev else None
    # (b) predictability vs age
    ax=fig.add_subplot(gs[0,1]); panel_label(ax,"b")
    if rec:
        age=np.array(rec["age"]); cb=np.array(rec["combo"])
        ax.scatter(age,cb,s=26,c=VIOLET,edgecolor="white",lw=0.4,alpha=0.9)
        m=np.isfinite(age)&np.isfinite(cb); z=np.polyfit(age[m],cb[m],1); xs=np.linspace(age.min(),age.max(),20)
        ax.plot(xs,z[0]*xs+z[1],"--",color=INK,lw=1.5)
        ax.text(0.05,0.05,f"rho={dev['age_corr']['combo']['rho']:+.2f}, p={dev['age_corr']['combo']['p']:.3f}",
                transform=ax.transAxes,fontsize=8,fontweight="bold")
    ax.set_xlabel("age (years)"); ax.set_ylabel("predictability (combo r)"); ax.set_title("Predictability rises with age",loc="left")

    # (c) symmetry vs age
    ax=fig.add_subplot(gs[0,2]); panel_label(ax,"c")
    if rec:
        age=np.array(rec["age"]); sy=np.array(rec["sym"])
        ax.scatter(age,sy,s=26,c=TEAL,edgecolor="white",lw=0.4,alpha=0.9)
        m=np.isfinite(age)&np.isfinite(sy); z=np.polyfit(age[m],sy[m],1); xs=np.linspace(age.min(),age.max(),20)
        ax.plot(xs,z[0]*xs+z[1],"--",color=INK,lw=1.5)
        ax.text(0.05,0.05,f"rho={dev['age_corr']['sym']['rho']:+.2f}, p={dev['age_corr']['sym']['p']:.3f}",
                transform=ax.transAxes,fontsize=8,fontweight="bold")
    ax.set_xlabel("age (years)"); ax.set_ylabel("operator symmetry"); ax.set_title("Operator becomes more symmetric",loc="left")

    # (d) predictability map: homotopic + distance/hemi
    ax=fig.add_subplot(gs[1,0]); panel_label(ax,"d")
    pm=loadjson("predictability_map.json")
    if pm:
        h=pm["homotopic"]
        labels=["near\nhomotopic","far from\nhomotopic"]; vals=[h["near_homotopic"]["r"],h["far_from_homotopic"]["r"]]
        ax.bar([0,1],vals,color=[TEAL,GREY],width=0.6,zorder=3)
        for i,v in enumerate(vals): ax.text(i,v+0.01,f"{v:.2f}",ha="center",fontsize=8,fontweight="bold")
        ax.set_ylim(0,0.8); ax.set_xticks([0,1]); ax.set_xticklabels(labels,fontsize=8); ax.set_ylabel("r(pred,meas)")
    ax.set_title("Signal survives at the mirror site",loc="left")

    # (e) responder detection
    ax=fig.add_subplot(gs[1,1]); panel_label(ax,"e")
    metrics=["ROC-AUC","precision@5"]; wm=[0.660,0.162]; cb=[0.852,0.578]; x=np.arange(2); w=0.36
    ax.bar(x-w/2,wm,w,color=GREY,label="within-mean",zorder=3); ax.bar(x+w/2,cb,w,color=AMBER,label="combo",zorder=3)
    for xi,(a,b) in enumerate(zip(wm,cb)):
        ax.text(xi-w/2,a+0.01,f"{a:.2f}",ha="center",fontsize=7); ax.text(xi+w/2,b+0.01,f"{b:.2f}",ha="center",fontsize=7)
    ax.set_ylim(0,1.0); ax.set_xticks(x); ax.set_xticklabels(metrics); ax.legend(frameon=False,fontsize=7)
    ax.set_ylabel("score"); ax.set_title("Responder detection (AUC 0.85)",loc="left")

    # (f) few-shot calibration
    ax=fig.add_subplot(gs[1,2]); panel_label(ax,"f")
    labels=["cross-site\nonly","own-pilot\nonly","blend\n(model+pilot)"]; vals=[0.713,0.877,0.899]
    ax.bar(range(3),vals,color=[BLUE,GREY,TEAL],width=0.62,zorder=3)
    for i,v in enumerate(vals): ax.text(i,v+0.008,f"{v:.3f}",ha="center",fontsize=8,fontweight="bold")
    ax.axhline(0.9,color=CORAL,ls="--",lw=1); ax.text(2.4,0.905,"0.90",fontsize=7,color=CORAL)
    ax.set_ylim(0.6,0.98); ax.set_xticks(range(3)); ax.set_xticklabels(labels,fontsize=7.5)
    ax.set_ylabel("r"); ax.set_title("Few-shot calibration reaches ~0.90",loc="left")
    fig.savefig(OUT/"jumbo_B.png",bbox_inches="tight"); plt.close(fig); print("jumbo_B")

if __name__=="__main__":
    jumbo_A(); jumbo_B(); print("saved ->",OUT)
