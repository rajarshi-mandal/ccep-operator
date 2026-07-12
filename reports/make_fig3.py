"""Figure 3 (elite) — Control & clinical: controllability, SOZ localization (ROC + brain)."""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fig_lib import (ROOT, REP, INK, TEAL, AMBER, NAVY, CORAL, GREY, LGREY, VIOLET, GREEN,
                     CMAP_AMP, panel, load_subject, glass_markers)
import sys; sys.path.insert(0, str(ROOT/"experiments"))
import ccep_operator_v2 as V2
from ccep_control import controllability
import ccep_soz as SZ
from nilearn import plotting
J=lambda n: json.loads((REP/f"{n}.json").read_text())
ctl=J("control")

fig=plt.figure(figsize=(11,3.0)); gs=GridSpec(1,5,figure=fig,wspace=0.5,left=0.05,right=0.99,top=0.84,bottom=0.2)

# A: controllability -> reach, example patient scatter + strip inset
axA=fig.add_subplot(gs[0,0]); panel(axA,"A")
ex=max(ctl["per_subject"],key=lambda p:(p.get("ctrl_reach_rho") or 0))
# recompute the per-site controllability vs reach for the example subject
sub_ex=ex["subject"].split("/")[-1]
cs,_=load_subject(sub_ex if sub_ex.startswith("sub-ccepAge") else "sub-ccepAgeUMCU48",
                  "ds004080")
keep=np.arange(len(cs.sites))[(np.isfinite(cs.reliability))&(cs.reliability>=0.3)]
cc,rr=[],[]
for s in keep:
    tr=[t for t in keep if t!=s]; A=V2._build_operator(cs,tr,"symmetric"); avg,_=controllability(A)
    pr=[a for a in cs.stim_idx[s] if a>=0]
    if pr: cc.append(float(np.mean(avg[pr]))); rr.append(float(np.nansum(cs.responses[s])))
cc=np.array(cc); rr=np.array(rr)
axA.scatter(cc,rr,s=12,color=TEAL,alpha=0.7,edgecolor="none")
b,a=np.polyfit(cc,rr,1); xs=np.linspace(cc.min(),cc.max(),20); axA.plot(xs,b*xs+a,color=CORAL,lw=1.4)
axA.set_xlabel("site controllability"); axA.set_ylabel("measured network reach"); axA.set_xticks([]); axA.set_yticks([])
axA.set_title("Controllability predicts\nreach (example patient)",fontsize=8,loc="left")
rho=np.array([p["ctrl_reach_rho"] for p in ctl["per_subject"] if p.get("ctrl_reach_rho") is not None])
axi=axA.inset_axes([0.58,0.14,0.38,0.32]); axi.hist(rho,bins=16,color=TEAL); axi.axvline(0,color=CORAL,ls="--",lw=0.8)
axi.set_title(f"{int((rho>0).sum())}/93",fontsize=6); axi.tick_params(labelsize=5); axi.set_yticks([])

# B: controllability on brain
axB=fig.add_subplot(gs[0,1]); panel(axB,"B")
A=V2._build_operator(cs,list(keep),"symmetric"); avg,_=controllability(A)
glass_markers(axB,cs.contact_xyz,avg,CMAP_AMP,display="z",size=16)
axB.set_title("Average controllability\non the montage",fontsize=8)

# ---- SOZ: features + labels + leave-one-subject-out logistic ROC ----
data,fnames=SZ.collect(); idx={f:i for i,f in enumerate(fnames)}
def loso_probs(feat):
    from sklearn.linear_model import LogisticRegression
    subj=[d for d in data if d["soz"].sum()>0 and (~d["soz"]).sum()>0]
    fi=[idx[f] for f in feat]; P,Y=[],[]
    for i,dt in enumerate(subj):
        Xtr=np.vstack([subj[j]["X"][:,fi] for j in range(len(subj)) if j!=i])
        ytr=np.concatenate([subj[j]["soz"] for j in range(len(subj)) if j!=i]).astype(int)
        mu,sd=np.nanmean(Xtr,0),np.nanstd(Xtr,0)+1e-9
        clf=LogisticRegression(max_iter=1000).fit(np.nan_to_num((Xtr-mu)/sd),ytr)
        P.append(clf.predict_proba(np.nan_to_num((dt["X"][:,fi]-mu)/sd))[:,1]); Y.append(dt["soz"])
    return np.concatenate(P),np.concatenate(Y).astype(int)
def roc(P,Y):
    th=np.unique(P)[::-1]; tpr=[0];fpr=[0]
    for t in th:
        pred=P>=t; tp=((pred)&(Y==1)).sum(); fp=((pred)&(Y==0)).sum()
        tpr.append(tp/max((Y==1).sum(),1)); fpr.append(fp/max((Y==0).sum(),1))
    tpr.append(1);fpr.append(1); return np.array(fpr),np.array(tpr)
op=["efferent_strength","avg_ctrb","modal_ctrb","asymmetry"]; ag=["afferent_strength","node_density","mean_dist"]
Pop,Y=loso_probs(op); fop,top=roc(Pop,Y)
Pag,_=loso_probs(ag); fag,tag=roc(Pag,Y)

# C: SOZ ROC
axC=fig.add_subplot(gs[0,2]); panel(axC,"C")
axC.plot(fop,top,color=TEAL,lw=1.8,label="operator (0.61)")
axC.plot(fag,tag,color=GREY,lw=1.6,label="amp+geom (0.56)")
axC.plot([0,1],[0,1],color=INK,ls="--",lw=0.7)
axC.set_xlabel("false positive rate"); axC.set_ylabel("true positive rate"); axC.legend(fontsize=6,loc="lower right")
axC.set_title("Localizes seizure-\nonset zone",fontsize=8,loc="left")

# D: SOZ on brain (focal-SOZ example) — colour by efferent, mark SOZ contacts
axD=fig.add_subplot(gs[0,3]); panel(axD,"D")
focal=[d for d in data if 5<=int(d["soz"].sum())<=12]
ex_soz=max(focal,key=lambda d:len(d["names"])) if focal else max(data,key=lambda d:d["soz"].sum())
cs2,_=load_subject(ex_soz["subject"],"ds004080")
lab=SZ.load_labels(ex_soz["subject"]); names=[str(n) for n in cs2.contacts]
soz=np.array([lab.get(n,(False,False))[0] for n in names])
eff=ex_soz["X"][:,idx["efferent_strength"]] if len(ex_soz["names"])==len(names) else np.zeros(len(names))
# map efferent (over labelled names) to full contact order
effmap={n:v for n,v in zip(ex_soz["names"],ex_soz["X"][:,idx["efferent_strength"]])}
effv=np.array([effmap.get(n,np.nan) for n in names]); effv=np.where(np.isfinite(effv),effv,np.nanmedian(effv))
disp=glass_markers(axD,cs2.contact_xyz,effv,CMAP_AMP,display="z",size=20)
if soz.any(): disp.add_markers(cs2.contact_xyz[soz],marker_color=[CORAL],marker_size=22,marker='o')
axD.set_title("Efferent strength;\nSOZ marked (red)",fontsize=8)

# E: targeting capture distribution
axE=fig.add_subplot(gs[0,4]); panel(axE,"E")
tg=ctl["targeting"]
per=[p for p in ctl["per_subject"]]
mc=[p["capture_model"] for p in per]; dc=[p["capture_distance"] for p in per]
axE.hist(dc,bins=14,color=NAVY,alpha=0.55,label=f"distance {tg['capture_distance_mean']:.2f}")
axE.hist(mc,bins=14,color=TEAL,alpha=0.55,label=f"operator {tg['capture_model_mean']:.2f}")
axE.set_xlabel("target-capture (1=oracle)"); axE.set_ylabel("patients"); axE.legend(fontsize=6)
axE.set_title("Single-site targeting:\nhonest tie",fontsize=8,loc="left")

fig.savefig(REP/"figs"/"Figure_3.png",bbox_inches="tight",facecolor="white"); print("saved Figure_3 (elite)")
