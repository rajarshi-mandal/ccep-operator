"""Figure 5 (elite) — External validation: operator vs F-TRACT, CCEP vs structure, TMS-EEG."""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fig_lib import (ROOT, REP, INK, TEAL, AMBER, NAVY, CORAL, GREY, LGREY, VIOLET, GREEN,
                     CMAP_AMP, panel)
import sys; sys.path.insert(0, str(ROOT/"experiments"))
J=lambda n: json.loads((REP/f"{n}.json").read_text())

def rank(x): return np.argsort(np.argsort(x)).astype(float)

fig=plt.figure(figsize=(11,3.0)); gs=GridSpec(1,4,figure=fig,wspace=0.42,left=0.055,right=0.99,top=0.84,bottom=0.2)

# ---- A: our operator vs F-TRACT (parcel-pair hexbin) ----
axA=fig.add_subplot(gs[0,0]); panel(axA,"A")
import ccep_ftract_crossmap as CM
header,pidx,ft_amp=CM.load_ft("amplitude"); n_p=len(header)
mapper=CM.build_mapper()
from data.ccep_pipeline import CCEPSubject
Gs=np.zeros((n_p,n_p)); Gc=np.zeros((n_p,n_p))
for p in sorted((ROOT/"data"/"processed"/"ds004080").glob("sub-*.npz")):
    cs=CCEPSubject.load(str(p)); keep=np.arange(len(cs.sites))[(np.isfinite(cs.reliability))&(cs.reliability>=0.3)]
    if len(keep)<6: continue
    pcol=np.array([pidx.get(pp,-1) if pp else -1 for pp in mapper(cs.contact_xyz)])
    R=cs.responses[keep].astype(float); Rz=(R-np.nanmean(R))/(np.nanstd(R)+1e-9)
    for i,s in enumerate(keep):
        sp=[pcol[a] for a in cs.stim_idx[s] if a>=0 and pcol[a]>=0]; row,ok=Rz[i],np.isfinite(Rz[i])
        for c in range(len(pcol)):
            if not ok[c] or pcol[c]<0: continue
            for pp in sp: Gs[pp,pcol[c]]+=row[c]; Gc[pp,pcol[c]]+=1
Gour=np.where(Gc>0,Gs/np.maximum(Gc,1),np.nan)
off=~np.eye(n_p,dtype=bool); m=off&np.isfinite(Gour)&np.isfinite(ft_amp)
xr=rank(Gour[m]); yr=rank(ft_amp[m])
axA.hexbin(xr,yr,gridsize=34,cmap=CMAP_AMP,mincnt=1,linewidths=0)
b,a=np.polyfit(xr,yr,1); axA.plot([xr.min(),xr.max()],[b*xr.min()+a,b*xr.max()+a],color=CORAL,lw=1.4)
rho=np.corrcoef(xr,yr)[0,1]
axA.set_xlabel("our operator (rank)"); axA.set_ylabel("F-TRACT 780pt (rank)"); axA.set_xticks([]); axA.set_yticks([])
axA.set_title(f"Recovers population\nstructure ($\\rho$={rho:.2f})",fontsize=8,loc="left")

# ---- B: CCEP effective vs DWI structural (Glasser hexbin) ----
axB=fig.add_subplot(gs[0,1]); panel(axB,"B")
import ccep_struct as ST
en_lab,SC=ST.load_enigma_sc(); hdr,prob=ST.load_ft_hcp("probability"); _,dist=ST.load_ft_hcp("euclidian_distance")
pos={p:i for i,p in enumerate(hdr)}; order=[pos.get(l) for l in en_lab]
keepi=[i for i,o in enumerate(order) if o is not None]; oidx=[order[i] for i in keepi]
P=prob[np.ix_(oidx,oidx)]; D=dist[np.ix_(oidx,oidx)]; S=SC[np.ix_(keepi,keepi)]
Psym=np.nanmean(np.dstack([P,P.T]),axis=2); Slog=np.where(S>0,np.log10(S+1),np.nan)
offb=~np.eye(S.shape[0],dtype=bool); mb=offb&np.isfinite(Psym)&np.isfinite(Slog)
xr=rank(Slog[mb]); yr=rank(Psym[mb])
axB.hexbin(xr,yr,gridsize=30,cmap=CMAP_AMP,mincnt=1,linewidths=0)
b,a=np.polyfit(xr,yr,1); axB.plot([xr.min(),xr.max()],[b*xr.min()+a,b*xr.max()+a],color=CORAL,lw=1.4)
rho=np.corrcoef(xr,yr)[0,1]
axB.set_xlabel("DWI structural (rank)"); axB.set_ylabel("CCEP effective (rank)"); axB.set_xticks([]); axB.set_yticks([])
axB.set_title(f"Tracks structure\n($\\rho$={rho:.2f})",fontsize=8,loc="left")

# ---- C: geometry dominance (raw vs distance-controlled) ----
axC=fig.add_subplot(gs[0,2]); panel(axC,"C")
st=J("struct"); raw=st["ages_15_100"]["rho_prob_struct"]; part=st["ages_15_100"]["partial_prob_struct_given_dist"]
axC.bar([0,1],[raw,part],color=[NAVY,GREY],width=0.5,zorder=3)
for i,v in enumerate([raw,part]): axC.text(i,v+0.005,f"{v:.2f}",ha="center",fontsize=8,fontweight="bold")
axC.set_ylim(0,0.26); axC.set_xticks([0,1]); axC.set_xticklabels(["raw","| distance"],fontsize=7)
axC.set_ylabel(r"CCEP $\sim$ structural $\rho$")
axC.set_title("Correspondence is\nalmost all geometric",fontsize=8,loc="left")

# ---- D: TMS-EEG per-subject dumbbell (distance vs CCEP, beyond) ----
axD=fig.add_subplot(gs[0,3]); panel(axD,"D")
tm=J("tmseeg"); per=tm.get("per_subject",[])
dist_r=[p.get("distance",0) for p in per]; part_r=[p.get("partial",0) for p in per]
o=np.argsort(part_r); x=np.arange(len(per))
for i,xi in enumerate(o): axD.plot([i,i],[dist_r[xi],part_r[xi]],color=LGREY,lw=0.6)
axD.scatter(x,np.array(dist_r)[o],s=18,color=NAVY,label="distance$\\to$TEP")
axD.scatter(x,np.array(part_r)[o],s=18,color=TEAL,label="CCEP | dist")
axD.axhline(0,color=CORAL,ls="--",lw=0.8)
axD.set_ylim(-0.6,0.7); axD.set_xlabel("TMS-EEG subjects"); axD.set_ylabel(r"$\rho$ with TEP"); axD.legend(fontsize=6,loc="lower right")
axD.set_title("TMS-EEG bridge:\ninconclusive (1/6)",fontsize=8,loc="left")

fig.savefig(REP/"figs"/"Figure_5.png",bbox_inches="tight",facecolor="white"); print("saved Figure_5 (elite)")
