"""Figure 2 (elite) — Mechanism: mono/poly-synaptic components, directionality, conduction law, latency gradient."""
import warnings; warnings.filterwarnings("ignore")
import json, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from fig_lib import (ROOT, REP, INK, TEAL, AMBER, NAVY, CORAL, GREY, LGREY, VIOLET, GREEN,
                     CMAP_AMP, CMAP_DIV, panel, load_subject, glass_markers)
import sys; sys.path.insert(0, str(ROOT/"experiments"))
from data.ccep_pipeline import CCEPSubject

DATASETS=["ds004774","ds004696","ds004457","ds003708","ds004080"]
def all_caches():
    return [p for ds in DATASETS for p in sorted((ROOT/"data"/"processed"/ds).glob("sub-*.npz"))]

# ---- real conduction-law data ----
lat_d,lat_t=[],[]
for p in all_caches():
    cs=CCEPSubject.load(str(p))
    if cs.latency is None or not np.size(cs.latency): continue
    keep=np.arange(len(cs.sites))[(np.isfinite(cs.reliability))&(cs.reliability>=0.3)]
    for s in keep:
        D=np.linalg.norm(cs.contact_xyz-cs.stim_xyz[s][None],axis=1); lat=cs.latency[s]
        m=np.isfinite(lat)&(lat>5)&(lat<150)&(D>3)&(cs.responses[s]>np.nanpercentile(cs.responses[s],60))
        lat_d.extend(D[m]); lat_t.extend(lat[m])
lat_d=np.array(lat_d); lat_t=np.array(lat_t)

# ---- per-subject directionality (from directed log) ----
fwd,trn=[],[]
for ln in (REP/"_directed_n93.txt").read_text().splitlines():
    m=re.match(r"^\S+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",ln)
    if m: fwd.append(float(m.group(2))); trn.append(float(m.group(3)))
fwd=np.array(fwd); trn=np.array(trn)

# ---- example subject for waveform + latency brain ----
cs,tr=load_subject("sub-ccepAgeUMCU48")
keep=np.arange(len(cs.sites))[(np.isfinite(cs.reliability))&(cs.reliability>=0.3)]
site=sorted([(int(np.isfinite(cs.responses[s]).sum()),s) for s in keep])[-8][1]

fig=plt.figure(figsize=(11,3.0))
gs=GridSpec(1,5,figure=fig,wspace=0.5,left=0.05,right=0.99,top=0.86,bottom=0.22)

# A: GMFA waveform with N1/N2 windows
axA=fig.add_subplot(gs[0,0]); panel(axA,"A")
if tr is not None:
    ts=list(tr["sites"]).index(cs.sites[site]) if cs.sites[site] in list(tr["sites"]) else 0
    T=tr["t_ms"]; Wt=np.abs(tr["traces"][ts]); gmfa=np.nanmean(Wt,axis=0)
    axA.axvspan(10,100,color=TEAL,alpha=0.12); axA.axvspan(100,300,color=VIOLET,alpha=0.12)
    axA.plot(T,gmfa,color=INK,lw=1.2)
    axA.text(55,axA.get_ylim()[1]*0.9,"N1",color=TEAL,fontsize=7,ha="center",fontweight="bold")
    axA.text(200,axA.get_ylim()[1]*0.9,"N2",color=VIOLET,fontsize=7,ha="center",fontweight="bold")
    axA.set_xlim(-5,320); axA.set_xlabel("time (ms)"); axA.set_ylabel("mean |response| (µV)")
axA.set_title("Early N1 vs late N2",fontsize=8,loc="left")
axi=axA.inset_axes([0.55,0.5,0.4,0.42])
axi.bar([0,1],[0.451,0.533],color=[TEAL,VIOLET],width=0.6)
axi.set_xticks([0,1]); axi.set_xticklabels(["N1","N2"],fontsize=6); axi.tick_params(labelsize=5)
axi.set_title("network incr.",fontsize=5.5); axi.set_ylim(0,0.6)

# B: directionality per-subject scatter
axB=fig.add_subplot(gs[0,1]); panel(axB,"B")
axB.scatter(trn,fwd,s=9,color=TEAL,alpha=0.7,edgecolor="none",zorder=3)
lim=[min(trn.min(),fwd.min())-0.02,max(trn.max(),fwd.max())+0.02]
axB.plot(lim,lim,color=GREY,ls="--",lw=0.8); axB.set_xlim(lim); axB.set_ylim(lim)
axB.set_xlabel("afferent (transpose) r"); axB.set_ylabel("efferent (forward) r")
axB.set_title(f"Orientation matters\n({int((fwd>trn).sum())}/93 efferent$>$afferent)",fontsize=8,loc="left")

# C: conduction law hexbin
axC=fig.add_subplot(gs[0,2]); panel(axC,"C")
axC.hexbin(lat_d,lat_t,gridsize=30,cmap=CMAP_AMP,mincnt=1,linewidths=0)
m=np.polyfit(lat_d,lat_t,1)[0]; vel=1/m; b1=np.polyfit(lat_d,lat_t,1)[1]
xs=np.linspace(np.percentile(lat_d,2),np.percentile(lat_d,98),50)
axC.plot(xs,m*xs+b1,color=CORAL,lw=1.6)
r=np.corrcoef(lat_d,lat_t)[0,1]
axC.set_ylim(0,np.percentile(lat_t,99)); axC.set_xlabel("distance from stim (mm)"); axC.set_ylabel("N1 latency (ms)")
axC.set_title("Conduction law",fontsize=8,loc="left")
axC.text(0.04,0.94,f"r={r:.2f}\n$\\approx${vel:.1f} mm/ms",transform=axC.transAxes,fontsize=6,va="top")

# D: latency gradient on brain
axD=fig.add_subplot(gs[0,3]); panel(axD,"D")
lat=cs.latency[site] if (cs.latency is not None and np.size(cs.latency)) else np.zeros(len(cs.contacts))
finite=np.isfinite(lat)&(lat>5)&(lat<150)
vals=np.where(finite,lat,np.nan)
from fig_lib import CMAP_TRACE
gm=glass_markers(axD,cs.contact_xyz[finite],lat[finite],CMAP_AMP,display="z",size=18,
                 vmin=np.nanpercentile(lat[finite],5),vmax=np.nanpercentile(lat[finite],95))
axD.set_title("Latency gradient\n(near=fast, far=slow)",fontsize=8)

# E: F-TRACT velocity replication
axE=fig.add_subplot(gs[0,4]); panel(axE,"E")
ft=json.loads((REP/"ftract.json").read_text())
ad=ft["conduction"]["ages_15_100"]["ftract_median_velocity_mm_per_ms"]; ch=ft["conduction"]["ages_0_15"]["ftract_median_velocity_mm_per_ms"]
axE.axhline(3.0,color=CORAL,ls="--",lw=1); axE.text(1.5,3.05,"ours (n=93)",color=CORAL,fontsize=6,ha="center")
axE.scatter([0,1],[ad,ch],s=60,color=[NAVY,"#5B7FA6"],zorder=3)
for i,v in enumerate([ad,ch]): axE.text(i,v+0.12,f"{v:.2f}",ha="center",fontsize=7,fontweight="bold")
axE.set_ylim(1.8,3.6); axE.set_xlim(-0.5,1.5); axE.set_xticks([0,1]); axE.set_xticklabels(["adult","child"],fontsize=7)
axE.set_ylabel("F-TRACT velocity (mm/ms)")
axE.set_title("Replicates at\n780 patients",fontsize=8,loc="left")

fig.savefig(REP/"figs"/"Figure_2.png",bbox_inches="tight",facecolor="white")
print("saved Figure_2 (elite); conduction ~%.1f mm/ms, dir %d/93"%(vel,int((fwd>trn).sum())))
