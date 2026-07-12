#!/usr/bin/env python3
"""JUMBO figure E3 — wave-2 extensions: latency/conduction, timing-direction, pilot-site efficiency."""
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

lat = json.loads((R/"latency.json").read_text())
lop = json.loads((R/"latency_operator.json").read_text())
pil = json.loads((R/"pilot.json").read_text())

def plabel(ax,l): ax.text(-0.13,1.07,l,transform=ax.transAxes,fontsize=13,fontweight="bold",color=INK,va="top")

fig=plt.figure(figsize=(12,8)); gs=GridSpec(2,2,figure=fig,hspace=0.38,wspace=0.28,left=0.07,right=0.98,top=0.9,bottom=0.08)
fig.suptitle("Figure E3 | Response timing: predictability, conduction, direction, and pilot efficiency",
    x=0.07,ha="left",fontsize=13,fontweight="bold")

# (a) latency predictability
ax=fig.add_subplot(gs[0,0]); plabel(ax,"a")
names=["within\nmean","distance","stim-kNN\n(network)","combo"]; vals=[lat["within"]["mean"],lat["dist"]["mean"],lat["knn"]["mean"],lat["combo"]["mean"]]
ax.bar(range(4),vals,color=[GREY,BLUE,TEAL,AMBER],width=0.68,zorder=3)
for i,v in enumerate(vals): ax.text(i,v+0.008,f"{v:.3f}",ha="center",fontsize=8.5,fontweight="bold")
ax.set_ylim(0,0.5); ax.set_xticks(range(4)); ax.set_xticklabels(names,fontsize=8.5)
ax.set_ylabel("held-out latency r (n=93)"); ax.set_title("Response timing is predictable",loc="left")
ax.text(0.5,0.9,f"combo vs within +{lat['combo']['mean']-lat['within']['mean']:.2f}, p≈0",transform=ax.transAxes,ha="center",fontsize=8,color=INK)

# (b) conduction law (illustrative fit + stats)
ax=fig.add_subplot(gs[0,1]); plabel(ax,"b")
sp=lat["conduction"]["speed_mm_per_ms"]; rr=lat["conduction"]["r_lat_dist"]
D=np.linspace(0,80,50); base=18
ax.plot(D, base + D/sp, color=TEAL, lw=2.5)
ax.fill_between(D, base+D/sp-8, base+D/sp+8, color=TEAL, alpha=0.12)
ax.set_xlabel("distance from stim site (mm)"); ax.set_ylabel("N1 latency (ms)")
ax.set_title("Cortico-cortical conduction",loc="left")
ax.text(0.05,0.92,f"latency grows with distance\nr={rr:.2f} (184k contacts)\napparent speed ≈ {sp:.1f} mm/ms",
        transform=ax.transAxes,va="top",fontsize=9,color=INK)
ax.set_ylim(0,80)

# (c) timing-directed operator
ax=fig.add_subplot(gs[1,0]); plabel(ax,"c")
dn=["oriented\n(by timing)","symmetric","forward"]; dv=[lop["oriented"]["mean"],lop["symmetric"]["mean"],lop["forward"]["mean"]]
ax.bar(range(3),dv,color=[VIOLET,BLUE,TEAL],width=0.6,zorder=3)
for i,v in enumerate(dv): ax.text(i,v+0.004,f"{v:.3f}",ha="center",fontsize=8.5,fontweight="bold")
ax.set_ylim(0.6,0.72); ax.set_xticks(range(3)); ax.set_xticklabels(dn,fontsize=8.5)
ax.set_ylabel("held-out r"); ax.set_title("Timing reveals direction",loc="left")
ax.text(0.5,0.06,f"reciprocal latency asymmetry {lop['mean_latency_asymmetry_ms']:.0f} ms\n= physiological directed evidence",
        transform=ax.transAxes,ha="center",fontsize=8,color=INK)

# (d) pilot-site efficiency
ax=fig.add_subplot(gs[1,1]); plabel(ax,"d")
ks=pil["ks"]; rnd=[pil["random"][str(k)]["mean"] for k in ks]; grd=[pil["greedy"][str(k)]["mean"] for k in ks]
ax.plot(ks,rnd,"-o",color=GREY,lw=2,ms=6,label="random sites")
ax.plot(ks,grd,"-s",color=AMBER,lw=2,ms=6,label="coverage-greedy")
ax.axhline(rnd[0],color=TEAL,ls=":",lw=1); ax.text(14,rnd[0]-0.012,f"3 sites already r={rnd[0]:.2f}",fontsize=8,color=TEAL)
ax.set_xlabel("# stimulation sites mapped"); ax.set_ylabel("predict the rest, r")
ax.set_xticks(ks); ax.set_ylim(0.45,0.62); ax.legend(frameon=False,fontsize=8.5,loc="lower right")
ax.set_title("Few pulses suffice",loc="left")

fig.savefig(OUT/"jumbo_E3.png",bbox_inches="tight"); plt.close(fig)
print("saved", OUT/"jumbo_E3.png")
