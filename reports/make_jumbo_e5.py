#!/usr/bin/env python3
"""JUMBO figure E5 — Tier-2 external-data validation: F-TRACT (780pt), structural grounding, animal."""
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

ft = json.loads((R/"ftract.json").read_text())
st = json.loads((R/"struct.json").read_text())
an = json.loads((R/"animal.json").read_text())

def plabel(ax,l): ax.text(-0.15,1.09,l,transform=ax.transAxes,fontsize=13,fontweight="bold",color=INK,va="top")

fig=plt.figure(figsize=(12,8)); gs=GridSpec(2,2,figure=fig,hspace=0.42,wspace=0.30,left=0.08,right=0.97,top=0.90,bottom=0.08)
fig.suptitle("Figure E5 | External validation beyond n=93: F-TRACT (780 patients), structure, and animal data",
    x=0.08,ha="left",fontsize=13,fontweight="bold")

# (a) F-TRACT conduction velocity replication
ax=fig.add_subplot(gs[0,0]); plabel(ax,"a")
ours=3.0; adult=ft["conduction"]["ages_15_100"]["ftract_median_velocity_mm_per_ms"]; child=ft["conduction"]["ages_0_15"]["ftract_median_velocity_mm_per_ms"]
ax.bar([0,1,2],[ours,adult,child],color=[AMBER,TEAL,BLUE],width=0.6,zorder=3)
for i,v in enumerate([ours,adult,child]): ax.text(i,v+0.05,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold")
ax.set_ylim(0,4.3); ax.set_xticks([0,1,2]); ax.set_xticklabels(["ours\n(n=93)","F-TRACT\nadults","F-TRACT\nchildren"],fontsize=8.5)
ax.set_ylabel("conduction velocity (mm/ms)")
ax.set_title("Conduction law replicates at 780 patients",loc="left")
rr=ft["conduction"]["ages_15_100"]["rho_dist_latency"]
ax.text(0.5,0.97,f"latency-distance rho={rr:.2f} (~20k pairs)\nvelocity rises with age (myelination)",transform=ax.transAxes,ha="center",va="top",fontsize=8)

# (b) F-TRACT: our claims replicate
ax=fig.add_subplot(gs[0,1]); plabel(ax,"b")
loc=-ft["amplitude_locality"]["rho_amp_distance"]; rec=ft["directionality"]["amplitude_reciprocity"]
ax.bar([0,1],[loc,rec],color=[TEAL,VIOLET],width=0.5,zorder=3)
for i,v in enumerate([loc,rec]): ax.text(i,v+0.008,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold")
ax.set_ylim(0,0.42); ax.set_xticks([0,1]); ax.set_xticklabels(["amplitude\nlocality |ρ|","reciprocal\ndominance"],fontsize=8.5)
ax.set_ylabel("F-TRACT effect (780 patients)")
ax.set_title("Locality & reciprocity replicate",loc="left")
ax.text(0.5,0.9,f"directional latency asymmetry\n{ft['directionality']['latency_asymmetry_ms']:.1f} ms (parcel scale)",transform=ax.transAxes,ha="center",va="top",fontsize=8)

# (c) structural grounding: geometry dominance
ax=fig.add_subplot(gs[1,0]); plabel(ax,"c")
raw=st["ages_15_100"]["rho_prob_struct"]; part=st["ages_15_100"]["partial_prob_struct_given_dist"]
ax.bar([0,1],[raw,part],color=[BLUE,GREY],width=0.5,zorder=3)
for i,v in enumerate([raw,part]): ax.text(i,v+0.006,f"{v:.2f}",ha="center",fontsize=9,fontweight="bold")
ax.set_ylim(0,0.26); ax.set_xticks([0,1]); ax.set_xticklabels(["raw\ncorrelation","controlling\nfor distance"],fontsize=8.5)
ax.set_ylabel("CCEP effective ~ DWI structural ρ")
ax.set_title("CCEP follows structure — via geometry",loc="left")
ax.text(0.5,0.9,"F-TRACT CCEP vs ENIGMA/HCP DWI (Glasser-360)\nstructural link is almost all geometric",transform=ax.transAxes,ha="center",va="top",fontsize=8)

# (d) animal identifiability
ax=fig.add_subplot(gs[1,1]); plabel(ax,"d")
ks=sorted(int(k) for k in an["reliability_vs_trials"].keys()); rv=[an["reliability_vs_trials"][str(k)] for k in ks]
ax.plot(ks,rv,"-o",color=TEAL,lw=2,ms=6,zorder=3)
ax.set_xlabel("trials per site"); ax.set_ylabel("split-half reliability")
ax.set_title("Animal: dense SITES, not trials, are the limit",loc="left")
ax.set_ylim(0,0.85)
sm=an["sites_per_session"]["median"]
ax.text(0.5,0.32,f"reliability rises with trials (0.30 to 0.73)\nyet {int(an['sites_per_session']['frac_under_6']*100)}% of sessions have <6 sites\n(median {sm:.0f}): operator NOT identifiable\n(confirms recovery simulation)",
        transform=ax.transAxes,ha="center",va="top",fontsize=8)

fig.savefig(OUT/"jumbo_E5.png",bbox_inches="tight"); plt.close(fig)
print("saved", OUT/"jumbo_E5.png")
