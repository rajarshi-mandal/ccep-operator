"""Shared helpers + aesthetic for the Brain Stimulation manuscript figures.
Rich, publication-grade panels: glass-brain electrode plots (nilearn), CCEP traces & heatmaps,
operator matrices, predicted-vs-measured maps, per-subject strip plots. Consistent palette."""
import warnings; warnings.filterwarnings("ignore")
import sys, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize

ROOT = Path("REDACTED/causal-dag-ssm")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
REP = ROOT / "reports"

# ---- palette ----
INK="#0E2233"; TEAL="#0FA3A3"; AMBER="#E8912A"; NAVY="#20456E"; CORAL="#E4572E"
VIOLET="#6C5CE7"; GREY="#9AA7B2"; LGREY="#D6DCE1"; GREEN="#2A9D5C"
plt.rcParams.update({
 "font.family":"sans-serif","font.sans-serif":["Helvetica Neue","Arial","DejaVu Sans"],
 "font.size":8,"axes.titlesize":9,"axes.labelsize":8,"axes.edgecolor":INK,"axes.linewidth":0.8,
 "xtick.color":INK,"ytick.color":INK,"text.color":INK,"xtick.labelsize":7,"ytick.labelsize":7,
 "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":300,"savefig.dpi":300,
 "legend.fontsize":7,"legend.frameon":False})
# amplitude colormap (white -> teal -> navy), diverging (coral/white/navy)
CMAP_AMP = LinearSegmentedColormap.from_list("amp", ["#FFFFFF","#BFE9E9","#0FA3A3","#20456E"])
CMAP_DIV = LinearSegmentedColormap.from_list("div", [NAVY,"#7FA8C9","#FFFFFF","#F0B27A",CORAL])
CMAP_TRACE = LinearSegmentedColormap.from_list("tr", [NAVY,"#6C93B8","#FFFFFF","#EBA96B",CORAL])

def panel(ax, letter, dx=-0.02, dy=1.04):
    ax.text(dx,dy,letter,transform=ax.transAxes,fontsize=12,fontweight="bold",va="bottom",ha="right")

def load_subject(sub_id, ds="ds004080"):
    from data.ccep_pipeline import CCEPSubject
    cs = CCEPSubject.load(str(ROOT/"data"/"processed"/ds/f"{sub_id}.npz"))
    tr_path = ROOT/"data"/"traces"/ds/f"{sub_id}.npz"
    tr = np.load(tr_path, allow_pickle=True) if tr_path.exists() else None
    return cs, tr

def per_subject_models():
    """Parse the operator_v2 log -> dict of arrays (within, distance, op_v1, op_v2) over 93 subjects."""
    rows=[]
    for ln in (REP/"_operator_v2_n93.txt").read_text().splitlines():
        m=re.match(r"^(\S+)\s+(\d+)\s+([+\-][\d.]+)\s+([+\-][\d.]+)\s+([+\-][\d.]+)\s+([+\-][\d.]+)", ln)
        if m: rows.append((m.group(1), float(m.group(3)),float(m.group(4)),float(m.group(5)),float(m.group(6))))
    within=np.array([r[1] for r in rows]); dist=np.array([r[2] for r in rows])
    v1=np.array([r[3] for r in rows]); v2=np.array([r[4] for r in rows])
    return dict(names=[r[0] for r in rows], within=within, distance=dist, op_v1=v1, op_v2=v2)

def glass_markers(ax, coords, values, cmap, vmin=None, vmax=None, size=28, display="l", hl=None, hlc=CORAL):
    """Electrodes on a glass brain (nilearn) into a given axes; optionally highlight a stim site."""
    from nilearn import plotting
    disp = plotting.plot_glass_brain(None, display_mode=display, axes=ax, alpha=0.06, black_bg=False)
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax
    disp.add_markers(coords, marker_color=cmap((np.clip(values,vmin,vmax)-vmin)/(vmax-vmin+1e-9)),
                     marker_size=size)
    if hl is not None:
        disp.add_markers(coords[hl].reshape(1,3), marker_color=[hlc], marker_size=size*3.2, marker='*')
    return disp
