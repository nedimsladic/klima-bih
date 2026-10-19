"""
Klima i nadmorska visina u BiH: udio Köppen-Geigerovih tipova po visinskim pojasevima i
visina granice kontinentalne klime (Dfb) po periodima i scenarijima.

Visina: Copernicus DEM GLO-90 (90 m), uprosječen na mrežu karata klimatskih tipova (1 km, Beck et al. 2023).

Upotreba:
    python visina_bih.py <dem_dir s Copernicus_DSM_COG_30_*.tif> <kg_tif_dir> <bih_boundary.geojson> <out_dir>
"""
import sys
import os
import csv
import glob

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from bih_stil import Bih, HIST, FUT, REF, SSP, SSP_COLORS, SSP_DASH, DPI, period_label, ssp_code

DEM_DIR, KG_DIR, BOUNDARY, OUT = sys.argv[1:5]
B = Bih(BOUNDARY, OUT)
KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
KG = {8: ("Csa", (255, 255, 0)), 9: ("Csb", (200, 200, 0)), 14: ("Cfa", (200, 255, 80)), 15: ("Cfb", (100, 255, 80)),
      18: ("Dsb", (200, 0, 200)), 25: ("Dfa", (0, 255, 255)), 26: ("Dfb", (55, 200, 255)), 27: ("Dfc", (0, 125, 125)),
      29: ("ET", (178, 178, 178))}
DESC = {"Csa": "Mediteranska, vruće ljeto", "Csb": "Mediteranska, toplo ljeto", "Cfa": "Umjereno topla, vruće ljeto",
        "Cfb": "Umjereno topla, toplo ljeto", "Dfa": "Kontinentalna, vruće ljeto", "Dfb": "Kontinentalna, toplo ljeto",
        "Dfc": "Subalpska (planinska)", "Dsb": "Kontinentalna, suho ljeto", "ET": "Planinska tundra"}


def kg(p, s):
    with rasterio.open(os.path.join(KG_DIR, p, *([s] if s else []), "koppen_geiger_0p00833333.tif")) as r:
        a, tr = mask(r, B.border.geometry, crop=True, nodata=0)
        return a[0], tr, r.crs


A0, TR, CRS = kg(REF, None)
srcs = [rasterio.open(f) for f in sorted(glob.glob(os.path.join(DEM_DIR, "*.tif")))]
dem, dtr = merge(srcs)
elev = np.full(A0.shape, np.nan, dtype="float32")
reproject(dem[0].astype("float32"), elev, src_transform=dtr, src_crs=srcs[0].crs, dst_transform=TR, dst_crs=CRS,
          resampling=Resampling.average, src_nodata=-32767, dst_nodata=np.nan)
lats = TR.f + TR.e * (np.arange(A0.shape[0]) + 0.5)
W = np.cos(np.deg2rad(lats))[:, None] * np.ones(A0.shape)
inside = A0 > 0
print("Visina u BiH:", int(np.nanmin(elev[inside])), "–", int(np.nanmax(elev[inside])), "m; srednja",
      int(np.nansum(elev[inside] * W[inside]) / W[inside].sum()), "m")

BANDS = np.arange(0, 2300, 100)
mid = BANDS[:-1] + 50
ORDER = [8, 9, 14, 15, 18, 25, 26, 27, 29]


def by_band(a):
    sh = np.zeros((len(ORDER), len(mid)))
    for j, (lo, hi) in enumerate(zip(BANDS[:-1], BANDS[1:])):
        m = inside & (elev >= lo) & (elev < hi)
        tot = W[m].sum()
        if tot == 0:
            continue
        for i, c in enumerate(ORDER):
            sh[i, j] = W[m & (a == c)].sum() / tot * 100
    return sh


def limit(a, cls=(26, 27, 29)):
    """Najniži visinski pojas od kojeg naviše kontinentalni i hladniji tipovi čine ≥ 50 % površine."""
    sh = by_band(a)
    cold = sh[[ORDER.index(c) for c in cls]].sum(0)
    ok = np.where(cold >= 50)[0]
    for j in ok:                                      # prvi pojas iznad kojeg svi viši pojasevi imaju ≥ 50 %
        if (cold[j:][np.isfinite(cold[j:])] >= 50).all():
            return float(BANDS[j])
    return np.nan


A = {k: (A0 if k == (REF, None) else kg(*k)[0]) for k in KEYS}
LIM = {k: limit(A[k]) for k in KEYS}
for k in [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]:
    print(period_label(k[0]), k[1] or "", "granica Dfb/Dfc:", LIM[k])
npix = np.array([inside[(elev >= lo) & (elev < hi)].sum() for lo, hi in zip(BANDS[:-1], BANDS[1:])])
top = mid[npix >= 20].max() + 50                     # gornja granica prikaza: pojasevi s bar 20 km²

# 1) udio tipova po visini: 1991–2020 i tri scenarija 2071–2099
for PF in FUT:
    cases = [((REF, None), "Opažanja 1991–2020")] + [((PF, s_), f"{period_label(PF)} · {SSP[s_].split(' · ')[0]}") for s_ in ("ssp126", "ssp245", "ssp585")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 8.4), sharey=True)
    used = set()
    for ax, (k, t) in zip(axes, cases):
        sh = by_band(A[k])
        left = np.zeros(len(mid))
        for i, c in enumerate(ORDER):
            if sh[i].max() < 0.5:
                continue
            used.add(c)
            ax.barh(mid, sh[i], left=left, height=96, color=tuple(np.array(KG[c][1]) / 255), edgecolor="none")
            left += sh[i]
        if np.isfinite(LIM[k]):
            ax.axhline(LIM[k], color="#111", lw=1.6, ls="--")
            ax.text(101, LIM[k], f"{LIM[k]:.0f} m", va="center", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 100); ax.set_ylim(0, top)
        ax.set_title(t, fontsize=14, loc="left", pad=10)
        ax.set_xlabel("udio površine u visinskom pojasu (%)", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors="#333", labelsize=10.5)
    axes[0].set_ylabel("nadmorska visina (m)", fontsize=12)
    hs = [Patch(facecolor=tuple(np.array(KG[c][1]) / 255), label=f"$\\mathbf{{{KG[c][0]}}}$  {DESC[KG[c][0]]}")
          for c in ORDER if c in used]
    hs.append(plt.Line2D([], [], color="#111", lw=1.6, ls="--", label="donja granica kontinentalne i planinske klime (Dfb, Dfc)"))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.78, bottom=0.25, wspace=0.12)
    x0 = 0.01
    fig.legend(handles=hs, loc="lower left", ncol=3, frameon=False, fontsize=11.5, bbox_to_anchor=(x0, 0.0),
               title="Klimatski tipovi prema Köppen-Geigerovoj klasifikaciji",
               title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
    fig.text(x0, 0.95, "Klima i nadmorska visina u Bosni i Hercegovini", fontsize=27, ha="left", va="baseline")
    fig.text(x0, 0.89, f"• Udio klimatskih tipova po visinskim pojasevima od 100 m, {period_label(PF)}", fontsize=27, ha="left", va="baseline")
    B.put_credit(fig, fig.transFigure, 0.99, 0.89, 15)
    fig.savefig(os.path.join(OUT, f"visina_tipovi_{PF}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ visina_tipovi_{PF}.png")

# 2) granica kroz vrijeme
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5, "2041_2070": 2055.5, "2071_2099": 2085}
fig, ax = plt.subplots(figsize=(12, 7.6))
r0 = LIM[(REF, None)]
for s in SSP:
    ys = [r0] + [LIM[(p, s)] for p in FUT]
    ax.plot([X[REF]] + [X[p] for p in FUT], ys, color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o",
            ms=5, mec="white", label=SSP[s].replace(" · ", " – "), zorder=3)
    if np.isfinite(ys[-1]):
        ax.annotate(f"{ys[-1]:.0f} m", (X["2071_2099"], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=10, color=SSP_COLORS[s])
ax.plot([X[p] for p in HIST], [LIM[(p, None)] for p in HIST], color="#222", lw=2.6, marker="o", ms=7, mec="white",
        label="Historijski periodi", zorder=4)
ax.annotate(f"{r0:.0f} m", (X[REF], r0), xytext=(-9, 8), textcoords="offset points", ha="right", fontsize=11)
ax.set_xticks(list(X.values())); ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in X])
ax.set_xlim(1905, 2099)
ax.set_ylabel("nadmorska visina (m)", fontsize=12)
ax.set_title("Donja granica kontinentalne i planinske klime (Dfb, Dfc)", fontsize=15, loc="left", pad=22)
ax.text(0, 1.02, "najniži visinski pojas od kojeg naviše Dfb, Dfc i ET čine najmanje polovinu površine", transform=ax.transAxes,
        fontsize=10.5, color="#666", va="bottom")
ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color("#888")
ax.legend(frameon=False, fontsize=10.5, loc="upper left", ncol=2)
fig.subplots_adjust(left=0.09, right=0.95, top=0.76, bottom=0.1)
fig.text(0.01, 0.95, "Klima i nadmorska visina u Bosni i Hercegovini", fontsize=24, ha="left", va="baseline")
fig.text(0.01, 0.89, "• Kontinentalna klima se povlači u planine", fontsize=24, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.99, 0.89, 13)
fig.savefig(os.path.join(OUT, "visina_granica.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ visina_granica.png")

with open(os.path.join(OUT, "visina_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij", "donja granica Dfb/Dfc/ET (m)"])
    for k in KEYS:
        w.writerow([period_label(k[0]), k[1].upper() if k[1] else "historijski",
                    "" if not np.isfinite(LIM[k]) else f"{LIM[k]:.0f}"])
print("✓ visina_bih.csv")
