"""
BiH i zemlje regije (Slovenija, Hrvatska, Srbija, Crna Gora, Kosovo, Albanija, Sjeverna Makedonija): Köppen-Geigerovi tipovi i udjeli površine.

Podaci: Beck et al. (2023), 1 km. Granice: geoBoundaries ADM0 (pojednostavljene).

Upotreba:
    python susjedi_bih.py <kg_tif_dir> <granice_dir s BIH/HRV/SRB/MNE/SVN/XKX/ALB/MKD.geojson> <out_dir>
"""
import sys
import os
import csv

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import contextily as ctx

from bih_stil import Bih, HIST, FUT, REF, SSP, DPI, period_label, ssp_code

KG_DIR, GDIR, OUT = sys.argv[1:4]
B = Bih(os.path.join(GDIR, "BIH.geojson"), OUT)
COUNTRIES = {"BIH": "Bosna i Hercegovina", "SVN": "Slovenija", "HRV": "Hrvatska", "SRB": "Srbija",
             "MNE": "Crna Gora", "XKX": "Kosovo", "ALB": "Albanija", "MKD": "Sj. Makedonija"}
G = {c: gpd.read_file(os.path.join(GDIR, f"{c}.geojson")).to_crs(4326) for c in COUNTRIES}
allg = gpd.GeoDataFrame(geometry=[g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union
                                  for g in G.values()], crs=4326)
xmin, ymin, xmax, ymax = allg.total_bounds
VIEW = (xmin - 0.2, xmax + 0.2, ymin - 0.2, ymax + 0.2)

KG = {8: ("Csa", (255, 255, 0)), 9: ("Csb", (200, 200, 0)), 14: ("Cfa", (200, 255, 80)), 15: ("Cfb", (100, 255, 80)),
      7: ("BSk", (255, 220, 100)), 6: ("BSh", (245, 165, 0)), 18: ("Dsb", (200, 0, 200)), 25: ("Dfa", (0, 255, 255)),
      26: ("Dfb", (55, 200, 255)), 27: ("Dfc", (0, 125, 125)), 29: ("ET", (178, 178, 178)),
      10: ("Csc", (150, 150, 0)), 17: ("Dsa", (255, 0, 255)), 19: ("Dsc", (150, 50, 150)), 30: ("EF", (102, 102, 102))}
DESC = {"Csa": "Mediteranska klima s vrućim ljetom", "Csb": "Mediteranska klima s toplim ljetom",
        "Cfa": "Umjereno topla vlažna klima s vrućim ljetom", "Cfb": "Umjereno topla vlažna klima s toplim ljetom",
        "BSk": "Hladna stepska klima", "BSh": "Vruća stepska klima", "Dsb": "Kontinentalna klima sa suhim ljetom",
        "Dfa": "Vlažna kontinentalna klima s vrućim ljetom", "Dfb": "Vlažna kontinentalna klima s toplim ljetom",
        "Dfc": "Subalpska (planinska) klima", "ET": "Planinska tundra", "Csc": "Mediteranska klima s hladnim ljetom",
        "Dsa": "Kontinentalna klima sa suhim, vrućim ljetom", "Dsc": "Subalpska klima sa suhim ljetom",
        "EF": "Vječni led"}
cols = [(1, 1, 1, 0)] * 31
for c, (_, rgb) in KG.items():
    cols[c] = tuple(v / 255 for v in rgb) + (1,)
cmap = ListedColormap(cols); norm = BoundaryNorm(np.arange(-0.5, 31.5, 1), cmap.N)

KEYS = [(REF, None)] + [("2071_2099", s) for s in ("ssp126", "ssp245", "ssp585")]


def path(p, s):
    return os.path.join(KG_DIR, p, *([s] if s else []), "koppen_geiger_0p00833333.tif")


def region(p, s):
    with rasterio.open(path(p, s)) as r:
        a, tr = mask(r, allg.geometry, crop=True, nodata=0)
    a = a[0]; h, w = a.shape
    return a, (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f)


def shares(p, s, c):
    with rasterio.open(path(p, s)) as r:
        a, tr = mask(r, G[c].geometry, crop=True, nodata=0)
    a = a[0]
    lats = tr.f + tr.e * (np.arange(a.shape[0]) + 0.5)
    wgt = np.cos(np.deg2rad(lats))[:, None] * np.ones(a.shape)
    tot = wgt[a > 0].sum()
    return {KG[k][0]: wgt[a == k].sum() / tot * 100 for k in np.unique(a[a > 0]) if k in KG}


# 1) karte regije
for PF in FUT:
    fig, axes = plt.subplots(2, 2, figsize=(16, 19))
    used = set()
    for ax, (p, s) in zip(axes.ravel(), [(REF, None)] + [(PF, s_) for s_ in ("ssp126", "ssp245", "ssp585")]):
        a, ext = region(p, s)
        used |= {KG[k][0] for k in np.unique(a) if k in KG}
        ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3])
        ax.set_aspect(1 / np.cos(np.deg2rad((ymin + ymax) / 2)))
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldShadedRelief, attribution=False, zoom=8)
        ax.imshow(np.ma.masked_equal(a, 0), extent=ext, cmap=cmap, norm=norm, interpolation="nearest", alpha=0.82, zorder=2)
        for c, g in G.items():
            g.boundary.plot(ax=ax, color="#111" if c == "BIH" else "#333", lw=1.6 if c == "BIH" else 0.8, zorder=3)
        halo = [pe.withStroke(linewidth=3, foreground="white")]
        for c, (lx, ly) in {"HRV": (15.6, 45.75), "SRB": (20.8, 44.4), "MNE": (19.25, 42.9), "BIH": (17.75, 44.25),
                            "SVN": (14.8, 46.15), "XKX": (20.9, 42.6), "ALB": (20.0, 40.9), "MKD": (21.7, 41.6)}.items():
            ax.text(lx, ly, COUNTRIES[c], fontsize=11, ha="center", zorder=5, path_effects=halo,
                    fontweight="bold" if c == "BIH" else "normal")
        ax.set_title(f"Referentni period {period_label(REF)}" if s is None else f"{period_label(p)} · {SSP[s]}",
                     fontsize=15, loc="left", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#888")
    order = [k for k in ["Csa", "Csb", "Csc", "BSh", "BSk", "Cfa", "Cfb", "Dsa", "Dsb", "Dsc", "Dfa", "Dfb", "Dfc", "ET", "EF"] if k in used]
    hs = [Patch(facecolor=tuple(v / 255 for v in dict((n, rgb) for n, rgb in KG.values())[k]), edgecolor="#444", lw=0.5,
                label=f"$\\mathbf{{{k}}}$  {DESC[k]}") for k in order]
    fig.legend(handles=hs, loc="lower left", ncol=3, frameon=False, fontsize=12, bbox_to_anchor=(0.01, 0.0),
               title="Klimatski tipovi prema Köppen-Geigerovoj klasifikaciji",
               title_fontproperties={"weight": "bold", "size": 14}, alignment="left")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.1, wspace=0.04, hspace=0.08)
    fig.text(0.01, 0.97, "Köppen-Geigerova klasifikacija klime", fontsize=27, ha="left", va="baseline")
    fig.text(0.01, 0.945, f"• BiH i zemlje regije: danas i {period_label(PF)}", fontsize=27, ha="left", va="baseline")
    B.put_credit(fig, fig.transFigure, 0.99, 0.945, 15)
    fig.savefig(os.path.join(OUT, f"susjedi_karte_{PF}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ susjedi_karte_{PF}.png")

# 2) udjeli po zemljama
ALL_KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in ("ssp126", "ssp245", "ssp585")]
S = {c: {k: shares(*k, c) for k in ALL_KEYS} for c in COUNTRIES}
classes = [k for k in ["Csa", "Csb", "Csc", "BSh", "BSk", "Cfa", "Cfb", "Dsa", "Dsb", "Dsc", "Dfa", "Dfb", "Dfc", "ET", "EF"]
           if any(S[c][key].get(k, 0) >= 0.3 for c in S for key in ALL_KEYS)]
rgbs = dict((n, tuple(v / 255 for v in rgb)) for n, rgb in KG.values())
fig, axes = plt.subplots(2, 4, figsize=(20, 14), sharey=True)
labels = [period_label(p)[2:4] + "–" + period_label(p)[-2:] if s is None
          else period_label(p)[2:4] + "–" + period_label(p)[-2:] + "\n" + ssp_code(s).replace("SSP", "") for p, s in ALL_KEYS]
for ax, c in zip(axes.ravel(), COUNTRIES):
    bottom = np.zeros(len(ALL_KEYS))
    for k in classes:
        v = np.array([S[c][key].get(k, 0) for key in ALL_KEYS])
        ax.bar(range(len(ALL_KEYS)), v, bottom=bottom, color=rgbs[k], edgecolor="white", lw=0.6, width=0.84, label=k)
        for i, (vv, b) in enumerate(zip(v, bottom)):
            if vv >= 12:
                ax.text(i, b + vv / 2, f"{vv:.0f}", ha="center", va="center", fontsize=8.5, color="#111")
        bottom += v
    ax.axvline(3.5, color="#333", lw=1, ls="--")
    ax.set_xticks(range(len(ALL_KEYS))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(COUNTRIES[c], fontsize=16, loc="left", pad=10, fontweight="bold" if c == "BIH" else "normal")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#333")
axes[0, 0].set_ylabel("Udio površine (%)", fontsize=12); axes[1, 0].set_ylabel("Udio površine (%)", fontsize=12)
hs = [Patch(facecolor=rgbs[k], edgecolor="#444", lw=0.5, label=f"$\\mathbf{{{k}}}$  {DESC[k]}") for k in classes]
fig.legend(handles=hs, loc="lower left", ncol=4, frameon=False, fontsize=11.5, bbox_to_anchor=(0.01, 0.02),
           title="Klimatski tipovi prema Köppen-Geigerovoj klasifikaciji",
           title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.subplots_adjust(left=0.045, right=0.99, top=0.87, bottom=0.17, wspace=0.06, hspace=0.32)
fig.text(0.01, 0.965, "Köppen-Geigerova klasifikacija klime", fontsize=27, ha="left", va="baseline")
fig.text(0.01, 0.93, "• Udio klimatskih tipova: BiH i zemlje regije (1901–2099)", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.99, 0.93, 15)
fig.savefig(os.path.join(OUT, "susjedi_udjeli.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ susjedi_udjeli.png")

with open(os.path.join(OUT, "susjedi_udjeli.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["zemlja", "period", "scenarij"] + classes)
    for c in COUNTRIES:
        for key in ALL_KEYS:
            w.writerow([COUNTRIES[c], period_label(key[0]), key[1].upper() if key[1] else "historijski"]
                       + [f"{S[c][key].get(k, 0):.1f}" for k in classes])
print("✓ susjedi_udjeli.csv")
for c in COUNTRIES:
    print(COUNTRIES[c], {k: round(v) for k, v in S[c][(REF, None)].items() if v > 3}, "→ SSP5-8.5",
          {k: round(v) for k, v in S[c][("2071_2099", "ssp585")].items() if v > 3})
