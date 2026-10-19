"""
Animacija Köppen-Geigerovih klimatskih tipova u BiH, 1901 → 2099, za odabrani scenarij.

Okviri: 1901–1930, 1931–1960, 1961–1990, 1991–2020, 2041–2070, 2071–2099. Između perioda
karta se postepeno pretapa. Izlaz: animirani WebP (za web stranicu) i GIF (za dijeljenje).

Upotreba:
    python animacija_bih.py <kg_tif_dir> <bih_boundary.geojson> <out_dir> [ssp245 ssp585 ...]
    python animacija_bih.py <kg_tif_dir> <bih_boundary.geojson> <out_dir> --regija <granice_dir> [ssp245 ...]

S --regija animacija pokriva BiH i 7 zemalja regije (granice_dir: BIH/SVN/HRV/SRB/MNE/XKX/ALB/MKD.geojson),
s glavnim gradovima, a izlaz je animacija_regija_<scenarij>.webp/.gif.
"""
import sys
import os
import io

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import contextily as ctx
from PIL import Image

from bih_stil import Bih, HIST, FUT, SSP, CITIES, PEAKS, period_label, AUTHOR, AUTHOR_COLOR

KG_DIR, BOUNDARY, OUT = sys.argv[1:4]
ARGS = sys.argv[4:]
REG = None
if "--regija" in ARGS:
    i = ARGS.index("--regija"); REG = ARGS[i + 1]; ARGS = ARGS[:i] + ARGS[i + 2:]
SCEN = ARGS or ["ssp245", "ssp585"]
B = Bih(BOUNDARY, OUT)
COUNTRIES = ["BIH", "SVN", "HRV", "SRB", "MNE", "XKX", "ALB", "MKD"]
if REG:
    G = {c: gpd.read_file(os.path.join(REG, f"{c}.geojson")).to_crs(4326) for c in COUNTRIES}
    UNION = gpd.GeoSeries([g.geometry.union_all() for g in G.values()], crs=4326).union_all()
    MASK = [UNION]
    x0, y0, x1, y1 = UNION.bounds
    VIEW = (x0 - 0.15, x1 + 0.15, y0 - 0.15, y1 + 0.15)
    ASPECT = 1 / np.cos(np.deg2rad((y0 + y1) / 2))
    ZOOM = 7
    PLACES = {"Ljubljana": (14.506, 46.051, "r"), "Zagreb": (15.982, 45.815, "r"), "Sarajevo": (18.413, 43.856, "r"),
              "Beograd": (20.457, 44.787, "r"), "Podgorica": (19.262, 42.441, "l"), "Priština": (21.166, 42.663, "r"),
              "Tirana": (19.819, 41.328, "l"), "Skoplje": (21.432, 41.996, "r"), "Split": (16.440, 43.508, "l"),
              "Novi Sad": (19.833, 45.267, "r"), "Banja Luka": (17.191, 44.772, "r"), "Niš": (21.896, 43.321, "r"),
              "Mostar": (17.808, 43.343, "r"), "Dubrovnik": (18.094, 42.650, "l")}
    TITLE = "BiH i zemlje regije"
    PREFIX = "animacija_regija_"
else:
    MASK = list(B.border.geometry)
    VIEW, ASPECT, ZOOM = B.view, B.aspect, 8
    PLACES = {n: (x, y, side) for n, (x, y, major, side) in CITIES.items() if major and n not in PEAKS}
    TITLE = "Bosna i Hercegovina"
    PREFIX = "animacija_"
KG = {8: ("Csa", (255, 255, 0)), 9: ("Csb", (200, 200, 0)), 14: ("Cfa", (200, 255, 80)), 15: ("Cfb", (100, 255, 80)),
      18: ("Dsb", (200, 0, 200)), 25: ("Dfa", (0, 255, 255)), 26: ("Dfb", (55, 200, 255)), 27: ("Dfc", (0, 125, 125)),
      29: ("ET", (178, 178, 178)), 6: ("BSh", (245, 165, 0)), 7: ("BSk", (255, 220, 100)),
      17: ("Dsa", (255, 0, 255)), 19: ("Dsc", (150, 50, 150))}
ORDER = [8, 9, 6, 7, 14, 15, 17, 18, 19, 25, 26, 27, 29]
DESC = {"Csa": "Mediteranska, vruće ljeto", "Cfa": "Umjereno topla, vruće ljeto", "Cfb": "Umjereno topla, toplo ljeto",
        "Dfb": "Kontinentalna, toplo ljeto", "Dfc": "Subalpska (planinska)"}
LUT = np.zeros((31, 3))
for c, (_, rgb) in KG.items():
    LUT[c] = np.array(rgb) / 255


def load(p, s):
    f = os.path.join(KG_DIR, p, *([s] if s else []), "koppen_geiger_0p00833333.tif")
    with rasterio.open(f) as r:
        a, tr = mask(r, MASK, crop=True, nodata=0)
    a = a[0]; h, w = a.shape
    return a, (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f)


# raspored (inči): naslov iznad karte, karta preko pune širine, vremenska traka ispod; bez praznih rubova
DPI = 110
MW = 9.4                                              # širina karte = širina slike
MH = MW * (VIEW[3] - VIEW[2]) * ASPECT / (VIEW[1] - VIEW[0])
HEAD, TL = 0.78, 0.5                                  # zaglavlje i vremenska traka
FIG_W, FIG_H = MW, HEAD + MH + TL


def render(rgb, ext, label, idx, scen, share, classes):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
    fy = lambda y: 1 - y / FIG_H                       # y od vrha, u inčima
    ax = fig.add_axes([0, fy(HEAD + MH), 1, MH / FIG_H])
    ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect(ASPECT, adjustable="datalim")
    ax.apply_aspect()                                   # karta ispunjava cijelu svoju površinu
    xl, yl = ax.get_xlim(), ax.get_ylim()               # podloga za malo veće područje, da rubovi ne ostanu prazni
    ax.set_xlim(xl[0] - 0.4, xl[1] + 0.4); ax.set_ylim(yl[0] - 0.4, yl[1] + 0.4)
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldShadedRelief, attribution=False, zoom=ZOOM)
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    alpha = (rgb.sum(axis=2) > 0).astype(float) * 0.85
    ax.imshow(np.dstack([rgb, alpha]), extent=ext, interpolation="nearest", zorder=2)
    if REG:
        for c, g in G.items():
            g.boundary.plot(ax=ax, color="#111" if c == "BIH" else "#333", lw=1.4 if c == "BIH" else 0.8, zorder=3)
    else:
        B.border.boundary.plot(ax=ax, color="#222", lw=1.2, zorder=3)
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    for name, (x, y, side) in PLACES.items():
        ax.plot(x, y, "o", ms=4.5, mfc="white", mec="#222", mew=0.8, zorder=4)
        ax.annotate(name, (x, y), xytext=(6 if side == "r" else -6, 0), textcoords="offset points",
                    ha="left" if side == "r" else "right", va="center_baseline", fontsize=10.5, zorder=5,
                    path_effects=halo)
    # period u crnom okviru, donji desni ugao karte
    ax.text(0.98, 0.025, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=26, fontweight="bold",
            color="white", zorder=10, bbox=dict(boxstyle="square,pad=0.35", facecolor="#111", edgecolor="none"))
    # legenda s udjelima, gornji desni ugao karte
    hs = [Patch(facecolor=tuple(np.array(KG[c][1]) / 255), edgecolor="#444", lw=0.5,
                label=f"{KG[c][0]} {share.get(c, 0):.0f} %") for c in classes]
    leg = ax.legend(handles=hs, loc="upper right", ncol=1, frameon=True, fontsize=11.5, handlelength=1.2,
                    borderaxespad=0.6, framealpha=0.92, edgecolor="none", fancybox=False)
    leg.set_zorder(10)
    # zaglavlje iznad karte
    x0, x1 = 0.12 / FIG_W, 1 - 0.12 / FIG_W
    fig.text(x0, fy(0.33), "Köppen-Geigerova klasifikacija klime", fontsize=20, va="baseline")
    fig.text(x0, fy(0.68), f"• {TITLE}, {SSP[scen].split(' · ')[0]}", fontsize=20, va="baseline")
    l1, l2 = AUTHOR.split("\n")
    fig.text(x1, fy(0.46), l1, fontsize=11, color=AUTHOR_COLOR, ha="right", va="baseline")
    fig.text(x1, fy(0.68), l2, fontsize=11, color=AUTHOR_COLOR, ha="right", va="baseline")
    # vremenska traka ispod karte
    tl = fig.add_axes([0.3 / FIG_W, fy(HEAD + MH + 0.24), 1 - 0.6 / FIG_W, 0.14 / FIG_H])
    tl.set_xlim(1901, 2099); tl.set_ylim(0, 1); tl.axis("off")
    mids = [1915.5, 1945.5, 1975.5, 2005.5, 2055.5, 2085]
    seg = np.interp(idx * 5, range(6), mids)
    tl.plot([mids[0], mids[-1]], [0.5, 0.5], color="#ccc", lw=3, solid_capstyle="round")
    tl.plot([mids[0], seg], [0.5, 0.5], color="#155e63", lw=3, solid_capstyle="round")
    for m, p in zip(mids, HIST + FUT):
        on = m <= seg + 0.1
        tl.plot(m, 0.5, "o", ms=8, mfc="#155e63" if on else "white", mec="#155e63", mew=1.5)
        tl.text(m, -0.4, period_label(p), ha="center", va="top", fontsize=10, color="#222" if on else "#888")
    tl.axvline(2030.5, color="#999", lw=1, ls=":")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def share_of(a, lats):
    w = np.cos(np.deg2rad(lats))[:, None] * np.ones(a.shape)
    tot = w[a > 0].sum()
    return {int(c): w[a == c].sum() / tot * 100 for c in np.unique(a[a > 0])}


for scen in SCEN:
    keys = [(p, None) for p in HIST] + [(p, scen) for p in FUT]
    maps = [load(*k) for k in keys]
    ext = maps[0][1]
    h = maps[0][0].shape[0]
    lats = ext[3] + (ext[2] - ext[3]) * (np.arange(h) + 0.5) / h
    allsh = [share_of(m[0], lats) for m in maps]
    classes = [c for c in ORDER if any(sh.get(c, 0) >= 0.5 for sh in allsh)]
    frames, durations = [], []
    n = len(keys)
    for i in range(n):
        a = maps[i][0]
        sh = share_of(a, lats)
        rgb = LUT[a]
        lab = period_label(keys[i][0])
        frames.append(render(rgb, ext, lab, i / (n - 1), scen, sh, classes)); durations.append(1800 if i in (3, n - 1) else 1300)
        if i < n - 1:                                      # pretapanje prema sljedećem periodu
            b = maps[i + 1][0]
            for t in (0.33, 0.67):
                mix = rgb * (1 - t) + LUT[b] * t
                frames.append(render(mix, ext, lab if t < 0.5 else period_label(keys[i + 1][0]),
                                     (i + t) / (n - 1), scen, sh if t < 0.5 else share_of(b, lats), classes))
                durations.append(140)
    webp = os.path.join(OUT, f"{PREFIX}{scen}.webp")
    frames[0].save(webp, save_all=True, append_images=frames[1:], duration=durations, loop=0, quality=82, method=6)
    gif = os.path.join(OUT, f"{PREFIX}{scen}.gif")
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print("✓", webp, round(os.path.getsize(webp) / 1e6, 1), "MB;", gif, round(os.path.getsize(gif) / 1e6, 1), "MB")
