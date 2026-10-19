"""
Jednostrani PDF (A4) za svaki od 78 većih gradova: klimatski tip, temperatura, vrući i mrazni dani, tropske noći
i snijeg za sve periode i tri scenarija, s grafikonima. Plus slika za dijeljenje linka (1200 × 630).

Upotreba:
    python pdf_gradovi_bih.py <gradovi.json> <bih_boundary.geojson> <out_pdf_dir> <og_png> <kg_karta_png>
"""
import sys
import os
import json
import unicodedata
import textwrap

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from PIL import Image

import bih_stil  # noqa: F401  (registruje Helvetica)
from bih_stil import SSP, SSP_COLORS, AUTHOR, AUTHOR_COLOR

DATA, BOUNDARY, OUTDIR, OG, KGMAP = sys.argv[1:6]
os.makedirs(OUTDIR, exist_ok=True)
D = json.load(open(DATA, encoding="utf-8"))
border = gpd.read_file(BOUNDARY).to_crs(4326)
xmin, ymin, xmax, ymax = border.total_bounds
k = 1000 / (xmax - xmin); c0 = np.cos(np.deg2rad((ymin + ymax) / 2))
KGC = {"Csa": "#ffff00", "Csb": "#c8c800", "Cfa": "#c8ff50", "Cfb": "#64ff50", "Dsb": "#c800c8", "Dfa": "#00ffff",
       "Dfb": "#37c8ff", "Dfc": "#007d7d", "ET": "#b2b2b2"}
NAMES = {"Csa": "Mediteranska, vruće ljeto", "Csb": "Mediteranska, toplo ljeto", "Cfa": "Umjereno topla, vruće ljeto",
         "Cfb": "Umjereno topla, toplo ljeto", "Dsb": "Kontinentalna, suho ljeto", "Dfa": "Kontinentalna, vruće ljeto",
         "Dfb": "Kontinentalna, toplo ljeto", "Dfc": "Subalpska (planinska)", "ET": "Planinska tundra"}
HIST = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, "1991_2020": 2005.5, "2041_2070": 2055.5, "2071_2099": 2085}
SEL = ["ssp126", "ssp245", "ssp585"]
VARS = [(0, "Srednja godišnja temperatura", "°C", 1), (1, "Vrući dani (≥ 30 °C)", "dana", 0),
        (2, "Mrazni dani (< 0 °C)", "dana", 0), (3, "Tropske noći (≥ 20 °C)", "noći", 0)]


def slug(n):
    return unicodedata.normalize("NFKD", n.replace("đ", "dj").replace("Đ", "Dj")).encode("ascii", "ignore").decode() \
        .lower().replace(" ", "-")


def dom(sh):
    return max(sh, key=sh.get) if sh else "–"


def fmt(v, d):
    return f"{v:.{d}f}".replace(".", ",")


geoms = border.geometry.union_all()
polys = geoms.geoms if geoms.geom_type == "MultiPolygon" else [geoms]
index = []
for m in D["m"]:
    d = m["d"]
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.06, 0.955, m["n"], fontsize=30, va="baseline")
    fig.text(0.06, 0.925, "Klima grada 1901–2099 · Köppen-Geigerova klasifikacija", fontsize=13, color="#444", va="baseline")
    fig.text(0.94, 0.955, AUTHOR.split("\n")[0], fontsize=9, color=AUTHOR_COLOR, ha="right", va="baseline")
    fig.text(0.94, 0.94, AUTHOR.split("\n")[1], fontsize=9, color=AUTHOR_COLOR, ha="right", va="baseline")
    # položaj na karti
    axm = fig.add_axes([0.56, 0.905, 0.12, 0.075])
    for pl in polys:
        xy = np.asarray(pl.exterior.coords)
        axm.fill(xy[:, 0], xy[:, 1], color="#e8ecea", ec="#999", lw=0.6)
    lon = xmin + m["cx"] / k; lat = ymax - m["cy"] * c0 / k
    axm.plot(lon, lat, "o", ms=7, mfc="#c0392b", mec="white", mew=1)
    axm.set_aspect(1 / c0); axm.axis("off")
    # sažetak
    now, fut = d["1991_2020|hist"], d["2071_2099|ssp245"]
    tn, tf = dom(now[5]), dom(fut[5])
    txt = (f"{m['n']} u periodu 1991–2020 ima klimu tipa {tn} ({NAMES.get(tn, tn).lower()}), srednju godišnju temperaturu "
           f"{fmt(now[0], 1)} °C i oko {now[1]:.0f} vrućih dana godišnje. Do 2071–2099 prema scenariju srednjih emisija (SSP2-4.5) "
           f"temperatura bi bila {fmt(fut[0], 1)} °C, s oko {fut[1]:.0f} vrućih dana" +
           (f", a klimatski tip {tf} ({NAMES.get(tf, tf).lower()})." if tf != tn else f"; klimatski tip ostaje {tn}."))
    if len(now) > 6:
        e0, e1 = now[6], fut[6]
        txt += (f" Potrebe za grijanjem (stepen-dani) bile bi {abs(e1[0] / e0[0] - 1) * 100:.0f} % "
                f"{'manje' if e1[0] < e0[0] else 'veće'}, a za hlađenjem {e0[1]:.0f} → {e1[1]:.0f} stepen-dana godišnje.")
    fig.text(0.06, 0.875, textwrap.fill(txt, 92), fontsize=10.5, va="top", color="#222", linespacing=1.45)
    # tabela
    rows = [("1961–1990", "1961_1990|hist"), ("1991–2020", "1991_2020|hist")] + \
           [(f"2041–2070 · {SSP[s].split(' · ')[0]}", f"2041_2070|{s}") for s in SEL] + \
           [(f"2071–2099 · {SSP[s].split(' · ')[0]}", f"2071_2099|{s}") for s in SEL]
    axt = fig.add_axes([0.06, 0.5, 0.88, 0.27]); axt.axis("off")
    heads = ["Period / scenarij", "Tip", "Temp. (°C)", "Vrući dani", "Mrazni dani", "Tropske noći", "Snijeg (dana)"]
    xs = [0, 0.32, 0.42, 0.54, 0.66, 0.78, 0.9]
    for x, h in zip(xs, heads):
        axt.text(x, 1, h, fontsize=9, fontweight="bold", va="top", color="#555")
    for i, (lab, kk) in enumerate(rows):
        y = 0.88 - i * 0.11
        v = d[kk]; t = dom(v[5])
        if i % 2 == 0:
            axt.add_patch(plt.Rectangle((-0.01, y - 0.075), 1.02, 0.1, color="#f2f4f2", zorder=0))
        axt.text(xs[0], y, lab, fontsize=10, va="center")
        axt.text(xs[1], y, t, fontsize=10, va="center", fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.25", fc=KGC.get(t, "#ddd"), ec="none"))
        for x, val, dd in zip(xs[2:], [v[0], v[1], v[2], v[3], v[4]], [1, 0, 0, 0, 0]):
            axt.text(x, y, fmt(val, dd), fontsize=10, va="center")
    # grafikoni
    for j, (vi, title, unit, dd) in enumerate(VARS):
        ax = fig.add_axes([0.08 + (j % 2) * 0.46, 0.3 - (j // 2) * 0.19, 0.38, 0.15])
        hy = [d[f"{p}|hist"][vi] for p in HIST]
        ax.plot([X[p] for p in HIST], hy, color="#222", lw=2, marker="o", ms=3.5)
        for s in SEL:
            ys = [hy[-1]] + [d[f"{p}|{s}"][vi] for p in ("2041_2070", "2071_2099")]
            ax.plot([X["1991_2020"], X["2041_2070"], X["2071_2099"]], ys, color=SSP_COLORS[s], lw=1.8, marker="o", ms=3)
        ax.set_title(title, fontsize=10.5, loc="left", pad=4)
        ax.set_xticks([1915.5, 1975.5, 2005.5, 2085]); ax.set_xticklabels(["1901–30", "1961–90", "1991–2020", "2071–99"], fontsize=7.5)
        ax.tick_params(labelsize=7.5); ax.set_ylabel(unit, fontsize=8)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    hs = [plt.Line2D([], [], color="#222", lw=2, label="Historijski periodi")] + \
         [plt.Line2D([], [], color=SSP_COLORS[s], lw=1.8, label=SSP[s].replace(" · ", " – ")) for s in SEL]
    fig.legend(handles=hs, loc="lower left", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.06, 0.035))
    fig.text(0.06, 0.012, "Izvori: Beck et al. (2023), Scientific Data 10:724; E-OBS v31.0e; GeoNames. Klimatski tip: kvadrat 5 × 5 km "
             "oko centra grada;\ntemperatura i dani: mreža od oko 10 km. Cijeli atlas: nedimsladic.github.io/klima-bih",
             fontsize=6.8, color="#666", linespacing=1.4)
    fn = f"klima_{slug(m['n'])}.pdf"
    with PdfPages(os.path.join(OUTDIR, fn)) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    index.append({"n": m["n"], "f": fn})
json.dump(index, open(os.path.join(OUTDIR, "index.json"), "w", encoding="utf-8"), ensure_ascii=False)
tot = sum(os.path.getsize(os.path.join(OUTDIR, f)) for f in os.listdir(OUTDIR))
print("✓", len(index), "PDF-ova,", round(tot / 1e6, 1), "MB")

# slika za dijeljenje linka (Open Graph, 1200 × 630)
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor="#f6f7f4")
fig.text(0.05, 0.8, "Klima Bosne\ni Hercegovine", fontsize=46, va="top", color="#18211f", linespacing=1.05)
fig.text(0.05, 0.47, "od 1901. do 2099.", fontsize=30, va="top", color="#155e63")
fig.text(0.05, 0.3, "Na slici: klimatski tipovi 2071–2099 prema SSP5-8.5.\nTemperatura, vrući i mrazni dani, kiša, gradovi.", fontsize=15,
         va="top", color="#444", linespacing=1.4)
fig.text(0.05, 0.08, "Analiza: Nedim Sladić · prateći Beck et al. (2018, 2023)", fontsize=12, color="#777")
im = Image.open(KGMAP)
im.seek(im.n_frames - 1)                             # posljednji okvir: 2071–2099
im = im.convert("RGB")
w, h = im.size
crop = im.crop((0, int(h * 0.075), w, int(h * 0.94)))   # bez zaglavlja i vremenske trake
ax = fig.add_axes([0.5, 0.0, 0.5, 1.0]); ax.imshow(crop, aspect="auto"); ax.axis("off")
fig.text(0.05, 0.62, "", fontsize=1)
fig.savefig(OG, dpi=100, facecolor="#f6f7f4")
plt.close(fig)
print("✓", OG)
