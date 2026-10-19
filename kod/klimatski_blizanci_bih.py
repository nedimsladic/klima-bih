"""
Klimatski blizanci: koji grad u Evropi i Mediteranu danas (1991–2020) ima klimu kakvu bi grad u BiH
mogao imati u budućnosti.

Za svaki grad: 12 mjesečnih temperatura i 12 mjesečnih padavina (√mm) iz Beck et al. (2023), 0,1°.
Svaka od 24 veličine standardizuje se na skupu kandidata, a blizanac je kandidat s najmanjom
euklidskom udaljenošću. Grad nikad nije sam sebi blizanac. Kandidati: gradovi s ≥ 10.000 stanovnika (Natural Earth) između 12° Z i
45° I te 30° i 62° S, plus 20 gradova BiH.

Kvalitet podudaranja: udaljenost blizanca podijeljena medijanom udaljenosti današnjeg grada u BiH do njegovog
najbližeg drugog grada (tipična razlika između dva slična grada danas). Do 1,5: dobro; 1,5–2,5: umjereno;
preko 2,5: slabo (danas nema grada s tako toplim ljetom uz takve zime i padavine).

Upotreba:
    python klimatski_blizanci_bih.py <climate_0p1_dir> <ne_populated_places.shp> <bih_boundary.geojson> <out_dir>
"""
import sys
import os
import csv
import json

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import contextily as ctx

from bih_stil import Bih, FUT, REF, SSP, SSP_COLORS, CITIES, PEAKS, DPI, period_label, ssp_code

CLIM, NE, BOUNDARY, OUT = sys.argv[1:5]
B = Bih(BOUNDARY, OUT)
DOM = dict(lon=(-12, 45), lat=(62, 30))

# Bosanski nazivi zemalja i nekih gradova
COUNTRY = {
    "Bosnia and Herzegovina": "BiH", "Croatia": "Hrvatska", "Serbia": "Srbija", "Montenegro": "Crna Gora",
    "Slovenia": "Slovenija", "Macedonia": "Sj. Makedonija", "North Macedonia": "Sj. Makedonija",
    "Albania": "Albanija", "Kosovo": "Kosovo", "Bulgaria": "Bugarska", "Romania": "Rumunija",
    "Hungary": "Mađarska", "Austria": "Austrija", "Italy": "Italija", "Greece": "Grčka", "Turkey": "Turska",
    "Spain": "Španija", "Portugal": "Portugal", "France": "Francuska", "Germany": "Njemačka",
    "Switzerland": "Švicarska", "Slovakia": "Slovačka", "Czech Republic": "Češka", "Czechia": "Češka",
    "Poland": "Poljska", "Ukraine": "Ukrajina", "Moldova": "Moldavija", "Russia": "Rusija",
    "Georgia": "Gruzija", "Armenia": "Armenija", "Azerbaijan": "Azerbejdžan", "Cyprus": "Kipar",
    "Syria": "Sirija", "Lebanon": "Liban", "Israel": "Izrael", "Jordan": "Jordan", "Iraq": "Irak",
    "Iran": "Iran", "Tunisia": "Tunis", "Algeria": "Alžir", "Morocco": "Maroko", "Libya": "Libija",
    "Egypt": "Egipat", "Malta": "Malta", "Belgium": "Belgija", "Netherlands": "Holandija",
    "United Kingdom": "UK", "Ireland": "Irska", "Denmark": "Danska", "Belarus": "Bjelorusija",
    "Lithuania": "Litvanija", "Latvia": "Latvija", "Estonia": "Estonija", "Sweden": "Švedska",
    "Norway": "Norveška", "Finland": "Finska", "Luxembourg": "Luksemburg", "Kazakhstan": "Kazahstan",
}
CITY = {"Naples": "Napulj", "Rome": "Rim", "Thessaloniki": "Solun", "Athens": "Atina", "Belgrade": "Beograd",
        "Bucharest": "Bukurešt", "Vienna": "Beč", "Milan": "Milano", "Venice": "Venecija", "Florence": "Firenca",
        "Turin": "Torino", "Genoa": "Đenova", "Lisbon": "Lisabon", "Seville": "Sevilja", "Istanbul": "Istanbul",
        "Tirana": "Tirana", "Skopje": "Skoplje", "Sofia": "Sofija", "Podgorica": "Podgorica",
        "Zagreb": "Zagreb", "Split": "Split", "Rijeka": "Rijeka", "Nis": "Niš", "Novi Sad": "Novi Sad",
        "Ljubljana": "Ljubljana", "Budapest": "Budimpešta", "Bari": "Bari", "Plovdiv": "Plovdiv", "Pec": "Peć"}


def load(p, s):
    f = os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")
    return xr.open_dataset(f).sel(lon=slice(*DOM["lon"]), lat=slice(*DOM["lat"]))


def feat(ds, lon, lat):
    pt = ds.sel(lon=xr.DataArray(lon, dims="c"), lat=xr.DataArray(lat, dims="c"), method="nearest")
    t = pt["air_temperature"].values.T                      # (c, 12)
    p = np.sqrt(np.maximum(pt["precipitation"].values.T, 0))
    return np.hstack([t, p])


ne = gpd.read_file(NE)
ne = ne[(ne.longitude.between(*DOM["lon"])) & (ne.latitude.between(DOM["lat"][1], DOM["lat"][0]))
        & (ne.pop_max >= 10000)].reset_index(drop=True)
bih_names = [n for n in CITIES if n not in PEAKS]
cand_names = [f"{CITY.get(r['name'], r['name'])} ({COUNTRY.get(r['adm0name'], r['adm0name'])})"
              for _, r in ne.iterrows()] + [f"{n} (BiH)" for n in bih_names]
cand_lon = np.r_[ne.longitude.values, [CITIES[n][0] for n in bih_names]]
cand_lat = np.r_[ne.latitude.values, [CITIES[n][1] for n in bih_names]]

REF_DS = load(REF, None)
C = feat(REF_DS, cand_lon, cand_lat)
ok = np.isfinite(C).all(axis=1)
C, cand_lon, cand_lat = C[ok], cand_lon[ok], cand_lat[ok]
cand_names = [n for n, o in zip(cand_names, ok) if o]
# ukloni duplikate imena (isti grad više puta)
_, uniq = np.unique(cand_names, return_index=True)
uniq = np.sort(uniq)
C, cand_lon, cand_lat = C[uniq], cand_lon[uniq], cand_lat[uniq]
cand_names = [cand_names[i] for i in uniq]
MU, SD = C.mean(axis=0), C.std(axis=0)
Cz = (C - MU) / SD
print("Kandidata:", len(cand_names))

SSPS = ["ssp119", "ssp126", "ssp434", "ssp245", "ssp460", "ssp370", "ssp585"]
SCEN = [(p, s) for p in ("2041_2070", "2071_2099") for s in SSPS]
blon = np.array([CITIES[n][0] for n in bih_names]); blat = np.array([CITIES[n][1] for n in bih_names])
SELF = [cand_names.index(f"{n} (BiH)") if f"{n} (BiH)" in cand_names else None for n in bih_names]


def match(F):
    d = np.sqrt(((F[:, None, :] - Cz[None, :, :]) ** 2).sum(axis=2))
    for i, j in enumerate(SELF):                   # grad nije sam sebi blizanac
        if j is not None:
            d[i, j] = np.inf
    return d


# kalibracija: današnji grad u BiH i njegov najbliži drugi grad
REF_B = feat(REF_DS, blon, blat)
BASE = float(np.median(match((REF_B - MU) / SD).min(axis=1)))
print("Medijan današnje udaljenosti do najbližeg drugog grada:", round(BASE, 2))
QUAL = [(1.5, "dobro", "#2e7d4f"), (2.5, "umjereno", "#c88a12"), (99, "slabo", "#b23b3b")]


def quality(r):
    return next(q for q in QUAL if r <= q[0])


res = {}
for p, s in SCEN:
    Fr = feat(load(p, s), blon, blat)
    d = match((Fr - MU) / SD)
    best = d.argmin(axis=1)
    res[(p, s)] = [(cand_names[b], float(cand_lon[b]), float(cand_lat[b]), float(d[i, b]) / BASE,
                    float(C[b, 6] - Fr[i, 6])) for i, b in enumerate(best)]
for i, n in enumerate(bih_names):
    print(n, "→", " | ".join(f"{res[k][i][0]} ({res[k][i][3]:.1f})" for k in SCEN[7:]))

# tabele: jedna po periodu, svih 7 scenarija; boja ćelije = kvalitet podudaranja
order = sorted(range(len(bih_names)), key=lambda i: -blat[i])
for per in ("2041_2070", "2071_2099"):
    keys = [(per, s) for s in SSPS]
    fig, ax = plt.subplots(figsize=(24, 11.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    x0, cw = 0.09, (1 - 0.09) / 7
    y0 = 0.955; dy = 0.92 / len(order)
    for j, k in enumerate(keys):
        ax.text(x0 + j * cw + 0.004, y0 + 0.01, SSP[k[1]].replace(" · ", "\n"), fontsize=11.5, fontweight="bold",
                va="bottom", ha="left", color=SSP_COLORS[k[1]])
    for r, i in enumerate(order):
        y = y0 - (r + 1) * dy
        ax.text(0.0, y + dy / 2, bih_names[i], fontsize=12.5, va="center")
        for j, k in enumerate(keys):
            name, _, _, q, dtj = res[k][i]
            lim, ql, col = quality(q)
            ax.add_patch(plt.Rectangle((x0 + j * cw, y + dy * 0.06), cw - 0.004, dy * 0.88, facecolor=col,
                                       alpha=0.13, edgecolor="none"))
            ax.add_patch(plt.Rectangle((x0 + j * cw, y + dy * 0.06), 0.0028, dy * 0.88, facecolor=col, edgecolor="none"))
            ax.text(x0 + j * cw + 0.007, y + dy / 2, name, fontsize=10.5, va="center",
                    color="#555" if name.endswith("(BiH)") else "#111")
    fig.text(0.01, 0.985, "Klimatski blizanci gradova u Bosni i Hercegovini", fontsize=27, ha="left", va="top")
    fig.text(0.01, 0.94, f"• Grad koji danas (1991–2020) ima klimu kakvu bi grad u BiH mogao imati {period_label(per)}",
             fontsize=22, ha="left", va="top")
    B.put_credit(fig, fig.transFigure, 0.99, 0.94, 15)
    hs = [Patch(facecolor=c, alpha=0.6, edgecolor="none", label=f"{l} podudaranje") for _, l, c in QUAL]
    fig.legend(handles=hs, loc="lower left", ncol=3, frameon=False, fontsize=12.5, bbox_to_anchor=(0.01, 0.0),
               title="Kvalitet (slabo: danas nema grada s tako toplim ljetom uz takve zime i padavine)",
               title_fontproperties={"weight": "bold", "size": 12.5}, alignment="left")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.09)
    fig.savefig(os.path.join(OUT, f"klimatski_blizanci_tabela_{per}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("✓", f"klimatski_blizanci_tabela_{per}.png")

# karta: linije od grada u BiH do blizanca (SSP2-4.5 i SSP5-8.5, 2041–2070 i 2071–2099)
fig, axes = plt.subplots(2, 2, figsize=(18, 14.4))
for ax, k in zip(axes.ravel(), [(p, s) for p in ("2041_2070", "2071_2099") for s in ("ssp245", "ssp585")]):
    ax.set_xlim(9.5, 24.5); ax.set_ylim(39.6, 46.6)
    ax.set_aspect(1 / np.cos(np.deg2rad(43)))
    ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.Esri.WorldShadedRelief, attribution=False, zoom=7)
    B.border.boundary.plot(ax=ax, color="#222", lw=0.9, zorder=3)
    halo = [pe.withStroke(linewidth=2.4, foreground="white")]
    seen = {}
    for i, n in enumerate(bih_names):
        tn, tx, ty, _, _ = res[k][i]
        ax.annotate("", xy=(tx, ty), xytext=(blon[i], blat[i]),
                    arrowprops=dict(arrowstyle="-|>", color=SSP_COLORS[k[1]], lw=1.2, alpha=0.75,
                                    shrinkA=0, shrinkB=3, connectionstyle="arc3,rad=0.15"), zorder=4)
        seen.setdefault(tn, (tx, ty))
    for tn, (tx, ty) in seen.items():
        ax.plot(tx, ty, "o", ms=6, mfc="white", mec=SSP_COLORS[k[1]], mew=1.6, zorder=5)
        if not tn.endswith("(BiH)"):
            ax.annotate(tn, (tx, ty), xytext=(7, 0), textcoords="offset points", fontsize=11,
                        va="center", zorder=6, path_effects=halo)
    for i, n in enumerate(bih_names):
        ax.plot(blon[i], blat[i], "o", ms=3.5, mfc="#222", mec="white", mew=0.6, zorder=5)
    ax.set_title(f"{period_label(k[0])} · {SSP[k[1]]}", fontsize=15, loc="left", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")
fig.subplots_adjust(left=0.01, right=0.99, top=0.89, bottom=0.01, wspace=0.04, hspace=0.1)
fig.text(0.01, 0.972, "Klimatski blizanci gradova u Bosni i Hercegovini", fontsize=27, ha="left", va="baseline")
fig.text(0.01, 0.938, "• Strelica vodi od grada u BiH do grada koji danas ima njegovu buduću klimu", fontsize=22,
         ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.99, 0.938, 15)
fig.savefig(os.path.join(OUT, "klimatski_blizanci_karta.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ klimatski_blizanci_karta.png")

with open(os.path.join(OUT, "klimatski_blizanci.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["grad u BiH", "period", "scenarij", "klimatski blizanac", "relativna udaljenost",
                "kvalitet", "razlika jul (°C, blizanac − budući grad)"])
    for i in order:
        for k in SCEN:
            name, _, _, q, dtj = res[k][i]
            w.writerow([bih_names[i], period_label(k[0]), k[1].upper(), name, f"{q:.2f}", quality(q)[1], f"{dtj:+.1f}"])
print("✓ klimatski_blizanci.csv")
