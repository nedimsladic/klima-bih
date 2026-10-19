"""
Riječni slivovi u BiH: Una, Vrbas, Bosna, Drina, Neretva i ostatak (direktni sliv Save, Trebišnjica, Cetina…).

Slivovi: HydroBASINS v1c (Lehner & Grill 2013), nivo 6, a za Vrbas nivo 7 (na nivou 6 Vrbas je spojen s glavnim
tokom Save; na nivou 7 sliv je uzvodno od Laktaša, oko 5.100 km²). Jedinica pripada slivu ako njen tok nizvodno
(NEXT_DOWN) prolazi kroz jedinicu s kontrolnom tačkom na toj rijeci. Za dio sliva unutar BiH:
udio klimatskih tipova (1 km), srednja godišnja temperatura i ljetni vodni bilans (0,1°), vrući dani (E-OBS + delta).

Upotreba:
    python slivovi_bih.py <hybas_lev06.shp> <bih_boundary.geojson> <kg_tif_dir> <climate_0p1_dir> <dani_svi.npz> <out_dir>
"""
import sys
import os
import csv

import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import contextily as ctx

from bih_stil import Bih, HIST, FUT, REF, SSP, SSP_COLORS, SSP_DASH, DPI, period_label, ssp_code

HB, BOUNDARY, KG_DIR, CLIM, NPZ, OUT = sys.argv[1:7]
B = Bih(BOUNDARY, OUT)
geom = B.border.geometry.union_all()
# kontrolne tačke na rijekama (malo uzvodno od ušća, da jedinica ne obuhvati glavni tok Save)
GAUGE = {"Una": (16.62, 45.10), "Bosna": (18.30, 44.95), "Drina": (19.27, 44.60), "Neretva": (17.72, 43.12)}
VRBAS7 = (17.31, 44.90)
COLORS = {"Una": "#4e79a7", "Vrbas": "#59a14f", "Bosna": "#e15759", "Drina": "#76b7b2", "Neretva": "#edc948",
          "Ostalo": "#bab0ac"}

hb = gpd.read_file(HB, bbox=(15.0, 42.0, 20.5, 46.0)).to_crs(4326)
down = dict(zip(hb.HYBAS_ID, hb.NEXT_DOWN))
gid = {r: hb[hb.contains(Point(*xy))].HYBAS_ID.iloc[0] for r, xy in GAUGE.items()}


def river(h):
    seen = 0
    while h and seen < 200:
        for r, g in gid.items():
            if h == g:
                return r
        h = down.get(h, 0); seen += 1
    return "Ostalo"


hb["sliv"] = hb.HYBAS_ID.map(river)
bas = hb.dissolve("sliv").reset_index()[["sliv", "geometry"]]
# Vrbas s nivoa 7: sve jedinice čiji tok nizvodno prolazi kroz jedinicu kod Laktaša
h7 = gpd.read_file(HB.replace("lev06", "lev07"), bbox=(15.0, 42.0, 20.5, 46.0)).to_crs(4326)
d7 = dict(zip(h7.HYBAS_ID, h7.NEXT_DOWN))
g7 = h7[h7.contains(Point(*VRBAS7))].HYBAS_ID.iloc[0]


def to_vrbas(h):
    for _ in range(200):
        if h == g7:
            return True
        h = d7.get(h, 0)
        if not h:
            return False
    return False


vrb = h7[h7.HYBAS_ID.map(to_vrbas)].geometry.union_all()
bas["geometry"] = bas.geometry.difference(vrb)
bas = pd.concat([bas, gpd.GeoDataFrame({"sliv": ["Vrbas"], "geometry": [vrb]}, crs=4326)], ignore_index=True)
bas["geometry"] = bas.geometry.intersection(geom)
bas = bas[~bas.geometry.is_empty].copy()
bas["km2"] = bas.to_crs(3035).area / 1e6
order = ["Una", "Vrbas", "Bosna", "Drina", "Neretva", "Ostalo"]
bas = bas.set_index("sliv").loc[[o for o in order if o in set(bas.sliv)]].reset_index()
print(bas[["sliv", "km2"]].round(0).to_string(index=False))

KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
TYPES = {8: "Csa", 9: "Csb", 14: "Cfa", 15: "Cfb", 25: "Dfa", 26: "Dfb", 27: "Dfc"}
KGC = {"Csa": "#ffff00", "Csb": "#c8c800", "Cfa": "#c8ff50", "Cfb": "#64ff50", "Dfa": "#00ffff", "Dfb": "#37c8ff", "Dfc": "#007d7d"}


def kg_shares(p, s, g):
    with rasterio.open(os.path.join(KG_DIR, p, *([s] if s else []), "koppen_geiger_0p00833333.tif")) as r:
        a, tr = mask(r, [g], crop=True, nodata=0)
    a = a[0]; lats = tr.f + tr.e * (np.arange(a.shape[0]) + 0.5)
    w = np.cos(np.deg2rad(lats))[:, None] * np.ones(a.shape); tot = w[a > 0].sum()
    return {TYPES[c]: w[a == c].sum() / tot * 100 for c in TYPES}


def field_mean(lon, lat, arr, g):
    res = abs(lon[1] - lon[0])
    ins = ~geometry_mask([g], arr.shape, from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res))
    w = np.cos(np.deg2rad(lat))[:, None] * ins
    ok = ins & np.isfinite(arr)
    return float((arr[ok] * w[ok]).sum() / w[ok].sum())


Z = np.load(NPZ)
DAYS = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
R = {}
for _, b in bas.iterrows():
    for key in KEYS:
        p, s = key
        ds = xr.open_dataset(os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")).sel(
            lon=slice(15.3, 20.0), lat=slice(45.6, 42.2))
        lon, lat = ds["lon"].values, ds["lat"].values
        t = ds["air_temperature"].values; pr = ds["precipitation"].values
        R[(b.sliv, key)] = {
            "t": field_mean(lon, lat, t.mean(0), b.geometry),
            "p_ljeto": field_mean(lon, lat, pr[5:8].sum(0), b.geometry),
            "vruci": field_mean(Z["lon"], Z["lat"], Z[f"vruci|{p}|{s or 'hist'}"], b.geometry),
            "kg": kg_shares(p, s, b.geometry),
        }
print("✓ izračunato")

# 1) karta slivova + udio tipova + temperatura
fig = plt.figure(figsize=(19, 13))
axm = fig.add_axes([0.0, 0.03, 0.45, 0.82], anchor="N")
axm.set_xlim(B.view[0], B.view[1]); axm.set_ylim(B.view[2], B.view[3]); axm.set_aspect(B.aspect)
ctx.add_basemap(axm, crs="EPSG:4326", source=ctx.providers.Esri.WorldShadedRelief, attribution=False, zoom=8)
for _, b in bas.iterrows():
    gpd.GeoSeries([b.geometry], crs=4326).plot(ax=axm, color=COLORS[b.sliv], alpha=0.7, edgecolor="white", lw=1, zorder=2)
    c = b.geometry.representative_point()
    axm.text(c.x, c.y, f"{b.sliv}\n{b.km2:,.0f} km²".replace(",", "."), ha="center", va="center", fontsize=12,
             fontweight="bold", zorder=4, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
B.border.boundary.plot(ax=axm, color="#222", lw=1.2, zorder=3)
axm.set_xticks([]); axm.set_yticks([])
for sp in axm.spines.values():
    sp.set_edgecolor("#888")
names = list(bas.sliv)
# udio klimatskih tipova: 1991–2020, 2041–2070 i 2071–2099 (SSP2-4.5 / SSP5-8.5)
axb = fig.add_axes([0.53, 0.43, 0.46, 0.42])
cols = [((REF, None), "1991–2020"), (("2041_2070", "ssp245"), "2041–70 SSP2-4.5"), (("2041_2070", "ssp585"), "2041–70 SSP5-8.5"),
        (("2071_2099", "ssp245"), "2071–99 SSP2-4.5"), (("2071_2099", "ssp585"), "2071–99 SSP5-8.5")]
y = 0; yt = []; yl = []
for n in names:
    for key, lab in cols:
        left = 0
        for c in ["Csa", "Csb", "Cfa", "Cfb", "Dfa", "Dfb", "Dfc"]:
            v = R[(n, key)]["kg"][c]
            axb.barh(y, v, left=left, color=KGC[c], height=0.82, edgecolor="white", lw=0.5)
            if v >= 14:
                axb.text(left + v / 2, y, f"{c} {v:.0f}", ha="center", va="center", fontsize=8.5)
            left += v
        yt.append(y); yl.append(f"{n} · {lab}" if lab == "1991–2020" else lab); y -= 1
    y -= 0.6
axb.set_yticks(yt); axb.set_yticklabels(yl, fontsize=8.5); axb.set_xlim(0, 100)
axb.set_xlabel("udio površine sliva u BiH (%)", fontsize=10.5)
axb.set_title("Klimatski tipovi po slivovima", fontsize=15, loc="left", pad=8)
axb.spines[["top", "right"]].set_visible(False)
# temperatura i ljetne padavine
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5, "2041_2070": 2055.5, "2071_2099": 2085}
for j, (var, title, ylab) in enumerate([("t", "Srednja godišnja temperatura", "°C"),
                                         ("p_ljeto", "Ljetne padavine (JJA)", "mm")]):
    ax = fig.add_axes([0.53 + j * 0.25, 0.07, 0.21, 0.24])
    for n in names:
        hy = [R[(n, (p, None))][var] for p in HIST]
        ax.plot([X[p] for p in HIST], hy, color=COLORS[n], lw=1.8, marker="o", ms=3.5)
        for s, ls in (("ssp245", "--"), ("ssp585", "-")):
            ys = [hy[-1]] + [R[(n, (p, s))][var] for p in FUT]
            ax.plot([X[REF]] + [X[p] for p in FUT], ys, color=COLORS[n], lw=1.8 if s == "ssp585" else 1.3, ls=ls)
    ax.set_title(title, fontsize=13.5, loc="left", pad=6)
    ax.set_xticks([X[p] for p in ("1901_1930", "1961_1990", REF, "2041_2070", "2071_2099")])
    ax.set_xticklabels(["1901–\n1930", "1961–\n1990", "1991–\n2020", "2041–\n2070", "2071–\n2099"], fontsize=9)
    ax.set_ylabel(ylab, fontsize=10.5); ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
fig.legend(handles=[plt.Line2D([], [], color="#555", lw=1.3, ls="--", label="SSP2-4.5"),
                    plt.Line2D([], [], color="#555", lw=1.8, label="SSP5-8.5")] +
           [Patch(color=COLORS[n], label=n) for n in names],
           loc="lower right", ncol=8, frameon=False, fontsize=10.5, bbox_to_anchor=(0.99, -0.01))
fig.text(0.01, 0.962, "Riječni slivovi u Bosni i Hercegovini", fontsize=27, ha="left", va="baseline")
fig.text(0.01, 0.922, "• Klima Une, Vrbasa, Bosne, Drine i Neretve (dio sliva u BiH)", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.99, 0.922, 15)
fig.savefig(os.path.join(OUT, "slivovi_bih.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ slivovi_bih.png")

with open(os.path.join(OUT, "slivovi_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sliv", "površina u BiH (km²)", "period", "scenarij", "srednja temperatura (°C)", "ljetne padavine (mm)",
                "vrući dani"] + list(KGC))
    for _, b in bas.iterrows():
        for key in KEYS:
            r = R[(b.sliv, key)]
            w.writerow([b.sliv, f"{b.km2:.0f}", period_label(key[0]), key[1].upper() if key[1] else "historijski",
                        f"{r['t']:.2f}", f"{r['p_ljeto']:.0f}", f"{r['vruci']:.1f}"] + [f"{r['kg'][c]:.1f}" for c in KGC])
print("✓ slivovi_bih.csv")
