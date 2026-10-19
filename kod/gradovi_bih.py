"""
Podaci za preglednik "Gradovi" na web stranici: veći gradovi BiH kao tačke.

Gradovi: naselja iz GeoNames (BA.txt) s ≥ 10.000 stanovnika (sjedišta, PPLC/PPLA*) i službeni gradovi koje
GeoNames navodi kao manje. Za svaki grad i svaki period/scenarij:
  udjeli Köppen-Geigerovih tipova u prozoru 5 × 5 km oko centra (1 km, Beck et al. 2023),
  srednja godišnja temperatura (0,1°), vrući i mrazni dani, tropske noći i dani sa snijegom (dani_svi.npz).

Upotreba:
    python gradovi_bih.py <BA.txt> <bih_boundary.geojson> <kg_tif_dir> <climate_0p1_dir> <dani_svi.npz> <out_json> <out_csv> [hladnoca_kisa.npz]

Sedmi element za svaki period (ako je dat hladnoca_kisa.npz): [stepen-dani grijanja, stepen-dani hlađenja,
sati zimskog hlađenja, Rx1day delta, Rx1day uz +7 %/°C].
"""
import sys
import os
import csv
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio

GN, BOUNDARY, KG_DIR, CLIM, NPZ, OUT_JSON, OUT_CSV = sys.argv[1:8]
HK = sys.argv[8] if len(sys.argv) > 8 else None
HIST = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
FUT = ["2041_2070", "2071_2099"]
SSP = ["ssp119", "ssp126", "ssp434", "ssp245", "ssp460", "ssp370", "ssp585"]
KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
TYPES = {8: "Csa", 9: "Csb", 14: "Cfa", 15: "Cfb", 18: "Dsb", 25: "Dfa", 26: "Dfb", 27: "Dfc", 29: "ET"}
EXTRA = ["Zvornik", "Široki Brijeg", "Lukavica", "Derventa", "Laktaši", "Srebrenik", "Stolac", "Čapljina",
         "Ljubuški", "Tešanj", "Modriča", "Teslić", "Prnjavor", "Kozarska Dubica", "Novi Grad", "Kiseljak",
         "Vitez", "Tomislavgrad", "Neum", "Gacko", "Pale", "Višegrad", "Ključ", "Bosanski Petrovac", "Jajce",
         "Kupres", "Posušje", "Grude", "Olovo", "Vareš", "Srebrenica", "Bratunac", "Vlasenica", "Sokolac",
         "Nevesinje", "Kalesija", "Banovići", "Maglaj", "Žepče", "Kotor Varoš", "Drvar", "Glamoč"]
RENAME = {"Lukavica": "Istočno Sarajevo"}

cols = ["id", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4",
        "pop", "elev", "dem", "tz", "mod"]
g = pd.read_csv(GN, sep="\t", names=cols, quoting=3, dtype={"a1": str, "a2": str})
g = g[g.fclass == "P"]
big = g[(g["pop"] >= 10000) & (g.fcode.isin(["PPLC", "PPLA", "PPLA2", "PPLA3", "PPL"])) & (g.fcode != "PPLL")]
big = big[big.name != "Stijena"]
rank = {"PPLC": 0, "PPLA": 1, "PPLA2": 2, "PPLA3": 3, "PPL": 4}
extra = []
for n in EXTRA:
    c = g[g.name == n].copy()
    if c.empty:
        print("  nije nađeno:", n); continue
    c["r"] = c.fcode.map(rank).fillna(9)
    extra.append(c.sort_values(["r", "pop"], ascending=[True, False]).iloc[[0]])
towns = pd.concat([big] + extra).drop_duplicates("id").drop_duplicates("name")
border = gpd.read_file(BOUNDARY).to_crs(4326)
geom = border.geometry.union_all() if hasattr(border.geometry, "union_all") else border.geometry.unary_union
towns = gpd.GeoDataFrame(towns, geometry=gpd.points_from_xy(towns.lon, towns.lat), crs=4326)
towns = towns[towns.within(geom)].copy()
towns["n"] = towns.name.map(lambda n: RENAME.get(n, n))
towns = towns.sort_values("n").reset_index(drop=True)
# gradovi koji već postoje u atlasu: iste koordinate kao u ostalim slikama i tabelama
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bih_stil import CITIES
for i, r in towns.iterrows():
    if r.n in CITIES:
        towns.loc[i, "lon"], towns.loc[i, "lat"] = CITIES[r.n][0], CITIES[r.n][1]
lon, lat = towns.lon.values, towns.lat.values
print("Gradova:", len(towns))

# 1) Köppen-Geiger: udjeli u prozoru 5 × 5 km
KGS = {}
for p, s in KEYS:
    with rasterio.open(os.path.join(KG_DIR, p, *([s] if s else []), "koppen_geiger_0p00833333.tif")) as r:
        rows, cc = rasterio.transform.rowcol(r.transform, lon, lat)
        out = []
        for rr, c in zip(rows, cc):
            w = r.read(1, window=rasterio.windows.Window(c - 2, rr - 2, 5, 5))
            w = w[w > 0]
            u, n = np.unique(w, return_counts=True)
            out.append({TYPES.get(int(k), str(k)): round(float(v) / w.size * 100, 1) for k, v in zip(u, n)})
    KGS[(p, s)] = out


def nearest(alon, alat, field):
    i = np.abs(alat[:, None] - lat[None, :]).argmin(axis=0)
    j = np.abs(alon[:, None] - lon[None, :]).argmin(axis=0)
    v = field[i, j]
    for k in np.where(~np.isfinite(v))[0]:
        w = field[max(i[k] - 1, 0):i[k] + 2, max(j[k] - 1, 0):j[k] + 2]
        v[k] = np.nanmean(w)
    return v


# 2) temperatura i 3) dani
T, D = {}, {}
Z = np.load(NPZ)
VARS = ["vruci", "mrazni", "tropske_noci", "snijeg"]
for p, s in KEYS:
    ds = xr.open_dataset(os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")).sel(
        lon=slice(15.3, 20.0), lat=slice(45.6, 42.2))
    T[(p, s)] = nearest(ds["lon"].values, ds["lat"].values, ds["air_temperature"].values.mean(axis=0))
    D[(p, s)] = [nearest(Z["lon"], Z["lat"], Z[f"{v}|{p}|{s or 'hist'}"]) for v in VARS]
E = {}
if HK:
    H = np.load(HK)
    for p, s in KEYS:
        k_ = f"{p}|{s or 'hist'}"
        E[(p, s)] = [nearest(Z["lon"], Z["lat"], Z[f"hdd|{k_}"]), nearest(Z["lon"], Z["lat"], Z[f"cdd|{k_}"])] + \
                    [nearest(H["lon"], H["lat"], H[f"{v}|{k_}"]) for v in ("chill", "rx1", "rx1_cc")]

# 4) projekcija za SVG (kao u opcine_bih.py)
xmin, ymin, xmax, ymax = border.total_bounds
k = 1000 / (xmax - xmin)
c0 = np.cos(np.deg2rad((ymin + ymax) / 2))
sx = lambda x: (x - xmin) * k
sy = lambda y: (ymax - y) * k / c0
simp = geom.simplify(0.003, preserve_topology=True)
polys = simp.geoms if simp.geom_type == "MultiPolygon" else [simp]
outline = "".join("M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in np.asarray(pl.exterior.coords)[:-1]) + "Z"
                  for pl in polys)

key_ids = [f"{p}|{s or 'hist'}" for p, s in KEYS]
m = []
for i, r in towns.iterrows():
    d = {kid: [round(float(T[key][i]), 1)] + [round(float(D[key][j][i]), 1) for j in range(4)] + [KGS[key][i]] +
         ([[round(float(E[key][j][i]), 1) for j in range(5)]] if E else [])
         for key, kid in zip(KEYS, key_ids)}
    m.append({"n": r.n, "pop": int(r["pop"]), "cx": round(sx(r.lon), 1), "cy": round(sy(r.lat), 1), "d": d})
os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
json.dump({"w": round(sx(xmax), 1), "h": round(sy(ymin), 1), "outline": outline, "keys": key_ids,
           "vars": ["t"] + VARS, "m": m}, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("✓", OUT_JSON, round(os.path.getsize(OUT_JSON) / 1e3), "kB")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["grad", "stanovništvo (GeoNames)", "period", "scenarij", "srednja godišnja temperatura (°C)",
                "vrući dani", "mrazni dani", "tropske noći", "dani sa snijegom", "preovlađujući klimatski tip",
                "udjeli tipova u krugu 5 × 5 km (%)"])
    for t in m:
        for kid in key_ids:
            p, s = kid.split("|"); v = t["d"][kid]; sh = v[-1]
            v = v[:6]
            w.writerow([t["n"], t["pop"], p.replace("_", "–"), s.upper() if s != "hist" else "historijski", *v[:-1],
                        max(sh, key=sh.get) if sh else "", "; ".join(f"{a} {b}" for a, b in sh.items())])
print("✓", OUT_CSV)
print(", ".join(towns.n))
