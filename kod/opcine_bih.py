"""
Podaci po općinama BiH za interaktivni preglednik na web stranici.

Za svaku općinu (geoBoundaries ADM3) i svaki period/scenarij:
  udio Köppen-Geigerovih tipova (1 km, Beck et al. 2023), srednja godišnja temperatura (0,1°),
  vrući, mrazni i dani sa snijegom te tropske noći (podaci/dani_svi.npz), stanovništvo (WorldPop 2020).

Upotreba:
    python opcine_bih.py <adm3.geojson> <kg_tif_dir> <climate_0p1_dir> <dani_svi.npz> <worldpop.tif> <out_json> <out_csv>
"""
import sys
import os
import csv
import json

import numpy as np
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import from_bounds

ADM, KG_DIR, CLIM, NPZ, POP, OUT_JSON, OUT_CSV = sys.argv[1:8]
HIST = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
FUT = ["2041_2070", "2071_2099"]
SSP = ["ssp119", "ssp126", "ssp434", "ssp245", "ssp460", "ssp370", "ssp585"]
KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
KG = {8: "Csa", 9: "Csb", 14: "Cfa", 15: "Cfb", 18: "Dsb", 25: "Dfa", 26: "Dfb", 27: "Dfc", 29: "ET"}

g = gpd.read_file(ADM).to_crs(4326)
c = g.to_crs(3035).geometry.centroid.to_crs(4326)
g["lon"], g["lat"] = c.x.values, c.y.values
g["km2"] = g.to_crs(3035).area / 1e6


def fix(row):
    n = row.shapeName
    if n == "Republika Srpska":            # u izvoru pogrešno imenovana; položaj i površina odgovaraju Višegradu
        return "Višegrad"
    if n == "Novi Grad":
        return "Novi Grad Sarajevo" if row.lon > 18 else "Novi Grad"
    return {"Doboj East": "Doboj Istok", "Brcko District": "Brčko distrikt", "Centar": "Centar Sarajevo",
            "Stari Grad": "Stari Grad Sarajevo", "Kupres (BiH)": "Kupres (FBiH)", "Kupres": "Kupres (RS)",
            "Trnovo (BiH)": "Trnovo (FBiH)", "Brod": "Brod", "Kupra na Uni": "Krupa na Uni"}.get(n, n)


g["name"] = g.apply(fix, axis=1)
g = g.sort_values("name").reset_index(drop=True)
N = len(g)
shapes = [(geom, i + 1) for i, geom in enumerate(g.geometry)]
print("Općina:", N)


def labels(transform, shape):
    lab = rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=False, dtype="int32")
    return lab


def zonal(lab, arr, lon, lat, fun="mean"):
    """Prosjek po općini; općine bez ćelije dobiju vrijednost najbliže ćelije centroidu."""
    out = np.full(N, np.nan)
    ok = np.isfinite(arr)
    for i in range(N):
        m = (lab == i + 1) & ok
        if m.any():
            out[i] = arr[m].mean() if fun == "mean" else arr[m].sum()
        else:
            r = int(np.argmin(np.abs(lat - g.lat[i]))); q = int(np.argmin(np.abs(lon - g.lon[i])))
            out[i] = arr[r, q]
    return out


# 1) Köppen-Geiger 1 km
xmin, ymin, xmax, ymax = g.total_bounds
kg = {}
for p, s in KEYS:
    f = os.path.join(KG_DIR, p, *([s] if s else []), "koppen_geiger_0p00833333.tif")
    with rasterio.open(f) as r:
        win = from_bounds(xmin, ymin, xmax, ymax, r.transform)
        a = r.read(1, window=win)
        tr = r.window_transform(win)
    if (p, s) == KEYS[0]:
        LAB_KG = labels(tr, a.shape)
    res = []
    for i in range(N):
        v = a[(LAB_KG == i + 1) & (a > 0)]
        if v.size == 0:
            res.append({}); continue
        codes, cnt = np.unique(v, return_counts=True)
        res.append({KG.get(int(cc), str(int(cc))): round(float(n) / v.size * 100, 1)
                    for cc, n in zip(codes, cnt) if n / v.size >= 0.005})
    kg[(p, s)] = res
print("✓ Köppen-Geiger")

# 2) temperatura 0,1°
T = {}
for p, s in KEYS:
    ds = xr.open_dataset(os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")).sel(
        lon=slice(15.3, 20.0), lat=slice(45.6, 42.2))
    lon, lat = ds["lon"].values, ds["lat"].values
    if (p, s) == KEYS[0]:
        res_ = abs(lon[1] - lon[0])
        LAB_T = labels(from_origin(lon[0] - res_ / 2, lat[0] + res_ / 2, res_, res_), (len(lat), len(lon)))
    T[(p, s)] = zonal(LAB_T, ds["air_temperature"].values.mean(axis=0), lon, lat)
print("✓ temperatura")

# 3) dani (E-OBS + delta)
Z = np.load(NPZ)
elon, elat = Z["lon"], Z["lat"]
eres = abs(elon[1] - elon[0])
LAB_E = labels(from_origin(elon[0] - eres / 2, elat[0] + eres / 2, eres, eres), (len(elat), len(elon)))
VARS = ["vruci", "mrazni", "tropske_noci", "snijeg"]
D = {(p, s): {v: zonal(LAB_E, Z[f"{v}|{p}|{s or 'hist'}"], elon, elat) for v in VARS} for p, s in KEYS}
print("✓ dani")

# 4) stanovništvo
with rasterio.open(POP) as r:
    pop = r.read(1).astype(float); ptr = r.transform
pop[(pop < 0) | ~np.isfinite(pop)] = 0
LAB_P = labels(ptr, pop.shape)
POPS = np.bincount(LAB_P.ravel(), weights=pop.ravel(), minlength=N + 1)[1:]

# 5) obrisi za SVG (jednostavna projekcija, y prema dolje)
lat0 = (ymin + ymax) / 2
k = 1000 / (xmax - xmin)
sx = lambda lon_: (lon_ - xmin) * k
sy = lambda lat_: (ymax - lat_) * k / np.cos(np.deg2rad(lat0))
W_, H_ = round(sx(xmax), 1), round(sy(ymin), 1)


def path(geom):
    geom = geom.simplify(0.004, preserve_topology=True)
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    out = []
    for poly in polys:
        for ring in [poly.exterior] + list(poly.interiors):
            xy = np.asarray(ring.coords)
            pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in xy[:-1])
            out.append("M" + pts.replace(" ", "L", 1).replace(" ", " L") + "Z")
    return "".join(out)


key_ids = [f"{p}|{s or 'hist'}" for p, s in KEYS]
munis = []
for i, row in g.iterrows():
    d = {}
    for (p, s), kid in zip(KEYS, key_ids):
        d[kid] = [round(float(T[(p, s)][i]), 1)] + [round(float(D[(p, s)][v][i]), 1) for v in VARS] + [kg[(p, s)][i]]
    munis.append({"n": row["name"], "pop": int(round(POPS[i], -1)), "km2": int(round(row.km2)),
                  "cx": round(sx(row.lon), 1), "cy": round(sy(row.lat), 1), "path": path(row.geometry), "d": d})

os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"w": W_, "h": H_, "keys": key_ids, "vars": ["t"] + VARS, "m": munis}, f, ensure_ascii=False,
              separators=(",", ":"))
print("✓", OUT_JSON, round(os.path.getsize(OUT_JSON) / 1e3), "kB")

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["općina", "stanovništvo (WorldPop 2020)", "površina (km²)", "period", "scenarij",
                "srednja godišnja temperatura (°C)", "vrući dani", "mrazni dani", "tropske noći",
                "dani sa snijegom", "preovlađujući klimatski tip", "udjeli tipova (%)"])
    for m in munis:
        for kid in key_ids:
            p, s = kid.split("|")
            v = m["d"][kid]
            shares = v[-1]
            dom = max(shares, key=shares.get) if shares else ""
            w.writerow([m["n"], m["pop"], m["km2"], p.replace("_", "–"), s.upper() if s != "hist" else "historijski",
                        *v[:-1], dom, "; ".join(f"{a} {b}" for a, b in shares.items())])
print("✓", OUT_CSV)
