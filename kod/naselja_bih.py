"""
Podaci za pretragu naselja ("Moje mjesto") na web stranici.

Za svako naselje iz GeoNames (BA.txt, klasa P) unutar granice BiH i za svaki period/scenarij:
  klimatski tip na 1 km (Beck et al. 2023), srednja godišnja temperatura (0,1°),
  vrući i mrazni dani, tropske noći i dani sa snijegom (podaci/dani_svi.npz, ~10 km).

Izlaz (u <out_dir>):
  naselja_indeks.json        [[naziv, općina, lon, lat, id], ...]
  naselja_<n>.json           podaci po 500 naselja: {id: {ključ: [t, vruci, mrazni, tropske_noci, snijeg, tip]}}

Upotreba:
    python naselja_bih.py <BA.txt> <adm3.geojson> <bih_boundary.geojson> <kg_tif_dir> <climate_0p1_dir> <dani_svi.npz> <out_dir>
"""
import sys
import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio

GN, ADM, BOUNDARY, KG_DIR, CLIM, NPZ, OUT = sys.argv[1:8]
HIST = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
FUT = ["2041_2070", "2071_2099"]
SSP = ["ssp119", "ssp126", "ssp434", "ssp245", "ssp460", "ssp370", "ssp585"]
KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
TYPES = {8: "Csa", 9: "Csb", 14: "Cfa", 15: "Cfb", 18: "Dsb", 25: "Dfa", 26: "Dfb", 27: "Dfc", 29: "ET"}
CHUNK = 500

cols = ["id", "name", "ascii", "alt", "lat", "lon", "fclass", "fcode", "cc", "cc2", "a1", "a2", "a3", "a4",
        "pop", "elev", "dem", "tz", "mod"]
g = pd.read_csv(GN, sep="\t", names=cols, quoting=3, dtype={"a1": str, "a2": str})
g = g[(g.fclass == "P") & (~g.fcode.isin(["PPLH", "PPLQ", "PPLW", "PPLX"]))].copy()
pts = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.lon, g.lat), crs=4326)
border = gpd.read_file(BOUNDARY).to_crs(4326)
pts = pts[pts.within(border.geometry.union_all() if hasattr(border.geometry, "union_all") else border.geometry.unary_union)]

adm = gpd.read_file(ADM).to_crs(4326)[["shapeName", "geometry"]]
REN = {"Republika Srpska": "Višegrad", "Doboj East": "Doboj Istok", "Brcko District": "Brčko distrikt",
       "Centar": "Centar Sarajevo", "Stari Grad": "Stari Grad Sarajevo", "Kupres (BiH)": "Kupres",
       "Trnovo (BiH)": "Trnovo", "Trnovo (RS)": "Trnovo (RS)", "Kupra na Uni": "Krupa na Uni"}
adm["opc"] = adm.shapeName.map(lambda n: REN.get(n, n))
pts = gpd.sjoin(pts, adm[["opc", "geometry"]], how="left", predicate="within").drop(columns="index_right")
pts["opc"] = pts["opc"].fillna("")
# isti naziv: prvo sjedišta (PPLC, PPLA…), zatim veća naselja, pa ostala po općini
RANK = {"PPLC": 0, "PPLA": 1, "PPLA2": 2, "PPLA3": 3}
pts["rank"] = pts.fcode.map(RANK).fillna(9)
pts = pts.drop_duplicates("id").sort_values(["name", "rank", "pop", "opc"], ascending=[True, True, False, True]).reset_index(drop=True)
pts["uid"] = np.arange(len(pts))
print("Naselja:", len(pts))
lon, lat = pts.lon.values, pts.lat.values

# 1) klimatski tip, 1 km
KG = {}
for p, s in KEYS:
    f = os.path.join(KG_DIR, p, *([s] if s else []), "koppen_geiger_0p00833333.tif")
    with rasterio.open(f) as r:
        rows, cols_ = rasterio.transform.rowcol(r.transform, lon, lat)
        rows, cols_ = np.asarray(rows), np.asarray(cols_)
        win = rasterio.windows.Window(min(cols_) - 2, min(rows) - 2, max(cols_) - min(cols_) + 5, max(rows) - min(rows) + 5)
        a = r.read(1, window=win)
    rr = np.array(rows) - (min(rows) - 2); cc = np.array(cols_) - (min(cols_) - 2)
    v = a[rr, cc]
    # ćelija bez vrijednosti (rub, jezero): uzmi najčešći tip u susjedstvu 3×3
    for i in np.where(v == 0)[0]:
        w = a[rr[i] - 1:rr[i] + 2, cc[i] - 1:cc[i] + 2]; w = w[w > 0]
        if w.size:
            u, n = np.unique(w, return_counts=True); v[i] = u[n.argmax()]
    KG[(p, s)] = v


# 2) temperatura, 0,1°
def nearest(arrlon, arrlat, field):
    i = np.abs(arrlat[:, None] - lat[None, :]).argmin(axis=0)
    j = np.abs(arrlon[:, None] - lon[None, :]).argmin(axis=0)
    v = field[i, j]
    bad = ~np.isfinite(v)
    for k in np.where(bad)[0]:
        w = field[max(i[k] - 1, 0):i[k] + 2, max(j[k] - 1, 0):j[k] + 2]
        v[k] = np.nanmean(w) if np.isfinite(w).any() else np.nan
    return v


T = {}
for p, s in KEYS:
    ds = xr.open_dataset(os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")).sel(
        lon=slice(15.3, 20.0), lat=slice(45.6, 42.2))
    T[(p, s)] = nearest(ds["lon"].values, ds["lat"].values, ds["air_temperature"].values.mean(axis=0))

# 3) dani
Z = np.load(NPZ)
VARS = ["vruci", "mrazni", "tropske_noci", "snijeg"]
D = {(p, s): [nearest(Z["lon"], Z["lat"], Z[f"{v}|{p}|{s or 'hist'}"]) for v in VARS] for p, s in KEYS}

# 4) izlaz
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.startswith("naselja_"):
        os.remove(os.path.join(OUT, f))
index = [[r["name"], r["opc"], round(float(r["lon"]), 4), round(float(r["lat"]), 4), int(r["uid"])]
         for _, r in pts.iterrows()]
json.dump({"types": list(TYPES.values()), "keys": [f"{p}|{s or 'hist'}" for p, s in KEYS], "n": index},
          open(os.path.join(OUT, "naselja_indeks.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
tcode = {c: i for i, c in enumerate(TYPES)}
for start in range(0, len(pts), CHUNK):
    chunk = {}
    for u in range(start, min(start + CHUNK, len(pts))):
        rec = []
        for k in KEYS:
            t = T[k][u]
            rec.append([None if not np.isfinite(t) else round(float(t), 1)] +
                       [None if not np.isfinite(D[k][j][u]) else round(float(D[k][j][u])) for j in range(4)] +
                       [tcode.get(int(KG[k][u]), -1)])
        chunk[u] = rec
    json.dump(chunk, open(os.path.join(OUT, f"naselja_{start // CHUNK}.json"), "w"), separators=(",", ":"))
tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.startswith("naselja_"))
print("✓ naselja:", len(pts), "mjesta,", len(os.listdir(OUT)), "datoteka u", OUT, round(tot / 1e6, 1), "MB")
ex = pts[pts.name.isin(["Sarajevo", "Mostar", "Lukavica", "Stolac"])].head(6)
for _, r in ex.iterrows():
    u = r.uid
    print(r["name"], "|", r.opc, "|", TYPES.get(int(KG[("1991_2020", None)][u])), "→", TYPES.get(int(KG[("2071_2099", "ssp585")][u])),
          "| T", round(float(T[("1991_2020", None)][u]), 1), "| vrući", round(float(D[("1991_2020", None)][0][u])))
