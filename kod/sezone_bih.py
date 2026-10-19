"""
Zima i ljeto u BiH: karte srednje temperature i padavina po SSP scenarijima.

Podaci: Beck et al. (2023) – climate_data_0p1.zip (mjesečne klimatologije, 0,1°, srednja vrijednost ansambla)

Upotreba:
    python sezone_bih.py <climate_0p1_dir> <bih_boundary.geojson> <out_dir>
"""
import sys
import os
import csv

import numpy as np
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

from bih_stil import Bih, HIST, FUT, REF, SSP, period_label, ssp_code

CLIM, BOUNDARY, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
B = Bih(BOUNDARY, OUT)
SEAS = {"zima": ("Zima (DJF)", [11, 0, 1]), "ljeto": ("Ljeto (JJA)", [5, 6, 7])}
SEL = ["ssp126", "ssp245", "ssp370", "ssp585"]


def load(p, s):
    f = os.path.join(CLIM, p, *([s] if s else []), f"ensemble_mean_{os.environ.get('KLIMA_RES', '0p1')}.nc")
    ds = xr.open_dataset(f).sel(lon=slice(15.3, 20.0), lat=slice(45.6, 42.2))
    return ds["air_temperature"].values, ds["precipitation"].values, ds["lon"].values, ds["lat"].values


keys = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
D = {k: load(*k) for k in keys}
_, _, LON, LAT = D[(REF, None)]
RES = abs(LON[1] - LON[0])
EXT = (LON[0] - RES / 2, LON[-1] + RES / 2, LAT[-1] - RES / 2, LAT[0] + RES / 2)
INSIDE = ~geometry_mask(B.border.geometry, (len(LAT), len(LON)),
                        from_origin(LON[0] - RES / 2, LAT[0] + RES / 2, RES, RES))
W = np.cos(np.deg2rad(LAT))[:, None] * INSIDE


def bmean(a):
    return float(np.nansum(np.where(INSIDE, a, 0) * W) / W.sum())


def tseas(k, se):
    return D[k][0][SEAS[se][1]].mean(axis=0)


def pseas(k, se):
    return D[k][1][SEAS[se][1]].sum(axis=0)


def binned(colors, bounds):
    cm = ListedColormap(colors)
    cm.set_under(colors[0]); cm.set_over(colors[-1])
    return cm, BoundaryNorm(bounds, cm.N)


# zimska skala: plavo ispod 0 °C, toplo iznad
wb = np.arange(-8, 11, 1)
neg = plt.get_cmap("Blues_r")(np.linspace(0.15, 0.85, 8))
pos = plt.get_cmap("YlOrRd")(np.linspace(0.05, 0.75, 10))
WIN = binned(list(neg) + list(pos), wb)
sb = np.arange(8, 31, 2)
SUM = binned(list(plt.get_cmap("YlOrRd")(np.linspace(0.05, 0.95, len(sb) - 1))), sb)
db = np.arange(0, 7.5, 0.5)
DT = binned(list(plt.get_cmap("Reds")(np.linspace(0.08, 0.95, len(db) - 1))), db)
pb = np.arange(-40, 45, 5)
DP = binned(list(plt.get_cmap("BrBG")(np.linspace(0.05, 0.95, len(pb) - 1))), pb)

rows = []
for se, cm, ticks in (("zima", WIN, wb[::2]), ("ljeto", SUM, sb)):
    name = SEAS[se][0]
    items = [(period_label(p), tseas((p, None), se), EXT, bmean(tseas((p, None), se))) for p in HIST]
    B.panel(items, "Temperatura zraka u Bosni i Hercegovini",
            f"• {name}: srednja temperatura, historijski periodi 1901–2020",
            f"{se}_temperatura_historijski.png", *cm, f"Srednja temperatura, {name[0].lower() + name[1:]} (°C)",
            fmt="{:.1f}", unit=" °C", ticks=ticks, extend="both")
    for p in FUT:
        items = [(f"Referentni period {period_label(REF)}", tseas((REF, None), se), EXT,
                  bmean(tseas((REF, None), se)))]
        items += [(SSP[s], tseas((p, s), se), EXT, bmean(tseas((p, s), se))) for s in SSP]
        B.panel(items, "Temperatura zraka u Bosni i Hercegovini",
                f"• {name}: srednja temperatura, projekcije za period {period_label(p)}",
                f"{se}_temperatura_{p}.png", *cm, f"Srednja temperatura, {name[0].lower() + name[1:]} (°C)",
                fmt="{:.1f}", unit=" °C", ticks=ticks, extend="both")
    for k in keys:
        rows.append([se, period_label(k[0]), k[1].upper() if k[1] else "historijski",
                     f"{bmean(tseas(k, se)):.2f}", f"{bmean(pseas(k, se)):.0f}"])

# promjene do 2071–2099: redovi zima/ljeto, kolone 4 scenarija
for PF in FUT:
    for var, cm, label, unit, fmt, fn, ticks in (
            ("t", DT, "Promjena srednje temperature (°C)", " °C", "+{:.1f}", "promjena_temperature_zima_ljeto", db[::2]),
            ("p", DP, "Promjena količine padavina (%)", " %", "{:+.0f}", "promjena_padavina_zima_ljeto", pb[::2])):
        items = []
        for se in SEAS:
            for s in SEL:
                if var == "t":
                    a = tseas((PF, s), se) - tseas((REF, None), se)
                    mean = bmean(tseas((PF, s), se)) - bmean(tseas((REF, None), se))
                else:
                    ref = pseas((REF, None), se)
                    a = (pseas((PF, s), se) / ref - 1) * 100
                    mean = (bmean(pseas((PF, s), se)) / bmean(ref) - 1) * 100
                items.append((f"{SEAS[se][0].split(' ')[0]} · {ssp_code(s)}", a, EXT, mean))
        what = "temperature" if var == "t" else "padavina"
        B.panel(items, "Zima i ljeto u Bosni i Hercegovini",
                f"• Promjena {what} do {period_label(PF)} u odnosu na 1991–2020",
                f"{fn}_{PF}.png", *cm, label, fmt=fmt, unit=unit, ticks=ticks, extend="both" if var == "p" else "max")

with open(os.path.join(OUT, "zima_ljeto_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["sezona", "period", "scenarij", "srednja temperatura (°C)", "padavine (mm)"])
    w.writerows(rows)
print("✓ zima_ljeto_bih.csv")
