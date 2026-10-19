"""
Ljetni vodni bilans u BiH: padavine minus potencijalna evapotranspiracija (P − PET), jun–avgust (JJA).

PET po Thornthwaiteu (1948) iz mjesečne temperature, s indeksom topline I iz perioda 1991–2020
za sve periode (da promjena PET-a odražava zagrijavanje). Podaci: Beck et al. (2023), 0,1°.

Upotreba:
    python vodni_bilans_bih.py <climate_0p1_dir> <bih_boundary.geojson> <out_dir>
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

from bih_stil import Bih, HIST, FUT, REF, SSP, SSP_COLORS, SSP_DASH, DPI, period_label

CLIM, BOUNDARY, OUT = sys.argv[1:4]
B = Bih(BOUNDARY, OUT)
JJA = [5, 6, 7]
DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
MIDDOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])


def load(p, s):
    f = os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")
    ds = xr.open_dataset(f).sel(lon=slice(15.3, 20.0), lat=slice(45.6, 42.2))
    return ds["air_temperature"].values, ds["precipitation"].values, ds["lon"].values, ds["lat"].values


KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
D = {k: load(*k) for k in KEYS}
_, _, LON, LAT = D[(REF, None)]
RES = abs(LON[1] - LON[0])
EXT = (LON[0] - RES / 2, LON[-1] + RES / 2, LAT[-1] - RES / 2, LAT[0] + RES / 2)
INSIDE = ~geometry_mask(B.border.geometry, (len(LAT), len(LON)),
                        from_origin(LON[0] - RES / 2, LAT[0] + RES / 2, RES, RES))
W = np.cos(np.deg2rad(LAT))[:, None] * INSIDE

# dužina dana (sati) sredinom mjeseca za svaku geografsku širinu
decl = 0.409 * np.sin(2 * np.pi * MIDDOY / 365 - 1.39)
phi = np.deg2rad(LAT)[None, :]
ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl)[:, None], -1, 1))
L = (24 / np.pi * ws)[:, :, None]                           # (12, lat, 1)

T0 = np.maximum(D[(REF, None)][0], 0)
I = ((T0 / 5) ** 1.514).sum(axis=0)
A = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239


def pet(t):
    t = np.maximum(t, 0)
    return 16 * (L / 12) * (DAYS[:, None, None] / 30) * (10 * t / np.maximum(I, 1e-6)) ** A


def wb(k):
    t, p = D[k][0], D[k][1]
    return (p[JJA] - pet(t)[JJA]).sum(axis=0)


R = {k: wb(k) for k in KEYS}


def bmean(a):
    return float(np.nansum(np.where(INSIDE, a, 0) * W) / W.sum())


bounds = [-350, -300, -250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250]
cols = list(plt.get_cmap("BrBG")(np.linspace(0.04, 0.96, len(bounds) - 1)))
cm = ListedColormap(cols); cm.set_under(cols[0]); cm.set_over(cols[-1])
nm = BoundaryNorm(bounds, cm.N)
LABEL = "Ljetni vodni bilans P − PET, JJA (mm)"

items = [(period_label(p), R[(p, None)], EXT, bmean(R[(p, None)])) for p in HIST]
B.panel(items, "Ljetni vodni bilans u Bosni i Hercegovini",
        "• Padavine minus isparavanje, ljeto (JJA), historijski periodi 1901–2020",
        "vodni_bilans_historijski.png", cm, nm, LABEL, fmt="{:+.0f}", unit=" mm", ticks=bounds, extend="both")
for p in FUT:
    items = [(f"Referentni period {period_label(REF)}", R[(REF, None)], EXT, bmean(R[(REF, None)]))]
    items += [(SSP[s], R[(p, s)], EXT, bmean(R[(p, s)])) for s in SSP]
    B.panel(items, "Ljetni vodni bilans u Bosni i Hercegovini",
            f"• Padavine minus isparavanje, ljeto (JJA), projekcije za period {period_label(p)}",
            f"vodni_bilans_{p}.png", cm, nm, LABEL, fmt="{:+.0f}", unit=" mm", ticks=bounds, extend="both")

# prosjek BiH: ljetne padavine, PET i bilans kroz vrijeme
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5,
     "2041_2070": 2055.5, "2071_2099": 2085}
fig, axes = plt.subplots(1, 3, figsize=(18, 7.4))
series = {
    "Ljetne padavine (JJA)": lambda k: bmean(D[k][1][JJA].sum(axis=0)),
    "Potencijalno isparavanje, PET (JJA)": lambda k: bmean(pet(D[k][0])[JJA].sum(axis=0)),
    "Vodni bilans P − PET (JJA)": lambda k: bmean(R[k]),
}
for ax, (title, f) in zip(axes, series.items()):
    r0 = f((REF, None))
    for s in SSP:
        ax.plot([X[REF]] + [X[p] for p in FUT], [r0] + [f((p, s)) for p in FUT], color=SSP_COLORS[s], lw=2.2,
                ls=SSP_DASH.get(s, "-"), marker="o", ms=5, mec="white", zorder=3)
    ax.plot([X[p] for p in HIST], [f((p, None)) for p in HIST], color="#222", lw=2.6, marker="o", ms=7,
            mec="white", zorder=4)
    ax.annotate(f"{r0:.0f} mm", (X[REF], r0), xytext=(-9, 8), textcoords="offset points", ha="right",
                fontsize=11, color="#222")
    if "bilans" in title:
        ax.axhline(0, color="#555", lw=1)
    ax.set_title(title, fontsize=16, loc="left", pad=12)
    ax.set_xticks(list(X.values()))
    ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in X])
    ax.set_xlim(1905, 2095)
    ax.set_ylabel("mm (prosjek BiH)", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=10.5)
hs = [plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=7, mec="white", label="Historijski periodi")]
hs += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                  mec="white", label=SSP[s].replace(" · ", " – ")) for s in SSP]
fig.subplots_adjust(left=0.05, right=0.98, top=0.76, bottom=0.24, wspace=0.22)
x0 = axes[0].get_position().x0 - 0.035
fig.legend(handles=hs, loc="lower left", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(x0, 0.0),
           title="Period / scenarij", title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.text(x0, 0.95, "Ljetni vodni bilans u Bosni i Hercegovini", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.885, "• Padavine, isparavanje i bilans, ljeto (JJA), 1901–2099", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.98, 0.885, 15)
fig.savefig(os.path.join(OUT, "vodni_bilans_prosjek_bih.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ vodni_bilans_prosjek_bih.png")

with open(os.path.join(OUT, "vodni_bilans_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij", "padavine JJA (mm)", "PET JJA (mm)", "P − PET JJA (mm)"])
    for k in KEYS:
        w.writerow([period_label(k[0]), k[1].upper() if k[1] else "historijski",
                    f"{series['Ljetne padavine (JJA)'](k):.0f}",
                    f"{series['Potencijalno isparavanje, PET (JJA)'](k):.0f}", f"{bmean(R[k]):.0f}"])
print("✓ vodni_bilans_bih.csv")
