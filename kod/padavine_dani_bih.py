"""
Dani s padavinama ≥ 1, ≥ 10 i ≥ 20 mm u BiH (1991–2020 i projekcije do 2099).

Metoda ("delta change", množenjem):
  1) Dnevne padavine (rr) 1991–2020 iz E-OBS v31.0e (0,1°, srednja vrijednost ansambla).
  2) Za svaki scenarij i mjesec: omjer srednjih mjesečnih padavina (period / 1991–2020) iz
     Beck et al. (2023), climate_data_0p1. Dnevne padavine tog mjeseca množe se tim omjerom.
  3) Brojanje dana na godinu.
Metoda mijenja količinu kiše u danima kad pada, ali ne mijenja broj dana bez kiše; zato
broj kišnih dana (≥ 1 mm) reaguje slabije nego što bi mogao u stvarnosti.

Upotreba:
    python padavine_dani_bih.py <eobs_dir> <climate_0p1_dir> <bih_boundary.geojson> <out_dir>
"""
import sys
import os
import csv
import glob

import numpy as np
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

from bih_stil import (Bih, HIST, FUT, REF, SSP, SSP_COLORS, SSP_DASH, CITIES, PEAKS, DPI,
                      period_label, ssp_code)

EOBS, CLIM, BOUNDARY, OUT = sys.argv[1:5]
B = Bih(BOUNDARY, OUT)
BOX = dict(lon=(15.3, 20.0), lat=(42.2, 45.6))

parts = []
for f in sorted(glob.glob(os.path.join(EOBS, "rr_ens_mean_0.1deg_reg_*_v31.0e.nc"))):
    ds = xr.open_dataset(f)
    lat = ds["latitude"].values
    latsl = slice(*BOX["lat"]) if lat[0] < lat[-1] else slice(*BOX["lat"][::-1])
    a = ds["rr"].sel(longitude=slice(*BOX["lon"]), latitude=latsl,
                     time=slice("1991-01-01", "2020-12-31"))
    if a.sizes["time"]:
        parts.append(a.load())
RR = xr.concat(parts, "time").sortby("latitude", ascending=False)
LON, LAT = RR["longitude"].values, RR["latitude"].values
RES = abs(LON[1] - LON[0])
EXT = (LON[0] - RES / 2, LON[-1] + RES / 2, LAT[-1] - RES / 2, LAT[0] + RES / 2)
INSIDE = ~geometry_mask(B.border.geometry, (len(LAT), len(LON)),
                        from_origin(LON[0] - RES / 2, LAT[0] + RES / 2, RES, RES))
W = np.cos(np.deg2rad(LAT))[:, None] * INSIDE
MONTH = RR["time"].dt.month.values - 1
NYEARS = 30
rr = RR.values
print("E-OBS rr:", rr.shape, "NaN unutar BiH:", int(np.isnan(rr[0][INSIDE]).sum()))


def clim(p, s):
    f = os.path.join(CLIM, p, *([s] if s else []), f"ensemble_mean_{os.environ.get('KLIMA_RES', '0p1')}.nc")
    a = xr.open_dataset(f)["precipitation"]
    return a.interp(lon=xr.DataArray(LON, dims="x"), lat=xr.DataArray(LAT, dims="y"),
                    method="nearest").transpose("time", "y", "x").values


P_REF = clim(REF, None)
FACTOR = {(p, s): np.clip(clim(p, s) / np.maximum(P_REF, 0.1), 0.3, 3.0)
          for p, s in [(h, None) for h in HIST if h != REF] + [(f, x) for f in FUT for x in SSP]}

IND = {
    "r1": ("Kišni dani", "dnevne padavine ≥ 1 mm", 1),
    "r10": ("Dani s jakim padavinama", "dnevne padavine ≥ 10 mm", 10),
    "r20": ("Dani s vrlo jakim padavinama", "dnevne padavine ≥ 20 mm", 20),
}


def count(factor=None):
    out = {k: np.zeros(rr.shape[1:]) for k in IND}
    for m in range(12):
        r = rr[MONTH == m]
        if factor is not None:
            r = r * factor[m]
        for k, (_, _, thr) in IND.items():
            out[k] += (r >= thr).sum(axis=0)
    for k in out:
        out[k] = np.where(np.isnan(rr[0]), np.nan, out[k] / NYEARS)
    return out


R = {(REF, None): count()}
for k, fac in FACTOR.items():
    R[k] = count(fac)


def bmean(a):
    return float(np.nansum(np.where(INSIDE, a, 0) * W) / W.sum())


def binned(cmap, bounds, lo=0.08, hi=0.95):
    cols = list(plt.get_cmap(cmap)(np.linspace(lo, hi, len(bounds) - 1)))
    cm = ListedColormap(cols)
    cm.set_over(cols[-1]); cm.set_under(cols[0])
    return cm, BoundaryNorm(bounds, cm.N)


SCALES = {
    "r1": binned("GnBu", [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]),
    "r10": binned("GnBu", [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]),
    "r20": binned("GnBu", [2, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30]),
}
DIFF = binned("BrBG", [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30], 0.05, 0.95)

for k, (name, rule, _) in IND.items():
    cm, nm = SCALES[k]
    items = [((f"Procjena {period_label(p)}" if p != REF else f"Opažanja {period_label(REF)} (E-OBS)"),
              R[(p, None)][k], EXT, bmean(R[(p, None)][k])) for p in HIST]
    B.panel(items, f"{name} u Bosni i Hercegovini",
            f"• {name} ({rule}), historijski periodi 1901–2020",
            f"padavine_{k}_historijski.png", cm, nm, f"{name} godišnje (broj dana)",
            fmt="{:.0f}", unit=" dana/god.", ticks=nm.boundaries, extend="both")
    for p in FUT:
        items = [(f"Opažanja {period_label(REF)} (E-OBS)", R[(REF, None)][k], EXT, bmean(R[(REF, None)][k]))]
        items += [(SSP[s], R[(p, s)][k], EXT, bmean(R[(p, s)][k])) for s in SSP]
        B.panel(items, f"{name} u Bosni i Hercegovini",
                f"• {name} ({rule}), projekcije za period {period_label(p)}",
                f"padavine_{k}_{p}.png", cm, nm, f"{name} godišnje (broj dana)",
                fmt="{:.0f}", unit=" dana/god.", ticks=nm.boundaries, extend="both")

# promjena u % do 2071–2099: redovi pokazatelji, kolone 4 scenarija
SEL = ["ssp126", "ssp245", "ssp370", "ssp585"]
for PF in FUT:
    items = []
    for k, (name, _, _) in IND.items():
        ref = R[(REF, None)][k]
        for s in SEL:
            a = (R[(PF, s)][k] / np.maximum(ref, 0.1) - 1) * 100
            items.append((f"Dani ≥ {IND[k][2]} mm · {ssp_code(s)}", a, EXT, (bmean(R[(PF, s)][k]) / bmean(ref) - 1) * 100))
    B.panel(items, "Dani s padavinama u Bosni i Hercegovini",
            f"• Promjena broja dana do {period_label(PF)} u odnosu na 1991–2020",
            f"padavine_promjena_dana_{PF}.png", *DIFF, "Promjena broja dana (%)", fmt="{:+.0f}", unit=" %",
            ticks=DIFF[1].boundaries, extend="both")

# prosjek BiH kroz vrijeme
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5,
     "2041_2070": 2055.5, "2071_2099": 2085}
fig, axes = plt.subplots(1, 3, figsize=(18, 7.4))
for ax, (k, (name, rule, _)) in zip(axes, IND.items()):
    r0 = bmean(R[(REF, None)][k])
    for s in SSP:
        ys = [r0] + [bmean(R[(p, s)][k]) for p in FUT]
        ax.plot([X[REF]] + [X[p] for p in FUT], ys, color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"),
                marker="o", ms=5, mec="white", zorder=3)
    ax.plot([X[p] for p in HIST], [bmean(R[(p, None)][k]) for p in HIST], color="#222", lw=2.6,
            marker="o", ms=7, mec="white", zorder=4)
    ax.annotate(f"{r0:.0f}", (X[REF], r0), xytext=(-9, 0), textcoords="offset points", ha="right",
                va="center", fontsize=11, color="#222")
    lo = min(bmean(R[key][k]) for key in R); hi = max(bmean(R[key][k]) for key in R)
    mid, half = (hi + lo) / 2, max((hi - lo) * 0.65, 6)  # najmanje ±6 dana, da sitne razlike ne izgledaju velike
    ax.set_ylim(max(mid - half, 0), mid + half)
    ax.set_title(name, fontsize=16, loc="left", pad=22)
    ax.text(0, 1.02, rule, transform=ax.transAxes, fontsize=11, color="#666", va="bottom")
    ax.set_xticks(list(X.values()))
    ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in X])
    ax.set_xlim(1905, 2092)
    ax.set_ylabel("Dana godišnje (prosjek BiH)", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=10.5)
hs = [plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=7, mec="white", label="Historijski periodi (1991–2020 opažanja E-OBS, ranije procjena)")]
hs += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                  mec="white", label=SSP[s].replace(" · ", " – ")) for s in SSP]
fig.subplots_adjust(left=0.05, right=0.98, top=0.76, bottom=0.24, wspace=0.22)
x0 = axes[0].get_position().x0 - 0.035
fig.legend(handles=hs, loc="lower left", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(x0, 0.0),
           title="Period / scenarij", title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.text(x0, 0.95, "Dani s padavinama u Bosni i Hercegovini", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.885, "• Prosječan broj dana godišnje (1991–2099)", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.98, 0.885, 15)
fig.savefig(os.path.join(OUT, "padavine_dani_prosjek_bih.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ padavine_dani_prosjek_bih.png")

with open(os.path.join(OUT, "padavine_dani_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij"] + [IND[k][0] + " (dana/god., prosjek BiH)" for k in IND])
    for key in [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]:
        w.writerow([period_label(key[0]), key[1].upper() if key[1] else ("opažanja (E-OBS)" if key[0] == REF else "procjena")]
                   + [f"{bmean(R[key][k]):.1f}" for k in IND])
print("✓ padavine_dani_bih.csv")
