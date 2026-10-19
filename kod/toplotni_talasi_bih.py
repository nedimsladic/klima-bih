"""
Toplotni talasi u BiH (1901–2099): niz od najmanje 5 uzastopnih dana u kojima je najviša dnevna temperatura
za 5 °C ili više iznad prosjeka (normale) za taj dan u godini. Normala: prosjek najviše dnevne temperature
1991–2020 (E-OBS) za svaki kalendarski dan, izglađen pomičnim prosjekom od 5 dana.
Broje se talasi koji počinju od maja do septembra (cijelom dužinom).

Ista "delta change" metoda kao dani_bih.py (E-OBS 1991–2020 + mjesečna promjena iz Beck et al. 2023).
Pokazatelji (godišnje, prosjek 30 godina):
  broj toplotnih talasa
  dani u toplotnim talasima
  najduži toplotni talas u prosječnoj godini (prosjek godišnjih maksimuma)

Upotreba:
    python toplotni_talasi_bih.py <eobs_dir> <climate_0p1_dir> <bih_boundary.geojson> <out_dir>
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
MINLEN, DT = 5, 5.0
SEASON = (4, 8)                                   # mjeseci 0-indeksirano: maj–septembar

parts = []
for f in sorted(glob.glob(os.path.join(EOBS, "tx_ens_mean_0.1deg_reg_*_v31.0e.nc"))):
    ds = xr.open_dataset(f)
    lat = ds["latitude"].values
    latsl = slice(*BOX["lat"]) if lat[0] < lat[-1] else slice(*BOX["lat"][::-1])
    a = ds["tx"].sel(longitude=slice(*BOX["lon"]), latitude=latsl, time=slice("1991-01-01", "2020-12-31"))
    if a.sizes["time"]:
        parts.append(a.load())
TX = xr.concat(parts, "time").sortby("latitude", ascending=False)
LON, LAT = TX["longitude"].values, TX["latitude"].values
RES = abs(LON[1] - LON[0])
EXT = (LON[0] - RES / 2, LON[-1] + RES / 2, LAT[-1] - RES / 2, LAT[0] + RES / 2)
INSIDE = ~geometry_mask(B.border.geometry, (len(LAT), len(LON)),
                        from_origin(LON[0] - RES / 2, LAT[0] + RES / 2, RES, RES))
W = np.cos(np.deg2rad(LAT))[:, None] * INSIDE
MONTH = TX["time"].dt.month.values - 1
YEAR = TX["time"].dt.year.values
YEARS = np.unique(YEAR)
tx = TX.values
# normala najviše temperature za svaki dan u godini (1991–2020), 29. februar = 28. februar
DOY = TX["time"].dt.dayofyear.values - 1
leap = TX["time"].dt.is_leap_year.values
DOY = np.where(leap & (DOY >= 59), DOY - 1, DOY)              # 0..364
norm = np.stack([np.nanmean(tx[DOY == d], axis=0) for d in range(365)])
norm = np.mean([np.roll(norm, k, axis=0) for k in (-2, -1, 0, 1, 2)], axis=0)   # 5-dnevni prosjek
THR = norm[DOY] + DT                                            # prag za svaki dan (t, y, x)


def clim(p, s):
    f = os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")
    a = xr.open_dataset(f)["air_temperature"]
    return a.interp(lon=xr.DataArray(LON, dims="x"), lat=xr.DataArray(LAT, dims="y"),
                    method="nearest").transpose("time", "y", "x").values


T_REF = clim(REF, None)
KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]


def waves(key):
    d = np.zeros((12,) + tx.shape[1:]) if key == (REF, None) else clim(*key) - T_REF
    hot = (tx + d[MONTH]) >= THR                          # (t, y, x); normala ostaje 1991–2020
    n_ev = np.zeros(tx.shape[1:]); n_day = np.zeros(tx.shape[1:]); longest = np.zeros(tx.shape[1:])
    for yr in YEARS:
        h = hot[YEAR == yr]; mo = MONTH[YEAR == yr]
        run = np.zeros(h.shape[1:]); ymax = np.zeros(h.shape[1:]); start = np.zeros(h.shape[1:], bool)
        for t in range(h.shape[0] + 1):
            cur = h[t] if t < h.shape[0] else np.zeros(h.shape[1:], bool)
            if t < h.shape[0]:                                  # talas se broji ako počne u sezoni (maj–septembar)
                start = np.where(cur & (run == 0), SEASON[0] <= mo[t] <= SEASON[1], start)
            ended = (~cur) & (run >= MINLEN) & start
            n_ev += ended; n_day += np.where(ended, run, 0)
            ymax = np.maximum(ymax, np.where(ended, run, 0))
            run = np.where(cur, run + 1, 0)
        longest += ymax
    ny = len(YEARS)
    nan = np.isnan(tx[0])
    return {"broj": np.where(nan, np.nan, n_ev / ny), "dani": np.where(nan, np.nan, n_day / ny),
            "najduzi": np.where(nan, np.nan, longest / ny)}


R = {k: waves(k) for k in KEYS}


def bmean(a):
    return float(np.nansum(np.where(INSIDE, a, 0) * W) / W.sum())


IND = {
    "broj": ("Broj toplotnih talasa", "maj–septembar; talas = najmanje 5 uzastopnih dana ≥ 5 °C iznad normale", "talasa/god.",
             [0, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 8, 10], "{:.1f}"),
    "dani": ("Dani u toplotnim talasima", "maj–septembar; dani unutar talasa od najmanje 5 dana ≥ 5 °C iznad normale", "dana/god.",
             [0, 1, 3, 5, 10, 20, 30, 40, 50, 60, 80], "{:.0f}"),
    "najduzi": ("Najduži toplotni talas", "maj–septembar; prosjek najdužeg niza dana ≥ 5 °C iznad normale (nizovi ≥ 5 dana)", "dana",
                [0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30], "{:.0f}"),
}


def binned(bounds):
    cols = list(plt.get_cmap("YlOrRd")(np.linspace(0.06, 0.97, len(bounds) - 1)))
    cm = ListedColormap(cols); cm.set_over(cols[-1])
    return cm, BoundaryNorm(bounds, cm.N)


for k, (name, rule, unit, bounds, fmt) in IND.items():
    cm, nm = binned(bounds)
    items = [((f"Procjena {period_label(p)}" if p != REF else f"Opažanja {period_label(REF)} (E-OBS)"),
              R[(p, None)][k], EXT, bmean(R[(p, None)][k])) for p in HIST]
    B.panel(items, f"{name} u Bosni i Hercegovini", f"• {name}, historijski periodi 1901–2020",
            f"talasi_{k}_historijski.png", cm, nm, f"{name} ({rule})", fmt=fmt, unit=" " + unit,
            ticks=bounds, extend="max")
    for p in FUT:
        items = [(f"Opažanja {period_label(REF)} (E-OBS)", R[(REF, None)][k], EXT, bmean(R[(REF, None)][k]))]
        items += [(SSP[s], R[(p, s)][k], EXT, bmean(R[(p, s)][k])) for s in SSP]
        B.panel(items, f"{name} u Bosni i Hercegovini", f"• {name}, projekcije za period {period_label(p)}",
                f"talasi_{k}_{p}.png", cm, nm, f"{name} ({rule})", fmt=fmt, unit=" " + unit,
                ticks=bounds, extend="max")

# prosjek BiH kroz vrijeme
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5,
     "2041_2070": 2055.5, "2071_2099": 2085}
fig, axes = plt.subplots(1, 3, figsize=(18, 7.4))
for ax, (k, (name, rule, unit, _, fmt)) in zip(axes, IND.items()):
    r0 = bmean(R[(REF, None)][k])
    for s in SSP:
        ys = [r0] + [bmean(R[(p, s)][k]) for p in FUT]
        ax.plot([X[REF]] + [X[p] for p in FUT], ys, color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"),
                marker="o", ms=5, mec="white", zorder=3)
        ax.annotate(fmt.format(ys[-1]), (X["2071_2099"], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=SSP_COLORS[s])
    ax.plot([X[p] for p in HIST], [bmean(R[(p, None)][k]) for p in HIST], color="#222", lw=2.6, marker="o",
            ms=7, mec="white", zorder=4)
    ax.annotate(fmt.format(r0), (X[REF], r0), xytext=(-9, 8), textcoords="offset points", ha="right",
                fontsize=11, color="#222")
    ax.set_title(name, fontsize=16, loc="left", pad=12)
    ax.set_xticks(list(X.values()))
    ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in X])
    ax.set_xlim(1905, 2097); ax.set_ylim(0, None)
    ax.set_ylabel(f"{unit} (prosjek BiH)", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=10.5)
hs = [plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=7, mec="white",
                 label="Historijski periodi (1991–2020 opažanja E-OBS, ranije procjena)")]
hs += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                  mec="white", label=SSP[s].replace(" · ", " – ")) for s in SSP]
fig.subplots_adjust(left=0.05, right=0.98, top=0.76, bottom=0.24, wspace=0.22)
x0 = axes[0].get_position().x0 - 0.035
fig.legend(handles=hs, loc="lower left", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(x0, 0.0),
           title="Period / scenarij", title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.text(x0, 0.95, "Toplotni talasi u Bosni i Hercegovini", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.885, "• Najmanje 5 dana zaredom ≥ 5 °C iznad normale (maj–sep), prosjek BiH", fontsize=27,
         ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.98, 0.885, 15)
fig.savefig(os.path.join(OUT, "talasi_prosjek_bih.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ talasi_prosjek_bih.png")

# gradovi: dani u talasima i najduži talas
places = sorted([n for n in CITIES if n not in PEAKS], key=lambda n: -CITIES[n][1])


def at(a, x, y):
    i = int(np.argmin(np.abs(LAT - y))); j = int(np.argmin(np.abs(LON - x)))
    v = a[i, j]
    return float(np.nanmean(a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]) if np.isnan(v) else v)


COLS = [(("1961_1990", None), "1961–\n1990"), ((REF, None), "1991–\n2020")] + \
       [((p, s), period_label(p)[:5] + period_label(p)[-2:] + "\n" + ssp_code(s)) for p in FUT for s in ("ssp126", "ssp245", "ssp585")]
fig, axes = plt.subplots(1, 2, figsize=(20, 11.5), sharey=True)
for ax, k in zip(axes, ["dani", "najduzi"]):
    name, rule, unit, bounds, fmt = IND[k]
    cm, nm = binned(bounds)
    M = np.array([[at(R[key][k], CITIES[n][0], CITIES[n][1]) for key, _ in COLS] for n in places])
    ax.imshow(M, cmap=cm, norm=nm, aspect="auto")
    for i in range(len(places)):
        for j in range(len(COLS)):
            v = M[i, j]; rgb = cm(nm(v))[:3]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=11,
                    color="white" if 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5 else "#111")
    for j in range(len(COLS) + 1):
        ax.axvline(j - 0.5, color="white", lw=1.5)
    for i in range(len(places) + 1):
        ax.axhline(i - 0.5, color="white", lw=1.5)
    ax.axvline(1.5, color="#333", lw=1.4); ax.axvline(4.5, color="#333", lw=1.4)
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c for _, c in COLS], fontsize=10)
    ax.xaxis.tick_top(); ax.tick_params(length=0, colors="#333")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{name} ({unit})", fontsize=16, loc="left", pad=46)
axes[0].set_yticks(range(len(places))); axes[0].set_yticklabels(places, fontsize=12.5)
fig.subplots_adjust(left=0.075, right=0.99, top=0.8, bottom=0.02, wspace=0.06)
fig.text(0.01, 0.955, "Toplotni talasi u gradovima BiH", fontsize=27, ha="left", va="baseline")
fig.text(0.01, 0.91, "• Dani u talasima i najduži talas, maj–septembar", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.99, 0.91, 15)
fig.savefig(os.path.join(OUT, "talasi_po_gradovima.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ talasi_po_gradovima.png")

with open(os.path.join(OUT, "toplotni_talasi_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij", "broj talasa/god.", "dani u talasima/god.", "najduži talas (dana)"])
    for key in KEYS:
        w.writerow([period_label(key[0]), key[1].upper() if key[1] else ("opažanja (E-OBS)" if key[0] == REF else "procjena")]
                   + [f"{bmean(R[key][k]):.2f}" for k in IND])
print("✓ toplotni_talasi_bih.csv")
