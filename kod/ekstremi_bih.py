"""
Tropske noći, stepen-dani grijanja i hlađenja i dani sa snježnim padavinama u BiH (1901–2099).

Ista "delta change" metoda kao dani_bih.py: dnevni E-OBS 1991–2020 (tn, tx, rr) pomjereni za
mjesečnu promjenu temperature, a padavine pomnožene mjesečnim omjerom iz Beck et al. (2023).

Pokazatelji (godišnje):
  tropske noći            tn ≥ 20 °C
  stepen-dani grijanja    Σ (18 − Tsr) za dane s Tsr ≤ 15 °C   (Eurostat)
  stepen-dani hlađenja    Σ (Tsr − 21) za dane s Tsr ≥ 24 °C   (Eurostat)
  dani sa snijegom        rr ≥ 1 mm i Tsr ≤ 0 °C (procjena dana sa snježnim padavinama)
gdje je Tsr = (tn + tx) / 2.

Uz karte, skripta sprema sve pokazatelje (i one iz dani_bih.py) u <out_dir>/../podaci/dani_svi.npz
za općine i izloženost stanovništva.

Upotreba:
    python ekstremi_bih.py <eobs_dir> <climate_0p1_dir> <bih_boundary.geojson> <out_dir>
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
import matplotlib.transforms as mtransforms

from bih_stil import (Bih, HIST, FUT, REF, SSP, SSP_COLORS, SSP_DASH, CITIES, PEAKS, DPI,
                      period_label, ssp_code)

EOBS, CLIM, BOUNDARY, OUT = sys.argv[1:5]
B = Bih(BOUNDARY, OUT)
BOX = dict(lon=(15.3, 20.0), lat=(42.2, 45.6))
RES_NAME = os.environ.get("KLIMA_RES", "0p1")


def eobs(var):
    parts = []
    for f in sorted(glob.glob(os.path.join(EOBS, f"{var}_ens_mean_0.1deg_reg_*_v31.0e.nc"))):
        ds = xr.open_dataset(f)
        lat = ds["latitude"].values
        latsl = slice(*BOX["lat"]) if lat[0] < lat[-1] else slice(*BOX["lat"][::-1])
        a = ds[var].sel(longitude=slice(*BOX["lon"]), latitude=latsl,
                        time=slice("1991-01-01", "2020-12-31"))
        if a.sizes["time"]:
            parts.append(a.load())
    return xr.concat(parts, "time").sortby("latitude", ascending=False)


TN, TX, RR = eobs("tn"), eobs("tx"), eobs("rr")
LON, LAT = TN["longitude"].values, TN["latitude"].values
RES = abs(LON[1] - LON[0])
EXT = (LON[0] - RES / 2, LON[-1] + RES / 2, LAT[-1] - RES / 2, LAT[0] + RES / 2)
INSIDE = ~geometry_mask(B.border.geometry, (len(LAT), len(LON)),
                        from_origin(LON[0] - RES / 2, LAT[0] + RES / 2, RES, RES))
W = np.cos(np.deg2rad(LAT))[:, None] * INSIDE
MONTH = TN["time"].dt.month.values - 1
NYEARS = 30
tn, tx, rr = TN.values, TX.values, RR.values


def clim(p, s, var):
    f = os.path.join(CLIM, p, *([s] if s else []), f"ensemble_mean_{RES_NAME}.nc")
    a = xr.open_dataset(f)[var]
    return a.interp(lon=xr.DataArray(LON, dims="x"), lat=xr.DataArray(LAT, dims="y"),
                    method="nearest").transpose("time", "y", "x").values


KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
T_REF, P_REF = clim(REF, None, "air_temperature"), clim(REF, None, "precipitation")

# svi pokazatelji: (naziv, opis, jedinica, funkcija(tn, tx, tm, rr) -> doprinos po danu)
IND = {
    "tropske_noci": ("Tropske noći", "najniža dnevna temperatura ≥ 20 °C", "dana",
                     lambda n, x, m, r: n >= 20),
    "hdd": ("Stepen-dani grijanja", "Σ (18 − Tsr) za dane s Tsr ≤ 15 °C", "°C·dan",
            lambda n, x, m, r: np.where(m <= 15, 18 - m, 0)),
    "cdd": ("Stepen-dani hlađenja", "Σ (Tsr − 21) za dane s Tsr ≥ 24 °C", "°C·dan",
            lambda n, x, m, r: np.where(m >= 24, m - 21, 0)),
    "snijeg": ("Dani sa snijegom", "padavine ≥ 1 mm i Tsr ≤ 0 °C (procjena)", "dana",
               lambda n, x, m, r: (r >= 1) & (m <= 0)),
}
# pokazatelji iz dani_bih.py i padavine_dani_bih.py, za općine i izloženost
EXTRA = {
    "mrazni": lambda n, x, m, r: n < 0,
    "ledeni": lambda n, x, m, r: x < 0,
    "vruci": lambda n, x, m, r: x >= 30,
    "vrlo_vruci": lambda n, x, m, r: x >= 35,
    "r10": lambda n, x, m, r: r >= 10,
}
ALL = {**{k: v[3] for k, v in IND.items()}, **EXTRA}


def compute(key):
    if key == (REF, None):
        dt = np.zeros((12,) + tn.shape[1:]); fp = np.ones((12,) + tn.shape[1:])
    else:
        dt = clim(*key, "air_temperature") - T_REF
        fp = np.clip(clim(*key, "precipitation") / np.maximum(P_REF, 0.1), 0.3, 3.0)
    out = {k: np.zeros(tn.shape[1:]) for k in ALL}
    for m in range(12):
        i = MONTH == m
        n, x, r = tn[i] + dt[m], tx[i] + dt[m], rr[i] * fp[m]
        tm = (n + x) / 2
        for k, f in ALL.items():
            out[k] += np.asarray(f(n, x, tm, r), dtype=float).sum(axis=0)
    for k in out:
        out[k] = np.where(np.isnan(tn[0]), np.nan, out[k] / NYEARS)
    return out


R = {k: compute(k) for k in KEYS}


def bmean(a):
    return float(np.nansum(np.where(INSIDE, a, 0) * W) / W.sum())


# spremi za općine i izloženost
DER = os.path.join(os.path.dirname(os.path.abspath(OUT)), "podaci")
os.makedirs(DER, exist_ok=True)
np.savez_compressed(os.path.join(DER, "dani_svi.npz"), lon=LON, lat=LAT,
                    **{f"{k}|{p}|{s or 'hist'}": R[(p, s)][k] for (p, s) in KEYS for k in ALL})
print("✓ podaci/dani_svi.npz")


def binned(cmap, bounds, lo=0.08, hi=0.95):
    cols = list(plt.get_cmap(cmap)(np.linspace(lo, hi, len(bounds) - 1)))
    cm = ListedColormap(cols)
    cm.set_over(cols[-1])
    return cm, BoundaryNorm(bounds, cm.N)


SCALES = {
    "tropske_noci": binned("RdPu", [0, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60]),
    "hdd": binned("PuBu", [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]),
    "cdd": binned("YlOrRd", [0, 5, 10, 25, 50, 100, 150, 200, 300, 400, 500]),
    "snijeg": binned("Blues", [0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80]),
}
FMT = {"tropske_noci": "{:.0f}", "hdd": "{:.0f}", "cdd": "{:.0f}", "snijeg": "{:.0f}"}
UNIT = {"dana": " dana/god.", "°C·dan": " °C·dan/god."}

for k, (name, rule, unit, _) in IND.items():
    cm, nm = SCALES[k]
    lab = f"{name} godišnje ({'broj dana' if unit == 'dana' else unit})"
    items = [((f"Procjena {period_label(p)}" if p != REF else f"Opažanja {period_label(REF)} (E-OBS)"),
              R[(p, None)][k], EXT, bmean(R[(p, None)][k])) for p in HIST]
    B.panel(items, f"{name} u Bosni i Hercegovini", f"• {name} ({rule}), historijski periodi 1901–2020",
            f"ekstremi_{k}_historijski.png", cm, nm, lab, fmt=FMT[k], unit=UNIT[unit],
            ticks=nm.boundaries, extend="max" if k != "hdd" else "both")
    for p in FUT:
        items = [(f"Opažanja {period_label(REF)} (E-OBS)", R[(REF, None)][k], EXT, bmean(R[(REF, None)][k]))]
        items += [(SSP[s], R[(p, s)][k], EXT, bmean(R[(p, s)][k])) for s in SSP]
        B.panel(items, f"{name} u Bosni i Hercegovini", f"• {name} ({rule}), projekcije za period {period_label(p)}",
                f"ekstremi_{k}_{p}.png", cm, nm, lab, fmt=FMT[k], unit=UNIT[unit],
                ticks=nm.boundaries, extend="max" if k != "hdd" else "both")

# prosjek BiH kroz vrijeme (2×2)
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5,
     "2041_2070": 2055.5, "2071_2099": 2085}
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
for ax, (k, (name, rule, unit, _)) in zip(axes.ravel(), IND.items()):
    r0 = bmean(R[(REF, None)][k])
    for s in SSP:
        ys = [r0] + [bmean(R[(p, s)][k]) for p in FUT]
        ax.plot([X[REF]] + [X[p] for p in FUT], ys, color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"),
                marker="o", ms=5, mec="white", zorder=3)
        ax.annotate(f"{ys[-1]:.0f}", (X["2071_2099"], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=SSP_COLORS[s])
    hy = [bmean(R[(p, None)][k]) for p in HIST]
    ax.plot([X[p] for p in HIST], hy, color="#222", lw=2.6, marker="o", ms=7, mec="white", zorder=4)
    ax.annotate(f"{r0:.0f}", (X[REF], r0), xytext=(-9, 8), textcoords="offset points", ha="right",
                fontsize=11, color="#222")
    ax.set_title(name, fontsize=16, loc="left", pad=22)
    ax.text(0, 1.02, rule, transform=ax.transAxes, fontsize=11, color="#666", va="bottom")
    ax.set_xticks(list(X.values()))
    ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in X])
    ax.set_xlim(1905, 2097)
    ax.set_ylim(0, None)
    ax.set_ylabel("Dana godišnje (prosjek BiH)" if unit == "dana" else "°C·dan godišnje (prosjek BiH)", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=10.5)
hs = [plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=7, mec="white",
                 label="Historijski periodi (1991–2020 opažanja E-OBS, ranije procjena)")]
hs += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                  mec="white", label=SSP[s].replace(" · ", " – ")) for s in SSP]
fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.17, hspace=0.42, wspace=0.18)
x0 = axes[0, 0].get_position().x0 - 0.04
fig.legend(handles=hs, loc="lower left", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(x0, 0.0),
           title="Period / scenarij", title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.text(x0, 0.955, "Tropske noći, grijanje, hlađenje i snijeg u BiH", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.905, "• Prosjek za BiH (1901–2099)", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.97, 0.905, 15)
fig.savefig(os.path.join(OUT, "ekstremi_prosjek_bih.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ ekstremi_prosjek_bih.png")

# gradovi i planine: 1961–1990, 1991–2020, 2071–2099 (SSP1-2.6, SSP2-4.5, SSP5-8.5)
COLS = [(("1961_1990", None), "1961–\n1990"), ((REF, None), "1991–\n2020")] + \
       [(("2071_2099", s), ssp_code(s).replace("SSP", "SSP\n")) for s in ("ssp126", "ssp245", "ssp585")]
places = (sorted([n for n in CITIES if n not in PEAKS], key=lambda n: -CITIES[n][1])
          + sorted(PEAKS, key=lambda n: -CITIES[n][1]))
seen, keep = set(), []
for n in places:
    cell = (int(np.argmin(np.abs(LAT - CITIES[n][1]))), int(np.argmin(np.abs(LON - CITIES[n][0]))))
    if cell not in seen:
        seen.add(cell); keep.append(n)
places = keep


def at(a, x, y):
    i = int(np.argmin(np.abs(LAT - y))); j = int(np.argmin(np.abs(LON - x)))
    v = a[i, j]
    return float(np.nanmean(a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]) if np.isnan(v) else v)


for PF in FUT:
    COLS = [(("1961_1990", None), "1961–\n1990"), ((REF, None), "1991–\n2020")] + \
           [((PF, s), ssp_code(s).replace("SSP", "SSP\n")) for s in ("ssp126", "ssp245", "ssp585")]
    nr = len(places)
    fig, axes = plt.subplots(1, 4, figsize=(20, 13.5), sharey=True)
    for ax, (k, (name, rule, unit, _)) in zip(axes, IND.items()):
        cm, nm = SCALES[k]
        M = np.array([[at(R[key][k], CITIES[n][0], CITIES[n][1]) for key, _ in COLS] for n in places])
        ax.imshow(M, cmap=cm, norm=nm, aspect="auto")
        for i in range(nr):
            for j in range(len(COLS)):
                v = M[i, j]
                rgb = cm(nm(v))[:3]
                dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10 if k != "hdd" else 9,
                        color="white" if dark else "#111")
        for j in range(len(COLS) + 1):
            ax.axvline(j - 0.5, color="white", lw=1.5)
        for i in range(nr + 1):
            ax.axhline(i - 0.5, color="white", lw=1.5)
        ax.axvline(1.5, color="#333", lw=1.4)
        ax.axhline(nr - len([n for n in places if n in PEAKS]) - 0.5, color="#333", lw=1.4)
        ax.set_xticks(range(len(COLS)))
        ax.set_xticklabels([c for _, c in COLS], fontsize=10.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, colors="#333")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(name, fontsize=17, loc="left", pad=54)
        ax.text(0, 1, rule, transform=mtransforms.offset_copy(ax.transAxes, fig=fig, y=36, units="points"),
                fontsize=10, color="#666", va="bottom")
    axes[0].set_yticks(range(nr))
    axes[0].set_yticklabels(places, fontsize=12)
    for t, n in zip(axes[0].get_yticklabels(), places):
        if n in PEAKS:
            t.set_fontstyle("italic")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.03, wspace=0.08)
    fig.text(0.01, 0.955, "Tropske noći, grijanje, hlađenje i snijeg", fontsize=27, ha="left", va="baseline")
    fig.text(0.01, 0.915, f"• Vrijednosti godišnje u gradovima i na planinama BiH, projekcije {period_label(PF)}", fontsize=27, ha="left", va="baseline")
    B.put_credit(fig, fig.transFigure, 0.99, 0.915, 15)
    fig.savefig(os.path.join(OUT, f"ekstremi_po_gradovima_{PF}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ ekstremi_po_gradovima_{PF}.png")

with open(os.path.join(OUT, "ekstremi_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij"] + [f"{IND[k][0]} ({IND[k][2]}/god., prosjek BiH)" for k in IND])
    for key in KEYS:
        w.writerow([period_label(key[0]), key[1].upper() if key[1] else ("opažanja (E-OBS)" if key[0] == REF else "procjena")]
                   + [f"{bmean(R[key][k]):.1f}" for k in IND])
print("✓ ekstremi_bih.csv")
