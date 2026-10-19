"""
Mrazni, ledeni, vrući i vrlo vrući dani u BiH (1991–2020 i projekcije do 2099).

Metoda ("delta change"):
  1) Dnevne minimalne (tn) i maksimalne (tx) temperature 1991–2020 iz E-OBS v31.0e (0,1°, srednja
     vrijednost ansambla; Cornes et al. 2018).
  2) Za svaki scenarij i mjesec: promjena srednje mjesečne temperature (period − 1991–2020) iz
     Beck et al. (2023), climate_data_0p1. Ista promjena dodaje se dnevnim tn i tx tog mjeseca.
  3) Brojanje dana na godinu.
Metoda pomjera cijelu raspodjelu i ne mijenja dnevnu varijabilnost.

Pokazatelji:
  mrazni dani      tn < 0 °C
  ledeni dani      tx < 0 °C
  vrući dani       tx ≥ 30 °C
  vrlo vrući dani  tx ≥ 35 °C

Upotreba:
    python dani_bih.py <eobs_dir> <climate_0p1_dir> <bih_boundary.geojson> <out_dir>
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
    a = xr.concat(parts, "time").sortby("latitude", ascending=False)
    return a


TN, TX = eobs("tn"), eobs("tx")
LON, LAT = TN["longitude"].values, TN["latitude"].values
RES = abs(LON[1] - LON[0])
EXT = (LON[0] - RES / 2, LON[-1] + RES / 2, LAT[-1] - RES / 2, LAT[0] + RES / 2)
INSIDE = ~geometry_mask(B.border.geometry, (len(LAT), len(LON)),
                        from_origin(LON[0] - RES / 2, LAT[0] + RES / 2, RES, RES))
W = np.cos(np.deg2rad(LAT))[:, None] * INSIDE
MONTH = TN["time"].dt.month.values - 1
NYEARS = 30
tn, tx = TN.values, TX.values
print("E-OBS:", tn.shape, "NaN unutar BiH:", int(np.isnan(tn[0][INSIDE]).sum()))


def clim(p, s):
    f = os.path.join(CLIM, p, *([s] if s else []), f"ensemble_mean_{os.environ.get('KLIMA_RES', '0p1')}.nc")
    a = xr.open_dataset(f)["air_temperature"]
    return a.interp(lon=xr.DataArray(LON, dims="x"), lat=xr.DataArray(LAT, dims="y"),
                    method="nearest").transpose("time", "y", "x").values


C_REF = clim(REF, None)
DELTA = {(p, None): clim(p, None) - C_REF for p in HIST if p != REF}
DELTA.update({(p, s): clim(p, s) - C_REF for p in FUT for s in SSP})

IND = {
    "mrazni": ("Mrazni dani", "najniža dnevna temperatura < 0 °C", lambda n, x: n < 0),
    "ledeni": ("Ledeni dani", "najviša dnevna temperatura < 0 °C", lambda n, x: x < 0),
    "vruci": ("Vrući dani", "najviša dnevna temperatura ≥ 30 °C", lambda n, x: x >= 30),
    "vrlo_vruci": ("Vrlo vrući dani", "najviša dnevna temperatura ≥ 35 °C", lambda n, x: x >= 35),
}


def count(delta=None):
    out = {k: np.zeros(tn.shape[1:]) for k in IND}
    for m in range(12):
        idx = MONTH == m
        d = 0 if delta is None else delta[m]
        n, x = tn[idx] + d, tx[idx] + d
        for k, (_, _, f) in IND.items():
            out[k] += f(n, x).sum(axis=0)
    for k in out:
        out[k] = np.where(np.isnan(tn[0]), np.nan, out[k] / NYEARS)
    return out


R = {(REF, None): count()}
for k, d in DELTA.items():
    R[k] = count(d)


def bmean(a):
    return float(np.nansum(np.where(INSIDE, a, 0) * W) / W.sum())


def binned(cmap, bounds, lo=0.08, hi=0.95):
    cols = list(plt.get_cmap(cmap)(np.linspace(lo, hi, len(bounds) - 1)))
    cm = ListedColormap(cols)
    cm.set_over(cols[-1])
    return cm, BoundaryNorm(bounds, cm.N)


SCALES = {
    "mrazni": binned("Blues", [0, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]),
    "ledeni": binned("PuBu", [0, 1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80]),
    "vruci": binned("YlOrRd", [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]),
    "vrlo_vruci": binned("Reds", [0, 0.5, 1, 2, 5, 10, 15, 20, 30, 40, 50]),
}

# 1) panel karata za svaki pokazatelj i period
for k, (name, rule, _) in IND.items():
    cm, nm = SCALES[k]
    items = [((f"Procjena {period_label(p)}" if p != REF else f"Opažanja {period_label(REF)} (E-OBS)"),
              R[(p, None)][k], EXT, bmean(R[(p, None)][k])) for p in HIST]
    B.panel(items, f"{name} u Bosni i Hercegovini",
            f"• {name} ({rule}), historijski periodi 1901–2020",
            f"dani_{k}_historijski.png", cm, nm, f"{name} godišnje (broj dana)",
            fmt="{:.0f}", unit=" dana/god.", ticks=nm.boundaries, extend="max")
    for p in FUT:
        items = [(f"Opažanja {period_label(REF)} (E-OBS)", R[(REF, None)][k], EXT, bmean(R[(REF, None)][k]))]
        items += [(SSP[s], R[(p, s)][k], EXT, bmean(R[(p, s)][k])) for s in SSP]
        B.panel(items, f"{name} u Bosni i Hercegovini",
                f"• {name} ({rule}), projekcije za period {period_label(p)}",
                f"dani_{k}_{p}.png", cm, nm, f"{name} godišnje (broj dana)",
                fmt="{:.0f}", unit=" dana/god.", ticks=nm.boundaries, extend="max")

# 2) prosjek za BiH: broj dana kroz vrijeme po scenariju (2×2)
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5,
     "2041_2070": 2055.5, "2071_2099": 2085}
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
for ax, (k, (name, rule, _)) in zip(axes.ravel(), IND.items()):
    r0 = bmean(R[(REF, None)][k])
    for s in SSP:
        ys = [r0] + [bmean(R[(p, s)][k]) for p in FUT]
        ax.plot([X[REF]] + [X[p] for p in FUT], ys, color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"),
                marker="o", ms=5, mec="white", zorder=3)
        ax.annotate(f"{ys[-1]:.0f}", (X["2071_2099"], ys[-1]), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=SSP_COLORS[s])
    ax.plot([X[p] for p in HIST], [bmean(R[(p, None)][k]) for p in HIST], color="#222", lw=2.6,
            marker="o", ms=7, mec="white", zorder=4)
    ax.annotate(f"{r0:.0f}", (X[REF], r0), xytext=(-9, 0), textcoords="offset points", ha="right",
                va="center", fontsize=11, color="#222")
    ax.set_title(f"{name}", fontsize=16, loc="left", pad=22)
    ax.text(0, 1.02, rule, transform=ax.transAxes, fontsize=11, color="#666", va="bottom")
    ax.set_xticks(list(X.values()))
    ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in X])
    ax.set_xlim(1905, 2094)
    ax.set_ylim(0, None)
    ax.set_ylabel("Dana godišnje (prosjek BiH)", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=10.5)
hs = [plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=7, mec="white", label="Historijski periodi (1991–2020 opažanja E-OBS, ranije procjena)")]
hs += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                  mec="white", label=SSP[s].replace(" · ", " – ")) for s in SSP]
fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.17, hspace=0.42, wspace=0.18)
fig.legend(handles=hs, loc="lower left", ncol=4, frameon=False, fontsize=12,
           bbox_to_anchor=(axes[0, 0].get_position().x0 - 0.04, 0.0), title="Period / scenarij",
           title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
x0 = axes[0, 0].get_position().x0 - 0.04
fig.text(x0, 0.955, "Mrazni, ledeni, vrući i vrlo vrući dani u BiH", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.905, "• Prosječan broj dana godišnje (1991–2099)", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.97, 0.905, 15)
fig.savefig(os.path.join(OUT, "dani_prosjek_bih.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ dani_prosjek_bih.png")


# 3) gradovi i planine: broj dana za 1991–2020 i 2071–2099 (SSP1-2.6, SSP2-4.5, SSP5-8.5)
def at(a, x, y):
    i = int(np.argmin(np.abs(LAT - y))); j = int(np.argmin(np.abs(LON - x)))
    v = a[i, j]
    if np.isnan(v):
        win = a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
        v = np.nanmean(win)
    return float(v)


COLS = [(("1961_1990", None), "1961–\n1990"), ((REF, None), "1991–\n2020")] + [(("2071_2099", s), ssp_code(s).replace("SSP", "SSP\n"))
                                          for s in ("ssp126", "ssp245", "ssp585")]
places = (sorted([n for n in CITIES if n not in PEAKS], key=lambda n: -CITIES[n][1])
          + sorted(PEAKS, key=lambda n: -CITIES[n][1]))
# mjesta u istoj ćeliji E-OBS (0,1°) dala bi iste brojeve; zadrži samo prvo (npr. Travnik/Vlašić)
_seen, _keep = set(), []
for n in places:
    cell = (int(np.argmin(np.abs(LAT - CITIES[n][1]))), int(np.argmin(np.abs(LON - CITIES[n][0]))))
    if cell in _seen:
        print("  izostavljeno (ista ćelija):", n)
        continue
    _seen.add(cell); _keep.append(n)
places = _keep
for PF in FUT:
    COLS = [(("1961_1990", None), "1961–\n1990"), ((REF, None), "1991–\n2020")] + \
           [((PF, s), ssp_code(s).replace("SSP", "SSP\n")) for s in ("ssp126", "ssp245", "ssp585")]
    nr = len(places)
    fig, axes = plt.subplots(1, 4, figsize=(20, 13.5), sharey=True)
    for ax, (k, (name, rule, _)) in zip(axes, IND.items()):
        cm, nm = SCALES[k]
        M = np.array([[at(R[key][k], CITIES[n][0], CITIES[n][1]) for key, _ in COLS] for n in places])
        ax.imshow(M, cmap=cm, norm=nm, aspect="auto")
        for i in range(nr):
            for j in range(len(COLS)):
                v = M[i, j]
                rgb = cm(nm(v))[:3]
                dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10.5,
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
                fontsize=10.5, color="#666", va="bottom")
    axes[0].set_yticks(range(nr))
    axes[0].set_yticklabels(places, fontsize=12)
    for t, n in zip(axes[0].get_yticklabels(), places):
        if n in PEAKS:
            t.set_fontstyle("italic")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.03, wspace=0.08)
    x0 = 0.01
    fig.text(x0, 0.955, "Mrazni, ledeni, vrući i vrlo vrući dani", fontsize=27, ha="left", va="baseline")
    fig.text(x0, 0.915, f"• Broj dana godišnje u gradovima i na planinama BiH, projekcije {period_label(PF)}", fontsize=27, ha="left", va="baseline")
    B.put_credit(fig, fig.transFigure, 0.99, 0.915, 15)
    fig.savefig(os.path.join(OUT, f"dani_po_gradovima_{PF}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ dani_po_gradovima_{PF}.png")

# 4) tabele
with open(os.path.join(OUT, "dani_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij"] + [IND[k][0] + " (dana/god., prosjek BiH)" for k in IND])
    for key in [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]:
        w.writerow([period_label(key[0]), key[1].upper() if key[1] else ("opažanja (E-OBS)" if key[0] == REF else "procjena")]
                   + [f"{bmean(R[key][k]):.1f}" for k in IND])
with open(os.path.join(OUT, "dani_po_gradovima.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    keys = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
    w.writerow(["mjesto", "pokazatelj"] + [period_label(p) + (" " + s.upper() if s else "") for p, s in keys])
    for n in places:
        for k in IND:
            w.writerow([n, IND[k][0]] + [f"{at(R[key][k], CITIES[n][0], CITIES[n][1]):.1f}" for key in keys])
print("✓ dani_bih.csv, dani_po_gradovima.csv")
