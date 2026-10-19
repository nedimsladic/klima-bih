"""
Zimsko hlađenje za voćke i ekstremne padavine u BiH (1901–2099).

Zimsko hlađenje (sati hlađenja): broj sati s temperaturom između 0 i 7,2 °C od 1. novembra do kraja februara
(Weinbergerov model). Satne temperature su procijenjene iz dnevnog Tmin/Tmax kosinusnom krivom (minimum u 3 h,
maksimum u 15 h). Šljiva i jabuka obično trebaju 800–1200 sati, trešnja 1000–1400.

Ekstremne padavine: Rx1day (najveća dnevna količina u godini) i Rx5day (najveća petodnevna), prosjek godina.
Dvije procjene budućnosti:
  1) "delta" – dnevne padavine pomnožene mjesečnim omjerom (kao kod ostalih pokazatelja); mijenja ekstreme
     koliko i prosjek, pa vjerovatno potcjenjuje porast.
  2) "Clausius-Clapeyron" – ekstremi rastu 7 % po °C lokalnog zagrijavanja (fizička gornja procjena za jake
     kiše), primijenjeno na vrijednosti iz 1991–2020.

Upotreba:
    python hladnoca_kisa_bih.py <eobs_dir> <climate_0p1_dir> <bih_boundary.geojson> <out_dir>
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

from bih_stil import (Bih, HIST, FUT, REF, SSP, SSP_COLORS, SSP_DASH, DPI, period_label, ssp_code)

EOBS, CLIM, BOUNDARY, OUT = sys.argv[1:5]
B = Bih(BOUNDARY, OUT)
BOX = dict(lon=(15.3, 20.0), lat=(42.2, 45.6))


def eobs(var):
    parts = []
    for f in sorted(glob.glob(os.path.join(EOBS, f"{var}_ens_mean_0.1deg_reg_*_v31.0e.nc"))):
        ds = xr.open_dataset(f)
        lat = ds["latitude"].values
        latsl = slice(*BOX["lat"]) if lat[0] < lat[-1] else slice(*BOX["lat"][::-1])
        a = ds[var].sel(longitude=slice(*BOX["lon"]), latitude=latsl, time=slice("1991-01-01", "2020-12-31"))
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
YEAR = TN["time"].dt.year.values
# zimska sezona: novembar–decembar pripadaju sljedećoj godini (sezona 1991/92 → 1992)
SEASON = np.where(MONTH >= 10, YEAR + 1, YEAR)
WINTER = (MONTH >= 10) | (MONTH <= 1)
tn, tx, rr = TN.values, TX.values, RR.values
NAN = np.isnan(tn[0])
HRS = np.cos(2 * np.pi * (np.arange(24) - 15) / 24)            # +1 u 15 h, −1 u 3 h


def clim(p, s, var):
    f = os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")
    a = xr.open_dataset(f)[var]
    return a.interp(lon=xr.DataArray(LON, dims="x"), lat=xr.DataArray(LAT, dims="y"),
                    method="nearest").transpose("time", "y", "x").values


KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
T_REF, P_REF = clim(REF, None, "air_temperature"), clim(REF, None, "precipitation")
seasons = np.unique(SEASON[WINTER]); seasons = seasons[(seasons > 1991) & (seasons <= 2020)]
years = np.unique(YEAR)


def compute(key):
    if key == (REF, None):
        dt = np.zeros((12,) + tn.shape[1:]); fp = np.ones((12,) + tn.shape[1:])
    else:
        dt = clim(*key, "air_temperature") - T_REF
        fp = np.clip(clim(*key, "precipitation") / np.maximum(P_REF, 0.1), 0.3, 3.0)
    # sati hlađenja
    idx = np.where(WINTER & np.isin(SEASON, seasons))[0]
    ch = np.zeros(tn.shape[1:])
    for chunk in np.array_split(idx, 20):
        d = dt[MONTH[chunk]]
        n, x = tn[chunk] + d, tx[chunk] + d
        mid, amp = (n + x) / 2, (x - n) / 2
        T = mid[..., None] + amp[..., None] * HRS            # (dani, y, x, 24)
        ch += ((T > 0) & (T < 7.2)).sum(axis=(0, 3))
    ch /= len(seasons)
    # ekstremne padavine (delta)
    r = rr * fp[MONTH]
    r5 = np.cumsum(np.concatenate([np.zeros((1,) + r.shape[1:]), np.nan_to_num(r)], axis=0), axis=0)
    r5 = r5[5:] - r5[:-5]
    y5 = YEAR[4:]
    rx1 = np.mean([r[YEAR == y].max(axis=0) for y in years], axis=0)
    rx5 = np.mean([r5[y5 == y].max(axis=0) for y in years], axis=0)
    dT = dt.mean(axis=0)
    return {"chill": np.where(NAN, np.nan, ch), "rx1": np.where(NAN, np.nan, rx1), "rx5": np.where(NAN, np.nan, rx5),
            "dT": dT}


R = {k: compute(k) for k in KEYS}
for k in KEYS:                                                  # Clausius-Clapeyron: +7 % po °C
    f = 1.07 ** R[k]["dT"]
    R[k]["rx1_cc"] = R[(REF, None)]["rx1"] * f
    R[k]["rx5_cc"] = R[(REF, None)]["rx5"] * f
DER = os.path.join(os.path.dirname(os.path.abspath(OUT)), "podaci")
np.savez_compressed(os.path.join(DER, "hladnoca_kisa.npz"), lon=LON, lat=LAT,
                    **{f"{v}|{p}|{s or 'hist'}": R[(p, s)][v] for (p, s) in KEYS
                       for v in ("chill", "rx1", "rx5", "rx1_cc", "rx5_cc")})
print("✓ podaci/hladnoca_kisa.npz")


def bmean(a):
    m = INSIDE & np.isfinite(a)
    return float((a[m] * W[m]).sum() / W[m].sum())


def binned(cmap, bounds, lo=0.08, hi=0.95):
    cols = list(plt.get_cmap(cmap)(np.linspace(lo, hi, len(bounds) - 1)))
    cm = ListedColormap(cols); cm.set_over(cols[-1]); cm.set_under(cols[0])
    return cm, BoundaryNorm(bounds, cm.N)


IND = {
    "chill": ("Zimsko hlađenje za voćke", "sati između 0 i 7,2 °C, novembar–februar", " sati",
              binned("BuPu", [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])),
    "rx1": ("Najjača dnevna kiša u godini", "Rx1day, prosjek godišnjih maksimuma (delta metoda)", " mm",
            binned("GnBu", [30, 40, 50, 60, 70, 80, 90, 100, 120, 140])),
    "rx5": ("Najjačih pet dana kiše u godini", "Rx5day, prosjek godišnjih maksimuma (delta metoda)", " mm",
            binned("GnBu", [50, 70, 90, 110, 130, 150, 170, 200, 230, 260])),
}
for k, (name, rule, unit, (cm, nm)) in IND.items():
    items = [((f"Procjena {period_label(p)}" if p != REF else f"Opažanja {period_label(REF)} (E-OBS)"),
              R[(p, None)][k], EXT, bmean(R[(p, None)][k])) for p in HIST]
    B.panel(items, f"{name} u Bosni i Hercegovini", f"• {name}, historijski periodi 1901–2020",
            f"hk_{k}_historijski.png", cm, nm, f"{name} ({rule})", fmt="{:.0f}", unit=unit, ticks=nm.boundaries, extend="both")
    for p in FUT:
        items = [(f"Opažanja {period_label(REF)} (E-OBS)", R[(REF, None)][k], EXT, bmean(R[(REF, None)][k]))]
        items += [(SSP[s], R[(p, s)][k], EXT, bmean(R[(p, s)][k])) for s in SSP]
        B.panel(items, f"{name} u Bosni i Hercegovini", f"• {name}, projekcije za period {period_label(p)}",
                f"hk_{k}_{p}.png", cm, nm, f"{name} ({rule})", fmt="{:.0f}", unit=unit, ticks=nm.boundaries, extend="both")

# prosjek BiH kroz vrijeme: hlađenje, Rx1day i Rx5day (delta i CC)
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5, "2041_2070": 2055.5, "2071_2099": 2085}
fig, axes = plt.subplots(1, 3, figsize=(19, 7.6))
for ax, (k, title, unit) in zip(axes, [("chill", "Zimsko hlađenje za voćke", "sati (nov–feb)"),
                                       ("rx1", "Najjača dnevna kiša u godini (Rx1day)", "mm"),
                                       ("rx5", "Najjačih pet dana kiše (Rx5day)", "mm")]):
    r0 = bmean(R[(REF, None)][k])
    for s in SSP:
        ys = [r0] + [bmean(R[(p, s)][k]) for p in FUT]
        ax.plot([X[REF]] + [X[p] for p in FUT], ys, color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"),
                marker="o", ms=5, mec="white", zorder=3)
        if k != "chill":
            yc = [r0] + [bmean(R[(p, s)][k + "_cc"]) for p in FUT]
            ax.plot([X[REF]] + [X[p] for p in FUT], yc, color=SSP_COLORS[s], lw=1.2, ls=":", zorder=2)
    ax.plot([X[p] for p in HIST], [bmean(R[(p, None)][k]) for p in HIST], color="#222", lw=2.6, marker="o",
            ms=7, mec="white", zorder=4)
    ax.annotate(f"{r0:.0f}", (X[REF], r0), xytext=(-9, 8), textcoords="offset points", ha="right", fontsize=11)
    if k == "chill":
        for y, lab in ((800, "šljiva, jabuka: oko 800–1200 sati"),):
            ax.axhline(y, color="#999", lw=1, ls="--"); ax.text(1908, y + 15, lab, fontsize=9.5, color="#666")
    ax.set_title(title, fontsize=15, loc="left", pad=12)
    ax.set_xticks(list(X.values())); ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in X])
    ax.set_xlim(1905, 2095)
    ax.set_ylabel(f"{unit} (prosjek BiH)", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color("#888")
hs = [plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=7, mec="white", label="Historijski periodi")]
hs += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), label=SSP[s].replace(" · ", " – ")) for s in SSP]
hs += [plt.Line2D([], [], color="#555", lw=1.2, ls=":", label="tačkasto: kiša raste 7 % po °C (Clausius-Clapeyron)")]
fig.subplots_adjust(left=0.05, right=0.98, top=0.76, bottom=0.26, wspace=0.22)
x0 = axes[0].get_position().x0 - 0.035
fig.legend(handles=hs, loc="lower left", ncol=3, frameon=False, fontsize=11.5, bbox_to_anchor=(x0, 0.0),
           title="Period / scenarij", title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.text(x0, 0.95, "Zima za voćke i jake kiše u BiH", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.885, "• Sati zimskog hlađenja i ekstremne padavine (1901–2099)", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.98, 0.885, 15)
fig.savefig(os.path.join(OUT, "hk_prosjek_bih.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ hk_prosjek_bih.png")

with open(os.path.join(OUT, "hladnoca_kisa_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij", "sati hlađenja (nov–feb)", "Rx1day delta (mm)", "Rx1day CC (mm)",
                "Rx5day delta (mm)", "Rx5day CC (mm)"])
    for key in KEYS:
        r = R[key]
        w.writerow([period_label(key[0]), key[1].upper() if key[1] else ("opažanja (E-OBS)" if key[0] == REF else "procjena"),
                    f"{bmean(r['chill']):.0f}", f"{bmean(r['rx1']):.1f}", f"{bmean(r['rx1_cc']):.1f}",
                    f"{bmean(r['rx5']):.1f}", f"{bmean(r['rx5_cc']):.1f}"])
print("✓ hladnoca_kisa_bih.csv")
