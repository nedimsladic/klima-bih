"""
Poljoprivreda, vinogradarstvo i skijaška sezona u BiH (1901–2099).

Ista "delta change" metoda kao dani_bih.py (dnevni E-OBS 1991–2020 + mjesečna promjena iz Beck et al. 2023).

Pokazatelji (prosjek 30 godina):
  vegetacijski period  ETCCDI GSL: od prvog niza 6 dana Tsr > 5 °C do prvog niza 6 dana Tsr < 5 °C nakon 1. jula (dana)
  posljednji proljetni mraz   prosječan datum posljednjeg dana s Tmin < 0 °C prije 1. jula
  prvi jesenji mraz           prosječan datum prvog dana s Tmin < 0 °C od 1. jula
  stepen-dani rasta    Σ max(Tsr − 10, 0), april–oktobar (°C·dan)
  Huglinov indeks      Σ ((Tsr − 10) + (Tmax − 10)) / 2 × K, april–septembar (K po geografskoj širini)
Skijaška sezona (decembar–mart) na planinskim centrima:
  dani sa snježnim padavinama   padavine ≥ 1 mm i Tsr ≤ 0 °C
  noći za umjetni snijeg        Tmin ≤ −2 °C (gruba procjena; prava granica zavisi i od vlažnosti zraka)

Upotreba:
    python agro_bih.py <eobs_dir> <climate_0p1_dir> <bih_boundary.geojson> <out_dir>
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
TIME = TN["time"]
MONTH = TIME.dt.month.values - 1
YEAR = TIME.dt.year.values
DOY = TIME.dt.dayofyear.values
YEARS = np.unique(YEAR)
tn, tx, rr = TN.values, TX.values, RR.values
NAN = np.isnan(tn[0])
KLAT = np.select([LAT < 42, LAT < 44, LAT < 46], [1.02, 1.03, 1.04], 1.05)[:, None] * np.ones(tn.shape[1:])


def clim(p, s, var):
    f = os.path.join(CLIM, p, *([s] if s else []), "ensemble_mean_0p1.nc")
    a = xr.open_dataset(f)[var]
    return a.interp(lon=xr.DataArray(LON, dims="x"), lat=xr.DataArray(LAT, dims="y"),
                    method="nearest").transpose("time", "y", "x").values


KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
T_REF, P_REF = clim(REF, None, "air_temperature"), clim(REF, None, "precipitation")


def runs(mask, n=6):
    """True na danu t ako su dani t..t+n-1 svi True (po vremenskoj osi 0)."""
    c = np.cumsum(np.concatenate([np.zeros((1,) + mask.shape[1:]), mask], axis=0), axis=0)
    out = np.zeros_like(mask)
    out[: mask.shape[0] - n + 1] = (c[n:] - c[:-n]) == n
    return out


def compute(key):
    if key == (REF, None):
        dt = np.zeros((12,) + tn.shape[1:]); fp = np.ones((12,) + tn.shape[1:])
    else:
        dt = clim(*key, "air_temperature") - T_REF
        fp = np.clip(clim(*key, "precipitation") / np.maximum(P_REF, 0.1), 0.3, 3.0)
    gsl, last, first, gdd, hug = (np.zeros(tn.shape[1:]) for _ in range(5))
    last_n, first_n = np.zeros(tn.shape[1:]), np.zeros(tn.shape[1:])
    ski_snow, ski_cold = np.zeros(tn.shape[1:]), np.zeros(tn.shape[1:])
    for y in YEARS:
        i = YEAR == y
        d = dt[MONTH[i]]
        n, x, r = tn[i] + d, tx[i] + d, rr[i] * fp[MONTH[i]]
        m = (n + x) / 2
        doy = DOY[i]
        nd = len(doy)
        idx = np.arange(nd)[:, None, None] * np.ones((1,) + n.shape[1:])
        # vegetacijski period (ETCCDI GSL)
        st = runs(m > 5)
        start = np.where(st.any(0), np.argmax(st, axis=0), nd)
        en = runs(m < 5) & (doy >= 182)[:, None, None]
        end = np.where(en.any(0), np.argmax(en, axis=0), nd)
        gsl += np.clip(end - start, 0, None)
        # mrazevi
        fr = n < 0
        spring = fr & (doy < 182)[:, None, None]
        has = spring.any(0)
        last += np.where(has, (np.where(spring, idx, -1).max(0) + 1), 0); last_n += has
        autumn = fr & (doy >= 182)[:, None, None]
        has = autumn.any(0)
        first += np.where(has, np.argmax(autumn, axis=0) + 1, 0); first_n += has
        # stepen-dani rasta i Huglin
        mo = MONTH[i]
        gs = ((mo >= 3) & (mo <= 9))[:, None, None]
        gdd += (np.maximum(m - 10, 0) * gs).sum(0)
        hs = ((mo >= 3) & (mo <= 8))[:, None, None]
        hug += (np.maximum(((m - 10) + (x - 10)) / 2, 0) * hs).sum(0) * KLAT
        # skijaška sezona: decembar te godine + januar–mart
        win = ((mo == 11) | (mo <= 2))[:, None, None]
        ski_snow += ((r >= 1) & (m <= 0) & win).sum(0)
        ski_cold += ((n <= -2) & win).sum(0)
    ny = len(YEARS)
    out = {"gsl": gsl / ny, "gdd": gdd / ny, "huglin": hug / ny,
           # prosječan datum: samo godine u kojima je mraza bilo; ako ga je bilo u < 1/3 godina, nema podatka
           "zadnji_mraz": np.where(last_n >= ny / 3, last / np.maximum(last_n, 1), np.nan),
           "prvi_mraz": np.where(first_n >= ny / 3, first / np.maximum(first_n, 1), np.nan),
           "ski_snijeg": ski_snow / ny, "ski_hladno": ski_cold / ny}
    return {k: np.where(NAN, np.nan, v) for k, v in out.items()}


R = {k: compute(k) for k in KEYS}
DER = os.path.join(os.path.dirname(os.path.abspath(OUT)), "podaci")
np.savez_compressed(os.path.join(DER, "agro_svi.npz"), lon=LON, lat=LAT,
                    **{f"{v}|{p}|{s or 'hist'}": R[(p, s)][v] for (p, s) in KEYS for v in R[KEYS[0]]})
print("✓ podaci/agro_svi.npz")


def bmean(a):
    m = INSIDE & np.isfinite(a)
    return float((a[m] * W[m]).sum() / W[m].sum())


MONTHS = ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "avg", "sep", "okt", "nov", "dec"]
CUM = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def date(doy):
    doy = int(round(doy)); m = int(np.searchsorted(CUM, doy, side="left")) - 1
    return f"{doy - CUM[m]}. {MONTHS[m]}"


def binned(cmap, bounds, lo=0.08, hi=0.95, rev=False):
    cols = list(plt.get_cmap(cmap)(np.linspace(lo, hi, len(bounds) - 1)))
    if rev:
        cols = cols[::-1]
    cm = ListedColormap(cols); cm.set_over(cols[-1]); cm.set_under(cols[0])
    return cm, BoundaryNorm(bounds, cm.N)


IND = {
    "gsl": ("Vegetacijski period", "dani od prvih 6 dana Tsr > 5 °C do prvih 6 dana Tsr < 5 °C u jesen", " dana",
            binned("YlGn", [150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]), "{:.0f}"),
    "zadnji_mraz": ("Posljednji proljetni mraz", "prosječan datum posljednjeg dana s Tmin < 0 °C", "",
                    binned("PuBu", [32, 46, 60, 74, 91, 105, 121, 135, 152, 166], rev=True), "date"),
    "prvi_mraz": ("Prvi jesenji mraz", "prosječan datum prvog dana s Tmin < 0 °C od 1. jula", "",
                  binned("PuBu", [244, 258, 274, 288, 305, 319, 335, 349, 365], rev=False), "date"),
    "gdd": ("Stepen-dani rasta", "Σ (Tsr − 10 °C), april–oktobar", " °C·dan",
            binned("YlOrBr", [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]), "{:.0f}"),
    "huglin": ("Huglinov indeks", "pogodnost za vinovu lozu, april–septembar", "",
               binned("RdPu", [1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]), "{:.0f}"),
}


def tick_labels(k, b):
    return [date(v) for v in b] if IND[k][4] == "date" else [f"{v:.0f}" for v in b]


for k, (name, rule, unit, (cm, nm), fmt) in IND.items():
    f = (lambda v: date(v)) if fmt == "date" else None
    ff = "{}" if fmt == "date" else fmt
    lab = f"{name} ({rule})"

    def mean_txt(a):
        v = bmean(a)
        return date(v) if fmt == "date" else v

    items = [((f"Procjena {period_label(p)}" if p != REF else f"Opažanja {period_label(REF)} (E-OBS)"),
              R[(p, None)][k], EXT, mean_txt(R[(p, None)][k])) for p in HIST]
    B.panel(items, f"{name} u Bosni i Hercegovini", f"• {name}, historijski periodi 1901–2020",
            f"agro_{k}_historijski.png", cm, nm, lab, fmt=ff, unit=unit, ticks=nm.boundaries, extend="both", ticklabels=tick_labels(k, nm.boundaries))
    for p in FUT:
        items = [(f"Opažanja {period_label(REF)} (E-OBS)", R[(REF, None)][k], EXT, mean_txt(R[(REF, None)][k]))]
        items += [(SSP[s], R[(p, s)][k], EXT, mean_txt(R[(p, s)][k])) for s in SSP]
        B.panel(items, f"{name} u Bosni i Hercegovini", f"• {name}, projekcije za period {period_label(p)}",
                f"agro_{k}_{p}.png", cm, nm, lab, fmt=ff, unit=unit, ticks=nm.boundaries, extend="both",
                ticklabels=tick_labels(k, nm.boundaries))
    if fmt == "date":                                      # datumi na skali boja umjesto rednih brojeva dana
        pass

# prosjek BiH kroz vrijeme (2×3: 5 pokazatelja + ski)
X = {"1901_1930": 1915.5, "1931_1960": 1945.5, "1961_1990": 1975.5, REF: 2005.5, "2041_2070": 2055.5, "2071_2099": 2085}
fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))
for ax, (k, (name, rule, unit, _, fmt)) in zip(axes.ravel(), IND.items()):
    r0 = bmean(R[(REF, None)][k])
    for s in SSP:
        ys = [r0] + [bmean(R[(p, s)][k]) for p in FUT]
        ax.plot([X[REF]] + [X[p] for p in FUT], ys, color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"),
                marker="o", ms=5, mec="white", zorder=3)
        ax.annotate(date(ys[-1]) if fmt == "date" else f"{ys[-1]:.0f}", (X["2071_2099"], ys[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9, color=SSP_COLORS[s])
    ax.plot([X[p] for p in HIST], [bmean(R[(p, None)][k]) for p in HIST], color="#222", lw=2.6, marker="o",
            ms=7, mec="white", zorder=4)
    ax.annotate(date(r0) if fmt == "date" else f"{r0:.0f}", (X[REF], r0), xytext=(-9, 8),
                textcoords="offset points", ha="right", fontsize=11, color="#222")
    ax.set_title(name, fontsize=16, loc="left", pad=22)
    ax.text(0, 1.02, rule, transform=ax.transAxes, fontsize=10, color="#666", va="bottom")
    ax.set_xticks(list(X.values())); ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in X])
    ax.set_xlim(1905, 2099)
    if fmt == "date":
        yt = ax.get_yticks(); ax.set_yticks(yt); ax.set_yticklabels([date(v) for v in yt])
        ax.set_ylabel("prosječan datum (prosjek BiH)", fontsize=11)
    else:
        ax.set_ylabel(f"{unit.strip() or 'indeks'} (prosjek BiH)", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=10)
# skijaška sezona: planinski centri, 1991–2020, 2041–2070 i 2071–2099
ax = axes.ravel()[5]
peaks = sorted(PEAKS)


def at(a, x, y):
    i = int(np.argmin(np.abs(LAT - y))); j = int(np.argmin(np.abs(LON - x)))
    v = a[i, j]
    return float(np.nanmean(a[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]) if np.isnan(v) else v)


cols_ = [((REF, None), "1991–2020", "#222"),
         (("2041_2070", "ssp245"), "2041–70 SSP2-4.5", "#f2b366"), (("2041_2070", "ssp585"), "2041–70 SSP5-8.5", "#e8868a"),
         (("2071_2099", "ssp245"), "2071–99 SSP2-4.5", "#c46e10"), (("2071_2099", "ssp585"), "2071–99 SSP5-8.5", SSP_COLORS["ssp585"])]
xs = np.arange(len(peaks)); w = 0.8 / len(cols_)
for j, (key, lab, col) in enumerate(cols_):
    v = [at(R[key]["ski_snijeg"], CITIES[pk][0], CITIES[pk][1]) for pk in peaks]
    ax.bar(xs + (j - (len(cols_) - 1) / 2) * w, v, width=w * 0.92, color=col, label=lab, zorder=3)
ax.set_xticks(xs); ax.set_xticklabels(peaks, fontsize=10.5)
ax.set_title("Skijaška sezona: dani sa snježnim padavinama", fontsize=16, loc="left", pad=22)
ax.text(0, 1.02, "decembar–mart; padavine ≥ 1 mm i Tsr ≤ 0 °C", transform=ax.transAxes, fontsize=10, color="#666", va="bottom")
ax.set_ylabel("dana u sezoni", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
ax.set_ylim(0, ax.get_ylim()[1] * 1.38)
ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color("#888")
hs = [plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=7, mec="white",
                 label="Historijski periodi (1991–2020 opažanja E-OBS, ranije procjena)")]
hs += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                  mec="white", label=SSP[s].replace(" · ", " – ")) for s in SSP]
fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.16, hspace=0.45, wspace=0.28)
x0 = axes[0, 0].get_position().x0 - 0.05
fig.legend(handles=hs, loc="lower left", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(x0, 0.0),
           title="Period / scenarij", title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.text(x0, 0.955, "Poljoprivreda, vinogradarstvo i skijanje u BiH", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.91, "• Prosjek za BiH i planinske centre (1901–2099)", fontsize=27, ha="left", va="baseline")
B.put_credit(fig, fig.transFigure, 0.97, 0.91, 15)
fig.savefig(os.path.join(OUT, "agro_prosjek_bih.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ agro_prosjek_bih.png")

# skijanje: tabela za planinske centre
with open(os.path.join(OUT, "skijanje_planine.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["planina", "period", "scenarij", "dani sa snježnim padavinama (dec–mar)", "noći Tmin ≤ −2 °C (dec–mar)"])
    for pk in peaks:
        for key in KEYS:
            w.writerow([pk, period_label(key[0]), key[1].upper() if key[1] else "historijski",
                        f"{at(R[key]['ski_snijeg'], *CITIES[pk][:2]):.1f}", f"{at(R[key]['ski_hladno'], *CITIES[pk][:2]):.1f}"])
with open(os.path.join(OUT, "agro_bih.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij", "vegetacijski period (dana)", "posljednji proljetni mraz", "prvi jesenji mraz",
                "stepen-dani rasta (°C·dan)", "Huglinov indeks"])
    for key in KEYS:
        r = R[key]
        w.writerow([period_label(key[0]), key[1].upper() if key[1] else ("opažanja (E-OBS)" if key[0] == REF else "procjena"),
                    f"{bmean(r['gsl']):.0f}", date(bmean(r["zadnji_mraz"])), date(bmean(r["prvi_mraz"])),
                    f"{bmean(r['gdd']):.0f}", f"{bmean(r['huglin']):.0f}"])
print("✓ agro_bih.csv, skijanje_planine.csv")
