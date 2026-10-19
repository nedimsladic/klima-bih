"""
Temperatura i padavine u Bosni i Hercegovini (1901–2099), prosjek za teritoriju BiH.

Podaci: Beck et al. (2023), Scientific Data 10, 724 – climate_data_0p5.zip
        (mjesečne klimatologije temperature zraka i padavina, srednja vrijednost ansambla)
Granica: geoBoundaries (gbOpen, BIH ADM0)

Upotreba:
    python temperatura_bih.py <climate_0p5_dir> <bih_boundary.geojson|.shp> <out_dir>
"""
import sys
import os
import csv
import json
import tempfile

import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import font_manager

CLIM_DIR, BOUNDARY, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
DPI = 600
AUTHOR = "Analiza: Nedim Sladić\n(prateći Beck et al. (2018, 2023))"
AUTHOR_COLOR = "#8a8a8a"

# Helvetica (macOS), isto kao u koppen_bih.py
HELV = "/System/Library/Fonts/Helvetica.ttc"
if os.path.exists(HELV):
    from fontTools.ttLib import TTCollection
    _fdir = tempfile.mkdtemp(prefix="helv_")
    for _i, _f in enumerate(TTCollection(HELV).fonts[:4]):
        _path = os.path.join(_fdir, f"Helvetica-{_i}.ttf")
        _f.save(_path)
        font_manager.fontManager.addfont(_path)
    plt.rcParams.update({"font.family": "Helvetica", "mathtext.fontset": "custom",
                         "mathtext.rm": "Helvetica", "mathtext.bf": "Helvetica:bold"})

HIST = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
FUT = ["2041_2070", "2071_2099"]
SSP = {
    "ssp119": "SSP1-1.9 · vrlo niske emisije",
    "ssp126": "SSP1-2.6 · niske emisije",
    "ssp434": "SSP4-3.4 · srednje niske emisije",
    "ssp245": "SSP2-4.5 · srednje emisije",
    "ssp460": "SSP4-6.0 · srednje visoke emisije",
    "ssp370": "SSP3-7.0 · visoke emisije",
    "ssp585": "SSP5-8.5 · vrlo visoke emisije",
}
SSP_COLORS = {"ssp119": "#00ADCF", "ssp126": "#173C66", "ssp434": "#4E9FD3", "ssp245": "#F79420",
              "ssp460": "#C0812B", "ssp370": "#E71D25", "ssp585": "#951B1E"}
SSP_DASH = {"ssp434": (0, (5, 2)), "ssp460": (0, (5, 2))}
MONTHS = ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "avg", "sep", "okt", "nov", "dec"]
SEASONS = {"Zima (DJF)": [11, 0, 1], "Proljeće (MAM)": [2, 3, 4],
           "Ljeto (JJA)": [5, 6, 7], "Jesen (SON)": [8, 9, 10]}
MID = {p: (int(p[:4]) + int(p[5:])) / 2 for p in HIST + FUT}


def period_label(p):
    a, b = p.split("_")
    return f"{a}–{b}"


border = gpd.read_file(BOUNDARY).to_crs(4326)


def bih_mean(path):
    """Mjesečne vrijednosti (12) temperature i padavina, ponderisano cos(lat), ćelije sa centrom u BiH."""
    ds = xr.open_dataset(path).sel(lon=slice(15, 20.5), lat=slice(46, 42))
    lon, lat = ds["lon"].values, ds["lat"].values
    inside = ~geometry_mask(border.geometry, (len(lat), len(lon)),
                            from_origin(lon[0] - 0.25, lat[0] + 0.25, 0.5, 0.5))
    w = np.cos(np.deg2rad(lat))[:, None] * inside
    out = {}
    for v in ("air_temperature", "precipitation"):
        a = ds[v].values
        out[v] = np.array([np.nansum(np.where(inside & np.isfinite(a[m]), a[m], 0) * w) /
                           (w * np.isfinite(a[m])).sum() for m in range(12)])
    return out


def path(p, s, kind):
    return os.path.join(CLIM_DIR, p, *([s] if s else []), f"ensemble_{kind}_0p5.nc")


keys = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
M = {k: bih_mean(path(*k, "mean")) for k in keys}
T = {k: M[k]["air_temperature"] for k in keys}
P = {k: M[k]["precipitation"] for k in keys}
REF = ("1991_2020", None)
tann = {k: T[k].mean() for k in keys}
# raspon među članovima ansambla: prosjek mjesečnih standardnih devijacija (približno σ godišnjeg prosjeka)
SD = {k: bih_mean(path(*k, "std"))["air_temperature"].mean() for k in keys}
BAND = ("ssp126", "ssp245", "ssp585")
pann = {k: P[k].sum() for k in keys}


def put_credit(fig, trans, x, y, fs):
    l1, l2 = AUTHOR.split("\n")
    kw = dict(ha="right", va="baseline", fontsize=fs, color=AUTHOR_COLOR)
    fig.text(x, y, l2, transform=trans, **kw)
    fig.text(x, y, l1, transform=mtransforms.offset_copy(trans, fig=fig, y=1.3 * fs, units="points"), **kw)


def header(fig, x0, sub, y1=0.955, y2=0.905):
    fig.text(x0, y1, "Temperatura i padavine u Bosni i Hercegovini", fontsize=27, ha="left",
             va="baseline")
    fig.text(x0, y2, sub, fontsize=27, ha="left", va="baseline")
    put_credit(fig, fig.transFigure, 0.99, y2, 15)


def style(ax):
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=10.5)


def ssp_handles(with_hist=True):
    hs = []
    if with_hist:
        hs.append(plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=6, mec="white",
                             label="Historijski period (opažanja)"))
    hs += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                      mec="white", label=SSP[s].replace(" · ", " – ")) for s in SSP]
    return hs


# 1) Srednja godišnja temperatura kroz vrijeme + zagrijavanje u odnosu na 1991–2020
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 8.2), gridspec_kw={"width_ratios": [1.35, 1]})
hx = [MID[p] for p in HIST]
hy = [tann[(p, None)] for p in HIST]
for s in SSP:
    a1.plot([hx[-1]] + [MID[p] for p in FUT], [hy[-1]] + [tann[(p, s)] for p in FUT],
            color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5, mec="white", zorder=3)
for s in BAND:                                      # ±1σ ansambla
    xx = [hx[-1]] + [MID[p] for p in FUT]
    yy = np.array([hy[-1]] + [tann[(p, s)] for p in FUT])
    ss = np.array([SD[REF]] + [SD[(p, s)] for p in FUT])
    a1.fill_between(xx, yy - ss, yy + ss, color=SSP_COLORS[s], alpha=0.13, lw=0, zorder=2)
a1.fill_between(hx, np.array(hy) - [SD[(p, None)] for p in HIST], np.array(hy) + [SD[(p, None)] for p in HIST],
                color="#222", alpha=0.12, lw=0, zorder=2)
a1.plot(hx, hy, color="#222", lw=2.6, marker="o", ms=6, mec="white", zorder=4)
for s in BAND:
    v = tann[("2071_2099", s)]
    a1.annotate(f"{v:.1f} °C", (MID["2071_2099"], v), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=11, color="#222")
a1.annotate(f"{hy[-1]:.1f} °C", (hx[-1], hy[-1]), xytext=(-8, 8), textcoords="offset points",
            ha="right", fontsize=11, color="#222")
a1.axvline(2020, color="#999", lw=1, ls=(0, (2, 2)), zorder=1)
a1.set_xlim(1905, 2102)
a1.set_xticks([MID[p] for p in HIST + FUT])
a1.set_xticklabels([period_label(p).replace("–", "–\n") for p in HIST + FUT])
a1.set_ylabel("Srednja godišnja temperatura (°C)", fontsize=12)
a1.set_title("Srednja godišnja temperatura (30-godišnji prosjek)", fontsize=15, loc="left", pad=12)
style(a1)

ref = tann[REF]
xs = np.arange(len(SSP))
wd = 0.38
for j, (p, alpha) in enumerate((("2041_2070", 0.55), ("2071_2099", 1.0))):
    vals = [tann[(p, s)] - ref for s in SSP]
    err = [SD[(p, s)] for s in SSP]
    bars = a2.bar(xs + (j - 0.5) * wd, vals, width=wd - 0.04, color=[SSP_COLORS[s] for s in SSP],
                  alpha=alpha, edgecolor="white", linewidth=0.8, zorder=3)
    a2.errorbar(xs + (j - 0.5) * wd, vals, yerr=err, fmt="none", ecolor="#333", elinewidth=1, capsize=3, zorder=4)
    for x, v, e in zip(xs + (j - 0.5) * wd, vals, err):
        a2.text(x, v + e + 0.06, f"+{v:.1f}", ha="center", va="bottom", fontsize=9.5, color="#222")
a2.set_xticks(xs)
a2.set_xticklabels([SSP[s].split(" · ")[0] for s in SSP], fontsize=10)
a2.set_ylabel("Zagrijavanje u odnosu na 1991–2020 (°C)", fontsize=12)
a2.set_title("Zagrijavanje do sredine i kraja vijeka", fontsize=15, loc="left", pad=12)
a2.set_ylim(0, max(tann[("2071_2099", s)] - ref + SD[("2071_2099", s)] for s in SSP) * 1.12)
style(a2)
a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#777", alpha=0.55, label="2041–2070"),
                   plt.Rectangle((0, 0), 1, 1, color="#777", label="2071–2099")],
          loc="upper left", frameon=False, fontsize=11.5)
a1.text(0.01, 0.97, "Osjenčeno: ±1 standardna devijacija ansambla", transform=a1.transAxes, fontsize=10.5,
        color="#555", va="top")
a2.text(0.99, 0.97, "Crtice: ±1 standardna devijacija ansambla", transform=a2.transAxes, fontsize=10.5,
        color="#555", va="top", ha="right")
fig.legend(handles=ssp_handles(), loc="lower left", ncol=4, frameon=False, fontsize=12,
           bbox_to_anchor=(a1.get_position().x0, -0.02), title="Period / scenarij",
           title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.2, wspace=0.18)
header(fig, a1.get_position().x0 - 0.035, "• Srednja godišnja temperatura (1901–2099)")
fig.savefig(os.path.join(OUT, "temperatura_godisnja.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ temperatura_godisnja.png")

# 2) Godišnji hod temperature + mjesečno zagrijavanje 2041–2070 i 2071–2099
fig, (b1, b2) = plt.subplots(1, 2, figsize=(17, 8.6), gridspec_kw={"width_ratios": [1, 1.1]})
mx = np.arange(12)
b1.plot(mx, T[("1961_1990", None)], color="#9a9a9a", lw=2.2, marker="o", ms=5, mec="white",
        label="1961–1990 (opažanja)")
b1.plot(mx, T[REF], color="#222", lw=2.6, marker="o", ms=6, mec="white", label="1991–2020 (opažanja)")
for s in ("ssp126", "ssp245", "ssp585"):
    b1.plot(mx, T[("2041_2070", s)], color=SSP_COLORS[s], lw=1.6, ls=(0, (4, 2)), marker="o", ms=3.5, mec="white",
            label=f"2041–2070 · {SSP[s].split(' · ')[0]}")
for s in ("ssp126", "ssp245", "ssp585"):
    b1.plot(mx, T[("2071_2099", s)], color=SSP_COLORS[s], lw=2.2, marker="o", ms=5, mec="white",
            label=f"2071–2099 · {SSP[s].split(' · ')[0]}")
b1.set_xticks(mx)
b1.set_xticklabels(MONTHS)
b1.set_ylabel("Srednja mjesečna temperatura (°C)", fontsize=12)
b1.set_title("Godišnji hod temperature", fontsize=15, loc="left", pad=12)
b1.legend(loc="upper left", frameon=False, fontsize=9.5, ncol=2)
lo_, hi_ = b1.get_ylim(); b1.set_ylim(lo_, hi_ + (hi_ - lo_) * 0.32)
style(b1)

grid = np.array([T[(p, s)] - T[REF] for p in FUT for s in SSP])
im = b2.imshow(grid, cmap="YlOrRd", vmin=0, vmax=np.ceil(grid.max()), aspect="auto")
for i in range(grid.shape[0]):
    for j in range(12):
        v = grid[i, j]
        b2.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                color="white" if v > grid.max() * 0.62 else "#222")
b2.set_xticks(mx)
b2.set_xticklabels(MONTHS)
b2.set_yticks(range(grid.shape[0]))
b2.set_yticklabels([f"{period_label(p)} · {SSP[s].split(' · ')[0]}" for p in FUT for s in SSP], fontsize=9.5)
b2.axhline(len(SSP) - 0.5, color="white", lw=4)
b2.tick_params(length=0, labelsize=10.5, colors="#333")
for sp in b2.spines.values():
    sp.set_visible(False)
b2.set_title("Zagrijavanje po mjesecima u odnosu na 1991–2020 (°C)", fontsize=15,
             loc="left", pad=12)
cb = fig.colorbar(im, ax=b2, fraction=0.035, pad=0.02)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=10)
cb.set_label("°C", fontsize=11)
fig.subplots_adjust(left=0.05, right=0.97, top=0.8, bottom=0.08, wspace=0.16)
header(fig, b1.get_position().x0 - 0.035, "• Godišnji hod temperature i zagrijavanje po mjesecima")
fig.savefig(os.path.join(OUT, "temperatura_mjesecna.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ temperatura_mjesecna.png")

# 3) Promjena temperature i padavina po godišnjim dobima, 2041–2070 i 2071–2099
fig, CC = plt.subplots(2, 2, figsize=(17, 13))
sx = np.arange(len(SEASONS))
wd = 0.8 / len(SSP)
for r, per in enumerate(FUT):
    c1, c2 = CC[r]
    for i, s in enumerate(SSP):
        dt = [T[(per, s)][idx].mean() - T[REF][idx].mean() for idx in SEASONS.values()]
        dp = [(P[(per, s)][idx].sum() / P[REF][idx].sum() - 1) * 100 for idx in SEASONS.values()]
        off = (i - (len(SSP) - 1) / 2) * wd
        c1.bar(sx + off, dt, width=wd * 0.92, color=SSP_COLORS[s], edgecolor="white", lw=0.6, zorder=3)
        c2.bar(sx + off, dp, width=wd * 0.92, color=SSP_COLORS[s], edgecolor="white", lw=0.6, zorder=3)
    for ax, lab, ttl in ((c1, "Promjena temperature (°C)", "Temperatura"),
                         (c2, "Promjena padavina (%)", "Padavine")):
        ax.set_xticks(sx)
        ax.set_xticklabels(list(SEASONS), fontsize=12)
        ax.set_ylabel(lab, fontsize=12)
        ax.set_title(f"{ttl}: {period_label(per)} u odnosu na 1991–2020", fontsize=15, loc="left", pad=12)
        ax.axhline(0, color="#555", lw=1, zorder=2)
        style(ax)
for j in range(2):
    lo = min(CC[0, j].get_ylim()[0], CC[1, j].get_ylim()[0]); hi = max(CC[0, j].get_ylim()[1], CC[1, j].get_ylim()[1])
    CC[0, j].set_ylim(lo, hi); CC[1, j].set_ylim(lo, hi)
c1 = CC[0, 0]
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=SSP_COLORS[s], label=SSP[s].replace(" · ", " – "))
                    for s in SSP], loc="lower left", ncol=4, frameon=False,
           fontsize=12, bbox_to_anchor=(c1.get_position().x0, -0.01), title="Scenarij",
           title_fontproperties={"weight": "bold", "size": 13}, alignment="left")
fig.subplots_adjust(left=0.05, right=0.99, top=0.885, bottom=0.12, wspace=0.18, hspace=0.32)
header(fig, c1.get_position().x0 - 0.035, "• Promjene po godišnjim dobima do sredine i kraja vijeka", 0.972, 0.944)
fig.savefig(os.path.join(OUT, "temperatura_padavine_sezone.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ temperatura_padavine_sezone.png")

# Tabela i JSON sažetak (za web stranicu)
with open(os.path.join(OUT, "temperatura_padavine.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij", "godišnja temperatura (°C)", "godišnje padavine (mm)"]
               + [f"T {m} (°C)" for m in MONTHS])
    for k in keys:
        p, s = k
        w.writerow([period_label(p), s.upper() if s else "historijski", f"{tann[k]:.2f}", f"{pann[k]:.0f}"]
                   + [f"{v:.2f}" for v in T[k]])
summary = {f"{p}|{s or 'hist'}": {"t": round(float(tann[(p, s)]), 2), "p": round(float(pann[(p, s)]))}
           for p, s in keys}
with open(os.path.join(OUT, "temperatura_sazetak.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print("✓ temperatura_padavine.csv")
