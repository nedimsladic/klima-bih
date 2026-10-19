"""
Köppen-Geiger klimatski tipovi za Bosnu i Hercegovinu (1901–2099).

Podaci: Beck et al. (2023), Scientific Data 10, 724
        https://doi.org/10.1038/s41597-023-02549-6  (figshare: koppen_geiger_tif.zip)
Granica: geoBoundaries (gbOpen, BIH ADM0)
Podloga: Esri World Shaded Relief (contextily)

Upotreba:
    python koppen_bih.py <kg_tif_dir> <bih_boundary.geojson|.shp> <out_dir> [kg_nc_dir]

kg_nc_dir (opcionalno): raspakovan koppen_geiger_nc.zip; iz njega se čita kg_confidence
i crtaju karte pouzdanosti klasifikacije.
"""
import sys
import os
import csv

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
import contextily as ctx
import tempfile
from matplotlib import font_manager

PEAKS = {"Bjelašnica", "Jahorina", "Vlašić", "Kupres", "Blidinje"}

# Helvetica (macOS): izdvoji obične i podebljane rezove iz .ttc i registruj ih za matplotlib
HELV = "/System/Library/Fonts/Helvetica.ttc"
if os.path.exists(HELV):
    from fontTools.ttLib import TTCollection
    _fdir = tempfile.mkdtemp(prefix="helv_")
    for _i, _f in enumerate(TTCollection(HELV).fonts[:4]):  # Regular, Bold, Oblique, Bold Oblique
        _path = os.path.join(_fdir, f"Helvetica-{_i}.ttf")
        _f.save(_path)
        font_manager.fontManager.addfont(_path)
    plt.rcParams.update({"font.family": "Helvetica", "mathtext.fontset": "custom",
                         "mathtext.rm": "Helvetica", "mathtext.bf": "Helvetica:bold"})

KG_DIR, BOUNDARY, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NC_DIR = sys.argv[4] if len(sys.argv) > 4 else None
RES = "0p00833333"  # ~1 km
DPI = 600
AUTHOR = "Analiza: Nedim Sladić\n(prateći Beck et al. (2018, 2023))"
AUTHOR_COLOR = "#8a8a8a"
os.makedirs(OUT, exist_ok=True)

# Kod -> (oznaka, opis na bosanskom, RGB iz Beck et al. 2023)
KG = {
    1: ("Af", "Tropska, prašumska", (0, 0, 255)),
    2: ("Am", "Tropska, monsunska", (0, 120, 255)),
    3: ("Aw", "Tropska, savanska", (70, 170, 250)),
    4: ("BWh", "Aridna, pustinjska, vruća", (255, 0, 0)),
    5: ("BWk", "Aridna, pustinjska, hladna", (255, 150, 150)),
    6: ("BSh", "Aridna, stepska, vruća", (245, 165, 0)),
    7: ("BSk", "Aridna, stepska, hladna", (255, 220, 100)),
    8: ("Csa", "Umjerena, suho ljeto, vruće ljeto", (255, 255, 0)),
    9: ("Csb", "Umjerena, suho ljeto, toplo ljeto", (200, 200, 0)),
    10: ("Csc", "Umjerena, suho ljeto, hladno ljeto", (150, 150, 0)),
    11: ("Cwa", "Umjerena, suha zima, vruće ljeto", (150, 255, 150)),
    12: ("Cwb", "Umjerena, suha zima, toplo ljeto", (100, 200, 100)),
    13: ("Cwc", "Umjerena, suha zima, hladno ljeto", (50, 150, 50)),
    14: ("Cfa", "Umjerena, bez sušnog perioda, vruće ljeto", (200, 255, 80)),
    15: ("Cfb", "Umjerena, bez sušnog perioda, toplo ljeto", (100, 255, 80)),
    16: ("Cfc", "Umjerena, bez sušnog perioda, hladno ljeto", (50, 200, 0)),
    17: ("Dsa", "Hladna, suho ljeto, vruće ljeto", (255, 0, 255)),
    18: ("Dsb", "Hladna, suho ljeto, toplo ljeto", (200, 0, 200)),
    19: ("Dsc", "Hladna, suho ljeto, hladno ljeto", (150, 50, 150)),
    20: ("Dsd", "Hladna, suho ljeto, vrlo hladna zima", (150, 100, 150)),
    21: ("Dwa", "Hladna, suha zima, vruće ljeto", (170, 175, 255)),
    22: ("Dwb", "Hladna, suha zima, toplo ljeto", (90, 120, 220)),
    23: ("Dwc", "Hladna, suha zima, hladno ljeto", (75, 80, 180)),
    24: ("Dwd", "Hladna, suha zima, vrlo hladna zima", (50, 0, 135)),
    25: ("Dfa", "Hladna, bez sušnog perioda, vruće ljeto", (0, 255, 255)),
    26: ("Dfb", "Hladna, bez sušnog perioda, toplo ljeto", (55, 200, 255)),
    27: ("Dfc", "Hladna, bez sušnog perioda, hladno ljeto", (0, 125, 125)),
    28: ("Dfd", "Hladna, bez sušnog perioda, vrlo hladna zima", (0, 70, 95)),
    29: ("ET", "Polarna, tundra", (178, 178, 178)),
    30: ("EF", "Polarna, vječni led", (102, 102, 102)),
}

# Opširniji opis klase za legende (granica C/D: najhladniji mjesec 0 °C; vruće ljeto: najtopliji mjesec ≥ 22 °C)
DESC = {
    8: "Mediteranska klima s vrućim ljetom – suho ljeto, najtopliji mjesec ≥ 22 °C",
    9: "Mediteranska klima s toplim ljetom – suho ljeto, najtopliji mjesec < 22 °C",
    14: "Umjereno topla vlažna klima s vrućim ljetom – bez sušnog perioda, najtopliji mjesec ≥ 22 °C",
    15: "Umjereno topla vlažna klima s toplim ljetom – bez sušnog perioda, najtopliji mjesec < 22 °C",
    18: "Kontinentalna klima sa suhim, toplim ljetom – najhladniji mjesec ≤ 0 °C",
    25: "Vlažna kontinentalna klima s vrućim ljetom – najhladniji mjesec ≤ 0 °C, najtopliji ≥ 22 °C",
    26: "Vlažna kontinentalna klima s toplim ljetom – najhladniji mjesec ≤ 0 °C, najtopliji < 22 °C",
    27: "Subalpska (planinska) klima – najhladniji mjesec ≤ 0 °C, manje od 4 mjeseca ≥ 10 °C",
    29: "Planinska tundra – najtopliji mjesec između 0 i 10 °C",
}


def class_label(c):
    return rf"$\mathbf{{{KG[c][0]}}}$  {DESC.get(c, KG[c][1]).split(' – ')[0]}"


LEGEND_TITLE = "Klimatski tipovi prema Köppen-Geigerovoj klasifikaciji"
TITLE_FONT = {"weight": "bold", "size": 15}

SSP = {  # poredano po jačini antropogenog forsiranja
    "ssp119": "SSP1-1.9 · vrlo niske emisije",
    "ssp126": "SSP1-2.6 · niske emisije",
    "ssp434": "SSP4-3.4 · srednje niske emisije",
    "ssp245": "SSP2-4.5 · srednje emisije",
    "ssp460": "SSP4-6.0 · srednje visoke emisije",
    "ssp370": "SSP3-7.0 · visoke emisije",
    "ssp585": "SSP5-8.5 · vrlo visoke emisije",
}
HIST = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
FUT = ["2041_2070", "2071_2099"]

# naziv: (lon, lat, na panelima?, strana oznake)
CITIES = {
    "Sarajevo": (18.413, 43.856, True, "r"), "Banja Luka": (17.191, 44.772, True, "r"),
    "Tuzla": (18.674, 44.538, True, "r"), "Zenica": (17.907, 44.204, True, "r"),
    "Mostar": (17.808, 43.343, True, "r"), "Bihać": (15.870, 44.817, True, "r"),
    "Bijeljina": (19.214, 44.757, True, "r"), "Trebinje": (18.344, 42.712, True, "r"),
    "Brčko": (18.810, 44.873, False, "r"), "Prijedor": (16.714, 44.980, False, "r"),
    "Doboj": (18.088, 44.735, False, "r"), "Cazin": (15.943, 44.967, False, "r"),
    "Travnik": (17.665, 44.226, False, "l"), "Bugojno": (17.450, 44.057, False, "r"),
    "Livno": (17.007, 43.827, False, "r"), "Konjic": (17.961, 43.652, False, "r"),
    "Goražde": (18.946, 43.668, False, "r"), "Foča": (18.776, 43.506, False, "r"),
    "Zvornik": (19.102, 44.386, False, "r"), "Gradiška": (17.255, 45.145, False, "r"),
    # planinski centri (približne lokacije skijališta), crtani trouglom
    "Bjelašnica": (18.266, 43.716, False, "l"), "Jahorina": (18.567, 43.737, False, "r"),
    "Vlašić": (17.648, 44.292, False, "r"), "Kupres": (17.282, 43.982, False, "r"),
    "Blidinje": (17.480, 43.620, False, "r"),
}

# Esri tile source; the shaded relief has no place-name labels, so all text stays in Bosnian
BASEMAP = ctx.providers.Esri.WorldShadedRelief

border = gpd.read_file(BOUNDARY).to_crs(4326)
xmin, ymin, xmax, ymax = border.total_bounds
pad = 0.12
EXTENT_VIEW = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)


def period_label(p):
    a, b = p.split("_")
    return f"{a}–{b}"


def load(path):
    with rasterio.open(path) as r:
        arr, tr = mask(r, border.geometry, crop=True, nodata=0, all_touched=False)
    arr = arr[0]
    h, w = arr.shape
    ext = (tr.c, tr.c + tr.a * w, tr.f + tr.e * h, tr.f)
    lats = tr.f + tr.e * (np.arange(h) + 0.5)
    return arr, ext, lats


# Build the list of scenarios
SCEN = [(p, None, os.path.join(KG_DIR, p, f"koppen_geiger_{RES}.tif")) for p in HIST]
for p in FUT:
    for s in SSP:
        SCEN.append((p, s, os.path.join(KG_DIR, p, s, f"koppen_geiger_{RES}.tif")))

data = {}
for p, s, f in SCEN:
    data[(p, s)] = load(f)

# Area-weighted shares (cos(lat) weighting for a regular lat/lon grid)
shares = {}
for k, (arr, _, lats) in data.items():
    wgt = np.cos(np.deg2rad(lats))[:, None] * np.ones_like(arr, dtype=float)
    tot = wgt[arr > 0].sum()
    shares[k] = {c: wgt[arr == c].sum() / tot * 100 for c in np.unique(arr[arr > 0])}

# Classes shown in legends: those covering ≥0.05 % in at least one map (drops 1–2 stray pixels)
present = sorted({c for sh in shares.values() for c, v in sh.items() if v >= 0.05})
codes = list(range(31))
cmap = ListedColormap([(1, 1, 1, 0)] + [tuple(v / 255 for v in KG[c][2]) + (1,) for c in range(1, 31)])
norm = BoundaryNorm(np.arange(-0.5, 31.5, 1), cmap.N)

ctx.set_cache_dir(os.path.join(OUT, ".tile_cache"))


def draw(ax, key, title, cities=True, small=False, major_only=False, layer=None):
    arr, ext, _ = data[key]
    lcmap, lnorm = cmap, norm
    if layer is not None:  # npr. pouzdanost: (niz, obuhvat, cmap, norm)
        arr, ext, lcmap, lnorm = layer
    ax.set_xlim(EXTENT_VIEW[0], EXTENT_VIEW[1])
    ax.set_ylim(EXTENT_VIEW[2], EXTENT_VIEW[3])
    ax.set_aspect(1 / np.cos(np.deg2rad((ymin + ymax) / 2)))
    ctx.add_basemap(ax, crs="EPSG:4326", source=BASEMAP, attribution=False,
                    zoom=9)  # Esri World Shaded Relief nema stvarne podloge iznad zoom 9 za ovo područje
    ax.imshow(np.ma.masked_equal(arr, 0) if layer is None else arr, extent=ext, cmap=lcmap, norm=lnorm,
              interpolation="nearest", alpha=0.78, zorder=2)
    border.boundary.plot(ax=ax, color="#222", linewidth=0.9 if small else 1.3, zorder=3)
    if cities:
        halo = [pe.withStroke(linewidth=2 if small else 2.4, foreground="white")]
        for name, (x, y, major, side) in CITIES.items():
            if (small or major_only) and not major and name not in PEAKS:
                continue
            capital = name == "Sarajevo"
            if capital:  # glavni grad: bijeli krug s crnom tačkom
                ax.plot(x, y, "o", ms=6 if small else 11, mfc="white", mec="black", mew=1.1, zorder=4)
                ax.plot(x, y, "o", ms=2.4 if small else 4.4, mfc="black", mec="black", zorder=4.1)
            elif name in PEAKS:  # planinski vrh: crni trougao
                ax.plot(x, y, "^", ms=4 if small else 8, mfc="#222", mec="white", mew=0.8, zorder=4)
            else:
                ax.plot(x, y, "o", ms=3 if small else 5.5, mfc="white", mec="#222", mew=0.8, zorder=4)
            # fiksni razmak u tačkama od oznake grada, vertikalno centrirano na osnovnu liniju
            gap = (5 if capital else 4) if small else (8.5 if capital else 6.5)
            ax.annotate(name, (x, y), xytext=(gap if side == "r" else -gap, 0),
                        textcoords="offset points", va="center_baseline",
                        ha="left" if side == "r" else "right",
                        fontsize=6.5 if small else 11, zorder=5, color="#111",
                        fontweight="bold" if capital else "normal", path_effects=halo)
    ax.set_title(title, fontsize=15 if small else 16, loc="left", pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")


def legend_handles(classes):
    return [Patch(facecolor=tuple(v / 255 for v in KG[c][2]), edgecolor="#444", linewidth=0.5,
                  label=class_label(c)) for c in classes]


def fname(key):
    p, s = key
    return f"kg_bih_{p}" + (f"_{s}" if s else "") + ".png"


def put_credit(fig, trans, x, y, fs):
    """Autor u dva reda, desno poravnato; donji red na istoj osnovnoj liniji kao red s • ."""
    l1, l2 = AUTHOR.split("\n")
    kw = dict(ha="right", va="baseline", fontsize=fs, color=AUTHOR_COLOR)
    fig.text(x, y, l2, transform=trans, **kw)
    fig.text(x, y, l1, transform=mtransforms.offset_copy(trans, fig=fig, y=1.3 * fs, units="points"), **kw)


# Traka s udjelima klimatskih tipova ispod karte
def share_bar(ax, key, fs=8.5, min_label=7, y=-0.085, h=0.055, segs=None):
    """segs: lista (udio %, boja, oznaka, boja teksta); zadano su klimatski tipovi."""
    if segs is None:
        segs = [(shares[key].get(c, 0), tuple(x / 255 for x in KG[c][2]), KG[c][0], "#111")
                for c in present]
    bar = ax.inset_axes([0, y, 1, h])
    left = 0
    for v, col, lab, tcol in segs:
        if v <= 0:
            continue
        bar.barh(0, v, left=left, color=col, height=1, edgecolor="white", linewidth=0.8)
        if v >= min_label:
            bar.text(left + v / 2, 0, f"{lab} {v:.0f} %", ha="center", va="center",
                     fontsize=fs, fontweight="bold", color=tcol)
        left += v
    bar.set_xlim(0, 100); bar.set_ylim(-0.5, 0.5)
    bar.axis("off")


# Pouzdanost klasifikacije (kg_confidence iz NetCDF datoteka, %)
CONF_BINS = [0, 50, 65, 80, 95, 100.01]
CONF_LABELS = ["< 50 %", "50–65 %", "65–80 %", "80–95 %", "≥ 95 %"]
CONF_COLORS = [plt.get_cmap("viridis")(v) for v in (0.0, 0.28, 0.52, 0.76, 0.98)]
CONF_TEXT = ["white", "white", "white", "#111", "#111"]
CONF_TITLE = "Pouzdanost klasifikacije (slaganje članova ansambla, %)"
conf_cmap = ListedColormap(CONF_COLORS)
conf_norm = BoundaryNorm(CONF_BINS, conf_cmap.N)
conf = {}


def conf_path(key):
    p, s = key
    return os.path.join(NC_DIR, p, *( [s] if s else []), f"koppen_geiger_{RES}.nc")


def load_conf(key):
    import xarray as xr
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    ds = xr.open_dataset(conf_path(key)).sel(lon=slice(xmin, xmax), lat=slice(ymax, ymin))
    c = ds["kg_confidence"].values.astype(float)
    lon, lat = ds["lon"].values, ds["lat"].values
    res = abs(lon[1] - lon[0])
    tr = from_origin(lon[0] - res / 2, lat[0] + res / 2, res, res)
    inside = ~geometry_mask(border.geometry, c.shape, tr)
    c = np.ma.masked_where(~inside | ~np.isfinite(c), c)
    ext = (lon[0] - res / 2, lon[-1] + res / 2, lat[-1] - res / 2, lat[0] + res / 2)
    wgt = np.cos(np.deg2rad(lat))[:, None] * np.ones_like(c.data)
    m = ~c.mask
    sh = [wgt[m & (c.data >= a) & (c.data < b)].sum() / wgt[m].sum() * 100
          for a, b in zip(CONF_BINS[:-1], CONF_BINS[1:])]
    return c, ext, sh


def conf_layer(key):
    c, ext, _ = conf[key]
    return c, ext, conf_cmap, conf_norm


def conf_segs(key):
    return [(v, col, lab.replace(" ", ""), t)
            for v, col, lab, t in zip(conf[key][2], CONF_COLORS, CONF_LABELS, CONF_TEXT)]


def conf_handles():
    return [Patch(facecolor=col, edgecolor="#444", linewidth=0.5, label=lab)
            for col, lab in zip(CONF_COLORS, CONF_LABELS)]


if NC_DIR:
    for key in data:
        conf[key] = load_conf(key)


# 1) Pojedinačne karte
for key in data:
    p, s = key
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    if s:
        code, desc = SSP[s].split(" · ")
        sub = f"• Projekcija za period {period_label(p)}: {code} ({desc})"
    else:
        sub = f"• Historijski period: {period_label(p)}"
    draw(ax, key, "")
    ax.text(0, 1.022, sub, transform=ax.transAxes, ha="left", va="baseline", fontsize=22)
    ax.set_title("Köppen-Geigerova klasifikacija klime", fontsize=22, loc="left", pad=40)
    # duga linija s • ne ostavlja mjesta desno, pa autor ide uz red s naslovom
    put_credit(fig, mtransforms.offset_copy(ax.transAxes, fig=fig, y=37, units="points"), 1, 1, 15)
    share_bar(ax, key, fs=12, min_label=5, y=-0.065, h=0.05)
    cls = [c for c in present if shares[key].get(c, 0) >= 0.05]
    lab = [f"{class_label(c)}  ({shares[key][c]:.1f} %)" for c in cls]
    hs = legend_handles(cls)
    for h, l in zip(hs, lab):
        h.set_label(l)
    ax.legend(handles=hs, loc="upper left", bbox_to_anchor=(0, -0.085), ncol=2, columnspacing=1.2,
              fontsize=11.5, frameon=False, alignment="left", borderaxespad=0,
              title=f"{LEGEND_TITLE} (%)", title_fontproperties=TITLE_FONT,
              handlelength=1.6, handleheight=1.0, labelspacing=0.55)
    fig.savefig(os.path.join(OUT, fname(key)), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("✓", fname(key))


# 2) Panelne karte
def panel(keys, titles, suptitle, out, ncols, classes=None, legend_ncol=2, bottom=None,
          conf=False, change=False):
    """Mreža karata s fiksnim rasporedom u inčima, bez praznog prostora između karata."""
    n = len(keys)
    nrows = int(np.ceil(n / ncols))
    classes = classes or present
    ratio = ((EXTENT_VIEW[3] - EXTENT_VIEW[2]) / np.cos(np.deg2rad((ymin + ymax) / 2))
             / (EXTENT_VIEW[1] - EXTENT_VIEW[0]))
    mw = 4.6; mh = mw * ratio                       # karta
    m, gx = 0.1, 0.18                                # vanjska margina, razmak između kolona
    head, ttl, bar, gy = 1.25, 0.38, 0.075 * mh, 0.12  # zaglavlje, naslov karte, traka, razmak redova
    handles = (conf_handles() if conf else change_handles() if change else legend_handles(classes))
    leg_rows = int(np.ceil(len(handles) / legend_ncol))
    leg = 0.55 + 0.27 * leg_rows
    W = 2 * m + ncols * mw + (ncols - 1) * gx
    H = m + head + nrows * (ttl + mh + bar + gy) + leg
    fig = plt.figure(figsize=(W, H))
    for i, (k, t) in enumerate(zip(keys, titles)):
        r, c = divmod(i, ncols)
        x0 = m + c * (mw + gx)
        y0 = H - (m + head + r * (ttl + mh + bar + gy) + ttl + mh)
        ax = fig.add_axes([x0 / W, y0 / H, mw / W, mh / H])
        if change and k[1] is not None:
            draw(ax, k, t, cities=True, small=True, layer=change_layer(k))
            share_bar(ax, k, fs=7.5, min_label=11, y=-0.075, h=0.06, segs=change_segs(k))
        elif conf:
            draw(ax, k, t, cities=True, small=True, layer=conf_layer(k))
            share_bar(ax, k, fs=7.5, min_label=17, y=-0.075, h=0.06, segs=conf_segs(k))
        else:
            draw(ax, k, t, cities=True, small=True)
            share_bar(ax, k, fs=7.5, min_label=11, y=-0.075, h=0.06)
    y_t = 1 - (m + 0.36) / H                     # osnovna linija naslova
    y_b = 1 - (m + 0.86) / H                     # osnovna linija reda s •
    fig.text(m / W, y_t, "Köppen-Geigerova klasifikacija klime", fontsize=27,
             ha="left", va="baseline")
    fig.text(m / W, y_b, suptitle, fontsize=27, ha="left", va="baseline")
    put_credit(fig, fig.transFigure, 1 - m / W, y_b, 15)
    fig.legend(handles=handles, loc="upper left", ncol=legend_ncol, fontsize=12,
               frameon=False, bbox_to_anchor=(m / W, (leg - 0.05) / H), alignment="left",
               title=CONF_TITLE if conf else CHANGE_TITLE if change else LEGEND_TITLE, title_fontproperties={"weight": "bold", "size": 14},
               handlelength=1.6, handleheight=1.0, labelspacing=0.55, columnspacing=2.5,
               borderaxespad=0, borderpad=0)
    fig.savefig(os.path.join(OUT, out), dpi=DPI, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("✓", out)


# Historijski periodi: mreža 2×2 u istom stilu kao panele s projekcijama
hist_cls = [c for c in present if any(shares[(p, None)].get(c, 0) >= 0.05 for p in HIST)]
panel([(p, None) for p in HIST], [period_label(p) for p in HIST],
      "• Historijski periodi: 1901–2020", "panel_historijski.png", 2,
      classes=hist_cls, legend_ncol=2)

for p in FUT:
    keys = [("1991_2020", None)] + [(p, s) for s in SSP]
    titles = ["Referentni period 1991–2020"] + [SSP[s] for s in SSP]
    panel(keys, titles,
          f"• Projekcije za period {period_label(p)} bazirano na 7 zajedničkih socioekonomskih scenarija (SSP)",
          f"panel_scenariji_{p}.png", 4, legend_ncol=4)

if NC_DIR:
    panel([(p, None) for p in HIST], [period_label(p) for p in HIST],
          "• Pouzdanost: 1901–2020", "panel_pouzdanost_historijski.png", 2,
          legend_ncol=5, conf=True)
    for p in FUT:
        keys = [("1991_2020", None)] + [(p, s) for s in SSP]
        titles = ["Referentni period 1991–2020"] + [SSP[s] for s in SSP]
        panel(keys, titles, f"• Pouzdanost projekcija za period {period_label(p)} (7 SSP scenarija)",
              f"panel_pouzdanost_{p}.png", 4, legend_ncol=5, conf=True)
    with open(os.path.join(OUT, "pouzdanost_klasifikacije.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["period", "scenarij", "srednja pouzdanost (%)"] + [f"udio {l}" for l in CONF_LABELS])
        for p, s in [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]:
            c, _, sh = conf[(p, s)]
            w.writerow([period_label(p), s.upper() if s else "historijski", f"{c.mean():.1f}"]
                       + [f"{v:.2f}" for v in sh])
    print("✓ pouzdanost_klasifikacije.csv")

# Promjena klimatskog tipa u odnosu na 1991–2020
REF = ("1991_2020", None)
NOCHANGE = 31
NOCHANGE_COLOR = "#d4d4d4"
CHANGE_TITLE = ("Promjena u odnosu na 1991–2020: siva = bez promjene, "
                "boja = novi klimatski tip")
change_cmap = ListedColormap(list(cmap.colors) + [NOCHANGE_COLOR])
change_norm = BoundaryNorm(np.arange(-0.5, 32.5, 1), change_cmap.N)


def change_arr(key):
    ref, fut = data[REF][0], data[key][0]
    out = np.where(fut == ref, NOCHANGE, fut)
    return np.ma.masked_where(fut == 0, out)


def change_layer(key):
    return change_arr(key), data[key][1], change_cmap, change_norm


def change_segs(key):
    a = change_arr(key)
    lats = data[key][2]
    wgt = np.cos(np.deg2rad(lats))[:, None] * np.ones(a.shape)
    m = ~a.mask
    tot = wgt[m].sum()
    segs = [(wgt[m & (a.data == NOCHANGE)].sum() / tot * 100, NOCHANGE_COLOR, "Bez promjene", "#111")]
    for c in present:
        v = wgt[m & (a.data == c)].sum() / tot * 100
        segs.append((v, tuple(x / 255 for x in KG[c][2]), f"u {KG[c][0]}", "#111"))
    return segs


def change_handles():
    hs = [Patch(facecolor=NOCHANGE_COLOR, edgecolor="#444", linewidth=0.5,
                label="Bez promjene klimatskog tipa")]
    used = sorted({c for p in FUT for s in SSP for c in np.unique(change_arr((p, s)).compressed())
                   if c != NOCHANGE})
    return hs + legend_handles(used)


for p in FUT:
    keys = [REF] + [(p, s) for s in SSP]
    titles = ["Referentni period 1991–2020"] + [SSP[s] for s in SSP]
    panel(keys, titles, f"• Gdje se klimatski tip mijenja do perioda {period_label(p)}",
          f"panel_promjena_{p}.png", 4, legend_ncol=4, change=True)
with open(os.path.join(OUT, "promjena_klimatskog_tipa.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij", "bez promjene (%)"] + [f"u {KG[c][0]} (%)" for c in present])
    for p in FUT:
        for s in SSP:
            w.writerow([period_label(p), s.upper()] + [f"{v:.2f}" for v, *_ in change_segs((p, s))])
print("✓ promjena_klimatskog_tipa.csv")

# 3) Udio površine po klimatskom tipu (stubičasti dijagram)
rows = [("1991_2020", None)] + [(p, s) for p in FUT for s in SSP]
rows = [(p, None) for p in HIST[:-1]] + rows
labels = [period_label(p) if s is None else f"{period_label(p)}\n{SSP[s].split(' · ')[0]}"
          for p, s in rows]
fig, ax = plt.subplots(figsize=(15, 6.5))
bottom = np.zeros(len(rows))
for c in present:
    vals = np.array([shares[k].get(c, 0) for k in rows])
    ax.bar(range(len(rows)), vals, bottom=bottom, color=tuple(v / 255 for v in KG[c][2]),
           edgecolor="white", linewidth=0.6, label=class_label(c), width=0.82)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v >= 6:
            ax.text(i, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="#111")
    bottom += vals
for xsep in (len(HIST) - 0.5, len(HIST) + len(SSP) - 0.5):
    ax.axvline(xsep, color="#333", lw=1, ls="--")
ax.text((len(HIST) - 1) / 2, 103, "Historijski", ha="center", fontsize=13)
ax.text(len(HIST) + (len(SSP) - 1) / 2, 103, "2041–2070", ha="center", fontsize=13)
ax.text(len(HIST) + len(SSP) + (len(SSP) - 1) / 2, 103, "2071–2099", ha="center", fontsize=13)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([period_label(p).replace("–", "–\n") if s is None else SSP[s].split(" · ")[0]
                    for p, s in rows], rotation=0,
                   ha="center", fontsize=9.5)
ax.set_ylabel("Udio površine BiH (%)")
ax.set_ylim(0, 108)
ax.set_xlim(-0.6, len(rows) - 0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Udio klimatskih tipova u ukupnoj površini Bosne i Hercegovine", fontsize=22,
             loc="left", pad=22)
put_credit(fig, mtransforms.offset_copy(ax.transAxes, fig=fig, y=22, units="points"), 1, 1, 15)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=4, fontsize=10, frameon=False,
          alignment="left", title=LEGEND_TITLE, title_fontproperties=TITLE_FONT, borderaxespad=0)
fig.savefig(os.path.join(OUT, "udio_klimatskih_tipova.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ udio_klimatskih_tipova.png")

with open(os.path.join(OUT, "udio_klimatskih_tipova.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["period", "scenarij"] + [KG[c][0] for c in present])
    for p, s in [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]:
        w.writerow([period_label(p), s.upper() if s else "historijski"]
                   + [f"{shares[(p, s)].get(c, 0):.2f}" for c in present])
print("✓ udio_klimatskih_tipova.csv")


# 4) Klimatski tip po gradovima (matrica grad × period/scenarij)
def city_class(key, x, y):
    arr, ext, _ = data[key]
    h, w = arr.shape
    r = int((ext[3] - y) / (ext[3] - ext[2]) * h)
    c = int((x - ext[0]) / (ext[1] - ext[0]) * w)
    v = arr[r, c]
    if v == 0:  # tačka tik uz granicu: uzmi najčešću vrijednost u susjedstvu 3×3
        win = arr[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
        vals, cnt = np.unique(win[win > 0], return_counts=True)
        v = vals[cnt.argmax()] if len(vals) else 0
    return int(v)


cols = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]
cities = (sorted([n for n in CITIES if n not in PEAKS], key=lambda n: -CITIES[n][1])  # gradovi, sjever→jug
          + sorted(PEAKS, key=lambda n: -CITIES[n][1]))                               # pa planine
M = np.array([[city_class(k, CITIES[n][0], CITIES[n][1]) for k in cols] for n in cities])
city_cls = [c for c in present if c in M]

nr, nc = M.shape
fig, ax = plt.subplots(figsize=(17, 11.5))
for i in range(nr):
    for j in range(nc):
        c = M[i, j]
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=tuple(v / 255 for v in KG[c][2]),
                                   edgecolor="white", linewidth=1.5))
        ref = M[i, len(HIST) - 1]
        ax.text(j + 0.5, i + 0.5, KG[c][0], ha="center", va="center", fontsize=10,
                fontweight="bold" if (j >= len(HIST) and c != ref) else "normal",
                color="white" if c in (27, 28, 24, 30) else "#111")
for xsep in (len(HIST), len(HIST) + len(SSP)):
    ax.axvline(xsep, color="#333", lw=1.4)
ax.set_xlim(0, nc); ax.set_ylim(nr, 0)
ax.set_yticks(np.arange(nr) + 0.5)
ax.set_yticklabels(cities, fontsize=12)
for t, n in zip(ax.get_yticklabels(), cities):
    if n in PEAKS:
        t.set_fontstyle("italic")
ax.axhline(nr - len(PEAKS), color="#333", lw=1.4)
ax.set_xticks(np.arange(nc) + 0.5)
ax.set_xticklabels([period_label(p).replace("–", "–\n") if s is None else SSP[s].split(" · ")[0]
                    for p, s in cols], fontsize=10)
ax.xaxis.tick_top()
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
for x0, txt in ((0, "Historijski periodi"), (len(HIST), "Projekcije 2041–2070"),
                (len(HIST) + len(SSP), "Projekcije 2071–2099")):
    ax.text(x0 + 0.05, -1.35, txt, fontsize=17, ha="left", va="bottom")
LX = -76  # lijevi rub imena gradova (tačke), da naslov, napomena i legenda budu poravnati s njima
title_tr = mtransforms.offset_copy(ax.transAxes, fig=fig, x=LX, y=128, units="points")
sub_tr = mtransforms.offset_copy(ax.transAxes, fig=fig, x=LX, y=98, units="points")
fig.text(0, 1, "Köppen-Geigerova klasifikacija klime", transform=title_tr, fontsize=27,
         ha="left", va="baseline")
fig.text(0, 1, "• Klimatski tip za bosanskohercegovačke gradove i planine", transform=sub_tr, fontsize=27,
         ha="left", va="baseline")
put_credit(fig, mtransforms.offset_copy(ax.transAxes, fig=fig, y=98, units="points"), 1, 1, 15)
ax.legend(handles=legend_handles(city_cls), loc="upper left", bbox_to_anchor=(0, -0.03),
          bbox_transform=mtransforms.offset_copy(ax.transAxes, fig=fig, x=LX, units="points"),
          ncol=3, fontsize=11, frameon=False, alignment="left", borderaxespad=0,
          title=LEGEND_TITLE, title_fontproperties=TITLE_FONT, columnspacing=2.0)
fig.savefig(os.path.join(OUT, "klima_po_gradovima.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ klima_po_gradovima.png")

with open(os.path.join(OUT, "klima_po_gradovima.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["grad", "lon", "lat"] + [period_label(p) + ("" if s is None else " " + s.upper())
                                         for p, s in cols])
    for n, row in zip(cities, M):
        w.writerow([n, CITIES[n][0], CITIES[n][1]] + [KG[c][0] for c in row])
print("✓ klima_po_gradovima.csv")


# 5) Linijski grafikon: udio klimatskih tipova kroz vrijeme, po scenarijima
SSP_COLORS = {  # boje scenarija prema IPCC AR6
    "ssp119": "#00ADCF", "ssp126": "#173C66", "ssp434": "#4E9FD3", "ssp245": "#F79420",
    "ssp460": "#C0812B", "ssp370": "#E71D25", "ssp585": "#951B1E",
}
SSP_DASH = {"ssp434": (0, (5, 2)), "ssp460": (0, (5, 2))}
MID = {p: (int(p[:4]) + int(p[5:])) / 2 for p in HIST + FUT}
line_cls = [c for c in present if max(sh.get(c, 0) for sh in shares.values()) >= 2]

fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
axes = axes.ravel()
for ax, c in zip(axes, line_cls):
    hx = [MID[p] for p in HIST]
    hy = [shares[(p, None)].get(c, 0) for p in HIST]
    for s in SSP:
        ax.plot([hx[-1]] + [MID[p] for p in FUT], [hy[-1]] + [shares[(p, s)].get(c, 0) for p in FUT],
                color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                mec="white", mew=1, zorder=3)
    ax.plot(hx, hy, color="#222", lw=2.6, marker="o", ms=6, mec="white", mew=1, zorder=4)
    ax.axvline(2020, color="#999", lw=1, ls=(0, (2, 2)), zorder=1)
    ax.set_title(f"$\\mathbf{{{KG[c][0]}}}$  {DESC.get(c, KG[c][1]).split(' – ')[0]}",
                 fontsize=13.5, loc="left", pad=10)
    ax.plot([0.009], [1.058], transform=ax.transAxes, clip_on=False, marker="o", ms=13,
            mfc=tuple(v / 255 for v in KG[c][2]), mec="#444", mew=0.6)
    ax.set_title(ax.get_title(loc="left"), fontsize=13.5, loc="left", pad=10, x=0.04)
    top = max(max(hy), max(shares[(p, s)].get(c, 0) for p in FUT for s in SSP))
    ax.set_ylim(0, top * 1.12 + 1)
    ax.set_xlim(1905, 2095)
    ax.set_xticks([MID[p] for p in HIST + FUT])
    ax.set_xticklabels([period_label(p).replace("–", "–\n") for p in HIST + FUT], fontsize=9.5)
    ax.set_ylabel("Udio površine BiH (%)", fontsize=10.5)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=9.5)
for ax in axes[len(line_cls):]:
    ax.axis("off")
lax = axes[len(line_cls)] if len(line_cls) < len(axes) else None
handles = [plt.Line2D([], [], color="#222", lw=2.6, marker="o", ms=6, mec="white",
                      label="Historijski period (opažanja)")]
handles += [plt.Line2D([], [], color=SSP_COLORS[s], lw=2.2, ls=SSP_DASH.get(s, "-"), marker="o", ms=5,
                       mec="white", label=SSP[s].replace(" · ", " – ")) for s in SSP]
(lax or axes[-1]).legend(handles=handles, loc="center left", frameon=False, fontsize=12.5,
                         title="Period / scenarij", title_fontproperties=TITLE_FONT,
                         alignment="left", handlelength=3, labelspacing=0.8)
fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.07, hspace=0.42, wspace=0.2)
x0 = axes[0].get_position().x0
fig.text(x0, 0.955, "Köppen-Geigerova klasifikacija klime", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.905, "• Promjena udjela klimatskih tipova u BiH (1901–2099)", fontsize=27,
         ha="left", va="baseline")
put_credit(fig, fig.transFigure, 0.99, 0.905, 15)
fig.savefig(os.path.join(OUT, "udio_klimatskih_tipova_linije.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ udio_klimatskih_tipova_linije.png")
