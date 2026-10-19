"""
Zajednički stil za karte BiH (font, gradovi, podloga, zaglavlje, panel karata s kontinuiranim poljem).
Koriste ga sezone_bih.py i dani_bih.py; izgled prati koppen_bih.py.
"""
import os
import tempfile

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import font_manager
import contextily as ctx

DPI = 600
AUTHOR = "Analiza: Nedim Sladić\n(prateći Beck et al. (2018, 2023))"
AUTHOR_COLOR = "#8a8a8a"

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
REF = "1991_2020"
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

# naziv: (lon, lat, na panelima?, strana oznake) – isto kao u koppen_bih.py
CITIES = {
    "Sarajevo": (18.413, 43.856, True, "r"), "Banja Luka": (17.191, 44.772, True, "r"),
    "Tuzla": (18.674, 44.538, True, "r"), "Zenica": (17.907, 44.204, True, "r"),
    "Mostar": (17.808, 43.343, True, "r"), "Bihać": (15.870, 44.817, True, "r"),
    "Bijeljina": (19.214, 44.757, True, "l"), "Trebinje": (18.344, 42.712, True, "r"),
    "Brčko": (18.810, 44.873, False, "r"), "Prijedor": (16.714, 44.980, False, "r"),
    "Doboj": (18.088, 44.735, False, "r"), "Cazin": (15.943, 44.967, False, "r"),
    "Travnik": (17.665, 44.226, False, "l"), "Bugojno": (17.450, 44.057, False, "r"),
    "Livno": (17.007, 43.827, False, "r"), "Konjic": (17.961, 43.652, False, "r"),
    "Goražde": (18.946, 43.668, False, "r"), "Foča": (18.776, 43.506, False, "r"),
    "Zvornik": (19.102, 44.386, False, "r"), "Gradiška": (17.255, 45.145, False, "r"),
    "Bjelašnica": (18.266, 43.716, False, "l"), "Jahorina": (18.567, 43.737, False, "r"),
    "Vlašić": (17.648, 44.292, False, "r"), "Kupres": (17.282, 43.982, False, "r"),
    "Blidinje": (17.480, 43.620, False, "r"),
}
PEAKS = {"Bjelašnica", "Jahorina", "Vlašić", "Kupres", "Blidinje"}
BASEMAP = ctx.providers.Esri.WorldShadedRelief


def period_label(p):
    a, b = p.split("_")
    return f"{a}–{b}"


def ssp_code(s):
    return SSP[s].split(" · ")[0]


def fill_nan(a, steps=6):
    """Popuni prazne ćelije (more uz obalu) prosjekom susjeda, da rubovi uz granicu nemaju rupe."""
    a = np.array(a, dtype=float)
    for _ in range(steps):
        miss = np.isnan(a)
        if not miss.any():
            break
        p = np.pad(a, 1, constant_values=np.nan)
        nb = np.stack([p[1 + dy:p.shape[0] - 1 + dy, 1 + dx:p.shape[1] - 1 + dx]
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
        cnt = (~np.isnan(nb)).sum(axis=0)
        mean = np.where(cnt > 0, np.nansum(nb, axis=0) / np.maximum(cnt, 1), np.nan)
        a = np.where(miss, mean, a)
    return a


class Bih:
    def __init__(self, boundary, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        ctx.set_cache_dir(os.path.join(out, ".tile_cache"))
        self.border = gpd.read_file(boundary).to_crs(4326)
        self.xmin, self.ymin, self.xmax, self.ymax = self.border.total_bounds
        pad = 0.12
        self.view = (self.xmin - pad, self.xmax + pad, self.ymin - pad, self.ymax + pad)
        self.aspect = 1 / np.cos(np.deg2rad((self.ymin + self.ymax) / 2))
        verts, codes = [], []
        geom = self.border.geometry.union_all() if hasattr(self.border.geometry, "union_all") \
            else self.border.geometry.unary_union
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            for ring in [poly.exterior] + list(poly.interiors):
                xy = np.asarray(ring.coords)
                verts += xy.tolist()
                codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
        self.clip = Path(verts, codes)

    # --- karta s kontinuiranim poljem, odsječenim po granici BiH
    def draw(self, ax, arr, ext, cmap, norm, title, small=True, cities=True):
        ax.set_xlim(self.view[0], self.view[1])
        ax.set_ylim(self.view[2], self.view[3])
        ax.set_aspect(self.aspect)
        ctx.add_basemap(ax, crs="EPSG:4326", source=BASEMAP, attribution=False, zoom=9)
        im = ax.imshow(fill_nan(arr), extent=ext, cmap=cmap, norm=norm, interpolation="bilinear",
                       alpha=0.88, zorder=2)
        im.set_clip_path(PathPatch(self.clip, transform=ax.transData, facecolor="none"))
        self.border.boundary.plot(ax=ax, color="#222", linewidth=0.9 if small else 1.3, zorder=3)
        if cities:
            self.cities(ax, small)
        ax.set_title(title, fontsize=15 if small else 16, loc="left", pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#888")
        return im

    def cities(self, ax, small=True):
        halo = [pe.withStroke(linewidth=2 if small else 2.4, foreground="white")]
        for name, (x, y, major, side) in CITIES.items():
            if small and not major and name not in PEAKS:
                continue
            capital = name == "Sarajevo"
            if capital:
                ax.plot(x, y, "o", ms=6 if small else 11, mfc="white", mec="black", mew=1.1, zorder=4)
                ax.plot(x, y, "o", ms=2.4 if small else 4.4, mfc="black", mec="black", zorder=4.1)
            elif name in PEAKS:
                ax.plot(x, y, "^", ms=4 if small else 8, mfc="#222", mec="white", mew=0.8, zorder=4)
            else:
                ax.plot(x, y, "o", ms=3 if small else 5.5, mfc="white", mec="#222", mew=0.8, zorder=4)
            gap = (5 if capital else 4) if small else (8.5 if capital else 6.5)
            ax.annotate(name, (x, y), xytext=(gap if side == "r" else -gap, 0),
                        textcoords="offset points", va="center_baseline",
                        ha="left" if side == "r" else "right",
                        fontsize=6.5 if small else 11, zorder=5, color="#111",
                        fontweight="bold" if capital else "normal", path_effects=halo)

    @staticmethod
    def put_credit(fig, trans, x, y, fs):
        l1, l2 = AUTHOR.split("\n")
        kw = dict(ha="right", va="baseline", fontsize=fs, color=AUTHOR_COLOR)
        fig.text(x, y, l2, transform=trans, **kw)
        fig.text(x, y, l1, transform=mtransforms.offset_copy(trans, fig=fig, y=1.3 * fs, units="points"), **kw)

    # --- mreža karata: naslov, red s •, karte, ispod svake prosjek za BiH, zajednička skala boja
    def panel(self, items, title, sub, out, cmap, norm, cbar_label, ncols=4, fmt="{:.0f}",
              unit="", ticks=None, extend="neither", ticklabels=None):
        """items: lista (naslov karte, niz, obuhvat, prosjek BiH ili None)."""
        n = len(items)
        nrows = int(np.ceil(n / ncols))
        ratio = (self.view[3] - self.view[2]) * self.aspect / (self.view[1] - self.view[0])
        mw = 4.6; mh = mw * ratio
        m, gx = 0.1, 0.18
        head, ttl, lab, gy = 1.25, 0.38, 0.32, 0.12
        cb = 1.05
        W = 2 * m + ncols * mw + (ncols - 1) * gx
        H = m + head + nrows * (ttl + mh + lab + gy) + cb
        fig = plt.figure(figsize=(W, H))
        im = None
        for i, (t, arr, ext, mean) in enumerate(items):
            r, c = divmod(i, ncols)
            x0 = m + c * (mw + gx)
            y0 = H - (m + head + r * (ttl + mh + lab + gy) + ttl + mh)
            ax = fig.add_axes([x0 / W, y0 / H, mw / W, mh / H])
            im = self.draw(ax, arr, ext, cmap, norm, t)
            if mean is not None:
                ax.text(0, -0.025, f"Prosjek BiH: {fmt.format(mean)}{unit}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=11, color="#222")
        y_t = 1 - (m + 0.36) / H
        y_b = 1 - (m + 0.86) / H
        fig.text(m / W, y_t, title, fontsize=27, ha="left", va="baseline")
        fig.text(m / W, y_b, sub, fontsize=27, ha="left", va="baseline")
        self.put_credit(fig, fig.transFigure, 1 - m / W, y_b, 15)
        cax = fig.add_axes([m / W, (0.42) / H, min(0.5, 7.5 / W), 0.16 / H])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks, extend=extend)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=11, length=3, colors="#333")
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
        fig.text(m / W, (0.42 + 0.16 + 0.12) / H, cbar_label, fontsize=14, fontweight="bold",
                 ha="left", va="bottom")
        fig.savefig(os.path.join(self.out, out), dpi=DPI, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print("✓", out)
