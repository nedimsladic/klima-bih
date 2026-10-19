"""
Gruba klimatska pogodnost za voće, povrće i ratarske kulture u BiH (samo temperatura, bez tla, vode i sorte).

Pokazatelji (prosjek 30 godina, ćelija 0,1°):
  GDD    stepen-dani rasta, april–oktobar, baza 10 °C                (agro_svi.npz)
  Huglin Huglinov indeks                                              (agro_svi.npz)
  bezmraz broj dana između posljednjeg proljetnog i prvog jesenjeg mraza (agro_svi.npz)
  ZM     prosječan dan u godini posljednjeg proljetnog mraza          (agro_svi.npz)
  chill  sati zimskog hlađenja 0–7,2 °C, novembar–februar             (hladnoca_kisa.npz)
  vrući dani (Tmax ≥ 30 °C), mrazni (Tmin < 0 °C), ledeni (Tmax < 0 °C) (dani_svi.npz)

Upotreba:
    python kulture_bih.py <agro_svi.npz> <dani_svi.npz> <hladnoca_kisa.npz> <bih_boundary.geojson> <out_dir>
"""
import sys
import os
import csv

import numpy as np
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from bih_stil import Bih, HIST, FUT, REF, SSP, DPI, period_label, ssp_code

A, D, H, BOUNDARY, OUT = sys.argv[1:6]
B = Bih(BOUNDARY, OUT)
A, D, H = np.load(A), np.load(D), np.load(H)
LON, LAT = A["lon"], A["lat"]
RES = abs(LON[1] - LON[0])
EXT = (LON[0] - RES / 2, LON[-1] + RES / 2, LAT[-1] - RES / 2, LAT[0] + RES / 2)
INSIDE = ~geometry_mask(B.border.geometry, (len(LAT), len(LON)),
                        from_origin(LON[0] - RES / 2, LAT[0] + RES / 2, RES, RES))
W = np.cos(np.deg2rad(LAT))[:, None] * INSIDE
KEYS = [(p, None) for p in HIST] + [(p, s) for p in FUT for s in SSP]


def bezmraz(k):
    zm, pm = A[f"zadnji_mraz|{k}"], A[f"prvi_mraz|{k}"]
    return np.where(np.isnan(zm) | np.isnan(pm), 365, pm - zm)


g = lambda k: A[f"gdd|{k}"]
hug = lambda k: A[f"huglin|{k}"]
chill = lambda k: H[f"chill|{k}"]
vruci = lambda k: D[f"vruci|{k}"]
mrazni = lambda k: D[f"mrazni|{k}"]
ledeni = lambda k: D[f"ledeni|{k}"]

# naziv, boja, pravilo, opis praga
GROUPS = {
    "voce": ("Klimatska pogodnost za voće", "Voće i vinova loza", {
        "jabuka": ("Jabuka", "#c0392b", lambda k: (chill(k) >= 1000) & (g(k) >= 900) & (vruci(k) <= 35),
                   "≥ 1000 sati hlađenja, GDD ≥ 900, ≤ 35 vrućih dana"),
        "sljiva": ("Šljiva", "#5b3a8c", lambda k: (chill(k) >= 800) & (g(k) >= 1000),
                   "≥ 800 sati hlađenja, GDD ≥ 1000"),
        "tresnja": ("Trešnja", "#a3123a", lambda k: (chill(k) >= 1100) & (g(k) >= 1000) & (vruci(k) <= 40),
                    "≥ 1100 sati hlađenja, GDD ≥ 1000, ≤ 40 vrućih dana"),
        "breskva": ("Breskva i marelica", "#ee8a3c", lambda k: (chill(k) >= 600) & (g(k) >= 1500) & (A[f"zadnji_mraz|{k}"] <= 100),
                    "≥ 600 sati hlađenja, GDD ≥ 1500, posljednji mraz do 10. aprila"),
        "vinova_loza": ("Vinova loza", "#8e3b74", lambda k: hug(k) >= 1500, "Huglinov indeks ≥ 1500"),
        "smokva": ("Smokva", "#7a5230", lambda k: (g(k) >= 1900) & (mrazni(k) <= 40) & (ledeni(k) <= 3),
                   "GDD ≥ 1900, ≤ 40 mraznih i ≤ 3 ledena dana"),
        "maslina": ("Maslina", "#6b8e23", lambda k: (hug(k) >= 2100) & (mrazni(k) <= 20),
                    "Huglinov indeks ≥ 2100, ≤ 20 mraznih dana"),
    }),
    "povrce": ("Klimatska pogodnost za povrće i ratarske kulture", "Povrće i ratarske kulture", {
        "kukuruz": ("Kukuruz za zrno", "#e0a526", lambda k: (g(k) >= 1250) & (bezmraz(k) >= 150),
                    "GDD ≥ 1250, ≥ 150 dana bez mraza"),
        "paradajz": ("Paradajz i paprika", "#d1452f", lambda k: (g(k) >= 1400) & (bezmraz(k) >= 160),
                     "na otvorenom, GDD ≥ 1400, ≥ 160 dana bez mraza"),
        "lubenica": ("Lubenica i dinja", "#2e8b57", lambda k: (g(k) >= 1700) & (bezmraz(k) >= 170),
                     "GDD ≥ 1700, ≥ 170 dana bez mraza"),
        "krompir": ("Krompir", "#a07a3c", lambda k: (g(k) >= 700) & (vruci(k) <= 20),
                    "GDD ≥ 700, ≤ 20 vrućih dana (vrućina smanjuje prinos)"),
        "kupus": ("Kupus i lisnato povrće", "#4f8f3a", lambda k: (bezmraz(k) >= 120) & (vruci(k) <= 30),
                  "≥ 120 dana bez mraza, ≤ 30 vrućih dana"),
    }),
}


def kid(p, s):
    return f"{p}|{s or 'hist'}"


OK = INSIDE & np.isfinite(A["gdd|1991_2020|hist"])


def share(m):
    return float((m & OK).astype(float)[OK].dot(W[OK]) / W[OK].sum() * 100)


ratio = (B.view[3] - B.view[2]) * B.aspect / (B.view[1] - B.view[0])
mw = 4.4; mh = mw * ratio
rows = []
for gk, (title, short, CROPS) in GROUPS.items():
    for PF in FUT:
        COLS = [((REF, None), "1991–2020")] + [((PF, s), f"{period_label(PF)} · {ssp_code(s)}")
                                                         for s in ("ssp126", "ssp245", "ssp585")]
        nrow = len(CROPS)
        fig = plt.figure(figsize=(4 * mw + 0.6, nrow * (mh + 0.55) + 2.9))
        FW, FH = fig.get_size_inches()
        for r, (c, (name, col, f, rule)) in enumerate(CROPS.items()):
            cm = ListedColormap([(1, 1, 1, 0), col]); nm = BoundaryNorm([-0.5, 0.5, 1.5], 2)
            for j, (key, ttl) in enumerate(COLS):
                m = f(kid(*key))
                arr = np.where(np.isnan(A[f"gdd|{kid(*key)}"]), np.nan, m.astype(float))
                x0 = 0.1 + j * (mw + 0.13); y0 = FH - (1.35 + (r + 1) * (mh + 0.55) - 0.1)
                ax = fig.add_axes([x0 / FW, y0 / FH, mw / FW, mh / FH])
                B.draw(ax, arr, EXT, cm, nm, f"{name} · {ttl}" if j == 0 else ttl, small=True, cities=False)
                ax.text(0.02, 0.03, f"{share(m):.0f} % površine", transform=ax.transAxes, fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.3", fc="white", alpha=0.9, ec="none"))
        fig.text(0.1 / FW, 1 - 0.45 / FH, title, fontsize=26, va="baseline")
        fig.text(0.1 / FW, 1 - 0.95 / FH, f"• Gdje temperatura odgovara kulturi, projekcije za period {period_label(PF)}",
                 fontsize=26, va="baseline")
        B.put_credit(fig, fig.transFigure, 1 - 0.1 / FW, 1 - 0.95 / FH, 15)
        fig.legend(handles=[Patch(color=col, label=f"{name}: {rule}") for name, col, _, rule in CROPS.values()],
                   loc="lower left", ncol=2, frameon=False, fontsize=11.5, bbox_to_anchor=(0.1 / FW, 0.003),
                   title="Kultura i prag (samo temperatura, bez tla i vode)",
                   title_fontproperties={"weight": "bold", "size": 12.5}, alignment="left")
        fn = f"kulture_{gk}_{PF}.png"
        fig.savefig(os.path.join(OUT, fn), dpi=DPI, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print("✓", fn)
    for c, (name, col, f, rule) in CROPS.items():
        for key in KEYS:
            rows.append([short, name, rule, period_label(key[0]), key[1].upper() if key[1] else "historijski",
                         f"{share(f(kid(*key))):.1f}"])
with open(os.path.join(OUT, "kulture_bih.csv"), "w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["grupa", "kultura", "prag", "period", "scenarij", "udio površine BiH s pogodnom temperaturom (%)"])
    w.writerows(rows)
for r in rows:
    if r[3] in ("1991–2020", "2041–2070", "2071–2099") and r[4] in ("historijski", "SSP126", "SSP245", "SSP585"):
        print(r[1], r[3], r[4], r[5])
