"""
Izloženost stanovništva BiH vrućim danima i tropskim noćima (1961–2099).

Stanovništvo: WorldPop 2020, 1 km (bih_ppp_2020_1km_Aggregated.tif), drži se nepromijenjenim
za sve periode, da razlike pokazuju samo promjenu klime.
Dani: podaci/dani_svi.npz iz ekstremi_bih.py (E-OBS 0,1° + delta metoda).

Upotreba:
    python izlozenost_bih.py <dani_svi.npz> <worldpop.tif> <out_dir>
"""
import sys
import os
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from bih_stil import Bih, FUT, REF, SSP, DPI, period_label, ssp_code, AUTHOR, AUTHOR_COLOR

NPZ, POP, OUT = sys.argv[1:4]
Z = np.load(NPZ)
LON, LAT = Z["lon"], Z["lat"]
with rasterio.open(POP) as r:
    pop = r.read(1).astype(float)
    tr = r.transform
pop[(pop < 0) | ~np.isfinite(pop)] = 0
rows, cols = np.indices(pop.shape)
plon = tr.c + (cols + 0.5) * tr.a
plat = tr.f + (rows + 0.5) * tr.e
ii = np.clip(np.searchsorted(-LAT, -plat), 0, len(LAT) - 1)
jj = np.clip(np.searchsorted(LON, plon), 0, len(LON) - 1)
# najbliža ćelija (searchsorted daje desnog susjeda; provjeri lijevog)
ii = np.where((ii > 0) & (np.abs(LAT[np.maximum(ii - 1, 0)] - plat) < np.abs(LAT[ii] - plat)), ii - 1, ii)
jj = np.where((jj > 0) & (np.abs(LON[np.maximum(jj - 1, 0)] - plon) < np.abs(LON[jj] - plon)), jj - 1, jj)
TOTAL = pop.sum()
print(f"Stanovništvo (WorldPop 2020): {TOTAL:,.0f}")


def field(k, p, s):
    a = Z[f"{k}|{p}|{s or 'hist'}"]
    v = a[ii, jj]
    return np.where(np.isfinite(v), v, 0)


COLS = [("1961_1990", None), (REF, None)] + [(p, s) for p in FUT for s in ("ssp126", "ssp245", "ssp585")]
LABELS = ["1961–\n1990", "1991–\n2020"] + [f"{period_label(p)}\n{ssp_code(s)}" for p in FUT
                                          for s in ("ssp126", "ssp245", "ssp585")]
IND = {
    "vruci": ("Vrući dani (najviša temperatura ≥ 30 °C)", [0, 10, 30, 60, 1e9],
              ["manje od 10 dana", "10–29 dana", "30–59 dana", "60 i više dana"],
              ["#fde9c8", "#f9b26b", "#e4572e", "#8e1b1b"]),
    "tropske_noci": ("Tropske noći (najniža temperatura ≥ 20 °C)", [0, 1, 10, 30, 1e9],
                     ["manje od 1 noći", "1–9 noći", "10–29 noći", "30 i više noći"],
                     ["#f6e4ef", "#e7a3c8", "#c03d8b", "#5e0f4c"]),
}

fig, axes = plt.subplots(1, 2, figsize=(18, 8.6))
rows_csv = []
for ax, (k, (title, bins, labs, colors)) in zip(axes, IND.items()):
    shares = []
    for p, s in COLS:
        v = field(k, p, s)
        sh = [pop[(v >= a) & (v < b)].sum() / TOTAL * 100 for a, b in zip(bins[:-1], bins[1:])]
        shares.append(sh)
        mean = (v * pop).sum() / TOTAL
        rows_csv.append([title.split(" (")[0], period_label(p), s.upper() if s else "historijski",
                         f"{mean:.1f}"] + [f"{x:.1f}" for x in sh])
    shares = np.array(shares)
    x = np.arange(len(COLS))
    bottom = np.zeros(len(COLS))
    for j, (lab, col) in enumerate(zip(labs, colors)):
        ax.bar(x, shares[:, j], bottom=bottom, color=col, edgecolor="white", lw=1, width=0.78, label=lab, zorder=3)
        for xi, (v, b) in enumerate(zip(shares[:, j], bottom)):
            if v >= 7:
                ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=10,
                        color="white" if j >= 2 else "#222")
        bottom += shares[:, j]
    ax.axvline(1.5, color="#333", lw=1, ls="--")
    ax.axvline(4.5, color="#999", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Udio stanovništva BiH (%)", fontsize=11.5)
    ax.set_title(title, fontsize=15, loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888")
    ax.tick_params(colors="#333", labelsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=4, frameon=False, fontsize=11,
              handlelength=1.2, columnspacing=1.4)
fig.subplots_adjust(left=0.05, right=0.99, top=0.78, bottom=0.2, wspace=0.16)
x0 = axes[0].get_position().x0 - 0.035
fig.text(x0, 0.95, "Izloženost stanovništva vrućinama u BiH", fontsize=27, ha="left", va="baseline")
fig.text(x0, 0.89, "• Koliki dio stanovništva živi s koliko vrućih dana i tropskih noći", fontsize=27,
         ha="left", va="baseline")
Bih.put_credit(fig, fig.transFigure, 0.99, 0.89, 15)
fig.savefig(os.path.join(OUT, "izlozenost_stanovnistva.png"), dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("✓ izlozenost_stanovnistva.png")

with open(os.path.join(OUT, "izlozenost_stanovnistva.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["pokazatelj", "period", "scenarij", "prosjek po stanovniku (dana/god.)",
                "udio razred 1 (%)", "udio razred 2 (%)", "udio razred 3 (%)", "udio razred 4 (%)"])
    w.writerows(rows_csv)
print("✓ izlozenost_stanovnistva.csv")
for r in rows_csv:
    print(r)
