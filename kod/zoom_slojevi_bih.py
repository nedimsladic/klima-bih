"""
Slojevi za zumabilnu kartu (Leaflet): Köppen-Geigerovi tipovi na 1 km za BiH i regiju, za sve periode i scenarije,
kao prozirne PNG slike (1 piksel = 1 ćelija od 1 km) i JSON s granicama.

Upotreba:
    python zoom_slojevi_bih.py <kg_tif_dir> <granice_dir s BIH/SVN/HRV/SRB/MNE/XKX/ALB/MKD.geojson> <out_dir>
"""
import sys
import os
import json

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform
from PIL import Image

KG_DIR, GDIR, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
HIST = ["1901_1930", "1931_1960", "1961_1990", "1991_2020"]
SSP = ["ssp119", "ssp126", "ssp434", "ssp245", "ssp460", "ssp370", "ssp585"]
KEYS = [(p, None) for p in HIST] + [(p, s) for p in ("2041_2070", "2071_2099") for s in SSP]
RGB = {1: (0, 0, 255), 2: (0, 120, 255), 3: (70, 170, 250), 4: (255, 0, 0), 5: (255, 150, 150), 6: (245, 165, 0),
       7: (255, 220, 100), 8: (255, 255, 0), 9: (200, 200, 0), 10: (150, 150, 0), 11: (150, 255, 150),
       12: (100, 200, 100), 13: (50, 150, 50), 14: (200, 255, 80), 15: (100, 255, 80), 16: (50, 200, 0),
       17: (255, 0, 255), 18: (200, 0, 200), 19: (150, 50, 150), 20: (150, 100, 150), 21: (170, 175, 255),
       22: (90, 120, 220), 23: (75, 80, 180), 24: (50, 0, 135), 25: (0, 255, 255), 26: (55, 200, 255),
       27: (0, 125, 125), 28: (0, 70, 95), 29: (178, 178, 178), 30: (102, 102, 102)}
LUT = np.zeros((31, 4), np.uint8)
for c, rgb in RGB.items():
    LUT[c] = (*rgb, 255)
countries = ["BIH", "SVN", "HRV", "SRB", "MNE", "XKX", "ALB", "MKD"]
geoms = [gpd.read_file(os.path.join(GDIR, f"{c}.geojson")).to_crs(4326).geometry.union_all() for c in countries]
meta = {}
for p, s in KEYS:
    with rasterio.open(os.path.join(KG_DIR, p, *([s] if s else []), "koppen_geiger_0p00833333.tif")) as r:
        a, tr = mask(r, geoms, crop=True, nodata=0)
    a = a[0]
    h, w = a.shape
    # u Web Mercator (EPSG:3857), da se slika tačno poklapa s podlogom na web karti
    left, top = tr.c, tr.f
    right, bottom = left + tr.a * w, top + tr.e * h
    dst_tr, dw, dh = calculate_default_transform("EPSG:4326", "EPSG:3857", w, h, left, bottom, right, top)
    b = np.zeros((dh, dw), np.uint8)
    reproject(a, b, src_transform=tr, src_crs="EPSG:4326", dst_transform=dst_tr, dst_crs="EPSG:3857",
              resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    name = f"kg_{p}_{s or 'hist'}.png"
    Image.fromarray(LUT[b], "RGBA").save(os.path.join(OUT, name), optimize=True)
    x0, y1 = dst_tr.c, dst_tr.f
    x1, y0 = x0 + dst_tr.a * dw, y1 + dst_tr.e * dh
    (lo0, lo1), (la0, la1) = transform("EPSG:3857", "EPSG:4326", [x0, x1], [y0, y1])
    meta = {"bounds": [[la0, lo0], [la1, lo1]]}
json.dump({**meta, "keys": [f"{p}|{s or 'hist'}" for p, s in KEYS]}, open(os.path.join(OUT, "zoom.json"), "w"))
tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
print("✓", len(KEYS), "slojeva,", round(tot / 1e6, 1), "MB; granice", meta["bounds"])
