"""
Pravi samostalnu verziju stranice za GitHub Pages: site/index.html iz web/klima_bih.html
(dodaje <!doctype>, <head> s meta oznakama i <body>), te kopira slike i podatke.

Upotreba:
    python build_site.py
"""
import os
import shutil
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "web", "klima_bih.html")
SITE = os.path.join(ROOT, "site")

HEAD = """<!doctype html>
<html lang="bs">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Atlas klime Bosne i Hercegovine 1901–2099: Köppen-Geigerovi klimatski tipovi, temperatura, zima i ljeto, mrazni, vrući i kišni dani po sedam SSP scenarija.">
<meta property="og:title" content="Klima Bosne i Hercegovine 1901–2099">
<meta property="og:description" content="Köppen-Geigerovi klimatski tipovi, temperatura, mrazni, vrući i kišni dani po sedam SSP scenarija. Analiza: Nedim Sladić.">
<meta property="og:type" content="website">
<meta property="og:url" content="https://nedimsladic.github.io/klima-bih/">
<meta property="og:image" content="https://nedimsladic.github.io/klima-bih/img/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://nedimsladic.github.io/klima-bih/img/og.png">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' height='16' rx='3' fill='%23155e63'/%3E%3Crect x='3' y='9' width='10' height='3' fill='%23c8ff50'/%3E%3Crect x='3' y='4' width='10' height='3' fill='%2337c8ff'/%3E%3C/svg%3E">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}</style>
"""

page = open(SRC, encoding="utf-8").read()
page = page.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
i = page.index("</style>", page.index("/* Raspored: atlas")) + len("</style>")
html = HEAD + page[:i] + "\n</head>\n<body>\n" + page[i:] + "\n</body>\n</html>\n"
os.makedirs(SITE, exist_ok=True)
open(os.path.join(SITE, "index.html"), "w", encoding="utf-8").write(html)
for sub in ("img", "data", "pdf"):
    src, dst = os.path.join(ROOT, "web", sub), os.path.join(SITE, sub)
    shutil.copytree(src, dst, dirs_exist_ok=True)
shutil.copy(os.path.join(ROOT, "web", "widget.html"), os.path.join(SITE, "widget.html"))
# kod analize: skripte, zavisnosti i granica BiH u site/kod/
KOD = os.path.join(SITE, "kod")
os.makedirs(KOD, exist_ok=True)
for f in glob.glob(os.path.join(ROOT, "*.py")) + [os.path.join(ROOT, n) for n in ("requirements.txt", "bih_granica.geojson")]:
    shutil.copy(f, KOD)
shutil.copy(os.path.join(ROOT, "KOD_README.md"), os.path.join(KOD, "README.md"))
print("✓ site/index.html,", len(os.listdir(os.path.join(SITE, "img"))), "slika,",
      len(os.listdir(os.path.join(SITE, "data"))), "datoteka podataka")
