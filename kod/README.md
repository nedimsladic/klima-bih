# Klima Bosne i Hercegovine 1901–2099: kod analize

Python skripte koje prave karte, grafikone, podatke i PDF-ove za atlas
[nedimsladic.github.io/klima-bih](https://nedimsladic.github.io/klima-bih/).
Analiza: Nedim Sladić, prateći Beck et al. (2018, 2023).

## Postavljanje

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Ulazni podaci (nisu u repozitoriju)

| Folder | Izvor |
|---|---|
| `kg/`, `kgnc/` | Beck et al. (2023), Köppen-Geiger 1 km 1901–2099, sa [GloH2O](https://www.gloh2o.org/koppen/) |
| `clim/`, `clim01/` | Beck et al. (2023), mjesečna temperatura i padavine (0,5° i 0,1°) |
| `eobs/` | E-OBS v31.0e, dnevni tn, tx, rr (0,1°), Copernicus / ECA&D, samo nekomercijalno |
| `dem/` | Copernicus DEM GLO-90 |
| `hybas/` | HydroBASINS v1c, nivo 6 i 7 |
| `pop/` | WorldPop 2020, 1 km |
| `region/`, `bih/`, `adm2/` | geoBoundaries (ADM0, ADM3) |
| `geonames/` | GeoNames BA.txt |
| `ne/` | Natural Earth populated places |

## Redoslijed

Svaka skripta u zaglavlju ima red `Upotreba:` s argumentima. Okvirno:

1. `koppen_bih.py`, `temperatura_bih.py`, `sezone_bih.py`: klimatski tipovi, temperatura, zima i ljeto
2. `dani_bih.py`, `ekstremi_bih.py`, `padavine_dani_bih.py`, `toplotni_talasi_bih.py`, `agro_bih.py`, `hladnoca_kisa_bih.py`: dnevni pokazatelji (E-OBS i delta metoda); pišu `podaci/*.npz`
3. `kulture_bih.py`, `vodni_bilans_bih.py`, `visina_bih.py`, `slivovi_bih.py`, `izlozenost_bih.py`, `susjedi_bih.py`, `klimatski_blizanci_bih.py`
4. `gradovi_bih.py`, `naselja_bih.py`, `zoom_slojevi_bih.py`, `pdf_gradovi_bih.py`, `animacija_bih.py`: podaci i datoteke za web stranicu
5. `build_site.py`: pravi stranicu za GitHub Pages

`bih_stil.py` sadrži zajednički stil (Helvetica, naslovi, gradovi i planine, paneli).
