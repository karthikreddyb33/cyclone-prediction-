# Cyclone Intensity Prediction — trained on 2013-2024, tested on 2025

Multimodal cyclone intensity estimation for the North Indian Ocean.

Estimates maximum sustained wind (knots) for Bay of Bengal / Arabian Sea
systems from a **satellite IR image + storm attributes**, and compares it with a
**tabular-only** baseline on the unseen **2025** season.

Training uses the 2013-2024 seasons; every model is then scored on the unseen
2025 season. The full dataset is rebuilt from public government sources with
one consistent pipeline.

## Data sources (all public, no login)

| What | Source | Script |
|---|---|---|
| Labels + track (wind, pressure, lat/lon, 3-hourly) | IMD RSMC New Delhi best track 1982-2026 — `rsmcnewdelhi.imd.gov.in` | `scripts/build_best_track.py` |
| IR imagery (11 µm brightness temperature, 0.07°, 3-hourly) | NOAA NCEI GridSat-B1 CDR, AWS mirror `noaa-cdr-gridsat-b1-pds` | `scripts/fetch_gridsat.py` |
| Sea surface temperature (0.25° daily) | NOAA OISST v2.1 via CoastWatch ERDDAP | `scripts/fetch_sst.py` |

Why GridSat and not INSAT-3D: ISRO MOSDAC lists every INSAT-3D/3DR file per
cyclone, but its THREDDS data services were returning server errors
(`datasetTracker_1 is not active`) and bulk orders need a MOSDAC account;
IMD's satellite archive needs a login. GridSat-B1 covers the region (Meteosat
IODC) for every year 2013-2025 and lines up with the 3-hourly best-track times.

## Layout

```
cyclone-prediction-/
├── cyclone_mm/
│   ├── config.py      paths, year splits, feature sets, crop size
│   ├── data.py        sample table, splits, TabularPreprocessor (impute+StandardScaler), CycloneDataset, transforms
│   ├── models.py      TabularModel (baseline), MultimodalModel (CNN backbone + MLP + fusion head)
│   └── engine.py      train / predict loops, early stopping, RMSE / MAE / R² / bias
├── scripts/
│   ├── build_best_track.py   IMD xlsx -> data/processed/best_track.csv
│   ├── fetch_gridsat.py      GridSat-B1 -> data/images/<storm>_<YYYYmmddHH>.npy (256×256 crops, K)
│   └── fetch_sst.py          OISST -> data/processed/sst.csv
├── train.py           train one model, saves outputs/<run>/{model.pt,metrics.json,history.csv,predictions_*.csv}
├── evaluate.py        scores every run on 2025 -> outputs/comparison_2025/
└── data/raw/best_tracks_1982_2026.xlsx
```

## Run it

```bash
git clone https://github.com/karthikreddyb33/cyclone-prediction-.git
cd cyclone-prediction-
pip install -r requirements.txt
python scripts/build_best_track.py
python scripts/fetch_sst.py
python scripts/fetch_gridsat.py --workers 10 --hours 0 12   # ~1.5 h; resumable
python scripts/fetch_gridsat.py --workers 10 --hours 6 18   # optional backfill, ~1.5 h

python train.py --model tabular    --features physical
python train.py --model multimodal --features physical    --backbone efficientnet_b0
python train.py --model tabular    --features no_pressure
python train.py --model multimodal --features no_pressure --backbone efficientnet_b0
python evaluate.py
```

GridSat files are one compressed chunk each (~30-45 MB, full globe), so every
timestamp is a full download; S3 throttles per connection, hence many workers.

## Design

* **Samples**: every best-track fix at 00/06/12/18 UTC that has an image crop.
  Both models are trained and scored on exactly the same rows.
* **Split**: development = 2013-2024, test = 2025.
  `--split storm` (default): 85/15 random split *grouped by storm*, so no
  storm has fixes in both train and validation (neighbouring 6-hourly fixes are
  near-duplicates; a row-level split would leak).
  `--split temporal`: train ≤ 2023, validate on 2024.
* **Tabular features**: lat, lon, central pressure, pressure drop, SST
  (box mean, previous day), translation speed + direction (sin/cos), month
  (sin/cos), hours since genesis, basin (BoB/AS), over-land flag. Median
  imputation and `StandardScaler` fitted on train only; target z-scored.
  The IMD Dvorak CI number is excluded: it is a direct intensity code.
* **Image**: 256×256 px (~18°) IR crop centred on the storm, inverted so cold
  cloud tops are bright, replicated to 3 channels, ImageNet-normalised.
  Train-time augmentation: random rotation ±180°, horizontal/vertical flips,
  resize to 224×224. `--no-flips` keeps rotation only (a single flip reverses
  the storm's spin direction).
* **Multimodal model**: torchvision EfficientNet-B0 (or `--backbone resnet50`,
  `efficientnet_b2`, `resnet18`), ImageNet weights, classifier removed →
  Linear-BN-ReLU-Dropout projection (256) ‖ tabular MLP (128→64, BN, Dropout)
  → fusion MLP (128) → wind. AdamW, backbone LR 1e-4, head LR 1e-3, OneCycle,
  SmoothL1 loss, early stopping on validation RMSE.
* **Baseline**: the same tabular MLP with a linear head.

### Read the results carefully

IMD derives best-track central pressure from wind through a wind-pressure
relationship, so the `physical` feature set (which the brief asked for) lets
the tabular model nearly read the answer off the pressure column. The
`no_pressure` set removes pressure and pressure drop; that comparison is the
honest measure of what the imagery contributes. In operational use the
pressure would not be known independently of the wind estimate either.

The 2025 season was weak (peak 60 kt, Shakhti), so the test set has no
severe cyclones; errors on 2025 say little about performance on very severe
storms.

## Results

Trained on CPU (Intel i-series, 10 threads, no GPU), seed 42, `--split storm`.
Proof of training: `outputs/training_report/report.html` (environment, timings,
curves, full per-epoch logs) and the raw logs in `outputs/training_logs/`.
Reproduce with `bash scripts/run_training.sh && python scripts/make_report.py`.

**Final run: all 2,006 image crops** (00/06/12/18 UTC), 2026-10-26 23:11 → 2026-10-27 01:23 IST.
2025 test season, 193 fixes from 15 systems, actual wind 20-60 kt:

| Model | Features | RMSE (kt) | MAE (kt) | R² | Bias (kt) |
|---|---|---:|---:|---:|---:|
| Tabular MLP | physical | 3.36 | 2.64 | 0.855 | +1.54 |
| Multimodal EfficientNet-B0 | physical | 4.11 | 3.37 | 0.783 | +1.09 |
| Multimodal EfficientNet-B0 | no_pressure | 8.49 | 6.59 | 0.072 | +4.34 |
| Tabular MLP | no_pressure | 18.74 | 12.15 | −3.518 | +10.50 |

**Earlier run: 1,563 crops** (all 00/12 UTC, part of 06/18), 2026-10-26 18:29 → 20:17 IST,
kept in `outputs/archive_run1_1563_crops/`:

| Model | Features | RMSE (kt) | MAE (kt) | R² |
|---|---|---:|---:|---:|
| Tabular MLP | physical | 3.36 | 2.61 | 0.855 |
| Multimodal EfficientNet-B0 | physical | 3.88 | 3.25 | 0.806 |
| Multimodal EfficientNet-B0 | no_pressure | 7.21 | 5.30 | 0.332 |
| Tabular MLP | no_pressure | 21.78 | 13.83 | −5.101 |

What the numbers say:

* **The image carries the intensity signal.** Without pressure, adding the IR
  image cuts the 2025 error from 18.7 to 8.5 kt RMSE (12.2 → 6.6 kt MAE). The
  tabular-only model has no way to tell a depression from a cyclone and
  over-predicts the weak 2025 storms by ~10 kt on average.
* **With pressure, the tabular model wins**, as expected: pressure is derived
  from wind in the IMD best track, so it nearly gives away the answer, and the
  image adds noise rather than information.
* **Run-to-run variation is large.** The no_pressure multimodal model scored 7.2 kt
  with fewer images and 8.5 kt with all of them, while its validation RMSE was
  similar (9.4 vs 9.9 kt). With one seed and a weak 193-fix test season, a
  ±1-2 kt difference is within noise; averaging several seeds would be needed
  before claiming one data set is better.

## Author

**karthikreddyb33** – [github.com/karthikreddyb33](https://github.com/karthikreddyb33)
