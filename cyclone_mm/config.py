"""Central configuration: paths, year splits and feature lists."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
IMAGES = DATA / "images"
OUTPUTS = ROOT / "outputs"

BEST_TRACK_XLSX = RAW / "best_tracks_1982_2026.xlsx"
TRACK_CSV = PROCESSED / "best_track.csv"          # parsed IMD best track
SAMPLES_CSV = PROCESSED / "samples.csv"           # track rows + image path + SST

FIRST_YEAR = 2013          # first year of the original (INSAT-era) dataset
TRAIN_LAST_YEAR = 2023     # train on <= 2023
VAL_YEAR = 2024            # validate on 2024 (temporal hold-out)
TEST_YEAR = 2025           # test on 2025
SYNOPTIC_HOURS = (0, 6, 12, 18)

TARGET = "wind_kt"

# GridSat-B1: 0.07 deg grid. 256 px ~= 17.9 deg box centred on the storm.
CROP_PX = 256
IMG_SIZE = 224
# Brightness temperature range (K) used to normalise IR images to [0, 1].
BT_MIN, BT_MAX = 170.0, 310.0

# Tabular feature sets.
#   "physical": what the spec asks for. Central pressure (ECP) and pressure
#               drop are in IMD best track derived from wind via a wind-pressure
#               relationship, so they leak the target almost perfectly.
#   "no_pressure": the same set without pressure terms -- an honest test of
#               what imagery adds.
FEATURE_SETS = {
    "physical": [
        "lat", "lon", "central_pressure", "pressure_drop", "sst",
        "trans_speed_kt", "trans_dir_sin", "trans_dir_cos",
        "month_sin", "month_cos", "hours_since_genesis", "basin_bob", "over_land",
    ],
    "no_pressure": [
        "lat", "lon", "sst",
        "trans_speed_kt", "trans_dir_sin", "trans_dir_cos",
        "month_sin", "month_cos", "hours_since_genesis", "basin_bob", "over_land",
    ],
}
