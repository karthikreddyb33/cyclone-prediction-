"""Parse the IMD RSMC New Delhi best-track workbook into one tidy CSV.

Source (public): https://rsmcnewdelhi.imd.gov.in/report.php?internal_menu=MzM=
One sheet per year; the layout drifts between years (serial/date/name only on
the first row of a storm in some years, every row in others; free-text remark
rows mixed in). This script normalises all of that and derives:
translation speed/direction, hours since genesis, cyclical month, basin flags.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclone_mm import config as C  # noqa: E402

COLS = ["serial", "basin", "name", "date", "time", "lat", "lon", "ci",
        "central_pressure", "wind_kt", "pressure_drop", "grade"]


def _storm_name(names: pd.Series) -> str:
    """First real name in a storm's rows; depressions are unnamed."""
    cands = names.dropna().astype(str).str.strip().str.upper()
    cands = cands[~cands.isin(["", "--", "NAN", "NAME"])]
    return cands.iloc[0] if len(cands) else "UNNAMED"


def _to_date(v):
    if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
        return pd.Timestamp(v).normalize()
    return pd.to_datetime(str(v).strip(), errors="coerce", dayfirst=True)


def _swap(d: pd.Timestamp) -> pd.Timestamp:
    return d.replace(month=d.day, day=d.month) if d.day <= 12 else d


def _fix_day_month_swaps(dates: pd.Series) -> pd.Series:
    """Excel stored some DD/MM dates as MM/DD -- sometimes a whole storm,
    sometimes single rows. A storm lasts days, not months, so: (1) choose the
    reading (as-is vs. swapped) with the shorter total span for the storm,
    then (2) fix stray rows by taking whichever reading is closest to the
    previous row."""
    asis = dates.map(_to_date)
    swapped = asis.map(lambda d: d if pd.isna(d) else _swap(d))
    span = lambda s: (s.max() - s.min()) if s.notna().any() else pd.Timedelta(0)
    base = swapped if span(swapped) < span(asis) else asis

    out, prev = [], None
    for d in base:
        if pd.notna(d) and prev is not None:
            d = min([d, _swap(d)], key=lambda c: abs((c - prev).days))
        out.append(d)
        prev = d if pd.notna(d) else prev
    return pd.Series(out, index=dates.index)


def parse_year(xls: pd.ExcelFile, year: int) -> pd.DataFrame:
    raw = pd.read_excel(xls, str(year), header=None)
    hdr = raw.index[raw[0].astype(str).str.startswith("Serial")][0]
    df = raw.iloc[hdr + 1:, :12].copy()
    df.columns = COLS

    # Over-land rows carry "Land" in the basin column instead of AS/BOB.
    basin = df["basin"].astype(str).str.strip().str.upper()
    df["over_land"] = basin.eq("LAND").astype(int)
    df["basin"] = basin.where(basin.isin(["AS", "BOB"]))

    df["serial"] = df["serial"].ffill()
    df["date"] = df["date"].ffill()
    df["basin"] = df.groupby("serial")["basin"].transform(lambda s: s.ffill().bfill())

    num = ["time", "lat", "lon", "central_pressure", "wind_kt", "pressure_drop"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "lat", "lon", "wind_kt"])  # drops remark rows
    df = df[(df["time"] % 100 == 0) & (df["time"] <= 2300)]

    df["date"] = df.groupby("serial")["date"].transform(_fix_day_month_swaps)
    df = df.dropna(subset=["date"])
    df["datetime"] = df["date"] + pd.to_timedelta(df["time"] // 100, unit="h")
    # Physically impossible values are typos in the sheet.
    df.loc[~df["central_pressure"].between(850, 1020), "central_pressure"] = np.nan
    df["year"] = year
    df["storm_id"] = [f"{year}_{int(s):02d}" for s in df["serial"]]
    df["name"] = df.groupby("storm_id")["name"].transform(_storm_name)
    return df[["storm_id", "year", "name", "basin", "datetime", "lat", "lon",
               "central_pressure", "pressure_drop", "wind_kt", "over_land"]]


def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["storm_id", "datetime"]).drop_duplicates(["storm_id", "datetime"])
    g = df.groupby("storm_id")
    dt_h = g["datetime"].diff().dt.total_seconds() / 3600
    dlat = g["lat"].diff()
    dlon = g["lon"].diff() * np.cos(np.deg2rad(df["lat"]))
    dist_nm = 60.0 * np.hypot(dlat, dlon)
    df["trans_speed_kt"] = (dist_nm / dt_h).replace([np.inf, -np.inf], np.nan).clip(upper=40)
    heading = np.arctan2(dlon, dlat)  # 0 = north, clockwise positive
    df["trans_dir_sin"], df["trans_dir_cos"] = np.sin(heading), np.cos(heading)
    # First fix of each storm has no previous position: use the next step's motion.
    for c in ["trans_speed_kt", "trans_dir_sin", "trans_dir_cos"]:
        df[c] = g[c].transform(lambda s: s.bfill()).fillna(0.0)
    df["hours_since_genesis"] = (df["datetime"] - g["datetime"].transform("min")).dt.total_seconds() / 3600
    m = df["datetime"].dt.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    df["basin_bob"] = (df["basin"] == "BOB").astype(int)
    return df


def main():
    xls = pd.ExcelFile(C.BEST_TRACK_XLSX)
    years = [y for y in range(C.FIRST_YEAR, C.TEST_YEAR + 1) if str(y) in xls.sheet_names]
    df = pd.concat([parse_year(xls, y) for y in years], ignore_index=True)
    df = add_motion_features(df)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TRACK_CSV, index=False)
    syn = df[df["datetime"].dt.hour.isin(C.SYNOPTIC_HOURS)]
    print(f"{len(df)} fixes, {df.storm_id.nunique()} storms -> {C.TRACK_CSV}")
    print(syn.groupby("year").agg(fixes=("wind_kt", "size"), storms=("storm_id", "nunique"),
                                  max_kt=("wind_kt", "max")).to_string())


if __name__ == "__main__":
    main()
