"""Sea surface temperature for each best-track fix from NOAA OISST v2.1.

Source (public): NOAA CoastWatch ERDDAP, dataset ncdcOisst21Agg_LonPM180
(0.25 deg daily). For each fix we take the mean SST in a box around the storm
centre on the *previous* day -- same-day SST under a cyclone is already cooled
by the storm's own mixing. Fixes over land get the widened-box value (or NaN,
imputed later from the training median; `over_land` is a separate feature).
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclone_mm import config as C  # noqa: E402

ERDDAP = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg_LonPM180.csv"
          "?sst[({t}T12:00:00Z)][(0.0)][({la0}):2:({la1})][({lo0}):2:({lo1})]")
OUT = C.PROCESSED / "sst.csv"


def box_mean(day: str, lat: float, lon: float, half: float) -> float:
    snap = lambda x: np.floor(x * 4) / 4 + 0.125  # OISST cell centres
    url = ERDDAP.format(t=day, la0=snap(lat - half), la1=snap(lat + half),
                        lo0=snap(lon - half), lo1=snap(lon + half))
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:  # ERDDAP returns 404 when every cell is NaN
                return np.nan
            r.raise_for_status()
            sst = pd.read_csv(io.StringIO(r.text), skiprows=[1])["sst"]
            return float(sst.mean()) if sst.notna().any() else np.nan
        except (requests.RequestException, ValueError):
            time.sleep(3 * 2 ** attempt)
    return np.nan


def sst_for(row) -> float:
    day = (row.datetime - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    v = box_mean(day, row.lat, row.lon, 1.5)
    return v if not np.isnan(v) else box_mean(day, row.lat, row.lon, 4.0)


def main():
    df = pd.read_csv(C.TRACK_CSV, parse_dates=["datetime"])
    df = df[df["datetime"].dt.hour.isin(C.SYNOPTIC_HOURS)].reset_index(drop=True)
    have = pd.read_csv(OUT, parse_dates=["datetime"]) if OUT.exists() else pd.DataFrame(columns=["storm_id", "datetime", "sst"])
    key = set(zip(have.storm_id, have.datetime))
    todo = df[[(s, t) not in key for s, t in zip(df.storm_id, df.datetime)]]
    print(f"{len(todo)} fixes need SST", flush=True)
    with ThreadPoolExecutor(6) as ex:
        vals = list(ex.map(sst_for, todo.itertuples()))
    new = todo[["storm_id", "datetime"]].assign(sst=vals)
    out = pd.concat([have, new], ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"saved {len(out)} rows -> {OUT}; missing {out.sst.isna().sum()}; "
          f"range {out.sst.min():.1f}-{out.sst.max():.1f} C")


if __name__ == "__main__":
    main()
