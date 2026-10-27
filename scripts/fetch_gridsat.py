"""Download GridSat-B1 IR brightness temperature and save storm-centred crops.

Source (public, NOAA NCEI CDR mirrored on AWS Open Data):
  https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/YYYY/GRIDSAT-B1.YYYY.MM.DD.HH.v02r01.nc
Each file is the whole globe in a single compressed HDF5 chunk (~30-40 MB), so
range reads don't help: we download, crop every storm active at that time,
and delete the file. Resumable -- finished crops are skipped on re-run.

Crop: CROP_PX x CROP_PX pixels of `irwin_cdr` (11 um window channel, K) on the
0.07 deg grid, north-up, centred on the IMD best-track position. Saved as
float16 .npy in data/images/<storm_id>_<YYYYmmddHH>.npy.
"""
import argparse
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cyclone_mm import config as C  # noqa: E402

URL = "https://noaa-cdr-gridsat-b1-pds.s3.amazonaws.com/data/{y}/GRIDSAT-B1.{y}.{m:02d}.{d:02d}.{h:02d}.v02r01.nc"
FILL = -31999


def crop_path(storm_id: str, ts: pd.Timestamp) -> Path:
    return C.IMAGES / f"{storm_id}_{ts:%Y%m%d%H}.npy"


def download(url: str, dest: Path, retries: int = 6) -> bool:
    """Stream to disk, resuming with HTTP Range after connection resets."""
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    return False
                if r.status_code == 416:  # already complete
                    return True
                r.raise_for_status()
                total = int(r.headers.get("Content-Range", "/0").split("/")[-1] or 0) or \
                    int(r.headers.get("Content-Length", 0))
                with open(dest, "ab" if r.status_code == 206 else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if not total or dest.stat().st_size >= total:
                return True
        except (requests.RequestException, OSError):
            pass
        time.sleep(min(60, 5 * 2 ** attempt))
    return False


def crop(h: h5py.File, lat: float, lon: float, half: int) -> np.ndarray:
    v = h["irwin_cdr"]
    lats, lons = h["lat"][:], h["lon"][:]
    i, j = int(np.abs(lats - lat).argmin()), int(np.abs(lons - lon).argmin())
    a = v[0, max(i - half, 0):i + half, max(j - half, 0):j + half].astype(np.float32)
    a[a == FILL] = np.nan
    a = a * np.ravel(v.attrs["scale_factor"])[0] + np.ravel(v.attrs["add_offset"])[0]
    out = np.full((2 * half, 2 * half), np.nan, np.float32)
    out[:a.shape[0], :a.shape[1]] = a
    return np.flipud(out).astype(np.float16)  # lat ascending -> north-up


def process(ts: pd.Timestamp, rows: pd.DataFrame, tmpdir: Path) -> list[dict]:
    url = URL.format(y=ts.year, m=ts.month, d=ts.day, h=ts.hour)
    nc = tmpdir / f"{ts:%Y%m%d%H}.nc"
    results = []
    try:
        if not download(url, nc):
            return [dict(storm_id=r.storm_id, datetime=ts, status="download_failed") for r in rows.itertuples()]
        with h5py.File(nc, "r") as h:
            for r in rows.itertuples():
                img = crop(h, r.lat, r.lon, C.CROP_PX // 2)
                np.save(crop_path(r.storm_id, ts), img)
                results.append(dict(storm_id=r.storm_id, datetime=ts, status="ok",
                                    nan_frac=float(np.isnan(img.astype(np.float32)).mean())))
    except OSError as e:  # corrupt/partial file: drop it so a re-run refetches
        results = [dict(storm_id=r.storm_id, datetime=ts, status=f"error: {e}") for r in rows.itertuples()]
    finally:
        nc.unlink(missing_ok=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--years", type=int, nargs="*", help="limit to these years")
    ap.add_argument("--hours", type=int, nargs="*", default=list(C.SYNOPTIC_HOURS),
                    help="UTC hours to fetch (e.g. 0 12 first, then 6 18)")
    ap.add_argument("--limit", type=int, help="max timestamps (for a quick test)")
    args = ap.parse_args()

    C.IMAGES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.TRACK_CSV, parse_dates=["datetime"])
    df = df[df["datetime"].dt.hour.isin(args.hours)]
    if args.years:
        df = df[df["year"].isin(args.years)]
    df = df[[not crop_path(s, t).exists() for s, t in zip(df.storm_id, df.datetime)]]
    # Newest first: the 2025 test set and 2024 validation set land before older training years.
    groups = sorted(df.groupby("datetime"), key=lambda g: g[0], reverse=True)
    if args.limit:
        groups = groups[:args.limit]
    print(f"{len(groups)} timestamps / {len(df)} crops to fetch", flush=True)

    log = C.PROCESSED / "gridsat_log.csv"
    done = 0
    with tempfile.TemporaryDirectory(dir=C.DATA) as tmp, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process, ts, rows, Path(tmp)) for ts, rows in groups]
        for f in as_completed(futs):
            res = f.result()
            pd.DataFrame(res).to_csv(log, mode="a", header=not log.exists(), index=False)
            done += 1
            bad = [r for r in res if r["status"] != "ok"]
            print(f"[{done}/{len(groups)}] {res[0]['datetime']:%Y-%m-%d %H}Z "
                  f"{len(res) - len(bad)} ok{' ' + bad[0]['status'] if bad else ''}", flush=True)


if __name__ == "__main__":
    main()
