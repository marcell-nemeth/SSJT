"""LargeST 2019: daily and hourly sums and counts of the valid 5-minute readings of every sensor.

Input (download from the LargeST repository, https://github.com/liuxu77/LargeST, Kaggle archive):
    ca_his_raw_2019.h5   5-minute flow readings of 8,600 California sensors in 2019 (pandas HDF, key 't')
    ca_meta.csv          sensor metadata (district, county, freeway, direction)
Usage:  python scripts/01_prepare_largest.py /path/to/ca_his_raw_2019.h5 /path/to/ca_meta.csv
Output: data/largest_2019_daily.npz   (s: 365 x 8600 sums, c: counts, ids)
        data/largest_2019_hourly.npz  (s: 8760 x 8600 sums, c: counts, ids)
        data/ca_meta.csv
The selection of sensors and the interpolation are done in ssjt.data.load_largest(_hourly).
"""
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def main(h5_path, meta_path):
    DATA.mkdir(exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        V = f["t/block0_values"]                      # (105120, 8600), 288 readings per day
        ids = f["t/block0_items"][:]
        n_days = V.shape[0] // 288
        N = V.shape[1]
        sd = np.zeros((n_days, N)); cd = np.zeros((n_days, N), np.int16)
        sh = np.zeros((24 * n_days, N), np.float32); ch = np.zeros((24 * n_days, N), np.int16)
        for d in range(n_days):
            X = V[d * 288:(d + 1) * 288]
            ok = np.isfinite(X)
            sd[d] = np.nansum(X, 0); cd[d] = ok.sum(0)
            Xh = X.reshape(24, 12, N)
            sh[d * 24:(d + 1) * 24] = np.nansum(Xh, 1); ch[d * 24:(d + 1) * 24] = np.isfinite(Xh).sum(1)
            if d % 50 == 0:
                print(f"day {d}/{n_days}", flush=True)
    np.savez_compressed(DATA / "largest_2019_daily.npz", s=sd, c=cd, ids=ids)
    np.savez_compressed(DATA / "largest_2019_hourly.npz", s=sh, c=ch, ids=ids)
    shutil.copy(meta_path, DATA / "ca_meta.csv")
    print("written", DATA / "largest_2019_daily.npz", DATA / "largest_2019_hourly.npz")


if __name__ == "__main__":
    main(*sys.argv[1:3])
