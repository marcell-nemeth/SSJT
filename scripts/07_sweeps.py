"""Runtime and memory usage as a function of N and of T (Figures 2 and 3).

N: 26 balanced trees (b = 2..7 children per parent, N = 511..262,143), standard normal residuals with T = 200,
   one forecast time step.
T: hourly BDG2 and LargeST, residuals and base forecasts (24 forecast time steps) of the first origin,
   T in {100, 200, 500, 1000, 2000, 4000, 6000}.
SSJT and every form of MinT-Fix / MinT-Shrink that fits into memory (dense up to N = 10,000, K x K up to K = 5,000).
Usage: python scripts/07_sweeps.py [N] [T]   -> results/sweep_N.csv, results/sweep_T.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from ssjt import experiment as X
from ssjt.hierarchy import Hierarchy

TREES = [(2, 8), (2, 10), (2, 12), (2, 14), (2, 15), (2, 16), (2, 17), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
         (4, 5), (4, 6), (4, 7), (4, 8), (5, 4), (5, 5), (5, 6), (5, 7), (6, 4), (6, 5), (6, 6), (7, 4), (7, 5), (7, 6)]


def split(m):
    est, form = (m.split("|") + [""])[:2]
    return est, form


def sweep_N():
    rows = []
    for b, d in sorted(TREES, key=lambda x: x[0] ** (x[1] + 1)):
        h = Hierarchy.balanced(b, d)
        rng = X.stable_seed("sweepN", b, d)
        E, Y = rng.standard_normal((200, h.N)), rng.uniform(50, 150, (h.N, 1))
        res = X.measure(E, Y, X.timing_fns(h, kxk_max_K=5000), reps=5 if h.N <= 20000 else 3)
        rows += [{"b": b, "d": d, "N": h.N, "K": h.K, "T": 200, "method": split(m)[0], "form": split(m)[1], **v}
                 for m, v in res.items()]
        print(f"N={h.N}", {m: round(v["median_s"] * 1e3, 1) for m, v in res.items()}, flush=True)
        pd.DataFrame(rows).to_csv(X.RESULTS / "sweep_N.csv", index=False)


def sweep_T():
    rows = []
    for tag in ("bdg2_hourly", "largest_hourly"):
        ds = X.load_dataset(tag)
        h = ds["h"]
        for T in (100, 200, 500, 1000, 2000, 4000, 6000):
            Yhat, E, _ = X.load_origin(ds, 0, window=T)
            res = X.measure(np.ascontiguousarray(E), Yhat, X.timing_fns(h, kxk_max_K=5000), reps=5)
            rows += [{"data": tag, "N": h.N, "K": h.K, "T": T, "method": split(m)[0], "form": split(m)[1], **v}
                     for m, v in res.items()]
            print(tag, T, {m: round(v["median_s"] * 1e3, 1) for m, v in res.items()}, flush=True)
            pd.DataFrame(rows).to_csv(X.RESULTS / "sweep_T.csv", index=False)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["N", "T"]:
        {"N": sweep_N, "T": sweep_T}[s]()
