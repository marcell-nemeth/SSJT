"""Runtime and peak memory usage on the real datasets (Tables 4 and A.8).

For every dataset, the base forecasts (all forecast time steps) and residuals of the first forecast origin are
reconciled by SSJT and by every form of MinT-Fix and MinT-Shrink that fits into memory. Median of 7 calls
(5 for N > 20,000); peak memory in a separate call.
Usage: python scripts/06_timing_real.py [tag ...]   -> results/timing_real.csv
Runtimes depend on the machine; the shipped file was measured on a 2-vCPU virtual machine with 7 GB memory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from ssjt import experiment as X

if __name__ == "__main__":
    tags = sys.argv[1:] or X.MAIN + X.SMALL
    rows = []
    for tag in tags:
        ds = X.load_dataset(tag)
        Yhat, E, _ = X.load_origin(ds, 0)
        h = ds["h"]
        res = X.measure(E, Yhat, X.timing_fns(h), reps=7 if h.N < 20000 else 5)
        for m, v in res.items():
            est, form = (m.split("|") + [""])[:2]
            rows.append({"data": tag, "N": h.N, "K": h.K, "T": E.shape[0], "method": est, "form": form, **v})
        print(tag, {m: round(v["median_s"] * 1e3, 1) for m, v in res.items()}, flush=True)
        pd.DataFrame(rows).to_csv(X.RESULTS / "timing_real.csv", index=False)
