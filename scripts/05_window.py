"""Effect of the number of time steps for the residuals calculation (Table 7): hourly BDG2 and LargeST,
4 origins, T in {500, 2000, 6000}; same base forecasts, only the covariance window changes.

Usage: python scripts/05_window.py   -> results/hourly_window.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from ssjt import experiment as X
from ssjt.mint import ss_lambda

if __name__ == "__main__":
    rows = []
    for tag in ("bdg2_hourly", "largest_hourly"):
        ds = X.load_dataset(tag)
        for o in range(len(ds["origins"])):
            for T in (500, 2000, 6000):
                Yhat, E, Yte = X.load_origin(ds, o, window=T)
                lam = ss_lambda(E)
                for m, r in X.reconcile_all(ds["h"], Yhat, E).items():
                    rows.append({"data": tag, "origin": o, "T": T, "method": m, "RMSE": X.rmse(r, Yte),
                                 "lambda_ss": lam})
            print(tag, "origin", o, "done", flush=True)
    pd.DataFrame(rows).to_csv(X.RESULTS / "hourly_window.csv", index=False)
