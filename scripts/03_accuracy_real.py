"""RMSE of Base, MinT-Fix, MinT-Shrink and SSJT on the real datasets (Tables 6 and A.9).

Usage: python scripts/03_accuracy_real.py [tag ...]      -> results/accuracy_real.csv
One row per dataset, origin and method: RMSE over all nodes of the tree and all forecast time steps.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

from ssjt import experiment as X

if __name__ == "__main__":
    tags = sys.argv[1:] or X.MAIN + X.SMALL
    out = X.RESULTS / "accuracy_real.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = []
    for tag in tags:
        ds = X.load_dataset(tag)
        for o in range(len(ds["origins"])):
            Yhat, E, Yte = X.load_origin(ds, o)
            for m, r in X.reconcile_all(ds["h"], Yhat, E).items():
                rows.append({"data": tag, "origin": o, "T": E.shape[0], "method": m, "RMSE": X.rmse(r, Yte)})
        print(tag, "done", flush=True)
    new = pd.DataFrame(rows)
    if len(old):
        new = pd.concat([old[~old.data.isin(tags)], new], ignore_index=True)
    new.to_csv(out, index=False)
