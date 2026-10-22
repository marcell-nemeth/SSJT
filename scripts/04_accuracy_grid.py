"""Synthetic grid (Table 5): RMSE of Base, MinT-Fix, MinT-Shrink and SSJT for every tree, sample length,
replicate and base model; uses the cached base forecasts of scripts/02 (grid_balanced, grid_imbalanced).
Also records the estimated shrinkage intensity of MinT-Shrink.

Usage: python scripts/04_accuracy_grid.py   -> results/grid_balanced.csv, results/grid_imbalanced.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from ssjt import experiment as X
from ssjt.mint import ss_lambda

if __name__ == "__main__":
    for kind in ("balanced", "imbalanced"):
        rows = []
        for topo, h in X.grid_topologies(kind):
            for T in X.GRID_T:
                for r in range(X.GRID_REPS):
                    Y = X.grid_series(h, T, X.GRID_H, X.stable_seed("grid", kind, topo, T, r))
                    Yte = Y[T:].T
                    for model in X.GRID_MODELS:
                        z = np.load(X.grid_cache(kind, topo, T, r, model))
                        Yhat, E = z["Yhat"], z["E"]
                        lam = ss_lambda(E)
                        for m, rec in X.reconcile_all(h, Yhat, E).items():
                            rows.append({"topology": topo, "N": h.N, "T": T, "rep": r, "model": model,
                                         "method": m, "RMSE": X.rmse(rec, Yte), "lambda_ss": lam})
            print(kind, topo, "done", flush=True)
        pd.DataFrame(rows).to_csv(X.RESULTS / f"grid_{kind}.csv", index=False)
