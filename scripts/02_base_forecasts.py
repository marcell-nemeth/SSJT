"""Base forecasts and in-sample one-step residuals for every dataset and forecast origin (cache/).

Usage: python scripts/02_base_forecasts.py [tag ...]
tags: Labour TourismSmall Wiki2 Traffic TourismLarge bdg2 largest bdg2_hourly largest_hourly m5_store m5_product
      grid_balanced grid_imbalanced            (default: all)
Existing cache files are not recomputed. Approximate times on 2 cores: small datasets and TourismLarge
a few minutes each, BDG2 daily ~10 min, LargeST daily ~1 h, hourly datasets several hours, M5 ~1.5 h per tree,
grid ~1.5 h. Model fitting uses StatsForecast (all cores).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from ssjt import experiment as X
from ssjt.forecasting import base_forecasts

ALL = X.SMALL + X.MAIN + ["grid_balanced", "grid_imbalanced"]


def real(tag):
    ds = X.load_dataset(tag)
    Y, H, s, W = ds["Y"], ds["H"], ds["season"], ds["W"]
    for o, t_end in enumerate(ds["origins"]):
        cf = ds["cache"](o, t_end)
        if cf.exists():
            continue
        t0 = time.time()
        Ytr = Y[:t_end] if W is None else Y[t_end - W:t_end]
        Yhat, E = base_forecasts(Ytr, H, ds["model"], s, ds["freq"])
        if tag.startswith("m5"):                  # stored in single precision (as used for the paper)
            np.savez_compressed(cf, Yhat=Yhat.astype(np.float32), E=E.astype(np.float32))
        else:
            np.savez_compressed(cf, Yhat=Yhat, E=E)
        print(f"{tag} origin {o}: {time.time() - t0:.0f} s, E {E.shape}", flush=True)


def grid(kind):
    for topo, h in X.grid_topologies(kind):
        for T in X.GRID_T:
            for r in range(X.GRID_REPS):
                Y = X.grid_series(h, T, X.GRID_H, X.stable_seed("grid", kind, topo, T, r))
                for model in X.GRID_MODELS:
                    cf = X.grid_cache(kind, topo, T, r, model)
                    if not cf.exists():
                        Yhat, E = base_forecasts(Y[:T], X.GRID_H, model, 1, "D")
                        np.savez_compressed(cf, Yhat=Yhat, E=E)
        print(f"grid {kind} {topo} done", flush=True)


if __name__ == "__main__":
    X.CACHE.mkdir(parents=True, exist_ok=True)
    for tag in sys.argv[1:] or ALL:
        grid(tag.split("_", 1)[1]) if tag.startswith("grid_") else real(tag)
