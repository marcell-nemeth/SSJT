"""Experiment protocol shared by the scripts: datasets, forecast origins, cached base forecasts, methods,
synthetic generators and the runtime / memory measurement."""
from __future__ import annotations

import gc
import os
import time
import tracemalloc
import zlib
from pathlib import Path
from typing import Callable, Dict

import numpy as np

from . import data as D
from .hierarchy import Hierarchy
from .mint import mint
from .ssjt import SSJT

REPO = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("SSJT_CACHE", REPO / "cache"))      # cached base forecasts (scripts/02)
RESULTS = REPO / "results"
COV_WINDOW = 2000          # at most the last 2,000 in-sample residuals are used for covariance estimation
LAMBDA = 0.1               # SSJT and MinT-Fix


def stable_seed(*keys) -> np.random.Generator:
    """Seeded generator that is identical across Python sessions."""
    return np.random.default_rng(np.random.SeedSequence([zlib.crc32(str(k).encode()) for k in keys]))


# ------------------------------------------------------------------------------------------- methods
def reconcile_all(h: Hierarchy, Yhat: np.ndarray, E: np.ndarray) -> Dict[str, np.ndarray]:
    """Base, MinT-Fix, MinT-Shrink and SSJT on one problem (Yhat: N x H, E: T x N, ordered as h.nodes)."""
    U = h.U
    return {"Base": Yhat,
            "MinT-Fix": mint(Yhat, U, E, LAMBDA)[0],
            "MinT-Shrink": mint(Yhat, U, E, "ss")[0],
            "SSJT": SSJT(h, E, LAMBDA).reconcile(Yhat)}


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(actual, float)) ** 2)))


# ------------------------------------------------------------------------------------------- real data
# tag: (loader, base model, season, H, fitting window W (None = expanding), list of forecast origins t_end)
def _hourly_origins(T):
    return [T - 24 - 168 * k for k in (3, 2, 1, 0)]


SMALL = ["Labour", "TourismSmall", "Wiki2", "Traffic"]
MAIN = ["TourismLarge", "bdg2", "largest", "bdg2_hourly", "largest_hourly", "m5_store", "m5_product"]
NAMES = {"Labour": "Labour", "TourismSmall": "TourismSmall", "Traffic": "Traffic", "Wiki2": "Wiki2",
         "TourismLarge": "TourismLarge", "bdg2": "BDG2 daily", "largest": "LargeST daily",
         "bdg2_hourly": "BDG2 hourly", "largest_hourly": "LargeST hourly",
         "m5_store": "M5 store tree", "m5_product": "M5 product tree"}


def hierarchicalforecast_origins(name: str, T_all: int, H: int, season: int):
    """10 rolling origins H apart, expanding window (fewer if the series are short, e.g. TourismSmall: 3)."""
    O = max(1, min(10, (T_all - max(3 * season, 24)) // H))
    return [T_all - (O - o) * H for o in range(O)]


def load_dataset(tag: str):
    """Returns dict(h, Y, model, season, H, W, origins, cache) with Y (T x N) ordered as h.nodes.
    cache(o, t_end) is the file of the cached base forecasts of origin o."""
    if tag in D.META:                      # hierarchicalforecast benchmark datasets
        ds = D.load(tag)
        H, s = ds.horizon, ds.season
        orig = hierarchicalforecast_origins(tag, ds.Y.shape[0], H, s)
        O = len(orig)
        return dict(h=ds.tree, Y=ds.Y, tree_cols=ds.tree_columns(), model="AutoETS", season=s, H=H, W=None,
                    freq=ds.freq, origins=orig, cache=lambda o, t: CACHE / f"{tag}_AutoETS_origin{o}_of{O}.npz")
    if tag in ("bdg2", "largest"):
        loader, model, W, orig = ((D.load_bdg2, "AutoETS", 364, [365 + 28 * k for k in range(13)]) if tag == "bdg2"
                                  else (D.load_largest, "ETS-ANA", 294, [301 + 7 * k for k in range(9)]))
        h, Y, _ = loader()
        return dict(h=h, Y=Y, tree_cols=None, model=model, season=7, H=7, W=W, freq="D", origins=orig,
                    cache=lambda o, t: CACHE / f"{tag}_{model}_W{W}_H7_t{t}.npz")
    if tag in ("bdg2_hourly", "largest_hourly"):
        h, Y, _ = (D.load_bdg2_hourly if tag == "bdg2_hourly" else D.load_largest_hourly)()
        return dict(h=h, Y=Y, tree_cols=None, model="ETS-ANA", season=24, H=24, W=6000, freq="h",
                    origins=_hourly_origins(Y.shape[0]),
                    cache=lambda o, t: CACHE / f"{tag}_ETS-ANA_W6000_H24_t{t}.npz")
    if tag in ("m5_store", "m5_product"):
        h, Y, _ = (D.load_m5 if tag == "m5_store" else D.load_m5_product)(None)
        T = Y.shape[0]
        pre = "m5" if tag == "m5_store" else "m5prod"
        return dict(h=h, Y=Y, tree_cols=None, model="AutoETS", season=7, H=28, W=1092, freq="D",
                    origins=[T - (3 - o) * 28 for o in range(3)],
                    cache=lambda o, t: CACHE / f"{pre}_W1092_origin{o}_of3.npz")
    raise ValueError(tag)


def load_origin(ds: dict, o: int, window: int = COV_WINDOW):
    """Base forecasts (N x H), residuals (last `window` rows, T x N) and test values (N x H) of origin o,
    restricted to the tree nodes."""
    t_end = ds["origins"][o]
    z = np.load(ds["cache"](o, t_end))
    Yhat, E = z["Yhat"].astype(float), z["E"].astype(float)
    Y = ds["Y"]
    if ds["tree_cols"] is not None:     # grouped dataset reduced to its tree
        tc = ds["tree_cols"]
        Yhat, E, Y = Yhat[tc], E[:, tc], Y[:, tc]
    return Yhat, E[-window:], Y[t_end:t_end + ds["H"]].T


# ------------------------------------------------------------------------------------------- synthetic
def grid_topologies(kind: str):
    """15 balanced and 17 imbalanced trees of the synthetic grid (Section 6.4)."""
    if kind == "balanced":
        combos = [(b, d) for b in (2, 3, 4, 5) for d in (2, 3, 4)] + [(2, 5), (2, 6), (3, 5)]
        return [(f"b{b}_d{d}", Hierarchy.balanced(b, d)) for b, d in combos]
    tops = []
    for b, d in [(b, d) for b in (3, 5, 7) for d in (3, 4, 5)]:
        for k in range(2):
            h = Hierarchy.imbalanced(b, d, stable_seed("imb", b, d, k))
            if h.N <= 900:
                tops.append((f"imb_b{b}_d{d}_{k}", h))
    return tops


def grid_series(h: Hierarchy, T: int, H: int, rng) -> np.ndarray:
    """Leaf random walks with drift, bounded below by 1, summed up the tree: (T + H) x N."""
    M = h.M
    X = np.empty((T + H, M))
    X[0] = rng.uniform(50, 150, M)
    drift = rng.uniform(-0.5, 1.5, M)
    for t in range(1, T + H):
        X[t] = X[t - 1] + drift + rng.normal(0, 2, M)
    X = np.maximum(X, 1.0)
    return (h.S @ X.T).T


GRID_T = (30, 60, 120, 250)
GRID_REPS = 5
GRID_H = 4
GRID_MODELS = ("AutoETS", "AutoARIMA")


def grid_cache(kind, topo, T, r, model):
    return CACHE / f"grid_{kind}_{topo}_T{T}_r{r}_{model}.npz"


# ------------------------------------------------------------------------------------------- timing
def measure(E: np.ndarray, Y: np.ndarray, fns: Dict[str, Callable], reps: int = 5,
            time_budget_s: float = 300.0) -> Dict[str, dict]:
    """Median wall-clock time over `reps` calls (fewer if one call exceeds the budget) and peak traced
    memory (tracemalloc) in a separate call. fns: {name: f(E, Y)}; f returns None if not run."""
    out = {}
    for m, f in fns.items():
        gc.collect()
        t0 = time.perf_counter(); r = f(E, Y); t1 = time.perf_counter() - t0
        if r is None:
            out[m] = {"median_s": np.nan, "min_s": np.nan, "max_s": np.nan, "reps": 0, "peak_MB": np.nan}
            continue
        ts = [t1]
        n = reps if t1 * reps < time_budget_s else max(1, int(time_budget_s // max(t1, 1e-9)))
        for _ in range(max(0, n - 1)):
            gc.collect()
            t0 = time.perf_counter(); f(E, Y); ts.append(time.perf_counter() - t0)
        del r
        gc.collect()
        tracemalloc.start()
        f(E, Y)
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        out[m] = {"median_s": float(np.median(ts)), "min_s": float(np.min(ts)), "max_s": float(np.max(ts)),
                  "reps": len(ts), "peak_MB": peak}
    return out


def timing_fns(h: Hierarchy, dense_max_N: int = 10000, kxk_max_K: int = 10000) -> Dict[str, Callable]:
    """SSJT and every form of MinT-Fix and MinT-Shrink that fits into memory."""
    U = h.U
    f = {"SSJT": lambda E, Y: SSJT(h, E, LAMBDA).reconcile(Y)}
    for est, lam in (("MinT-Shrink", "ss"), ("MinT-Fix", LAMBDA)):
        if h.N <= dense_max_N:
            f[f"{est}|dense"] = lambda E, Y, lam=lam: mint(Y, U, E, lam, "dense")[0]
        if h.K <= kxk_max_K:
            f[f"{est}|kxk"] = lambda E, Y, lam=lam: mint(Y, U, E, lam, "kxk")[0]
        f[f"{est}|txt"] = lambda E, Y, lam=lam: mint(Y, U, E, lam, "txt")[0]
    return f
