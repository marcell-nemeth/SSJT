"""Dataset loaders. Every loader returns the hierarchy as a single-parent tree and the data ordered as its nodes.

* Labour, TourismSmall, TourismLarge, Traffic, Wiki2: benchmark archive of Nixtla's hierarchicalforecast
  (data/datasets.zip, https://nixtla-public.s3.amazonaws.com/hierarchical-data/datasets.zip).
* M5: data/m5.zip (sales_train_evaluation.csv and calendar.csv), two trees (store tree, product tree).
* BDG2: data/bdg2_electricity_cleaned.csv and data/bdg2_metadata.csv (Building Data Genome Project 2).
* LargeST: data/largest_2019_daily.npz, data/largest_2019_hourly.npz (made by scripts/01_prepare_largest.py)
  and data/ca_meta.csv.
The data directory is <repository>/data, or the directory given by the environment variable SSJT_DATA.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .hierarchy import Hierarchy, constraint_matrix_from_S, tree_from_summing_matrix

import os
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SSJT_DATA", REPO / "data"))
DEFAULT_ZIP = DATA / "datasets.zip"

# frequency, seasonal period and horizon as defined by datasetsforecast.hierarchical
META = {
    "Labour":       dict(freq="MS", season=12, horizon=8),
    "TourismSmall": dict(freq="QS", season=4, horizon=4),
    "TourismLarge": dict(freq="MS", season=12, horizon=12),
    "Traffic":      dict(freq="D", season=7, horizon=14),
    "Wiki2":        dict(freq="D", season=7, horizon=14),
}


@dataclass
class Dataset:
    name: str
    Y: np.ndarray                 # (T, N) all nodes of the original structure
    dates: pd.DatetimeIndex
    S_df: pd.DataFrame            # (N x M) summing matrix, index = node names
    tree: Hierarchy               # single-parent tree used by all methods
    freq: str
    season: int
    horizon: int

    @property
    def nodes(self):
        return list(self.S_df.index)

    @property
    def is_bottom(self) -> np.ndarray:
        return self.S_df.index.isin(self.S_df.columns)

    @property
    def U_full(self) -> np.ndarray:
        return constraint_matrix_from_S(self.S_df.values, self.is_bottom)

    def tree_columns(self) -> np.ndarray:
        """Positions of the tree nodes in the full node order."""
        pos = {n: i for i, n in enumerate(self.nodes)}
        return np.array([pos[n] for n in self.tree.nodes])

    def summary(self) -> dict:
        s = self.tree.summary()
        return {"dataset": self.name, "N": len(self.nodes), "N_tree": s["N"], "M": self.S_df.shape[1],
                "levels_tree": s["levels"], "b_min": s["b_min"], "b_mean": round(s["b_mean"], 2),
                "b_max": s["b_max"], "K": s["K"], "T": len(self.dates), "freq": self.freq,
                "season": self.season, "horizon": self.horizon,
                "dropped_by_tree_conversion": len(self.nodes) - s["N"]}


def _read(zf: zipfile.ZipFile, name: str, fname: str) -> pd.DataFrame:
    cands = [n for n in zf.namelist() if n.endswith(f"{name}/{fname}")]
    if not cands:
        raise FileNotFoundError(f"{name}/{fname} not found in archive")
    with zf.open(cands[0]) as fh:
        return pd.read_csv(io.BytesIO(fh.read()), index_col=0)


def load(name: str, zip_path: Optional[Path] = None) -> Dataset:
    zip_path = Path(zip_path or DEFAULT_ZIP)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} not found. Download https://nixtla-public.s3.amazonaws.com/hierarchical-data/"
            "datasets.zip and place it there.")
    with zipfile.ZipFile(zip_path) as zf:
        S_df = _read(zf, name, "agg_mat.csv")
        Y_df = _read(zf, name, "data.csv")
    Y_df.index = pd.to_datetime(Y_df.index)
    if name == "Labour":                       # as in datasetsforecast: exclude the COVID period
        Y_df = Y_df[Y_df.index < "2020-01-01"]
    Y_df = Y_df[list(S_df.index)]
    S_df = S_df.astype(float)
    tree = tree_from_summing_matrix(S_df, name=f"{name}-tree")
    m = META[name]
    return Dataset(name, Y_df.to_numpy(float), Y_df.index, S_df, tree, m["freq"], m["season"], m["horizon"])


# ---------------------------------------------------------------------------------------- M5
M5_ZIP = DATA / "m5.zip"
M5_SOURCE = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"


def load_m5(zip_path: Optional[Path] = None, store: Optional[str] = None):
    """M5 (Walmart unit sales, 30,490 item-store series, 1,941 days) as a single-parent tree
        Total -> State (3) -> Store (10) -> Store x Category (30) -> Store x Department (70) -> Item x Store.
    The grouped M5 aggregates that do not fit this tree (category, department, state x category,
    state x department, item, item x state) are not used. With ``store`` given, only that store's
    subtree (Store -> Store x Category -> Store x Department -> items) is returned.

    Returns (hier, Y, dates) with Y of shape (T, N) ordered as hier.nodes.
    """
    zip_path = Path(zip_path or M5_ZIP)
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found; download {M5_SOURCE}")
    with zipfile.ZipFile(zip_path) as zf:
        name = [n for n in zf.namelist() if n.endswith("sales_train_evaluation.csv")][0]
        with zf.open(name) as fh:
            sales = pd.read_csv(fh)
        with zf.open([n for n in zf.namelist() if n.endswith("calendar.csv")][0]) as fh:
            cal = pd.read_csv(fh)
    if store is not None:
        sales = sales[sales.store_id == store]
    dcols = [c for c in sales.columns if c.startswith("d_")]
    cal_dates = pd.to_datetime(cal["date"]).to_numpy()            # calendar row k is day d_{k+1}
    dates = pd.DatetimeIndex(cal_dates[[int(c[2:]) - 1 for c in dcols]])
    X = sales[dcols].to_numpy(float).T                          # T x M
    st, sto = sales.state_id.to_numpy(), sales.store_id.to_numpy()
    cat, dep = sales.cat_id.to_numpy(), sales.dept_id.to_numpy()
    leaf = (sales.item_id + "|" + sales.store_id).to_numpy()
    lv_store = sto
    lv_cat = np.char.add(np.char.add(sto.astype(str), "|"), cat.astype(str))
    lv_dep = np.char.add(np.char.add(sto.astype(str), "|"), dep.astype(str))
    parent = {}
    if store is None:
        for s_, t_ in zip(st, sto):
            parent[t_] = s_
            parent[s_] = "Total"
    for a, b in zip(lv_cat, lv_store):
        parent[a] = b
    for a, b in zip(lv_dep, lv_cat):
        parent[a] = b
    for a, b in zip(leaf, lv_dep):
        parent[a] = b
    tops = (["Total"] + sorted(set(st)) if store is None else []) + sorted(set(lv_store))
    nodes = tops + sorted(set(lv_cat)) + sorted(set(lv_dep)) + list(leaf)
    hier = Hierarchy.from_parent_map(parent, nodes, name="M5" if store is None else f"M5-{store}")
    # aggregate the bottom series up the tree
    Y = np.zeros((X.shape[0], hier.N))
    Y[:, hier.leaf_index()] = X
    for p in hier.post_order:
        Y[:, hier.idx[p]] = Y[:, [hier.idx[c] for c in hier.children[p]]].sum(axis=1)
    return hier, Y, dates


def load_m5_product(zip_path: Optional[Path] = None):
    """M5 as the product tree built from official M5 aggregation levels:
        Total -> Category (3) -> Department (7) -> Item (3,049) -> Item x State (9,147) -> Item x Store.
    Leaves are the same 30,490 item-store series as in ``load_m5``. Returns (hier, Y, dates)."""
    zip_path = Path(zip_path or M5_ZIP)
    with zipfile.ZipFile(zip_path) as zf:
        name = [n for n in zf.namelist() if n.endswith("sales_train_evaluation.csv")][0]
        with zf.open(name) as fh:
            sales = pd.read_csv(fh)
        with zf.open([n for n in zf.namelist() if n.endswith("calendar.csv")][0]) as fh:
            cal = pd.read_csv(fh)
    dcols = [c for c in sales.columns if c.startswith("d_")]
    cal_dates = pd.to_datetime(cal["date"]).to_numpy()
    dates = pd.DatetimeIndex(cal_dates[[int(c[2:]) - 1 for c in dcols]])
    X = sales[dcols].to_numpy(float).T
    cat = ("C:" + sales.cat_id).to_numpy()
    dep = ("D:" + sales.dept_id).to_numpy()
    item = ("I:" + sales.item_id).to_numpy()
    ist = ("IS:" + sales.item_id + "|" + sales.state_id).to_numpy()
    leaf = (sales.item_id + "|" + sales.store_id).to_numpy()
    parent = {}
    for c in set(cat):
        parent[c] = "Total"
    for a, b in zip(dep, cat):
        parent[a] = b
    for a, b in zip(item, dep):
        parent[a] = b
    for a, b in zip(ist, item):
        parent[a] = b
    for a, b in zip(leaf, ist):
        parent[a] = b
    nodes = ["Total"] + sorted(set(cat)) + sorted(set(dep)) + sorted(set(item)) + sorted(set(ist)) + list(leaf)
    hier = Hierarchy.from_parent_map(parent, nodes, name="M5-product")
    Y = np.zeros((X.shape[0], hier.N))
    Y[:, hier.leaf_index()] = X
    for p in hier.post_order:
        Y[:, hier.idx[p]] = Y[:, [hier.idx[c] for c in hier.children[p]]].sum(axis=1)
    return hier, Y, dates


# ---------------------------------------------------------------------------------------- BDG2, LargeST
# LargeST (Liu et al., NeurIPS 2023; CC BY-NC 4.0): California PeMS traffic flow in 2019, 8,600 sensors,
# 5-minute readings summed per day / hour by scripts/01_prepare_largest.py (sum and count of valid readings).
# Tree: Total -> PeMS district -> county -> freeway and direction -> sensor. Sensors kept: >= 90% valid
# readings on >= 95% of the days. Daily total = sum / count * 288 on days with >= 90% valid readings, other
# days linearly interpolated (hourly: >= 9 of 12 readings, scaled to 12).
#
# BDG2 (Miller et al., Sci. Data 2020; CC BY 4.0): hourly electricity of non-residential buildings,
# 2016-2017 (cleaned data set). Tree: Total -> site -> primary space usage -> building. Meters kept:
# >= 95% non-missing days; gaps of up to 3 days interpolated, meters with longer gaps dropped.

def _tree_from_paths(paths, leaf_names, X):
    """paths: list of tuples (level1, level2, ...) for each leaf (excluding Total); X: T x M leaf data."""
    parent, aggs = {}, []
    for pth, leaf in zip(paths, leaf_names):
        prev = "Total"
        for k in range(len(pth)):
            node = "|".join(pth[:k + 1])
            if node not in parent:
                parent[node] = prev
                aggs.append(node)
            prev = node
        parent[leaf] = prev
    nodes = ["Total"] + aggs + list(leaf_names)
    h = Hierarchy.from_parent_map(parent, nodes, name="tree")
    Y = np.zeros((X.shape[0], h.N))
    Y[:, h.leaf_index()] = X
    for p in h.post_order:
        Y[:, h.idx[p]] = Y[:, [h.idx[c] for c in h.children[p]]].sum(axis=1)
    return h, Y


def load_largest(min_read=0.9, min_days=0.95):
    z = np.load(DATA / "largest_2019_daily.npz")
    s, c, ids = z["s"], z["c"].astype(float), z["ids"].astype(str)
    meta = pd.read_csv(DATA / "ca_meta.csv")
    meta["ID"] = meta["ID"].astype(str)
    ok_day = c >= min_read * 288
    keep = ok_day.mean(0) >= min_days
    daily = np.where(ok_day, s / np.maximum(c, 1) * 288, np.nan)[:, keep]
    daily = pd.DataFrame(daily).interpolate(limit_direction="both").to_numpy()
    m = meta.set_index("ID").loc[ids[keep]]
    paths = [(f"D{d}", f"D{d}|{cty}", f"D{d}|{cty}|{fwy}") for d, cty, fwy in zip(m.District, m.County, m.Fwy)]
    paths = [(a, b, c_) for a, b, c_ in paths]
    leaf = ["S" + i for i in ids[keep]]
    order = np.lexsort((np.array(leaf), np.array([p[2] for p in paths])))
    h, Y = _tree_from_paths([paths[i] for i in order], [leaf[i] for i in order], daily[:, order])
    h.name = "LargeST-2019"
    dates = pd.date_range("2019-01-01", periods=Y.shape[0], freq="D")
    return h, Y, dates


def load_bdg2(min_days=0.95):
    e = pd.read_csv(DATA / "bdg2_electricity_cleaned.csv", parse_dates=["timestamp"]).set_index("timestamp")
    meta = pd.read_csv(DATA / "bdg2_metadata.csv").set_index("building_id")
    d = e.resample("D").sum(min_count=20)
    good = d.columns[d.notna().mean() >= min_days]
    d = d[good].interpolate(limit=3).dropna(axis=1)
    site = meta.loc[d.columns, "site_id"].astype(str).to_numpy()
    use = meta.loc[d.columns, "primaryspaceusage"].fillna("NA").astype(str).to_numpy()
    paths = [(s, f"{s}|{u}") for s, u in zip(site, use)]
    leaf = ["B:" + c for c in d.columns]
    order = np.lexsort((np.array(leaf), np.array([p[1] for p in paths])))
    h, Y = _tree_from_paths([paths[i] for i in order], [leaf[i] for i in order], d.to_numpy()[:, order])
    h.name = "BDG2"
    return h, Y, d.index


# ------------------------------------------------------------------ hourly versions (long histories)
def load_largest_hourly(min_read=0.9, min_days=0.95, min_hour_read=9):
    """Hourly flow totals for the same sensors as load_largest (daily coverage rule). An hour with at least
    `min_hour_read` of 12 valid 5-minute readings is scaled to 12 readings; other hours are interpolated."""
    zd = np.load(DATA / "largest_2019_daily.npz")
    keep = (zd["c"].astype(float) >= min_read * 288).mean(0) >= min_days
    z = np.load(DATA / "largest_2019_hourly.npz")
    s, c, ids = z["s"].astype(float), z["c"].astype(float), z["ids"].astype(str)
    hourly = np.where(c >= min_hour_read, s / np.maximum(c, 1) * 12, np.nan)[:, keep]
    hourly = pd.DataFrame(hourly).interpolate(limit_direction="both").to_numpy()
    meta = pd.read_csv(DATA / "ca_meta.csv"); meta["ID"] = meta["ID"].astype(str)
    m = meta.set_index("ID").loc[ids[keep]]
    paths = [(f"D{d}", f"D{d}|{cty}", f"D{d}|{cty}|{fwy}") for d, cty, fwy in zip(m.District, m.County, m.Fwy)]
    leaf = ["S" + i for i in ids[keep]]
    order = np.lexsort((np.array(leaf), np.array([p[2] for p in paths])))
    h, Y = _tree_from_paths([paths[i] for i in order], [leaf[i] for i in order], hourly[:, order])
    h.name = "LargeST-2019-hourly"
    return h, Y, pd.date_range("2019-01-01", periods=Y.shape[0], freq="h")


def load_bdg2_hourly(min_days=0.95):
    """Hourly electricity for the same buildings as load_bdg2; gaps interpolated."""
    e = pd.read_csv(DATA / "bdg2_electricity_cleaned.csv", parse_dates=["timestamp"]).set_index("timestamp")
    meta = pd.read_csv(DATA / "bdg2_metadata.csv").set_index("building_id")
    d = e.resample("D").sum(min_count=20)
    good = d.columns[d.notna().mean() >= min_days]
    d = d[good].interpolate(limit=3).dropna(axis=1)
    hx = e[d.columns].interpolate(limit_direction="both")
    site = meta.loc[hx.columns, "site_id"].astype(str).to_numpy()
    use = meta.loc[hx.columns, "primaryspaceusage"].fillna("NA").astype(str).to_numpy()
    paths = [(s, f"{s}|{u}") for s, u in zip(site, use)]
    leaf = ["B:" + c for c in hx.columns]
    order = np.lexsort((np.array(leaf), np.array([p[1] for p in paths])))
    h, Y = _tree_from_paths([paths[i] for i in order], [leaf[i] for i in order], hx.to_numpy()[:, order])
    h.name = "BDG2-hourly"
    return h, Y, hx.index
