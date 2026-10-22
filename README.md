# SSJT — Structurally Shrunk Junction Tree reconciliation

Code, data preparation and results for the article

> M. Németh, G. Benedek, G. Szűcs. *Scalable Hierarchical Time Series Reconciliation via Structurally Shrunk Junction Trees.* Submitted to Neurocomputing.

SSJT makes the forecasts of a hierarchy of time series coherent. It splits the tree into parent–children families and applies a Minimum Trace (MinT) correction within each family, in one upward and one downward traversal; the output is exactly coherent. Each family covariance matrix is shrunk towards a structural target, and a convex safeguard keeps every correction between the parent forecast and the sum of its children's forecasts. The quantities needed by the two traversals are computed without forming any covariance matrix: runtime O(T·N), memory O(N), independent of the branching factor (N time series, T residuals).

## Installation

```bash
git clone https://github.com/marcell-nemeth/SSJT.git && cd SSJT
python -m venv .venv && source .venv/bin/activate      # Python >= 3.10
pip install -r requirements.txt
pip install -e .
pytest                                                 # 14 tests, a few seconds
```

`requirements.txt` pins the versions used for the results in `results/` (Python 3.11).

## Using the method

```python
import numpy as np
from ssjt import SSJT, Hierarchy, mint

h = Hierarchy.from_parent_map({"A": "Total", "B": "Total", "a1": "A", "a2": "A", "b1": "B", "b2": "B"})
E = ...      # (T x N) in-sample one-step residuals of the base models, columns ordered as h.nodes
Yhat = ...   # (N,) or (N x H) base forecasts, rows ordered as h.nodes

Y_ssjt = SSJT(h, E, lam=0.1).reconcile(Yhat)          # SSJT (matrix-free, convex safeguard)
Y_shrink, lam = mint(Yhat, h.U, E, lam="ss")           # MinT-Shrink (Schafer-Strimmer intensity)
Y_fix, _ = mint(Yhat, h.U, E, lam=0.1, form="txt")     # MinT-Fix in the T x T (Woodbury) form
```

`SSJT(h, E)` runs Phase I (residual statistics) once; `reconcile` can then be called for any number of forecast vectors. `mint(..., form=...)` computes MinT in the dense, K × K or T × T form (Section 5.2 of the article); all forms give the same result.

## Repository layout

| Path | Content |
|---|---|
| `src/ssjt/ssjt.py` | SSJT (Algorithm 1): Phase I in one pass over the residuals, upward and downward passes, convex safeguard |
| `src/ssjt/mint.py` | MinT-Fix / MinT-Shrink in the dense, K × K and T × T forms; Schäfer–Strimmer shrinkage intensity |
| `src/ssjt/hierarchy.py` | `Hierarchy` (single-parent tree), generators, conversion of a grouped structure to a tree |
| `src/ssjt/data.py` | loaders of all datasets |
| `src/ssjt/forecasting.py` | base forecasts and residuals with StatsForecast |
| `src/ssjt/experiment.py` | protocol shared by the scripts: origins, windows, cache files, methods, synthetic generators, timing |
| `scripts/` | numbered scripts that reproduce every result, table and figure (below) |
| `results/` | the results reported in the article (CSV) |
| `tables/`, `figures/` | LaTeX tables and figures of the article, generated from `results/` |
| `tests/` | correctness tests (matrix-free SSJT against explicit family matrices, coherence, MinT forms) |

## Data

| Dataset | Source | How to obtain |
|---|---|---|
| Labour, TourismSmall, TourismLarge, Traffic, Wiki2 | hierarchicalforecast benchmark archive (Nixtla) | shipped as `data/datasets.zip` |
| M5 (store tree, product tree) | M5 competition (Walmart), Nixtla mirror | `python scripts/00_download_data.py` |
| BDG2 (daily, hourly) | Building Data Genome Project 2, CC BY 4.0 | `python scripts/00_download_data.py` |
| LargeST (daily, hourly) | LargeST benchmark (CA, 2019), CC BY-NC 4.0 | manual download, then `scripts/01_prepare_largest.py` |

`00_download_data.py` downloads the files into `data/` and checks their SHA-256 hashes. For LargeST, download `ca_his_raw_2019.h5` and `ca_meta.csv` from the LargeST repository (https://github.com/liuxu77/LargeST, Kaggle archive), then run

```bash
python scripts/01_prepare_largest.py /path/to/ca_his_raw_2019.h5 /path/to/ca_meta.csv
```

which writes the daily and hourly sums and counts of the valid 5-minute readings to `data/`. The selection of sensors and buildings and the construction of the trees are in `src/ssjt/data.py` and are described in Section 6.3 of the article. The environment variable `SSJT_DATA` can point to another data directory.

## Reproducing the results

The scripts run in order; each writes its output to `results/`, `tables/` or `figures/`. `run_all.sh` runs all of them.

| Script | Produces | Article | Time on 2 cores |
|---|---|---|---|
| `00_download_data.py` | `data/` | — | minutes |
| `01_prepare_largest.py` | `data/largest_2019_*.npz` | — | ~15 min |
| `02_base_forecasts.py [tag ...]` | `cache/` (base forecasts and residuals of every origin) | §6.3, §6.4 | several hours in total (see the script) |
| `03_accuracy_real.py` | `results/accuracy_real.csv` | Tables 6, A.9 | ~5 min |
| `04_accuracy_grid.py` | `results/grid_balanced.csv`, `results/grid_imbalanced.csv` | Table 5 | ~3 min |
| `05_window.py` | `results/hourly_window.csv` | Table 7 | ~5 min |
| `06_timing_real.py` | `results/timing_real.csv` | Tables 4, A.8 | ~30 min |
| `07_sweeps.py` | `results/sweep_N.csv`, `results/sweep_T.csv` | Figures 2, 3 | ~25 min |
| `08_tables.py` | `tables/*.tex` | Tables 3–7, A.8, A.9 | < 1 min |
| `09_figures.py` | `figures/sweep_N.pdf`, `figures/sweep_T.pdf` | Figures 2, 3 | < 1 min |
| `10_diagnostics.py` | `results/diagnostics.txt` | numbers in §7.2 | ~1 min |

To rebuild only the tables and figures from the shipped results:

```bash
python scripts/08_tables.py && python scripts/09_figures.py
```

**Base forecasts.** `02_base_forecasts.py` fits the base models (StatsForecast `AutoETS`, `AutoARIMA` or ETS(A,N,A)) at every forecast origin and stores the forecasts and in-sample one-step residuals in `cache/` (about 4 GB, not in the repository). The cache files are not recomputed if they exist; set `SSJT_CACHE` to use another directory. All later accuracy results follow deterministically from the cache.

**Accuracy.** `03`–`05` reproduce `results/accuracy_real.csv`, `results/grid_*.csv` and `results/hourly_window.csv` exactly (to floating-point rounding, < 1e-12 relative) from the cached base forecasts. The synthetic grid series are generated with fixed seeds (`ssjt.experiment.stable_seed`).

**Runtime and memory.** Runtimes and peak memory depend on the machine. The shipped files were measured on a virtual machine with 2 vCPUs (Intel Xeon) and 7 GB of memory, where repeated measurements of the same task differed by up to a factor of two. `06` and `07` run SSJT and every form of MinT that fits into memory; the tables and figures use the fastest (runtime) and the least memory-consuming (memory) form. `results/timing_real.csv` also contains, in the rows with `impl = standard`, measurements of an earlier implementation of the K × K and T × T forms that keeps a full copy of the residual matrix (same results, more memory); `impl = lean` and `dense` are the implementations in `src/ssjt/mint.py` (new runs of `06` write these without the `impl` column). Table 4 takes the minimum over all rows of a form, as in the article.

## Result files

| File | Columns |
|---|---|
| `accuracy_real.csv` | dataset, origin, number of residuals T, method, RMSE over all nodes and forecast time steps |
| `grid_balanced.csv`, `grid_imbalanced.csv` | tree, N, T, replicate, base model, method, RMSE, estimated MinT-Shrink intensity |
| `hourly_window.csv` | dataset, origin, T, method, RMSE, estimated MinT-Shrink intensity |
| `timing_real.csv` | dataset, N, K, T, method, MinT form, implementation, median / min / max runtime (s), repetitions, peak memory (MB) |
| `sweep_N.csv`, `sweep_T.csv` | as `timing_real.csv`, for the synthetic trees (b, d) and for the hourly datasets |
| `diagnostics.txt` | estimated shrinkage intensity on the grid, pair counts and correlation stability on BDG2 |

## Citation

See `CITATION.cff`. Please also cite the data sources: hierarchicalforecast (Olivares et al., 2022), BDG2 (Miller et al., 2020), LargeST (Liu et al., 2023) and M5 (Makridakis et al., 2022).

## License

Code: MIT (see `LICENSE`). The datasets keep their own licenses (BDG2: CC BY 4.0; LargeST: CC BY-NC 4.0; M5: terms of the M5 competition).
