#!/usr/bin/env bash
# Reproduces every result, table and figure of the article. Data must be in data/ (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
python scripts/00_download_data.py
# LargeST must be prepared once by hand: python scripts/01_prepare_largest.py ca_his_raw_2019.h5 ca_meta.csv
python scripts/02_base_forecasts.py
python scripts/03_accuracy_real.py
python scripts/04_accuracy_grid.py
python scripts/05_window.py
python scripts/06_timing_real.py
python scripts/07_sweeps.py
python scripts/08_tables.py
python scripts/09_figures.py
python scripts/10_diagnostics.py
