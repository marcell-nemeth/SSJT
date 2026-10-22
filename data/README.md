# Data

| File | Origin | In the repository |
|---|---|---|
| `datasets.zip` | hierarchicalforecast benchmark archive (Nixtla), https://nixtla-public.s3.amazonaws.com/hierarchical-data/datasets.zip | yes |
| `m5.zip` | M5 competition data, Nixtla mirror https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip | no, `scripts/00_download_data.py` |
| `bdg2_electricity_cleaned.csv`, `bdg2_metadata.csv` | Building Data Genome Project 2 (CC BY 4.0), https://github.com/buds-lab/building-data-genome-project-2 | no, `scripts/00_download_data.py` |
| `largest_2019_daily.npz`, `largest_2019_hourly.npz`, `ca_meta.csv` | LargeST (CC BY-NC 4.0), https://github.com/liuxu77/LargeST | no, `scripts/01_prepare_largest.py` |

SHA-256 of the prepared LargeST files (as used for the article):

    largest_2019_daily.npz   a8876c04c1d46eec8548878a00ca18858f259c896e6342aaa164c24c32662b27
    largest_2019_hourly.npz  abc6a7ecab4289b82e944f626f51f90ad92feee303f1a269be5d48bc796ed569
    ca_meta.csv              57c5edb5e6f1d1802440426f2bd07f54d812859135989c740ab96b1448b087ee

The hashes of the downloaded files are checked by `scripts/00_download_data.py`.
Labour is restricted to observations before 2020-01-01, as in `datasetsforecast`.
