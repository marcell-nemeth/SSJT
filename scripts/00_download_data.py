"""Download the public datasets into data/ and check their SHA-256 hashes.

* data/datasets.zip  hierarchicalforecast benchmark archive (Labour, TourismSmall, TourismLarge, Traffic, Wiki2);
                     shipped with the repository, downloaded only if missing
* data/m5.zip        M5 competition data (mirror of Nixtla); only sales_train_evaluation.csv and calendar.csv are read
* data/bdg2_electricity_cleaned.csv, data/bdg2_metadata.csv   Building Data Genome Project 2 (CC BY 4.0)
LargeST cannot be downloaded automatically; see README.md and scripts/01_prepare_largest.py.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
BDG2 = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"
FILES = {
    "datasets.zip": ("https://nixtla-public.s3.amazonaws.com/hierarchical-data/datasets.zip",
                     "6512d9aa80f111ee26480bc6f3f4eb3b5655d4ceecc384933100edc85adf704b"),
    "m5.zip": ("https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip",
               "cc704ba15d6802f8262e6ec7d4c6041e4ad6366a94365e8c84f721e450eed774"),
    "bdg2_electricity_cleaned.csv": (f"{BDG2}/meters/cleaned/electricity_cleaned.csv",
                                     "b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3"),
    "bdg2_metadata.csv": (f"{BDG2}/metadata/metadata.csv",
                          "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    ok = True
    for name, (url, digest) in FILES.items():
        p = DATA / name
        if not p.exists():
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, p)
        h = sha256(p)
        good = h == digest
        ok &= good
        print(f"{name}: {'OK' if good else 'HASH MISMATCH ' + h}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
