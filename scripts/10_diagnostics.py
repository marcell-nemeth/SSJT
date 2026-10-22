"""Numbers quoted in the text of Section 7.2 that are not in a table.

* distribution of the estimated MinT-Shrink intensity on the synthetic grid (AutoETS base forecasts)
* hourly BDG2: number of pairs of time series and of pairs within a family; stability of the residual
  correlations (correlation between the correlations estimated from the first and the last 2,000 hours of the
  6,000-hour window, per origin)
Usage: python scripts/10_diagnostics.py   (after scripts/04)  -> results/diagnostics.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from ssjt import experiment as X

if __name__ == "__main__":
    out = []
    g = pd.concat([pd.read_csv(X.RESULTS / f"grid_{k}.csv") for k in ("balanced", "imbalanced")])
    lam = g[(g.model == "AutoETS") & (g.method == "SSJT")].lambda_ss
    out.append(f"grid, AutoETS: {len(lam)} runs, estimated lambda median {lam.median():.3f}, "
               f"10th-90th percentile {lam.quantile(0.1):.3f}-{lam.quantile(0.9):.3f}")
    ds = X.load_dataset("bdg2_hourly")
    h = ds["h"]
    within = sum((len(c) + 1) * len(c) // 2 for c in h.children.values() if c)
    out.append(f"BDG2: N = {h.N}, pairs = {h.N * (h.N - 1) // 2}, pairs within a family = {within}")
    for o in range(len(ds["origins"])):
        _, E, _ = X.load_origin(ds, o, window=6000)
        def corr(A):
            A = A - A.mean(0)
            s = np.sqrt((A ** 2).sum(0)); s[s == 0] = 1
            C = (A / s).T @ (A / s)
            return C[np.triu_indices(C.shape[0], 1)]
        r = np.corrcoef(corr(E[:2000]), corr(E[-2000:]))[0, 1]
        out.append(f"BDG2 origin {o}: correlation of residual correlations, first vs last 2000 h = {r:.3f}")
    txt = "\n".join(out)
    print(txt)
    (X.RESULTS / "diagnostics.txt").write_text(txt + "\n")
