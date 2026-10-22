"""Figures 2 and 3 from results/sweep_N.csv and results/sweep_T.csv (written to figures/).

For MinT-Fix and MinT-Shrink the runtime of the fastest and the memory usage of the least memory-consuming
form is plotted; the dense form of MinT-Shrink is shown separately.
Usage: python scripts/09_figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ssjt import experiment as X

OUT = X.REPO / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ST = {"SSJT": dict(color="#1b6ca8", marker="o"), "MinT-Fix": dict(color="#e08214", marker="s"),
      "MinT-Shrink": dict(color="#7b3294", marker="^"), "MinT-Shrink (dense)": dict(color="#888888", marker="x")}


def best(df, keys):
    rows = []
    for k, g in df.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        for est, gg in g.groupby("method"):
            rows.append({**dict(zip(keys, k)), "fam": est, "t": gg.median_s.min() * 1e3, "m": gg.peak_MB.min()})
        dd = g[(g.method == "MinT-Shrink") & (g.form == "dense")]
        if len(dd):
            rows.append({**dict(zip(keys, k)), "fam": "MinT-Shrink (dense)", "t": dd.median_s.iloc[0] * 1e3,
                         "m": dd.peak_MB.iloc[0]})
    return pd.DataFrame(rows)


def fig_N():
    B = best(pd.read_csv(X.RESULTS / "sweep_N.csv"), ["N"])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for fam, st in ST.items():
        q = B[B.fam == fam].sort_values("N")
        ax[0].plot(q.N, q.t, label=fam, ms=4, ls="none", **st)
        ax[1].plot(q.N, q.m, label=fam, ms=4, ls="none", **st)
    for a, yl in zip(ax, ["runtime (ms)", "peak memory usage (MB)"]):
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("number of time series $N$"); a.set_ylabel(yl)
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "sweep_N.pdf"); fig.savefig(OUT / "sweep_N.png", dpi=150)


def fig_T():
    B = best(pd.read_csv(X.RESULTS / "sweep_T.csv"), ["data", "T"])
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    for j, (d, name, N) in enumerate([("bdg2_hourly", "BDG2 hourly ($N = 856$)", 856),
                                      ("largest_hourly", "LargeST hourly ($N = 8\\,661$)", 8661)]):
        for fam in ["SSJT", "MinT-Fix", "MinT-Shrink"]:
            q = B[(B.data == d) & (B.fam == fam)].sort_values("T")
            ax[0, j].plot(q["T"], q.t, label=fam, ms=4, lw=1.2, **ST[fam])
            ax[1, j].plot(q["T"], q.m, label=fam, ms=4, lw=1.2, **ST[fam])
        ax[0, j].set_title(name, fontsize=9)
        for i in range(2):
            a = ax[i, j]; a.set_xscale("log"); a.set_yscale("log"); a.grid(alpha=0.3)
            if N <= 6000:
                a.axvline(N, color="k", lw=0.6, ls=":")
        ax[1, j].set_xlabel("$T$")
    ax[0, 0].set_ylabel("runtime (ms)"); ax[1, 0].set_ylabel("peak memory usage (MB)")
    ax[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "sweep_T.pdf"); fig.savefig(OUT / "sweep_T.png", dpi=150)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_N(); fig_T()
    print("figures written to", OUT)
