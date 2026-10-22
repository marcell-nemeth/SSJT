"""LaTeX tables of the article from results/*.csv (written to tables/).

  t_spec.tex          Table 3   dataset specification
  t_timing.tex        Table 4   runtime / memory on the real datasets, ratios MinT / SSJT (form in superscript)
  t_grid.tex          Table 5   synthetic grid: mean RMSE and relative difference of SSJT (Wilcoxon, Holm)
  t_real.tex          Table 6   RMSE on the real datasets
  t_Tacc.tex          Table 7   RMSE on the hourly datasets for three covariance windows
  t_timing_small.tex  Table A.8 runtime / memory on the small datasets
  t_real_small.tex    Table A.9 RMSE on the small datasets
Usage: python scripts/08_tables.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ssjt import experiment as X

R = X.RESULTS
OUT = X.REPO / "tables"
MOD = {"TourismLarge": "AutoETS", "bdg2": "AutoETS", "largest": "ETS(A,N,A)", "bdg2_hourly": "ETS(A,N,A)",
       "largest_hourly": "ETS(A,N,A)", "m5_store": "AutoETS", "m5_product": "AutoETS"}
MOD.update({d: "AutoETS" for d in X.SMALL})
M = ["Base", "MinT-Fix", "MinT-Shrink", "SSJT"]


def th(n): return f"{int(n):,}".replace(",", r"\,")
def dec(x, d=1): return f"{x:.{d}f}"
def B(s, c): return r"\textbf{" + s + "}" if c else s


def pct(x, d=1):
    return B(f"{x:+.{d}f}".replace("-", "$-$"), x <= -0.5 * 10 ** (-d))


def ratio(v):
    return B(dec(v, 1) if v < 10 else f"{v:.0f}", v >= 1.05)


def num(v):
    if v >= 1000: return th(round(v))
    if v >= 100: return dec(v, 1)
    if v >= 1: return dec(v, 2)
    return dec(v, 3)


def table(cols, head, body, size=r"\footnotesize", sep=None):
    return "\n".join([size + (rf"\setlength{{\tabcolsep}}{{{sep}}}" if sep else ""),
                      r"\begin{tabular}{" + cols + "}", r"\toprule"] + head + [r"\midrule"] + body +
                     [r"\bottomrule", r"\end{tabular}"]) + "\n"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def write(name, txt):
    (OUT / name).write_text(txt)


# ------------------------------------------------------------------------------------------- timing
def timing_tables():
    t = pd.read_csv(R / "timing_real.csv")
    L = {"dense": "D", "kxk": "K", "txt": "T"}

    def best(d, T):
        g = t[(t.data == d) & (t["T"] == T)]
        s = g[g.method == "SSJT"].iloc[0]
        r = {"N": s.N, "K": s.K, "T": T, "ms": s.median_s * 1e3, "MB": s.peak_MB}
        for f in ["MinT-Shrink", "MinT-Fix"]:
            gg = g[g.method == f]
            r[f + "_t"] = gg.median_s.min() / s.median_s
            r[f + "_m"] = gg.peak_MB.min() / s.peak_MB
            r[f + "_tf"] = L[gg.loc[gg.median_s.idxmin(), "form"]]
            r[f + "_mf"] = L[gg.loc[gg.peak_MB.idxmin(), "form"]]
        return r

    def row(d):
        T = 2000 if "hourly" in d else int(t[t.data == d]["T"].iloc[0])
        r = best(d, T)
        c = [X.NAMES[d], th(r["N"]), th(r["K"]), th(r["T"]), num(r["ms"]), num(r["MB"])]
        c += [ratio(r[f + x]) + r"$^{\mathrm{" + r[f + x + "f"] + "}}$" for f in ["MinT-Shrink", "MinT-Fix"]
              for x in ["_t", "_m"]]
        return " & ".join(c) + r"\\"

    head = [r" & & & & \multicolumn{2}{c}{SSJT} & \multicolumn{2}{c}{MinT-Shrink / SSJT} & "
            r"\multicolumn{2}{c}{MinT-Fix / SSJT}\\",
            r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
            r"Dataset & $N$ & $K$ & $T$ & time (ms) & mem. (MB) & time & mem. & time & mem.\\"]
    write("t_timing.tex", table("l rrr rr cc cc", head, [row(d) for d in X.MAIN], sep="4pt"))
    write("t_timing_small.tex", table("l rrr rr cc cc", head, [row(d) for d in X.SMALL], sep="4pt"))


# ------------------------------------------------------------------------------------------- accuracy, real
def real_tables():
    D = pd.read_csv(R / "accuracy_real.csv")
    A = D.groupby(["data", "method"]).RMSE.mean().unstack()
    ori = D.groupby("data").origin.nunique()

    def rows(ds, with_pct):
        out = []
        for d in ds:
            r = A.loc[d]; bst = min(r[m] for m in M)
            c = [X.NAMES[d], MOD[d], str(ori[d])] + [B(num(r[m]), r[m] == bst) for m in M]
            if with_pct:
                c += [pct((r["SSJT"] / r[m] - 1) * 100) for m in M[:-1]]
            out.append(" & ".join(c) + r"\\")
        return out

    head = [r" & & & \multicolumn{4}{c}{RMSE (all nodes)} & \multicolumn{3}{c}{$\Delta$ of SSJT (\%) against}\\",
            r"\cmidrule(lr){4-7}\cmidrule(lr){8-10}",
            r"Dataset & Base model & $O$ & Base & MinT-Fix & MinT-Shrink & SSJT & Base & MinT-Fix & MinT-Shrink\\"]
    write("t_real.tex", table("l l r rrrr rrr", head, rows(X.MAIN, True), sep="4pt"))
    write("t_real_small.tex", table("l l r rrrr",
                                    [r"Dataset & Base model & $O$ & Base & MinT-Fix & MinT-Shrink & SSJT\\"],
                                    rows(X.SMALL, False)))


# ------------------------------------------------------------------------------------------- window
def window_table():
    H = pd.read_csv(R / "hourly_window.csv")
    body = []
    for d in ["bdg2_hourly", "largest_hourly"]:
        for T in (500, 2000, 6000):
            g = H[(H.data == d) & (H["T"] == T)]
            v = {m: g[g.method == m].RMSE.mean() for m in ["MinT-Fix", "MinT-Shrink", "SSJT"]}
            bst = min(v.values())
            body.append(" & ".join([X.NAMES[d], th(T), dec(g.lambda_ss.groupby(g.origin).first().mean(), 4)] +
                                   [B(num(v[m]), v[m] == bst) for m in v]) + r"\\")
        base = H[(H.data == d) & (H.method == "Base")].groupby("origin").RMSE.first().mean()
        body.append(r"\multicolumn{3}{l}{\quad base forecasts (independent of $T$)} & \multicolumn{3}{c}{"
                    + num(base) + r"}\\")
        if d == "bdg2_hourly":
            body.append(r"\midrule")
    write("t_Tacc.tex", table("l r r rrr", [r"Dataset & $T$ & estimated $\lambda$ & MinT-Fix & MinT-Shrink & SSJT\\"],
                              body, r"\small"))


# ------------------------------------------------------------------------------------------- grid
def grid_table():
    body, pv, cells = [], [], []
    G = {k: pd.read_csv(R / f"grid_{k}.csv") for k in ("balanced", "imbalanced")}
    ref = ["Base", "MinT-Fix", "MinT-Shrink"]
    for k, d in G.items():
        p = d.pivot_table(index=["topology", "T", "rep", "model"], columns="method", values="RMSE")
        for model in X.GRID_MODELS:
            q = p.xs(model, level="model")
            for m in ref:
                pv.append(wilcoxon(q["SSJT"], q[m]).pvalue)
                cells.append(((q["SSJT"].mean() / q[m].mean() - 1) * 100, len(q)))
    ph = holm(pv); i = 0
    for k, kn in [("balanced", "Balanced"), ("imbalanced", "Imbalanced")]:
        for model in X.GRID_MODELS:
            qm = G[k][G[k].model == model].groupby("method").RMSE.mean()
            v = [qm[m] for m in M]; bst = min(v); cs = []
            for _ in ref:
                rel, n = cells[i]
                cs.append(pct(rel, 2) + ("$^{\\ast}$" if ph[i] < 0.05 else "")); i += 1
            body.append(" & ".join([kn, model, str(n)] + [B(dec(x, 3), x == bst) for x in v] + cs) + r"\\")
    head = [r" & & & \multicolumn{4}{c}{Mean RMSE (all nodes)} & \multicolumn{3}{c}{$\Delta$ of SSJT (\%) against}\\",
            r"\cmidrule(lr){4-7}\cmidrule(lr){8-10}",
            r"Trees & Base model & Runs & Base & MinT-Fix & MinT-Shrink & SSJT & Base & MinT-Fix & MinT-Shrink\\"]
    write("t_grid.tex", table("l l r rrrr rrr", head, body, sep="4pt"))


# ------------------------------------------------------------------------------------------- specification
def spec_table():
    freq = {"MS": "monthly", "QS": "quarterly", "D": "daily", "h": "hourly"}
    body = []
    order = ["TourismLarge", "bdg2", "bdg2_hourly", "largest", "largest_hourly", "m5_store", "m5_product"]
    for d in order + [None] + X.SMALL:
        if d is None:
            body.append(r"\midrule"); continue
        ds = X.load_dataset(d)
        h = ds["h"]
        br = h.branching[h.branching > 0]
        depth = int(h.depth) + 1
        Ts = sorted({X.load_origin(ds, o)[1].shape[0] for o in range(len(ds["origins"]))})
        Tr = th(Ts[0]) if len(Ts) == 1 else f"{Ts[0]}--{Ts[-1]}"
        name = X.NAMES[d] + ("$^{a}$" if d == "TourismLarge" else "")
        if "hourly" in d:
            Tr += "$^{b}$"
        body.append(" & ".join([name, th(h.N), th(h.K), th(h.M), str(depth),
                                f"{br.min()} / {br.mean():.1f} / {br.max()}", th(ds["Y"].shape[0]),
                                f"{freq[ds['freq']]} ({ds['season']})", str(ds["H"]), str(len(ds["origins"])),
                                Tr, MOD[d]]) + r"\\")
    head = [r"Dataset & $N$ & $K$ & $M$ & Levels & $b$ (min / mean / max) & Length & Frequency (season) & "
            r"$H$ & $O$ & $T$ & Base model\\"]
    write("t_spec.tex", table("l rrrr l r l rr r l", head, body, r"\scriptsize", "3pt"))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    grid_table(); timing_tables(); real_tables(); window_table(); spec_table()
    print("tables written to", OUT)
