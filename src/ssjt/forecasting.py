"""Base forecasts and in-sample residuals with StatsForecast (Section 6.3)."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("NIXTLA_ID_AS_COL", "1")


def base_forecasts(Y_train: np.ndarray, h: int, model: str, season: int = 1, freq: str = "D",
                   n_jobs: int = -1, return_sigma: bool = False, batch: int = 4000):
    """Fit one univariate model per column of ``Y_train`` (T x N).

    Returns
    -------
    Yhat : (N, h) forecasts
    E    : (T', N) in-sample one-step residuals (rows with missing fitted values dropped;
           any remaining gaps set to 0)
    Sigma: (N, h) h-step standard deviations implied by each base model's 68.27% prediction interval
           (only if ``return_sigma``)
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS

    T, N = Y_train.shape
    if N > batch:                          # bounded memory for large hierarchies (e.g. M5)
        parts = [base_forecasts(Y_train[:, i:i + batch], h, model, season, freq, n_jobs, return_sigma, batch)
                 for i in range(0, N, batch)]
        lead = min(p[1].shape[0] for p in parts)
        E = np.hstack([p[1][-lead:] for p in parts])
        out = (np.vstack([p[0] for p in parts]), E)
        return out + ((np.vstack([p[2] for p in parts]),) if return_sigma else ())
    if model == "AutoETS":
        mdl = AutoETS(season_length=season)
    elif model == "ETS-ANA":                 # fixed ETS(A,N,A); used where AutoETS is too slow (LargeST)
        mdl = AutoETS(season_length=season, model="ANA")
    elif model == "AutoARIMA":
        mdl = AutoARIMA(season_length=season)
    else:
        raise ValueError(model)
    ds = pd.date_range("2000-01-01", periods=T, freq=freq)
    df = pd.DataFrame({"unique_id": np.repeat(np.arange(N), T), "ds": np.tile(ds, N),
                       "y": Y_train.T.ravel()})
    sf = StatsForecast(models=[mdl], freq=freq, n_jobs=n_jobs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = sf.forecast(df=df, h=h, fitted=True, **({"level": [68.27]} if return_sigma else {}))
        fv = sf.forecast_fitted_values()
    col = [c for c in fc.columns if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c][0]
    fc = fc.sort_values(["unique_id", "ds"])
    Yhat = fc[col].to_numpy().reshape(N, h)
    fv = fv.sort_values(["unique_id", "ds"])
    fitted = fv[col].to_numpy().reshape(N, T).T
    E = Y_train - fitted
    bad = ~np.isfinite(E)
    lead = 0
    while lead < T and bad[lead].any():
        lead += 1
    E = E[lead:]
    E[~np.isfinite(E)] = 0.0
    Yhat[~np.isfinite(Yhat)] = 0.0
    if return_sigma:
        Sig = ((fc[f"{col}-hi-68.27"] - fc[f"{col}-lo-68.27"]).to_numpy() / 2.0).reshape(N, h)
        Sig[~np.isfinite(Sig)] = 0.0
        return Yhat, E, Sig
    return Yhat, E
