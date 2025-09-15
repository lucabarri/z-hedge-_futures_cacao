from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Hedge:
    alpha: float
    beta: float


def ols_hedge(cc_prices: pd.Series, c_prices: pd.Series) -> Hedge:
    df = pd.DataFrame({"cc": cc_prices, "c": c_prices}).dropna()
    x = np.vstack([np.ones(len(df)), df["c"].values]).T
    y = df["cc"].values
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    return Hedge(alpha=alpha, beta=beta)


def spread_series(cc_prices: pd.Series, c_prices: pd.Series, hedge: Hedge) -> pd.Series:
    df = pd.DataFrame({"cc": cc_prices, "c": c_prices}).dropna()
    sp = df["cc"] - (hedge.alpha + hedge.beta * df["c"])
    sp.name = "spread"
    return sp


def zscore(series: pd.Series, window: int = 252) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std(ddof=0)
    return (series - roll_mean) / roll_std


def adf_pvalue(series: pd.Series) -> float:
    from statsmodels.tsa.stattools import adfuller

    s = series.dropna()
    if len(s) < 20:
        raise ValueError("Series too short for ADF test (need >=20 observations).")
    result = adfuller(s, autolag="AIC")
    return float(result[1])


def cointegration_test(cc_prices: pd.Series, c_prices: pd.Series) -> Tuple[Hedge, pd.Series, Optional[float]]:
    df = pd.DataFrame({"cc": cc_prices, "c": c_prices}).dropna()
    hedge = ols_hedge(df["cc"], df["c"])
    sp = spread_series(df["cc"], df["c"], hedge)
    pval = adf_pvalue(sp)
    return hedge, sp, pval
