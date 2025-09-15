from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Config
import data_io
import preprocess
import spread
import fx as fx_mod


@dataclass
class EvalParams:
    lag: int = 15
    horizon: int = 20
    hac_maxlags: Optional[int] = None  # if None, set to horizon//2 or 1


def _hac_lags(ep: EvalParams) -> int:
    return max(1, ep.horizon // 2) if ep.hac_maxlags is None else ep.hac_maxlags


def load_prices(
    cfg: Optional[Config] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    if cfg is None:
        cfg = Config()
    root = cfg.paths.root
    cc = data_io.load_local_default_cc(root)
    c = data_io.load_local_default_c(root)
    if cc is None or c is None:
        raise FileNotFoundError(
            "Missing dati_cacao CSVs. Expected under data/processed/. Run scripts/build_datasets.py if you have data/raw/dati_cacao.xlsx."
        )
    # Normalize
    cc = cc[["date", "close"]].copy()
    c = c[["date", "close"]].copy()
    cc["date"] = pd.to_datetime(cc["date"])  # type: ignore[index]
    c["date"] = pd.to_datetime(c["date"])  # type: ignore[index]

    # Attempt FX conversion for C
    fx_df = fx_mod.get_gbpusd_fx(
        cache_path=cfg.paths.data_external / cfg.fx.cache_filename,
        ticker=cfg.fx.ticker,
        start=cfg.fx.start,
        end=cfg.fx.end,
    )
    c_conv = preprocess.merge_c_with_fx(c, fx_df)
    fx_available = ("usdgbp_rate" in c_conv.columns) and bool(
        c_conv["usdgbp_rate"].notna().any()
    )
    # If FX available, use USD close
    if fx_available and "close_usd" in c_conv.columns:
        c_conv["close"] = c_conv["close_usd"]
    return cc, c_conv, fx_available


def forward_return_over_window(
    s: pd.Series, start_date: pd.Timestamp, lag: int, horizon: int
) -> float:
    s = s.sort_index()
    # backfill to next trading day for anchor
    start_idx = s.index.get_indexer([start_date], method="backfill")[0] + lag
    end_idx = start_idx + horizon
    if end_idx >= len(s):
        return np.nan
    p0 = float(s.iloc[start_idx])
    p1 = float(s.iloc[end_idx])
    if p0 <= 0:
        return np.nan
    return p1 / p0 - 1.0


def compute_beta_for_spread(cc: pd.DataFrame, c_usd: pd.DataFrame) -> float:
    cc_al, c_al, _ = preprocess.align_common_dates(cc, c_usd)
    h = spread.ols_hedge(cc_al["close"], c_al["close"])
    return float(h.beta)


def compute_forward_returns(
    signals: pd.DataFrame,
    cc: pd.DataFrame,
    c: pd.DataFrame,
    ep: EvalParams,
    fx_available: bool,
) -> pd.DataFrame:
    """Attach forward returns for CC, C, and USD spread to each signal row."""
    out = signals.copy()
    out["issue_date"] = pd.to_datetime(out["issue_date"])  # type: ignore[index]
    cc_s = cc.set_index("date")["close"].astype(float)
    c_s = c.set_index("date")["close"].astype(float)

    # Optional spread beta if FX available
    beta = compute_beta_for_spread(cc, c) if fx_available else np.nan

    fwd_cc = []
    fwd_c = []
    fwd_spread = []
    for dt in out["issue_date"]:
        r_cc = forward_return_over_window(cc_s, dt, ep.lag, ep.horizon)
        r_c = forward_return_over_window(c_s, dt, ep.lag, ep.horizon)
        fwd_cc.append(r_cc)
        fwd_c.append(r_c)
        if fx_available and np.isfinite(beta):
            if np.isfinite(r_cc) and np.isfinite(r_c):
                fwd_spread.append(r_cc - beta * r_c)
            else:
                fwd_spread.append(np.nan)
        else:
            fwd_spread.append(np.nan)

    out["fwd_cc"] = fwd_cc
    out["fwd_c"] = fwd_c
    out["fwd_spread_usd"] = fwd_spread
    out["beta_spread"] = beta
    return out


def bucket_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute bucket means for ≤20, (20,80), ≥80 percentiles."""
    low = df[df["percentile"] <= 20]
    mid = df[(df["percentile"] > 20) & (df["percentile"] < 80)]
    high = df[df["percentile"] >= 80]
    return pd.DataFrame(
        {
            "bucket": ["<=20", "20-80", ">=80"],
            col: [low[col].mean(), mid[col].mean(), high[col].mean()],
            "count": [len(low), len(mid), len(high)],
        }
    )


def correlations(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for t in targets:
        s = df[["anom_signed", t]].dropna()
        if len(s) < 3:
            rows.append(
                {
                    "target": t,
                    "pearson_r": np.nan,
                    "pearson_p": np.nan,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                }
            )
            continue
        pr = pearsonr(s["anom_signed"], s[t])
        sr = spearmanr(s["anom_signed"], s[t])
        rows.append(
            {
                "target": t,
                "pearson_r": pr.statistic,
                "pearson_p": pr.pvalue,
                "spearman_r": sr.correlation,
                "spearman_p": sr.pvalue,
            }
        )
    return pd.DataFrame(rows)


def fit_ols_nw(
    df: pd.DataFrame, target: str, with_month_fe: bool, ep: EvalParams
) -> Tuple[Optional[object], pd.DataFrame]:
    try:
        import statsmodels.api as sm
    except Exception:
        return None, pd.DataFrame()

    d = df.copy().dropna(
        subset=[target, "anom_signed", "valid_month"]
    )  # valid_month should exist
    if len(d) < 10:
        return None, pd.DataFrame()

    X_parts = [d[["anom_signed"]]]
    if with_month_fe:
        # month-of-year FE from valid_month
        d["mon"] = d["valid_month"].str.slice(5, 7)
        mon_d = pd.get_dummies(d["mon"], prefix="m", drop_first=True)
        X_parts.append(mon_d)
    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X)
    y = d[target].astype(float)

    model = sm.OLS(y, X, missing="drop")
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": _hac_lags(ep)})
    # Build a tidy table
    params = res.params
    bse = res.bse
    tvals = res.tvalues
    pvals = res.pvalues
    out = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "std_err": bse.values,
            "t": tvals.values,
            "p": pvals.values,
        }
    )
    out.loc[out["term"] == "const", "term"] = "intercept"
    out.attrs["adj_r2"] = float(res.rsquared_adj)
    return res, out
