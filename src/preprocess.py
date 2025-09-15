from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataPair:
    cc: pd.DataFrame
    c: pd.DataFrame


def merge_c_with_fx(c_df: pd.DataFrame, fx_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge C contract with FX and convert close GBP->USD.

    Requires a valid FX DataFrame; raises if missing.
    """
    if fx_df is None or fx_df.empty:
        raise ValueError("FX data is required to convert C (GBP) to USD. Run scripts/build_datasets.py.")
    df = c_df.copy()
    merged = pd.merge(df, fx_df, on="date", how="left")
    merged["usdgbp_rate"] = merged["usdgbp_rate"].ffill()
    merged["close_original_gbp"] = merged["close"]
    merged["close_usd"] = merged["close_original_gbp"] * merged["usdgbp_rate"]
    # Replace close with USD for downstream analysis
    merged["close"] = merged["close_usd"]
    return merged


def align_common_dates(cc_df: pd.DataFrame, c_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Intersect dates and return filtered copies with a shared index."""
    cc = cc_df.copy()
    c = c_df.copy()
    common = pd.Index(sorted(set(cc["date"]) & set(c["date"]))).astype("datetime64[ns]")
    cc = cc[cc["date"].isin(common)].sort_values("date").reset_index(drop=True)
    c = c[c["date"].isin(common)].sort_values("date").reset_index(drop=True)
    return cc, c, common


def add_returns(df: pd.DataFrame, price_col: str = "close", prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}returns"] = out[price_col].pct_change(fill_method=None)
    out[f"{prefix}log_returns"] = np.log(out[price_col] / out[price_col].shift(1))
    return out
