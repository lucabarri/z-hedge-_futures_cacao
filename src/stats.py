from __future__ import annotations

from typing import Tuple

import pandas as pd


def basic_summary(df: pd.DataFrame, price_col: str = "close", volume_col: str | None = None) -> dict:
    out = {
        "n_obs": int(len(df)),
        "start": str(df["date"].min()) if "date" in df.columns else "",
        "end": str(df["date"].max()) if "date" in df.columns else "",
        "mean_price": float(pd.to_numeric(df[price_col], errors="coerce").mean()),
        "std_price": float(pd.to_numeric(df[price_col], errors="coerce").std(ddof=0)),
    }
    if volume_col and volume_col in df.columns:
        out["mean_volume"] = float(pd.to_numeric(df[volume_col], errors="coerce").mean())
    return out


def overlaps_and_corr(cc: pd.DataFrame, c: pd.DataFrame) -> Tuple[int, float, float]:
    x = cc.set_index("date")["close"].astype(float)
    y = c.set_index("date")["close"].astype(float)
    idx = x.index.intersection(y.index)
    x = x.loc[idx]
    y = y.loc[idx]
    overlaps = int(len(idx))
    price_corr = float(x.corr(y)) if overlaps > 1 else float("nan")
    r_x = x.pct_change(fill_method=None)
    r_y = y.pct_change(fill_method=None)
    ret_corr = float(r_x.corr(r_y)) if overlaps > 2 else float("nan")
    return overlaps, price_corr, ret_corr


def to_dataframe(cc_sum: dict, c_sum: dict) -> pd.DataFrame:
    return pd.DataFrame([{"contract": "CC", **cc_sum}, {"contract": "C", **c_sum}])
