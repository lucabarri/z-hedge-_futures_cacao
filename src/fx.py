from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def load_cached_fx(cache_path: Path) -> Optional[pd.DataFrame]:
    if cache_path.exists():
        fx = pd.read_csv(cache_path)
        fx.columns = [c.strip().lower() for c in fx.columns]
        if "date" not in fx.columns:
            fx = fx.rename(columns={fx.columns[0]: "date"})
        if "usdgbp_rate" not in fx.columns:
            # Try generic column naming
            for candidate in ("close", "rate", "usdgbp", "gbpusd"):
                if candidate in fx.columns:
                    fx = fx.rename(columns={candidate: "usdgbp_rate"})
                    break
        fx["date"] = pd.to_datetime(fx["date"])  # type: ignore[index]
        fx = fx.sort_values("date").reset_index(drop=True)
        return fx[["date", "usdgbp_rate"]]
    return None


def download_fx(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download GBPUSD FX daily closes via yfinance.

    Returns DataFrame with columns ['date', 'usdgbp_rate'] or None on failure.
    """
    fx = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if fx is None or fx.empty:
        return None
    fx = fx[["Close"]].reset_index()
    fx.columns = ["date", "usdgbp_rate"]
    fx["date"] = pd.to_datetime(fx["date"])  # type: ignore[index]
    fx = fx.sort_values("date").reset_index(drop=True)
    return fx


def get_gbpusd_fx(
    cache_path: Path,
    ticker: str,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Load GBPUSD daily FX rates.

    Always attempts a fresh download and overwrites the cache; if the download
    fails, falls back to the cached file if present.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fx = download_fx(ticker=ticker, start=start, end=end)
    if fx is not None and not fx.empty:
        fx.to_csv(cache_path, index=False)
        return fx
    return load_cached_fx(cache_path)
