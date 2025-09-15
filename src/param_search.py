from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import preprocess, spread, backtest


@dataclass
class Split:
    train_start: str
    train_end: str
    test_start: str
    test_end: Optional[str] = None  # None = to end


def _mask_by_dates(idx: pd.DatetimeIndex, start: str, end: Optional[str]) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else None
    if end_ts is None:
        return (idx >= start_ts)
    return (idx >= start_ts) & (idx <= end_ts)


def _metrics_from_slice(pnl: pd.Series, pos: pd.Series, ann_factor: int = 252) -> dict:
    s = pnl.dropna()
    if s.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan, "max_drawdown": np.nan, "trades": 0}
    ann_ret = float(s.mean() * ann_factor)
    ann_vol = float(s.std(ddof=0) * np.sqrt(ann_factor))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    eq = (1.0 + s.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min())
    # Count trades within the same slice
    turns = (pos != pos.shift(1)).astype(float)
    turns = turns.loc[s.index]
    trades = int(turns.sum(skipna=True))
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd, "trades": trades}


def run_grid_search(
    cc: pd.DataFrame,
    c: pd.DataFrame,
    split: Split,
    entry_grid: Iterable[float],
    exit_grid: Iterable[float],
    stop_grid: Iterable[float],
    z_window: int = 252,
    cost_bps_per_leg: float = 1.0,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[pd.DataFrame, Tuple[float, float, float], spread.Hedge, pd.Series]:
    """Run a parameter grid search for z-score thresholds and return a summary table.

    Returns (summary_df, best_triple, hedge, z_series)
    """
    # Align common dates and set index
    cc_al, c_al, _ = preprocess.align_common_dates(cc, c)
    cc_al = cc_al.set_index("date").sort_index()
    c_al = c_al.set_index("date").sort_index()

    # Hedge parameters from training period only
    train_mask = _mask_by_dates(cc_al.index, split.train_start, split.train_end)
    cc_train = cc_al.loc[train_mask]
    c_train = c_al.loc[train_mask]
    h = spread.ols_hedge(cc_train["close"], c_train["close"])  # alpha, beta

    # Build price-level spread with training hedge across full series and zscore
    sp = spread.spread_series(cc_al["close"], c_al["close"], h)
    z = spread.zscore(sp, window=z_window)

    # Prepare masks on the z index (since pnl/positions align to z)
    train_mask_z = _mask_by_dates(z.index, split.train_start, split.train_end)
    test_mask_z = _mask_by_dates(z.index, split.test_start, split.test_end)

    results = []
    for entry in entry_grid:
        for exit_ in exit_grid:
            for stop in stop_grid:
                rules = backtest.Rules(entry_z=entry, exit_z=exit_, stop_z=stop, cost_bps_per_leg=cost_bps_per_leg)
                res = backtest.backtest_spread(
                    cc_close=cc_al["close"],
                    c_close=c_al["close"],
                    z=z,
                    beta=h.beta,
                    rules=rules,
                )
                # Slice metrics
                train_stats = _metrics_from_slice(res.pnl.loc[train_mask_z], res.positions.loc[train_mask_z], ann_factor=rules.ann_factor)
                test_stats = _metrics_from_slice(res.pnl.loc[test_mask_z], res.positions.loc[test_mask_z], ann_factor=rules.ann_factor)

                row = {
                    "entry_z": entry,
                    "exit_z": exit_,
                    "stop_z": stop,
                    "train_ann_return": train_stats["ann_return"],
                    "train_ann_vol": train_stats["ann_vol"],
                    "train_sharpe": train_stats["sharpe"],
                    "train_max_drawdown": train_stats["max_drawdown"],
                    "train_trades": train_stats["trades"],
                    "test_ann_return": test_stats["ann_return"],
                    "test_ann_vol": test_stats["ann_vol"],
                    "test_sharpe": test_stats["sharpe"],
                    "test_max_drawdown": test_stats["max_drawdown"],
                    "test_trades": test_stats["trades"],
                }
                results.append(row)
                if progress is not None:
                    progress(1)

    summary = pd.DataFrame(results)
    # Pick best by training Sharpe, tie-break by train_ann_return
    summary = summary.sort_values(["train_sharpe", "train_ann_return"], ascending=[False, False]).reset_index(drop=True)
    if summary.empty:
        best = (np.nan, np.nan, np.nan)
    else:
        r0 = summary.iloc[0]
        best = (float(r0["entry_z"]), float(r0["exit_z"]), float(r0["stop_z"]))
    return summary, best, h, z


def run_grid_search_causal(
    cc: pd.DataFrame,
    c: pd.DataFrame,
    split: Split,
    entry_grid: Iterable[float],
    exit_grid: Iterable[float],
    stop_grid: Iterable[float],
    z_window: int = 252,
    cost_bps_per_leg: float = 1.0,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[pd.DataFrame, Tuple[float, float, float], spread.Hedge, pd.Series]:
    """Grid search using causal alignment for signal z (CC_{t-1} vs CUSD_t with FX_{t-1})."""
    # Align common dates
    cc_al, c_al, _ = preprocess.align_common_dates(cc, c)
    cc_ser = cc_al.set_index("date")["close"].astype(float).sort_index()
    # Build CUSD signal with lagged FX if available
    if "close_original_gbp" in c_al.columns and "usdgbp_rate" in c_al.columns:
        c_usd_sig = (c_al.set_index("date")["close_original_gbp"].astype(float) * c_al.set_index("date")["usdgbp_rate"].astype(float).shift(1)).dropna()
    else:
        c_usd_sig = c_al.set_index("date")["close"].astype(float)
    cc_sig = cc_ser.shift(1).dropna()
    idx = cc_sig.index.intersection(c_usd_sig.index)
    cc_sig = cc_sig.loc[idx]
    c_usd_sig = c_usd_sig.loc[idx]

    # Masks on signal index
    train_mask = _mask_by_dates(idx, split.train_start, split.train_end)
    test_mask = _mask_by_dates(idx, split.test_start, split.test_end)

    # Hedge on signal series (training only)
    h = spread.ols_hedge(cc_sig.loc[train_mask], c_usd_sig.loc[train_mask])
    sp = cc_sig - (h.alpha + h.beta * c_usd_sig)
    z = spread.zscore(sp, window=z_window)

    # Realized close series aligned to z index
    cc_full = cc_ser.reindex(z.index)
    c_full = c_al.set_index("date")["close"].astype(float).reindex(z.index)

    results = []
    for entry in entry_grid:
        for exit_ in exit_grid:
            for stop in stop_grid:
                rules = backtest.Rules(entry_z=entry, exit_z=exit_, stop_z=stop, cost_bps_per_leg=cost_bps_per_leg)
                res = backtest.backtest_spread(
                    cc_close=cc_full,
                    c_close=c_full,
                    z=z,
                    beta=h.beta,
                    rules=rules,
                )
                train_stats = _metrics_from_slice(res.pnl.loc[train_mask], res.positions.loc[train_mask], ann_factor=rules.ann_factor)
                test_stats = _metrics_from_slice(res.pnl.loc[test_mask], res.positions.loc[test_mask], ann_factor=rules.ann_factor)
                results.append({
                    "entry_z": entry,
                    "exit_z": exit_,
                    "stop_z": stop,
                    "train_ann_return": train_stats["ann_return"],
                    "train_ann_vol": train_stats["ann_vol"],
                    "train_sharpe": train_stats["sharpe"],
                    "train_max_drawdown": train_stats["max_drawdown"],
                    "train_trades": train_stats["trades"],
                    "test_ann_return": test_stats["ann_return"],
                    "test_ann_vol": test_stats["ann_vol"],
                    "test_sharpe": test_stats["sharpe"],
                    "test_max_drawdown": test_stats["max_drawdown"],
                    "test_trades": test_stats["trades"],
                })
                if progress is not None:
                    progress(1)

    summary = pd.DataFrame(results).sort_values(["train_sharpe", "train_ann_return"], ascending=[False, False]).reset_index(drop=True)
    if summary.empty:
        best = (np.nan, np.nan, np.nan)
    else:
        r0 = summary.iloc[0]
        best = (float(r0["entry_z"]), float(r0["exit_z"]), float(r0["stop_z"]))
    return summary, best, h, z
