from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class Rules:
    entry_z: float = 1.5
    exit_z: float = 0.5
    stop_z: float = 4.0
    ann_factor: int = 252
    # Cost per position change, expressed in basis points applied to total notional (|1| + |beta|)
    # 0.12% per trade = 12 bps
    cost_bps_per_leg: float = 12.0


@dataclass
class Results:
    equity: pd.Series
    pnl: pd.Series
    positions: pd.Series
    stats: Dict[str, float]


def compute_spread_return(
    cc_close: pd.Series, c_close: pd.Series, beta: float
) -> pd.Series:
    r_cc = cc_close.pct_change(fill_method=None)
    r_c = c_close.pct_change(fill_method=None)
    spr = r_cc - beta * r_c
    spr.name = "spread_return"
    return spr


def positions_from_z(z: pd.Series, entry: float, exit: float, stop: float) -> pd.Series:
    pos = pd.Series(0.0, index=z.index)
    current = 0.0
    for i, val in enumerate(z.values):
        if not np.isfinite(val):
            pos.iloc[i] = current
            continue
        if current == 0.0:
            if val >= entry:
                current = -1.0
            elif val <= -entry:
                current = 1.0
        else:
            if abs(val) <= exit:
                current = 0.0
            elif abs(val) >= stop:
                current = 0.0
        pos.iloc[i] = current
    pos.name = "position"
    return pos


def backtest_spread(
    cc_close: pd.Series,
    c_close: pd.Series,
    z: pd.Series,
    beta: float,
    rules: Optional[Rules] = None,
) -> Results:
    if rules is None:
        rules = Rules()

    spr = compute_spread_return(cc_close, c_close, beta)
    pos = positions_from_z(z, rules.entry_z, rules.exit_z, rules.stop_z)
    # Align spread returns to signal index to avoid misalignment
    spr = spr.reindex(pos.index)
    pos = pos.shift(1).fillna(0.0)
    gross = pos.abs() * (1.0 + abs(beta))
    turns = (pos != pos.shift(1)).astype(float)
    cost = turns * (rules.cost_bps_per_leg * 1e-4) * gross
    pnl = pos * spr - cost
    pnl.name = "pnl"
    equity = (1.0 + pnl.fillna(0.0)).cumprod()
    equity.name = "equity"

    ret = pnl
    ann_ret = float(ret.mean() * rules.ann_factor)
    ann_vol = float(ret.std(ddof=0) * np.sqrt(rules.ann_factor))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = equity / equity.cummax() - 1.0
    max_dd = float(dd.min())
    stats = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "trades": int(turns.sum(skipna=True)),
    }
    return Results(equity=equity, pnl=pnl, positions=pos, stats=stats)

