#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import argparse
from typing import List, Tuple

# Ensure src/ is on path when running from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import Config
from rain_signal_eval import load_prices
from param_search import Split, run_grid_search, run_grid_search_causal


def _parse_grid(s: str) -> List[float]:
    if s is None:
        return []
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_range(s: str) -> Tuple[float, float]:
    lo, hi = s.split(":")
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune z-score thresholds for spread backtest with train/test split")
    parser.add_argument("--train-start", default="2011-01-01")
    parser.add_argument("--train-end", default="2017-12-31")
    parser.add_argument("--test-start", default="2018-01-01")
    parser.add_argument("--test-end", default=None)
    # Provide ranges + grid-n for search (explicit grids removed for simplicity)
    parser.add_argument("--entry-range", default="1.0:3.0", help="Entry z range as lo:hi (used if grid not provided)")
    parser.add_argument("--exit-range", default="0.1:1.0", help="Exit z range as lo:hi (used if grid not provided)")
    parser.add_argument("--stop-range", default="2.0:5.0", help="Stop z range as lo:hi (used if grid not provided)")
    parser.add_argument("--grid-n", type=int, default=5, help="Number of points per range (e.g., 5 → 5x5x5)")
    parser.add_argument("--z-window", type=int, default=252)
    # Cost fixed project-wide in code to simplify usage (12 bps per leg)
    args = parser.parse_args()

    cfg = Config()
    (cfg.paths.results / "backtests" / "tuning").mkdir(parents=True, exist_ok=True)

    # Load prices (+FX fresh)
    cc, c, fx_available = load_prices(cfg)
    if not fx_available:
        print("Warning: FX not available; using C in native GBP. USD spread results may be degraded.")

    split = Split(train_start=args.train_start, train_end=args.train_end, test_start=args.test_start, test_end=args.test_end)
    lo, hi = _parse_range(args.entry_range)
    entry = list(np.linspace(lo, hi, num=args.grid_n))
    lo, hi = _parse_range(args.exit_range)
    exit_ = list(np.linspace(lo, hi, num=args.grid_n))
    lo, hi = _parse_range(args.stop_range)
    stop = list(np.linspace(lo, hi, num=args.grid_n))

    total_points = len(entry) * len(exit_) * len(stop)
    bar = tqdm(total=total_points * 2, desc="Tuning (standard+causal)")
    def bump(n=1):
        try:
            bar.update(n)
        except Exception:
            pass

    summary, best, hedge, z = run_grid_search(
        cc=cc,
        c=c,
        split=split,
        entry_grid=entry,
        exit_grid=exit_,
        stop_grid=stop,
        z_window=args.z_window,
        cost_bps_per_leg=12.0,
        progress=bump,
    )

    out_dir = cfg.paths.results / "backtests" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary_standard.csv", index=False)
    best_df = pd.DataFrame({"entry_z": [best[0]], "exit_z": [best[1]], "stop_z": [best[2]], "beta": [hedge.beta], "alpha": [hedge.alpha]})
    best_df.to_csv(out_dir / "best_params_standard.csv", index=False)

    # Causal grid search
    summary_c, best_c, hedge_c, _ = run_grid_search_causal(
        cc=cc,
        c=c,
        split=split,
        entry_grid=entry,
        exit_grid=exit_,
        stop_grid=stop,
        z_window=args.z_window,
        cost_bps_per_leg=12.0,
        progress=bump,
    )
    summary_c.to_csv(out_dir / "summary_causal.csv", index=False)
    best_df_c = pd.DataFrame({"entry_z": [best_c[0]], "exit_z": [best_c[1]], "stop_z": [best_c[2]], "beta": [hedge_c.beta], "alpha": [hedge_c.alpha]})
    best_df_c.to_csv(out_dir / "best_params_causal.csv", index=False)

    bar.close()
    print(f"Grid size: {len(entry)} x {len(exit_)} x {len(stop)} = {total_points}")
    print("Saved:", out_dir / "summary_standard.csv")
    print("Saved:", out_dir / "summary_causal.csv")
    print("Best params (standard):", best)
    print("Best params (causal):", best_c)


if __name__ == "__main__":
    main()
