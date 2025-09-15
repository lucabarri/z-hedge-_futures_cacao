#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import argparse

import pandas as pd

# Ensure src/ is on path when running from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Plotting disabled: no figure generation/saving

from config import Config
from rainfall import build_observed_signals_csv, load_monthly_precip_csv
from rain_signal_eval import (
    EvalParams,
    load_prices,
    compute_forward_returns,
    bucket_stats,
    correlations,
    fit_ols_nw,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rainfall signals vs C/CC/spread"
    )
    parser.add_argument("--lag", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=20)
    # Month fixed effects disabled by default to keep outputs light and consistent
    args = parser.parse_args()

    cfg = Config()
    # Ensure dirs (organized outputs)
    (cfg.paths.results / "signals").mkdir(parents=True, exist_ok=True)
    tables_dir = cfg.paths.tables / "rain_signals"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Ensure rainfall signals exist; if not, build from local monthly POWER CSV
    sig_path = cfg.paths.results / "signals" / "rain_percentile_signals.csv"
    if not sig_path.exists():
        print("Signals not found; building from local monthly POWER CSV...")
        try:
            monthly = load_monthly_precip_csv(cfg)
        except FileNotFoundError:
            # Try to build tidy monthly from raw POWER point file
            from power_point import build_power_point_tidy

            raw_dir = cfg.paths.root / "data" / "raw"
            cands = sorted(raw_dir.glob("POWER_Point_Monthly_*.csv"))
            if not cands:
                raise SystemExit(
                    "No local monthly precip CSV and no POWER point raw file found. Place POWER_Point_Monthly_*.csv under data/raw/."
                )
            tidy_path = (
                cfg.paths.root / "data" / "derived" / "power_monthly_civ_gha.csv"
            )
            build_power_point_tidy(cands[-1], tidy_path)
            monthly = pd.read_csv(tidy_path)
        tmp = cfg.paths.root / "data" / "derived" / "power_monthly_civ_gha.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        monthly.to_csv(tmp, index=False)
        sig_path = build_observed_signals_csv(cfg, monthly_csv=tmp)
    signals = pd.read_csv(sig_path)

    # Load prices (C, CC) and FX if allowed
    cc, c, fx_available = load_prices(cfg)

    # Compute forward returns for each signal
    ep = EvalParams(lag=args.lag, horizon=args.horizon)
    panel = compute_forward_returns(signals, cc, c, ep, fx_available)
    # Save merged panel
    out_panel = cfg.paths.results / "signals" / "rain_signals_panel.csv"
    panel.to_csv(out_panel, index=False)

    # Bucket stats and correlations
    tbls = {}
    for target in ["fwd_cc", "fwd_c", "fwd_spread_usd"]:
        tbls[target] = bucket_stats(panel, target)
        tbls[target].to_csv(tables_dir / f"bucket_{target}.csv", index=False)

    corr = correlations(panel, ["fwd_cc", "fwd_c", "fwd_spread_usd"])
    corr.to_csv(tables_dir / "correlations.csv", index=False)

    # OLS with HAC
    panel["valid_month"] = panel["valid_month"].astype(str)
    for target in ["fwd_cc", "fwd_c", "fwd_spread_usd"]:
        res, tidy = fit_ols_nw(panel, target, with_month_fe=False, ep=ep)
        if tidy is not None and not tidy.empty:
            # Include adj R2 in a footer row
            adj = tidy.attrs.get("adj_r2")
            tidy.to_csv(tables_dir / f"ols_{target}.csv", index=False)
            if adj is not None:
                with open(tables_dir / f"ols_{target}.csv", "a", encoding="utf-8") as f:
                    f.write(f"adj_r2,{adj}\n")

    print("Saved:", out_panel)
    print("Saved tables to:", tables_dir)


if __name__ == "__main__":
    main()
