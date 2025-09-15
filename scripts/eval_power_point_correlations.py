#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import argparse

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

"""Evaluate correlations between POWER climate and returns (tables only)."""
# Ensure src/ is on path when running from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config

FEATURES = ["precip_mm", "gwetroot", "t2m_max"]
TARGETS = ["ret_cc_m", "ret_c_usd_m", "ret_spread_m"]


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in FEATURES:
        for targ in TARGETS:
            s = df[[feat, targ]].dropna()
            if len(s) < 3:
                pr = (np.nan, np.nan)
                sr = (np.nan, np.nan)
            else:
                pr = pearsonr(s[feat], s[targ])
                sr = spearmanr(s[feat], s[targ])
            rows.append(
                {
                    "feature": feat,
                    "target": targ,
                    "N": len(s),
                    "pearson_r": pr[0],
                    "pearson_p": pr[1],
                    "spearman_r": sr[0],
                    "spearman_p": sr[1],
                }
            )
    return pd.DataFrame(rows)


# Bucket tables removed to keep outputs light


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate correlations between POWER point climate and monthly returns"
    )
    # Buckets disabled; only correlations written
    # Run one or more lags in a single call, e.g., --lags 0 1 2 (defaults to 0)
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=None,
        help="List of lags to evaluate (space-separated). Default: 0",
    )
    args = parser.parse_args()

    cfg = Config()
    panel_path = REPO_ROOT / "results" / "signals" / "power_point_panel.csv"
    if not panel_path.exists():
        raise SystemExit(
            "Panel not found. Run scripts/build_power_point_panel.py first."
        )
    df = pd.read_csv(panel_path, parse_dates=["date"])  # type: ignore[list-item]

    # Determine lags to run
    lags = args.lags if args.lags is not None else [0]
    for lag in lags:
        dfl = df.copy()
        # Apply lag to targets: feature_t vs return_{t+lag}
        if lag and lag > 0:
            for targ in TARGETS:
                dfl[targ] = dfl[targ].shift(-lag)

        # Output dirs organized by domain and lag
        lag_dir = f"lag{lag}" if lag else "lag0"
        tables_dir = cfg.paths.tables / "power" / lag_dir
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Correlations (tabular only) + meta JSON
        corr = correlations(dfl)
        corr_path = tables_dir / "correlations.csv"
        corr.to_csv(corr_path, index=False)
        try:
            any_sig = bool((corr[["pearson_p", "spearman_p"]] < 0.05).any(axis=None))
            meta = {"any_significant": any_sig, "rows": int(len(corr))}
            (tables_dir / "correlations_meta.json").write_text(
                __import__("json").dumps(meta), encoding="utf-8"
            )
        except Exception:
            pass

        # Buckets disabled

        print("Saved tables to:", tables_dir)


if __name__ == "__main__":
    main()
