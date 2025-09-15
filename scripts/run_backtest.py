#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import argparse
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

# Ensure src/ is on path when running from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config
import data_io, fx, preprocess, spread, backtest, reporting


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pairs backtest; optionally apply best params and training-only hedge; compare standard vs causal alignment"
    )
    # Always use tuned best-params if available under results/backtests/tuning/
    # Best params are read from the default tuning paths if --use-best-params is set
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--cost-bps", type=float, default=12.0)
    parser.add_argument(
        "--compare-causal",
        action="store_true",
        help="Also compute causal (time-aligned) variant and save under *_causal.csv",
    )
    # Optional threshold tilting based on monthly precip with causal lag (auto-configured)
    parser.add_argument(
        "--tilt",
        action="store_true",
        help="Enable auto threshold tilting using precip_mm signal (lag and direction inferred from training window)",
    )
    args = parser.parse_args()

    cfg = Config()

    cc = data_io.load_local_default_cc(cfg.paths.root)
    c = data_io.load_local_default_c(cfg.paths.root)
    if cc is None or c is None:
        raise SystemExit(
            "Missing dati_cacao CSVs. Expected under data/processed/. Run scripts/build_datasets.py if you have data/raw/dati_cacao.xlsx."
        )

    fx_df = fx.get_gbpusd_fx(
        cfg.paths.data_external / cfg.fx.cache_filename,
        cfg.fx.ticker,
        cfg.fx.start,
        cfg.fx.end,
    )
    c_conv = preprocess.merge_c_with_fx(c, fx_df)
    cc_al, c_al, _ = preprocess.align_common_dates(cc, c_conv)
    cc_al = cc_al.set_index("date").sort_index()
    c_al = c_al.set_index("date").sort_index()

    # Hedge: training-only if requested
    if args.train_start and args.train_end:
        h = spread.ols_hedge(
            cc_al.loc[args.train_start : args.train_end, "close"],
            c_al.loc[args.train_start : args.train_end, "close"],
        )
    else:
        h = spread.ols_hedge(cc_al["close"], c_al["close"])
    # Z-score window fixed for simplicity
    z_window = 252
    sp = spread.spread_series(cc_al["close"], c_al["close"], h)
    z = spread.zscore(sp, window=z_window)

    # Optional: build tilted z based on monthly precip signal (auto lag/sign)
    def build_tilted_z(
        z_in: pd.Series, index_for_counts: pd.DatetimeIndex
    ) -> tuple[pd.Series, dict]:
        meta: dict = {}
        regime_daily = pd.Series(0, index=index_for_counts)
        if not args.tilt:
            return z_in, meta
        panel_path = Path(cfg.paths.results / "signals" / "power_point_panel.csv")
        if not panel_path.exists():
            raise SystemExit("[tilt] Missing power_point_panel.csv. Run scripts/build_power_point_panel.py first.")
        panel = pd.read_csv(panel_path, parse_dates=["date"])  # type: ignore[list-item]
        panel = panel.sort_values("date").reset_index(drop=True)
        feat = panel["precip_mm"].astype(float)
        targ = panel["ret_spread_m"].astype(float)
        dates_m = panel["date"]

        # Select training window on monthly panel if provided
        if args.train_start and args.train_end:
            mask_train_m = (dates_m >= pd.Timestamp(args.train_start)) & (
                dates_m <= pd.Timestamp(args.train_end)
            )
        else:
            mask_train_m = pd.Series(True, index=panel.index)

        best_lag = None
        best_p = None
        best_sign = 0.0
        min_obs = 36
        for lag in range(0, 3):
            f = feat.copy()
            r = targ.copy().shift(-lag)
            d = pd.DataFrame({"f": f, "r": r})
            d = d.loc[mask_train_m].dropna()
            if len(d) < min_obs:
                continue
            try:
                corr, pval = pearsonr(d["f"], d["r"])  # type: ignore
            except Exception:
                continue
            if best_p is None or (pval < best_p):
                best_p = float(pval)
                best_lag = int(lag)
                best_sign = float(
                    np.sign(corr) if np.isfinite(corr) and corr != 0 else 0.0
                )

        if best_lag is None:
            best_lag = 2
            # infer sign on full window at fallback lag
            d_full = pd.DataFrame({"f": feat, "r": targ.shift(-best_lag)}).dropna()
            try:
                corr_full, _ = pearsonr(d_full["f"], d_full["r"])  # type: ignore
                best_sign = float(
                    np.sign(corr_full)
                    if np.isfinite(corr_full) and corr_full != 0
                    else 0.0
                )
            except Exception:
                best_sign = 0.0

        # Quantiles from training window (fallback to full if insufficient)
        f_train = feat.loc[mask_train_m]
        if f_train.dropna().size < min_obs:
            f_train = feat
        q_lo, q_hi = float(f_train.quantile(0.2)), float(f_train.quantile(0.8))
        # Monthly regime: +1 long-favored, -1 short-favored, 0 neutral
        regime_m = pd.Series(0, index=panel.index, dtype=int)
        if best_sign >= 0:
            regime_m = np.where(feat >= q_hi, 1, np.where(feat <= q_lo, -1, 0))
        else:
            regime_m = np.where(feat >= q_hi, -1, np.where(feat <= q_lo, 1, 0))
        regime_m = pd.Series(regime_m, index=dates_m)
        # Shift by lag (feature at t applies to returns at t+lag)
        regime_m_shift = regime_m.shift(best_lag)
        # Map to daily by forward-fill
        regime_daily = (
            regime_m_shift.reindex(z_in.index, method="ffill").fillna(0).astype(int)
        )

        # Apply tilting to z
        favor = 0.8
        unfavor = 1.2
        z_out = z_in.copy()
        # Long-favored days: make long entries easier (neg z / favor) and short harder (pos z / unfavor)
        mask_long = regime_daily == 1
        if mask_long.any():
            mpos = mask_long & (z_out > 0)
            mneg = mask_long & (z_out < 0)
            z_out.loc[mpos] = z_out.loc[mpos] / unfavor
            z_out.loc[mneg] = z_out.loc[mneg] / favor
        # Short-favored days: make short entries easier (pos z / favor) and long harder (neg z / unfavor)
        mask_short = regime_daily == -1
        if mask_short.any():
            mpos = mask_short & (z_out > 0)
            mneg = mask_short & (z_out < 0)
            z_out.loc[mpos] = z_out.loc[mpos] / favor
            z_out.loc[mneg] = z_out.loc[mneg] / unfavor

        # Meta
        meta = {
            "tilt_on": True,
            "tilt_lag": int(best_lag),
            "tilt_sign": "positive" if best_sign >= 0 else "negative",
            "tilt_q_lo": 0.2,
            "tilt_q_hi": 0.8,
            "tilt_favor": favor,
            "tilt_unfavor": unfavor,
            "tilt_days_long_fav": int((regime_daily == 1).sum()),
            "tilt_days_short_fav": int((regime_daily == -1).sum()),
            "tilt_days_neutral": int((regime_daily == 0).sum()),
        }
        return z_out, meta

    z_std, tilt_meta_std = build_tilted_z(z, z.index)

    # Thresholds: allow distinct best-params for standard vs causal
    def _load_best(path_str: str) -> tuple[float | None, float | None, float | None]:
        try:
            p = Path(path_str)
            if not p.exists():
                return (None, None, None)
            best = pd.read_csv(p).iloc[0]
            e = float(best["entry_z"]) if "entry_z" in best else None
            x = float(best["exit_z"]) if "exit_z" in best else None
            s = float(best["stop_z"]) if "stop_z" in best else None
            return (e, x, s)
        except Exception:
            return (None, None, None)

    # Initialize thresholds (None → will use defaults unless best-params requested)
    entry_std = exit_std = stop_std = None
    entry_cau = exit_cau = stop_cau = None

    # Always try to load best params from tuning outputs
    std_best = REPO_ROOT / "results" / "backtests" / "tuning" / "best_params_standard.csv"
    cau_best = REPO_ROOT / "results" / "backtests" / "tuning" / "best_params_causal.csv"
    e_s, x_s, s_s = _load_best(str(std_best))
    e_c, x_c, s_c = _load_best(str(cau_best))
    if e_s is not None:
        entry_std = e_s
    if x_s is not None:
        exit_std = x_s
    if s_s is not None:
        stop_std = s_s
    if e_c is not None:
        entry_cau = e_c
    if x_c is not None:
        exit_cau = x_c
    if s_c is not None:
        stop_cau = s_c

    # Fill remaining with defaults
    defaults = backtest.Rules()
    if entry_std is None:
        entry_std = defaults.entry_z
    if exit_std is None:
        exit_std = defaults.exit_z
    if stop_std is None:
        stop_std = defaults.stop_z
    if entry_cau is None:
        entry_cau = defaults.entry_z
    if exit_cau is None:
        exit_cau = defaults.exit_z
    if stop_cau is None:
        stop_cau = defaults.stop_z

    rules_std = backtest.Rules(
        entry_z=entry_std,
        exit_z=exit_std,
        stop_z=stop_std,
        cost_bps_per_leg=args.cost_bps,
    )

    def metrics_slice(s: pd.Series, ann_factor: int) -> dict:
        s = s.dropna()
        if s.empty:
            return {
                "ann_return": float("nan"),
                "ann_vol": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
            }
        ann_ret = float(s.mean() * ann_factor)
        ann_vol = float(s.std(ddof=0) * (ann_factor**0.5))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        eq = (1.0 + s).cumprod()
        max_dd = float((eq / eq.cummax() - 1.0).min())
        return {
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        }

    # Organized output directories
    base_or_tilt = "tilted" if args.tilt else "base"
    std_dir = cfg.paths.backtests / "standard" / base_or_tilt
    std_dir.mkdir(parents=True, exist_ok=True)

    # STANDARD FLOW
    res_std = backtest.backtest_spread(
        cc_al["close"], c_al["close"], z_std, beta=h.beta, rules=rules_std
    )
    res_std.equity.to_csv(std_dir / "equity.csv", index_label="date")
    res_std.pnl.to_csv(std_dir / "pnl.csv", index_label="date")
    res_std.positions.to_csv(std_dir / "positions.csv", index_label="date")
    # No duplicate saves; organized layout used instead
    # ADF p-value (optional)
    adf_note = ""
    pval = None
    _, _, pval = spread.cointegration_test(cc_al["close"], c_al["close"]) 
    meta_std = {
        "mode": "standard",
        "entry_z": entry_std,
        "exit_z": exit_std,
        "stop_z": stop_std,
        "beta": float(h.beta),
        "alpha": float(h.alpha),
        "z_window": z_window,
        "cost_bps": args.cost_bps,
        "adf_pvalue": pval if pval is not None else "",
        "adf_note": adf_note,
    }
    if tilt_meta_std:
        meta_std.update(tilt_meta_std)
    metrics_std_path = std_dir / "metrics.csv"
    if args.train_start and args.train_end and args.test_start and args.test_end:
        idx = res_std.pnl.index
        train_mask = (idx >= pd.Timestamp(args.train_start)) & (
            idx <= pd.Timestamp(args.train_end)
        )
        test_mask = (idx >= pd.Timestamp(args.test_start)) & (
            idx <= pd.Timestamp(args.test_end)
        )
        train_m = metrics_slice(res_std.pnl.loc[train_mask], rules_std.ann_factor)
        test_m = metrics_slice(res_std.pnl.loc[test_mask], rules_std.ann_factor)
        reporting.save_metrics(
            meta_std
            | {f"train_{k}": v for k, v in train_m.items()}
            | {f"test_{k}": v for k, v in test_m.items()},
            metrics_std_path,
        )
    else:
        reporting.save_metrics(meta_std | res_std.stats, metrics_std_path)

    # CAUSAL FLOW (if requested): decision at London t using C_t*FX_{t-1} vs CC_{t-1}; execute at t+1
    if args.compare_causal:
        cau_dir = cfg.paths.backtests / "causal" / base_or_tilt
        cau_dir.mkdir(parents=True, exist_ok=True)
        # Build aligned series for signal
        fx_series = fx_df.set_index("date")["usdgbp_rate"].astype(float).sort_index()
        c_gbp = c_al["close_original_gbp"].astype(float)
        c_usd_sig = (c_gbp * fx_series.shift(1)).dropna()
        cc_sig = cc_al["close"].shift(1).dropna()
        common_sig = c_usd_sig.index.intersection(cc_sig.index)
        c_usd_sig = c_usd_sig.loc[common_sig]
        cc_sig = cc_sig.loc[common_sig]
        # Hedge for causal (training-only if provided, on aligned series)
        if args.train_start and args.train_end:
            h_causal = spread.ols_hedge(
                cc_sig.loc[args.train_start : args.train_end],
                c_usd_sig.loc[args.train_start : args.train_end],
            )
        else:
            h_causal = spread.ols_hedge(cc_sig, c_usd_sig)
        sp_causal = cc_sig - (h_causal.alpha + h_causal.beta * c_usd_sig)
        z_causal = spread.zscore(sp_causal, window=z_window)
        # Apply tilting to causal z as well, using same monthly regime mapping aligned to its index
        z_causal_tilt, tilt_meta_c = build_tilted_z(z_causal, z_causal.index)
        # ADF p-value on causal spread (optional)
        adf_note_c = ""
        pval_c = None
        pval_c = spread.adf_pvalue(sp_causal)
        # Backtest with causal z, realized returns from full series
        rules_causal = backtest.Rules(
            entry_z=entry_cau,
            exit_z=exit_cau,
            stop_z=stop_cau,
            cost_bps_per_leg=args.cost_bps,
        )
        res_causal = backtest.backtest_spread(
            cc_al["close"],
            c_al["close"],
            z_causal_tilt,
            beta=h_causal.beta,
            rules=rules_causal,
        )
        res_causal.equity.to_csv(cau_dir / "equity.csv", index_label="date")
        res_causal.pnl.to_csv(cau_dir / "pnl.csv", index_label="date")
        res_causal.positions.to_csv(cau_dir / "positions.csv", index_label="date")
        # No duplicate saves; organized layout used instead
        # Metrics causal
        meta_c = {
            "mode": "causal",
            "entry_z": entry_cau,
            "exit_z": exit_cau,
            "stop_z": stop_cau,
            "beta": float(h_causal.beta),
            "alpha": float(h_causal.alpha),
            "z_window": z_window,
            "cost_bps": args.cost_bps,
            "adf_pvalue": pval_c if pval_c is not None else "",
            "adf_note": adf_note_c,
        }
        if tilt_meta_c:
            meta_c.update(tilt_meta_c)
        metrics_causal_path = cau_dir / "metrics.csv"
        if args.train_start and args.train_end and args.test_start and args.test_end:
            idx2 = res_causal.pnl.index
            train_mask2 = (idx2 >= pd.Timestamp(args.train_start)) & (
                idx2 <= pd.Timestamp(args.train_end)
            )
            test_mask2 = (idx2 >= pd.Timestamp(args.test_start)) & (
                idx2 <= pd.Timestamp(args.test_end)
            )
            train_m2 = metrics_slice(
                res_causal.pnl.loc[train_mask2], rules_causal.ann_factor
            )
            test_m2 = metrics_slice(
                res_causal.pnl.loc[test_mask2], rules_causal.ann_factor
            )
            reporting.save_metrics(
                meta_c
                | {f"train_{k}": v for k, v in train_m2.items()}
                | {f"test_{k}": v for k, v in test_m2.items()},
                metrics_causal_path,
            )
        else:
            reporting.save_metrics(meta_c | res_causal.stats, metrics_causal_path)

    print("Saved standard to:", std_dir)
    if args.compare_causal:
        print("Saved causal to:", cau_dir)


if __name__ == "__main__":
    main()
