from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import Config
import data_io
import preprocess
import spread
import fx as fx_mod


MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _find_header_start(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("PARAMETER,"):
            return i
    raise ValueError("PARAMETER header not found in POWER CSV")


def parse_power_point_csv(raw_path: Path) -> pd.DataFrame:
    """Parse a POWER Point Monthly CSV into tidy long format and pivot to wide columns.

    Returns DataFrame with columns: date (month-end), precip_mm, gwetroot, t2m_max.
    - PRECTOTCORR (mm/day) is converted to monthly totals (mm).
    - Missing sentinel -999.0 is treated as NaN.
    """
    start = _find_header_start(raw_path)
    df = pd.read_csv(raw_path, skiprows=start)
    # Keep only needed columns
    cols = ["PARAMETER", "YEAR"] + MONTHS  # ignore ANN
    df = df[cols]
    # Melt to long
    long = df.melt(id_vars=["PARAMETER", "YEAR"], value_vars=MONTHS, var_name="mon", value_name="value")
    # Drop missing sentinel
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long.loc[long["value"] <= -998.9, "value"] = np.nan
    # Build date (use month-end)
    mon_map = {m: i + 1 for i, m in enumerate(MONTHS)}
    long["month"] = long["mon"].map(mon_map)
    long["date"] = pd.to_datetime(long["YEAR"].astype(int).astype(str) + "-" + long["month"].astype(int).astype(str).str.zfill(2)) + pd.offsets.MonthEnd(0)
    # Pivot to columns per parameter
    piv = long.pivot_table(index="date", columns="PARAMETER", values="value", aggfunc="first").sort_index()
    # Convert precip to monthly totals (mm): PRECTOTCORR is mm/day monthly mean
    if "PRECTOTCORR" in piv.columns:
        days = piv.index.days_in_month
        piv["precip_mm"] = piv["PRECTOTCORR"].astype(float) * days
    else:
        piv["precip_mm"] = np.nan
    # Rename other columns
    if "GWETROOT" in piv.columns:
        piv["gwetroot"] = piv["GWETROOT"].astype(float)
    else:
        piv["gwetroot"] = np.nan
    if "T2M_MAX" in piv.columns:
        piv["t2m_max"] = piv["T2M_MAX"].astype(float)
    else:
        piv["t2m_max"] = np.nan
    out = piv[["precip_mm", "gwetroot", "t2m_max"]].reset_index()
    # Add canonical month (YYYY-MM) for downstream consumers
    out["month"] = out["date"].dt.strftime("%Y-%m")
    return out


def build_power_point_tidy(raw_path: Path, out_path: Path) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tidy = parse_power_point_csv(raw_path)
    tidy.to_csv(out_path, index=False)
    return tidy


def monthly_price_panel(cfg: Optional[Config] = None) -> Tuple[pd.DataFrame, bool, float]:
    """Build monthly CC and C (USD) close series and returns, and a USD spread return.

    Returns (panel_df, fx_available, beta)
    Columns: date, cc_close_m, c_usd_close_m, ret_cc_m, ret_c_usd_m, ret_spread_m
    """
    if cfg is None:
        cfg = Config()
    root = cfg.paths.root
    cc = data_io.load_local_default_cc(root)
    c = data_io.load_local_default_c(root)
    if cc is None or c is None:
        raise FileNotFoundError("Missing dati_cacao CSVs. Expected under data/processed/. Run scripts/build_datasets.py if you have data/raw/dati_cacao.xlsx.")
    cc["date"] = pd.to_datetime(cc["date"])  # type: ignore[index]
    c["date"] = pd.to_datetime(c["date"])  # type: ignore[index]

    # FX and conversion
    fx_df = fx_mod.get_gbpusd_fx(cfg.paths.data_external / cfg.fx.cache_filename, cfg.fx.ticker, cfg.fx.start, cfg.fx.end)
    c_conv = preprocess.merge_c_with_fx(c, fx_df)
    fx_available = ("usdgbp_rate" in c_conv.columns) and bool(c_conv["usdgbp_rate"].notna().any())
    c_conv["close_usd"] = c_conv["close_usd"].astype(float)

    # Monthly resample (last obs in month)
    cc_m = cc.set_index("date")["close"].astype(float).resample("ME").last()
    c_usd_m = c_conv.set_index("date")["close_usd"].astype(float).resample("ME").last()
    panel = pd.concat({"cc_close_m": cc_m, "c_usd_close_m": c_usd_m}, axis=1).dropna()
    panel = panel.reset_index().rename(columns={"index": "date"})
    panel["ret_cc_m"] = panel["cc_close_m"].pct_change()
    panel["ret_c_usd_m"] = panel["c_usd_close_m"].pct_change()

    # Beta for spread (entire sample; acceptable for correlation work)
    # Use price levels on overlapping monthly data
    monthly_cc = panel.set_index("date")["cc_close_m"]
    monthly_c = panel.set_index("date")["c_usd_close_m"]
    h = spread.ols_hedge(monthly_cc, monthly_c)
    beta = float(h.beta)
    panel["ret_spread_m"] = panel["ret_cc_m"] - beta * panel["ret_c_usd_m"]
    return panel, fx_available, beta
