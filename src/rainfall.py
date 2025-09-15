from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from config import Config


POWER_CSV_NAME = "power_monthly_civ_gha.csv"


def load_monthly_precip_csv(cfg: Optional[Config] = None) -> pd.DataFrame:
    if cfg is None:
        cfg = Config()
    p = cfg.paths.root / "data" / "derived" / POWER_CSV_NAME
    if not p.exists():
        raise FileNotFoundError(
            "Monthly precip CSV not found. Build it from your local POWER point file (e.g., run scripts/build_power_point_panel.py)."
        )
    return pd.read_csv(p)


@dataclass
class Climatology:
    base: pd.DataFrame  # columns: month, precip_mm

    def percentile_of(self, valid_month: str, value_mm: float) -> float:
        # Use 1981-2020 subset and same calendar month
        ser = self.base.copy()
        if "month" not in ser.columns:
            raise KeyError("Climatology base requires 'month' column (YYYY-MM)")
        ser["year"] = ser["month"].str.slice(0, 4).astype(int)
        ser["mon"] = ser["month"].str.slice(5, 7).astype(int)
        vm = pd.Period(valid_month, freq="M")
        mon = int(vm.strftime("%m"))
        mask = (ser["year"] >= 1981) & (ser["year"] <= 2020) & (ser["mon"] == mon)
        ref = ser.loc[mask, "precip_mm"].dropna().values
        if ref.size == 0:
            return float("nan")
        return float((ref <= value_mm).mean() * 100.0)


def compute_climatology(cfg: Optional[Config] = None) -> Climatology:
    return Climatology(base=load_monthly_precip_csv(cfg))


def build_observed_signals_csv(
    cfg: Optional[Config] = None, monthly_csv: Optional[Path] = None
) -> Path:
    """Create signals from observed rainfall (no forecast)."""
    if cfg is None:
        cfg = Config()
    if monthly_csv is not None:
        df = pd.read_csv(monthly_csv)
    else:
        df = load_monthly_precip_csv(cfg)
    if "month" not in df.columns:
        raise KeyError("Expected 'month' column (YYYY-MM) in monthly precip CSV")
    clim = compute_climatology(cfg)
    rows = []
    for m, mm in zip(df["month"], df["precip_mm"]):
        vm = pd.Period(m, freq="M")
        issue_dt = vm.to_timestamp(how="end").date()
        pct = clim.percentile_of(m, float(mm))
        rows.append(
            {
                "issue_date": issue_dt.strftime("%Y-%m-%d"),
                "valid_month": m,
                "percentile": pct,
                "dry_signal": 1 if pct <= 20 else 0,
                "wet_signal": 1 if pct >= 80 else 0,
                "anom_signed": 50.0 - pct,
            }
        )
    sig = pd.DataFrame(rows)
    out_dir = cfg.paths.results / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rain_percentile_signals.csv"
    sig.to_csv(out_path, index=False)
    return out_path
