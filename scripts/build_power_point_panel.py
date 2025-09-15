#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is on path when running from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from config import Config
from power_point import build_power_point_tidy, monthly_price_panel


def find_power_point_file(root: Path) -> Path:
    raw_dir = root / "data" / "raw"
    candidates = list(raw_dir.glob("POWER_Point_Monthly_*.csv"))
    if not candidates:
        raise SystemExit("No POWER_Point_Monthly_*.csv found under data/raw/")
    # Pick the latest by name
    return sorted(candidates)[-1]


def main() -> None:
    cfg = Config()
    # 1) Parse and tidy POWER point file
    raw = find_power_point_file(cfg.paths.root)
    # Canonical monthly POWER file name
    tidy_out = cfg.paths.root / "data" / "derived" / "power_monthly_civ_gha.csv"
    tidy = build_power_point_tidy(raw, tidy_out)

    # 2) Build monthly price panel (with FX conversion)
    price_panel, fx_available, beta = monthly_price_panel(cfg)

    # 3) Join climate with monthly panel
    # Align dates: both at month-end
    merged = pd.merge(price_panel, tidy, on="date", how="inner")

    out_dir = cfg.paths.root / "results" / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_panel = out_dir / "power_point_panel.csv"
    merged.to_csv(out_panel, index=False)

    print("Saved:", tidy_out)
    print("Saved:", out_panel)


if __name__ == "__main__":
    main()
