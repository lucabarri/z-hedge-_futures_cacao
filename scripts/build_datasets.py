#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

# Ensure src/ is on path when running from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config
import data_io, fx


def main() -> None:

    cfg = Config()

    xlsx = Path("data/raw/dati_cacao.xlsx")
    if xlsx.exists():
        data_io.convert_excel_sheets_to_csv(xlsx, cfg.paths.data_processed)
        print("Converted Excel to CSVs in", cfg.paths.data_processed)
    else:
        print("Excel file not found; skipping conversion.")

    fx_df = fx.get_gbpusd_fx(
        cache_path=cfg.paths.data_external / cfg.fx.cache_filename,
        ticker=cfg.fx.ticker,
        start=cfg.fx.start,
        end=cfg.fx.end,
    )
    if fx_df is not None:
        print("FX cache ready:", cfg.paths.data_external / cfg.fx.cache_filename)
    else:
        print("FX cache missing; run with network in a separate step if needed.")


if __name__ == "__main__":
    main()
