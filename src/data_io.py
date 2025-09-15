from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def read_cc_csv(path: Path) -> pd.DataFrame:
    """Read ICE US Cocoa (CC) OHLCV CSV.

    Expected columns: date/date-like, open, high, low, close, contract_code, VOLUME
    """
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])  # type: ignore[index]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def read_c_csv(path: Path) -> pd.DataFrame:
    """Read ICE London Cocoa (C) close-only CSV.

    Expected columns: Date/date, close (GBP native; conversion handled elsewhere).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])  # type: ignore[index]
    # Normalize close column name
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_local_default_cc(root: Path) -> Optional[pd.DataFrame]:
    """Load CC CSV strictly from data/processed.

    Expected path: data/processed/dati_cacao_CC.csv
    Returns None if missing.
    """
    p = root / "data" / "processed" / "dati_cacao_CC.csv"
    if p.exists():
        return read_cc_csv(p)
    return None


def load_local_default_c(root: Path) -> Optional[pd.DataFrame]:
    """Load C CSV strictly from data/processed.

    Expected path: data/processed/dati_cacao_C.csv
    Returns None if missing.
    """
    p = root / "data" / "processed" / "dati_cacao_C.csv"
    if p.exists():
        return read_c_csv(p)
    return None


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def convert_excel_sheets_to_csv(xlsx_path: Path, out_dir: Path) -> Tuple[Path, Path]:
    """Convert provided Excel sheets into two CSV files (C and CC) under out_dir.

    Only recognized sheets are written: 'CC' and 'C' (case-insensitive). Any other
    sheets are ignored. Raises if neither CC nor C could be found.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    xl = pd.ExcelFile(xlsx_path)
    c_csv = out_dir / "dati_cacao_C.csv"
    cc_csv = out_dir / "dati_cacao_CC.csv"
    wrote_c = False
    wrote_cc = False
    for sheet in xl.sheet_names:
        s_norm = str(sheet).strip().lower().replace(" ", "")
        if s_norm == "cc":
            df = pd.read_excel(xlsx_path, sheet_name=sheet)
            df.to_csv(cc_csv, index=False)
            wrote_cc = True
        elif s_norm == "c":
            df = pd.read_excel(xlsx_path, sheet_name=sheet)
            df.to_csv(c_csv, index=False)
            wrote_c = True
        else:
            continue
    if not (wrote_c and wrote_cc):
        raise ValueError("Expected sheets named 'C' and 'CC' in the Excel file.")
    return c_csv, cc_csv
