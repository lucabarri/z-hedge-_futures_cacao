from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_csv(path, header=False)

