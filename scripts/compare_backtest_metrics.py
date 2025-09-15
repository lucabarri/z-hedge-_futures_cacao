#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Ensure src/ is on path when running from repo
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import Config


def main() -> None:
    cfg = Config()
    bt = cfg.paths.backtests
    candidates = {
        "standard": bt / "standard" / "base" / "metrics.csv",
        "standard_tilted": bt / "standard" / "tilted" / "metrics.csv",
        "causal": bt / "causal" / "base" / "metrics.csv",
        "causal_tilted": bt / "causal" / "tilted" / "metrics.csv",
    }
    rows = []
    for name, path in candidates.items():
        if not path.exists():
            continue
        s = pd.read_csv(path, header=None, names=["key", "value"])
        s = s.set_index("key")["value"].to_dict()
        s["variant"] = name
        rows.append(s)
    if not rows:
        print("No metrics files found under", bt)
        sys.exit(0)
    df = pd.DataFrame(rows)
    out_dir = cfg.paths.tables
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "backtest_compare.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)


if __name__ == "__main__":
    main()
