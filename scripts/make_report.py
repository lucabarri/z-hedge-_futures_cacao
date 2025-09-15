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


def main() -> None:
    cfg = Config()
    report = cfg.paths.results / "REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    content = []

    content.append("# Cocoa Contracts Analysis")
    content.append("")
    content.append(
        "This report summarizes the comparison between ICE US Cocoa (CC) and London Cocoa (C) and a simple cross-market mean-reversion backtest."
    )
    content.append("")
    content.append(f"- Summary table: `{(cfg.paths.tables / 'summary.csv').as_posix()}`")
    content.append(f"- Overlap/correlation: `{(cfg.paths.tables / 'overlap_corr.csv').as_posix()}`")

    # Backtest metrics files (standard vs causal, plus tilted variants if present)
    std_metrics = cfg.paths.backtests / "standard" / "base" / "metrics.csv"
    cau_metrics = cfg.paths.backtests / "causal" / "base" / "metrics.csv"
    std_tilt = cfg.paths.backtests / "standard" / "tilted" / "metrics.csv"
    cau_tilt = cfg.paths.backtests / "causal" / "tilted" / "metrics.csv"
    items = [("Standard metrics", std_metrics), ("Causal metrics", cau_metrics)]
    if std_tilt.exists():
        items.append(("Standard metrics (tilted)", std_tilt))
    if cau_tilt.exists():
        items.append(("Causal metrics (tilted)", cau_tilt))
    for label, path in items:
        content.append(f"- {label}: `{path.as_posix()}`")

    # Equity curves
    eq_std = cfg.paths.backtests / "standard" / "base" / "equity.csv"
    eq_cau = cfg.paths.backtests / "causal" / "base" / "equity.csv"
    if eq_std.exists():
        content.append(f"- Equity (standard): `{eq_std.as_posix()}`")
    if eq_cau.exists():
        content.append(f"- Equity (causal): `{eq_cau.as_posix()}`")

    report.write_text("\n".join(content), encoding="utf-8")
    print("Saved:", report)


if __name__ == "__main__":
    main()

