from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Paths:
    root: Path = field(default_factory=lambda: Path(".").resolve())
    data_raw: Path = field(init=False)
    data_external: Path = field(init=False)
    data_processed: Path = field(init=False)
    results: Path = field(init=False)
    tables: Path = field(init=False)
    backtests: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_raw = self.root / "data" / "raw"
        self.data_external = self.root / "data" / "external"
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results"
        self.tables = self.results / "tables"
        self.backtests = self.results / "backtests"


@dataclass
class AnalysisParams:
    # Rolling window for spread mean/std
    spread_window: int = 252
    z_entry: float = 1.0
    z_exit: float = 10.0
    z_stop: float = 4.17
    transaction_cost_bps: float = 1.0  # per leg, round-trip assumed 2x
    annualization_factor: int = 252


@dataclass
class FXParams:
    ticker: str = "GBPUSD=X"
    start: str = "2007-01-01"
    end: str = "2030-01-01"
    cache_filename: str = "gbpusd.csv"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    analysis: AnalysisParams = field(default_factory=AnalysisParams)
    fx: FXParams = field(default_factory=FXParams)
