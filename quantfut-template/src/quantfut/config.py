from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import yaml
from pathlib import Path

@dataclass
class Instrument:
    symbol: str
    multiplier: float
    tick_size: float

@dataclass
class Settings:
    initial_cash: float
    commission_per_contract: float
    slippage_bps: float
    timezone: str
    data_path: str
    instruments: Dict[str, Instrument]

def load_settings(path: str | Path) -> Settings:
    """
    Load YAML settings to strongly-typed Settings.
    WHY: Validate required keys early; fail fast.
    """
    p = Path(path)
    cfg = yaml.safe_load(p.read_text())
    instruments = {
        sym: Instrument(symbol=sym,
                        multiplier=float(meta["multiplier"]),
                        tick_size=float(meta["tick_size"]))
        for sym, meta in cfg.get("instruments", {}).items()
    }
    return Settings(
        initial_cash=float(cfg["initial_cash"]),
        commission_per_contract=float(cfg["commission_per_contract"]),
        slippage_bps=float(cfg["slippage_bps"]),
        timezone=str(cfg["timezone"]),
        data_path=str(cfg["data_path"]),
        instruments=instruments,
    )
