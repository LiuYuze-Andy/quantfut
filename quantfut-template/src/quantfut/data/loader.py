from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional

REQUIRED_COLS = ["datetime", "open", "high", "low", "close", "volume"]

def load_csv(path: str | Path, tz: Optional[str] = "UTC") -> pd.DataFrame:
    """
    Load OHLCV CSV to a tz-aware DataFrame indexed by datetime.
    WHY: Standardize time index for consistent backtesting semantics.
    """
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False, infer_datetime_format=True)
    if tz:
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize(tz)
        else:
            df["datetime"] = df["datetime"].dt.tz_convert(tz)
    df = df.set_index("datetime").sort_index()
    df = df.astype({
        "open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"
    })
    return df
