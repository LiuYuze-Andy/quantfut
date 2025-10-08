from __future__ import annotations
import pandas as pd
import numpy as np

def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range.
    WHY: Enables volatility-aware sizing in future extensions.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = np.maximum.reduce([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ])
    return tr.rolling(window, min_periods=window).mean()
