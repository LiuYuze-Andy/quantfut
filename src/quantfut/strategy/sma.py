from __future__ import annotations
import pandas as pd
from .base import Strategy

class SMACrossover(Strategy):
    def __init__(self, short: int = 10, long: int = 30):
        if short <= 0 or long <= 0 or short >= long:
            raise ValueError("Require 0 < short < long")
        self.short = int(short)
        self.long = int(long)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        price = data["close"]
        ma_s = price.rolling(self.short, min_periods=self.short).mean()
        ma_l = price.rolling(self.long, min_periods=self.long).mean()
        raw = (ma_s > ma_l).astype(int) - (ma_s < ma_l).astype(int)
        sig = raw.replace({pd.NA: 0}).fillna(0).astype(int)
        sig.name = "signal"
        return sig
