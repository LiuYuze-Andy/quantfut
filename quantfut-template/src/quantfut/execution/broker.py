from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import pandas as pd

Side = Literal["BUY", "SELL"]

@dataclass
class Fill:
    dt: pd.Timestamp
    side: Side
    price: float
    qty: int
    commission: float
    slippage: float

class PaperBroker:
    """
    Simple broker that fills at next bar's open with slippage bps & per-side commission.
    WHY: Deterministic, delay-aware execution for backtests & paper trading parity.
    """
    def __init__(self, commission_per_contract: float, slippage_bps: float):
        self.commission = float(commission_per_contract)
        self.slippage_bps = float(slippage_bps)

    def _apply_slippage(self, price: float, side: Side) -> float:
        bps = self.slippage_bps / 10_000.0
        return price * (1.0 + bps) if side == "BUY" else price * (1.0 - bps)

    def market(self, when: pd.Timestamp, next_open: float, side: Side, qty: int) -> Fill:
        px = self._apply_slippage(next_open, side)
        return Fill(
            dt=when, side=side, price=float(px), qty=int(qty),
            commission=self.commission * abs(qty),
            slippage=abs(px - next_open) * abs(qty),
        )
