from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
from ..execution.broker import PaperBroker, Fill
from ..config import Instrument

@dataclass
class Trade:
    entry_dt: pd.Timestamp
    exit_dt: pd.Timestamp | None
    side: str
    qty: int
    entry_price: float
    exit_price: float | None
    pnl: float | None

@dataclass
class Metrics:
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int

class Backtester:
    """
    Vectorized daily mark-to-market with next-open execution of prior-day signal.
    WHY: Prevents look-ahead while keeping sim simple & fast.
    """
    def __init__(
        self,
        data: pd.DataFrame,
        instrument: Instrument,
        broker: PaperBroker,
        initial_cash: float = 100_000.0,
        contracts: int = 1,
    ):
        self.data = data.copy()
        self.instrument = instrument
        self.broker = broker
        self.initial_cash = float(initial_cash)
        self.contracts = int(contracts)

    def run(self, signals: pd.Series) -> Dict[str, object]:
        df = self.data
        if not signals.index.equals(df.index):
            signals = signals.reindex(df.index).fillna(0).astype(int)
        sig_exec = signals.shift(1).fillna(0).astype(int)

        opens = df["open"].copy()
        closes = df["close"].copy()

        pos = np.zeros(len(df), dtype=int)
        cash = np.zeros(len(df), dtype=float)
        equity = np.zeros(len(df), dtype=float)
        cash[0] = self.initial_cash
        fills: List[Fill] = []
        trades: List[Trade] = []
        current_trade: Trade | None = None

        for i in range(1, len(df)):
            dt = df.index[i]
            prev_pos = pos[i-1]
            target_pos = sig_exec.iat[i] * self.contracts

            delta = target_pos - prev_pos
            if delta != 0:
                side = "BUY" if delta > 0 else "SELL"
                fill = self.broker.market(when=dt, next_open=opens.iat[i], side=side, qty=abs(delta))
                fills.append(fill)
                signed = +1 if side == "SELL" else -1
                cash[i-1] += signed * fill.price * fill.qty * self.instrument.multiplier
                cash[i-1] -= fill.commission

                if prev_pos == 0:
                    current_trade = Trade(
                        entry_dt=dt, exit_dt=None,
                        side="LONG" if target_pos > 0 else "SHORT",
                        qty=abs(target_pos),
                        entry_price=fill.price, exit_price=None, pnl=None
                    )
                    trades.append(current_trade)
                else:
                    if target_pos == 0 or np.sign(target_pos) != np.sign(prev_pos):
                        if current_trade is not None:
                            current_trade.exit_dt = dt
                            current_trade.exit_price = fill.price

            mtm = prev_pos * (closes.iat[i] - closes.iat[i-1]) * self.instrument.multiplier
            cash[i] = cash[i-1] + mtm
            pos[i] = target_pos

            if current_trade and current_trade.exit_dt is not None and current_trade.pnl is None:
                signed = +1 if current_trade.side == "LONG" else -1
                realized = signed * (current_trade.exit_price - current_trade.entry_price)                            * current_trade.qty * self.instrument.multiplier
                current_trade.pnl = realized

        equity = cash

        eq = pd.Series(equity, index=df.index, name="equity")
        ret = eq.pct_change().fillna(0.0)
        ann_factor = 252
        total_return = (eq.iat[-1] / eq.iat[0]) - 1.0 if eq.iat[0] != 0 else 0.0
        annual_return = (1 + total_return) ** (ann_factor / max(len(eq)-1, 1)) - 1.0
        annual_vol = ret.std(ddof=0) * np.sqrt(ann_factor)
        sharpe = (ret.mean() / (ret.std(ddof=0) + 1e-12)) * np.sqrt(ann_factor)
        dd = (eq / eq.cummax() - 1.0).min()
        max_drawdown = float(abs(dd))
        trade_pnls = [t.pnl for t in trades if t.pnl is not None]
        win_rate = (np.mean([p > 0 for p in trade_pnls]) if trade_pnls else 0.0)
        metrics = Metrics(
            total_return=float(total_return),
            annual_return=float(annual_return),
            annual_vol=float(annual_vol),
            sharpe=float(sharpe),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            n_trades=int(len(trades))
        )

        results = {
            "equity": eq,
            "returns": ret,
            "fills": fills,
            "trades": trades,
            "metrics": metrics,
        }
        return results
