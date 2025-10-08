# src/quantfut/portfolio/backtester.py
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
    side: str                 # LONG/SHORT
    qty: int                  # 进场时目标张数（简化：不随中途加减变化）
    entry_price: float
    exit_price: float | None
    pnl: float | None         # 已平仓才有值

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
    逐日盯市（futures-style）：仅费用在成交时计入现金；PnL 以 close-close 计提到现金。
    WHY: 期货没有名义本金现金流，避免权益巨大跳变导致指标失真。
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

        # 延迟执行：t-1 信号在 t 开盘执行
        sig_exec = signals.shift(1).fillna(0).astype(int)

        opens = df["open"].to_numpy()
        closes = df["close"].to_numpy()
        idx = df.index

        n = len(df)
        pos = np.zeros(n, dtype=int)
        cash = np.zeros(n, dtype=float)
        cash[0] = self.initial_cash

        fills: List[Fill] = []
        trades: List[Trade] = []
        current_trade: Trade | None = None
        mult = self.instrument.multiplier

        for i in range(1, n):
            dt = idx[i]
            prev_pos = pos[i - 1]
            target_pos = int(sig_exec.iat[i]) * self.contracts

            # —— 在今日开盘按目标仓位调仓（仅计佣金；不扣名义本金）——
            delta = target_pos - prev_pos
            if delta != 0:
                side = "BUY" if delta > 0 else "SELL"
                fill = self.broker.market(when=dt, next_open=float(opens[i]), side=side, qty=abs(delta))
                fills.append(fill)

                # 期货式：只扣佣金；名义本金不入现金流
                cash[i - 1] -= fill.commission

                prev_sign = np.sign(prev_pos)
                tgt_sign = np.sign(target_pos)

                # 平仓或翻仓：先关旧 trade
                if prev_pos != 0 and (target_pos == 0 or tgt_sign != prev_sign):
                    if current_trade is not None and current_trade.exit_dt is None:
                        current_trade.exit_dt = dt
                        current_trade.exit_price = fill.price
                        signed = +1 if current_trade.side == "LONG" else -1
                        # 以进出场价计算的已实现盈亏（仅用于报告；逐日盯市已在 cash 中累计）
                        realized = signed * (current_trade.exit_price - current_trade.entry_price) \
                                   * current_trade.qty * mult
                        current_trade.pnl = float(realized)
                    current_trade = None  # 旧单已结束

                # 开新仓（包括翻仓后的新方向）
                if target_pos != 0 and (prev_pos == 0 or tgt_sign != prev_sign):
                    current_trade = Trade(
                        entry_dt=dt,
                        exit_dt=None,
                        side="LONG" if target_pos > 0 else "SHORT",
                        qty=abs(target_pos),
                        entry_price=fill.price,
                        exit_price=None,
                        pnl=None,
                    )
                    trades.append(current_trade)

                # 同向加减仓：不新开 Trade（简化）；已实现盈亏仍由逐日盯市承担

            # —— 逐日盯市：昨日持仓对收盘价变动的 PnL 计入现金 ——
            mtm = prev_pos * (closes[i] - closes[i - 1]) * mult
            cash[i] = cash[i - 1] + mtm
            pos[i] = target_pos

        # 权益即现金（逐日盯市后，现金已包含累计 PnL 与费用）
        eq = pd.Series(cash, index=idx, name="equity")
        ret = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        ann_factor = 252
        total_return = float((eq.iat[-1] / eq.iat[0]) - 1.0) if eq.iat[0] != 0 else 0.0
        # 使用等效年化：按交易日数换算
        life_days = max(len(eq) - 1, 1)
        annual_return = float((1 + total_return) ** (ann_factor / life_days) - 1.0)
        annual_vol = float(ret.std(ddof=0) * np.sqrt(ann_factor))
        sharpe = float((ret.mean() / (ret.std(ddof=0) + 1e-12)) * np.sqrt(ann_factor))
        max_drawdown = float(abs((eq / eq.cummax() - 1.0).min()))

        closed = [t for t in trades if t.pnl is not None]
        win_rate = float(np.mean([t.pnl > 0 for t in closed]) if closed else 0.0)
        n_trades = int(len(closed))

        metrics = Metrics(
            total_return=total_return,
            annual_return=annual_return,
            annual_vol=annual_vol,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            n_trades=n_trades,
        )

        return {
            "equity": eq,
            "returns": ret,
            "fills": fills,
            "trades": trades,
            "metrics": metrics,
        }
