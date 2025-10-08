import pandas as pd
import numpy as np
from src.quantfut.portfolio.backtester import Backtester
from src.quantfut.execution.broker import PaperBroker
from src.quantfut.config import Instrument
from src.quantfut.strategy.sma import SMACrossover

def make_df(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    price = 100 + np.linspace(0, 10, n)
    df = pd.DataFrame({
        "open": price, "high": price+1, "low": price-1, "close": price, "volume": 1000
    }, index=idx)
    return df

def test_backtester_runs():
    df = make_df()
    strat = SMACrossover(short=5, long=20)
    sig = strat.generate_signals(df)
    bt = Backtester(
        data=df,
        instrument=Instrument(symbol="TEST", multiplier=10.0, tick_size=0.01),
        broker=PaperBroker(commission_per_contract=0.5, slippage_bps=0.0),
        initial_cash=10_000,
        contracts=1,
    )
    res = bt.run(sig)
    assert "equity" in res and len(res["equity"]) == len(df)
    assert res["metrics"].n_trades >= 0
