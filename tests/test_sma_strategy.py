import pandas as pd
from quantfut.strategy.sma import SMACrossover

def test_sma_signals_shape():
    idx = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    df = pd.DataFrame({"close": range(100)}, index=idx)
    strat = SMACrossover(short=2, long=3)
    sig = strat.generate_signals(df)
    assert set(sig.unique()).issubset({-1,0,1})
    assert sig.index.equals(df.index)
