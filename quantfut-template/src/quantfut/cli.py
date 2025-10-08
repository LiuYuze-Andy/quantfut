from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from .config import load_settings
from .data.loader import load_csv
from .execution.broker import PaperBroker
from .portfolio.backtester import Backtester
from .strategy.sma import SMACrossover

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def cmd_gen_sample(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(42)
    n = int(args.rows)
    dt_index = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    price = 3000 + np.cumsum(rng.normal(0, 10, size=n))
    close = pd.Series(price, index=dt_index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_, close) + rng.uniform(0, 5, size=n)
    low = np.minimum(open_, close) - rng.uniform(0, 5, size=n)
    vol = rng.integers(1_000, 10_000, size=n)

    df = pd.DataFrame({
        "datetime": dt_index,
        "open": open_.values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "volume": vol,
    })
    out = Path(args.out or f"data/{args.symbol}.csv")
    _ensure_parent(out)
    df.to_csv(out, index=False)
    print(f"Sample data written: {out}")

def cmd_backtest(args: argparse.Namespace) -> None:
    settings = load_settings(args.config)
    sym = args.symbol
    inst_cfg = settings.instruments.get(sym)
    if inst_cfg is None:
        raise SystemExit(f"Instrument '{sym}' not found in config.")

    csv_path = Path(settings.data_path) / f"{sym}.csv"
    df = load_csv(csv_path, tz=settings.timezone)

    if args.strategy == "sma":
        strategy = SMACrossover(short=args.short, long=args.long)
    else:
        raise SystemExit(f"Unknown strategy: {args.strategy}")

    sig = strategy.generate_signals(df)

    broker = PaperBroker(
        commission_per_contract=settings.commission_per_contract,
        slippage_bps=settings.slippage_bps,
    )

    bt = Backtester(
        data=df,
        instrument=inst_cfg,
        broker=broker,
        initial_cash=settings.initial_cash,
        contracts=args.contracts,
    )

    res = bt.run(sig)
    eq = res["equity"]
    metr = res["metrics"]

    outdir = Path(args.outdir or "runs") / sym
    _ensure_parent(outdir)
    eq.to_csv(outdir / "equity.csv", header=True)
    pd.Series(sig, name="signal").to_csv(outdir / "signals.csv", header=True)

    print(f"[{sym}] Backtest done.")
    print(f"Equity final: {eq.iloc[-1]:.2f}")
    print(f"Metrics: "
          f"TotRet={metr.total_return:.2%} "
          f"AnnRet={metr.annual_return:.2%} "
          f"AnnVol={metr.annual_vol:.2%} "
          f"Sharpe={metr.sharpe:.2f} "
          f"MaxDD={metr.max_drawdown:.2%} "
          f"WinRate={metr.win_rate:.2%} "
          f"Trades={metr.n_trades}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantfut")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-sample", help="Generate synthetic OHLCV CSV")
    g.add_argument("--symbol", type=str, default="ES")
    g.add_argument("--rows", type=int, default=1500)
    g.add_argument("--out", type=str, default=None)
    g.set_defaults(func=cmd_gen_sample)

    b = sub.add_parser("backtest", help="Run backtest")
    b.add_argument("--config", type=str, required=True)
    b.add_argument("--symbol", type=str, required=True)
    b.add_argument("--strategy", type=str, choices=["sma"], default="sma")
    b.add_argument("--short", type=int, default=10)
    b.add_argument("--long", type=int, default=30)
    b.add_argument("--contracts", type=int, default=1)
    b.add_argument("--outdir", type=str, default=None)
    b.set_defaults(func=cmd_backtest)

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
