# quantfut
Minimal futures trading system (Python) with backtester, SMA strategy, paper broker, CLI, tests, CI, Docker.

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
python -m quantfut.cli gen-sample --symbol ES --rows 1500 --out data/ES.csv
python -m quantfut.cli backtest --config config/example.yaml --symbol ES --strategy sma --short 10 --long 30
```

## Repo bootstrap to GitHub
```bash
git init
git add .
git commit -m "init: quantfut"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Data format
CSV with header: `datetime,open,high,low,close,volume` (datetime parsable). Timezone optional.

## Disclaimer
For education only. Not investment advice.
