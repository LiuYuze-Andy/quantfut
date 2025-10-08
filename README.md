# Quantfut

**Quantfut** is a lightweight Python framework for **quantitative strategy backtesting**.  
It can generate synthetic futures data, execute SMA strategies, and produce performance reports in both **CSV and chart form**.

---

##  Features
- Generate synthetic futures market data for testing and prototyping.  
- Run single or multiple SMA (Simple Moving Average) strategies via CLI.  
- Automatically generate CSV reports and performance visualizations.  
- Modular, extensible design with configuration and testing support.  

**Author:** Yuze Liu  
**Future Work:** Multi-factor strategy comparison, live data feed integration.

---

##  Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (development mode)
pip install -e .[dev]

# 3. Generate synthetic ES futures data
python -m quantfut.cli gen-sample --symbol ES --rows 1500 --out data/ES.csv

# 4. Run a simple SMA backtest (10/30 crossover)
python -m quantfut.cli backtest --config config/example.yaml --symbol ES --strategy sma --short 10 --long 30
