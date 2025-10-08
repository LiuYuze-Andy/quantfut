FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .
COPY src /app/src
COPY config /app/config
CMD ["python", "-m", "quantfut.cli", "backtest", "--config", "config/example.yaml", "--symbol", "ES", "--strategy", "sma", "--short", "10", "--long", "30"]
