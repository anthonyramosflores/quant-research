"""
Data source abstraction.

Design goal: the backtester should never care *where* bars came from.
Every loader returns a validated OHLCV DataFrame via `load(symbol, start, end)`.

Today this ships with a CSV-backed source (works fully offline, good for
this environment's network restrictions). When you add a live-data source
later (e.g. a broker API or a vendor feed), implement `DataSource` and the
rest of the stack (backtester, strategies, risk tools) needs zero changes.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

import pandas as pd

from data.schema import validate_ohlcv


class DataSource(abc.ABC):
    """Interface every market-data source must implement."""

    @abc.abstractmethod
    def load(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return a validated OHLCV DataFrame indexed by date, ascending."""
        raise NotImplementedError


class CSVDataSource(DataSource):
    """
    Loads OHLCV bars from local CSVs.

    Expects one file per symbol at `{data_dir}/{symbol}.csv` with at least
    columns: date, open, high, low, close, volume. This is intentionally
    the simplest possible source — point it at data you've exported from
    any vendor and it works, with no API keys or network calls required.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def _path_for(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol.upper()}.csv"

    def load(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        path = self._path_for(symbol)
        if not path.exists():
            raise FileNotFoundError(
                f"No CSV found for '{symbol}' at {path}. "
                f"Add a CSV with columns date,open,high,low,close,volume."
            )

        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = validate_ohlcv(df, symbol=symbol)

        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index <= pd.Timestamp(end)]

        if df.empty:
            raise ValueError(
                f"No rows for '{symbol}' in range [{start}, {end}]"
            )
        return df


def synthetic_gbm_series(
    symbol: str,
    start: str,
    periods: int,
    start_price: float = 100.0,
    annual_drift: float = 0.07,
    annual_vol: float = 0.20,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic daily OHLCV series via Geometric Brownian Motion.

    This is NOT for evaluating whether a strategy has a real edge — GBM has
    no exploitable structure by construction (no momentum, no mean
    reversion, no vol clustering). It exists purely so the backtesting
    engine, portfolio simulation, and tests can run deterministically
    without needing real market data or network access. Use CSVDataSource
    (or a future live source) for anything you intend to draw research
    conclusions from.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dt = 1 / 252
    n = periods

    shocks = rng.normal(
        loc=(annual_drift - 0.5 * annual_vol**2) * dt,
        scale=annual_vol * (dt**0.5),
        size=n,
    )
    close = start_price * np.exp(np.cumsum(shocks))
    close = np.insert(close, 0, start_price)[:n]

    dates = pd.bdate_range(start=start, periods=n)
    intraday_range = np.abs(rng.normal(0, annual_vol * (dt**0.5), n)) * close

    open_ = close * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) + intraday_range * 0.5
    low = np.minimum(open_, close) - intraday_range * 0.5
    low = np.clip(low, a_min=0.01, a_max=None)
    volume = rng.integers(1_000_000, 5_000_000, n)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    return validate_ohlcv(df, symbol=symbol)
