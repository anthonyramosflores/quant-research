"""
Strategy interface.

A Strategy consumes historical OHLCV data and produces a target position
(a weight in [-1, 1]: 1 = fully long, -1 = fully short, 0 = flat) for each
bar. It must NOT use any information from bars after the one it's
deciding on — the engine enforces this structurally by shifting signals
forward one bar before applying them (see backtesting/engine.py), but
strategies should still be written as if only `data.loc[:t]` were visible,
since indicator lookback windows etc. are the strategy's own responsibility.
"""

from __future__ import annotations

import abc

import pandas as pd


class Strategy(abc.ABC):
    """Base class for all strategies."""

    name: str = "unnamed_strategy"

    @abc.abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Given a full OHLCV DataFrame, return a Series of target position
        weights (float, typically in [-1, 1]) indexed the same as `data`.

        Implementations should only use `data[col].rolling(...)`,
        `.expanding()`, or similar backward-looking operations. Do not use
        `.shift(-n)` or any forward-looking transformation here — the
        engine assumes the signal at index t was computable using only
        data available through t.
        """
        raise NotImplementedError


class BuyAndHold(Strategy):
    """Baseline: fully long from the first bar. Useful as a benchmark."""

    name = "buy_and_hold"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index)


class MovingAverageCrossover(Strategy):
    """
    Classic trend-following baseline: long when fast MA > slow MA,
    flat otherwise. Included as a reference implementation, not because
    it's assumed to have an edge — treat it as a research hypothesis to
    test, like anything else.
    """

    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast window must be < slow window")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        fast_ma = data["close"].rolling(self.fast).mean()
        slow_ma = data["close"].rolling(self.slow).mean()
        signal = (fast_ma > slow_ma).astype(float)
        signal[fast_ma.isna() | slow_ma.isna()] = 0.0
        return signal
