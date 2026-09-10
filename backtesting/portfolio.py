"""
Portfolio simulation.

Turns a price series + target-weight signal series into an equity curve,
accounting for transaction costs. This is where look-ahead bias would most
easily sneak in, so the shift-by-one-bar step is isolated and tested (see
tests/test_backtesting.py::test_signal_is_lagged_before_use).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.execution import ExecutionCosts


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series  # actual position held during each bar (lagged signal)
    trades: pd.Series  # position change at each bar (0 = no trade)
    costs: pd.Series  # $ cost incurred at each bar
    starting_capital: float


def simulate_portfolio(
    prices: pd.Series,
    signals: pd.Series,
    starting_capital: float = 10_000.0,
    costs: ExecutionCosts | None = None,
) -> PortfolioResult:
    """
    Simulate a single-asset portfolio given close prices and target-weight
    signals.

    Critical bias-avoidance step: the signal computed using data through
    bar t is only ever *acted on* starting at bar t+1 (`signals.shift(1)`).
    A strategy deciding "go long" using today's close cannot also earn
    today's return from that decision — that would be look-ahead bias.
    """
    if costs is None:
        costs = ExecutionCosts()

    prices, signals = prices.align(signals, join="inner")
    if prices.empty:
        raise ValueError("prices and signals have no overlapping dates")

    # The position actually HELD during bar t is yesterday's signal.
    position = signals.shift(1).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    gross_returns = position * asset_returns

    trades = position.diff().fillna(position.iloc[0])

    equity = pd.Series(index=prices.index, dtype=float)
    equity.iloc[0] = starting_capital
    cost_series = pd.Series(0.0, index=prices.index)

    capital = starting_capital
    for i in range(len(prices)):
        if i == 0:
            equity.iloc[0] = capital
            continue

        capital *= 1 + gross_returns.iloc[i]

        trade_notional = abs(trades.iloc[i]) * capital
        cost = costs.cost_for_trade(trade_notional)
        capital -= cost
        cost_series.iloc[i] = cost

        equity.iloc[i] = capital

    net_returns = equity.pct_change().fillna(0.0)

    return PortfolioResult(
        equity_curve=equity,
        returns=net_returns,
        positions=position,
        trades=trades,
        costs=cost_series,
        starting_capital=starting_capital,
    )
