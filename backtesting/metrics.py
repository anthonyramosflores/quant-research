"""
Performance metrics computed from a returns series.

All functions take a pandas Series of periodic (default: daily) returns.
Nothing here fabricates or assumes profitability — these are just the
standard measuring sticks; interpretation is on you.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if len(equity_curve) < 2:
        return float("nan")
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = len(equity_curve) / periods_per_year
    if years <= 0 or total_return <= 0:
        return float("nan")
    return total_return ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sharpe ratio. risk_free_rate is annualized."""
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return float("nan")
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def win_rate(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return float("nan")
    return (nonzero > 0).sum() / len(nonzero)


def expectancy(returns: pd.Series) -> float:
    """
    Average return per active (nonzero) period — a rough proxy for
    per-trade expectancy when using daily rebalanced weights rather than
    discrete trades.
    """
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return float("nan")
    return nonzero.mean()


@dataclass
class PerformanceReport:
    cagr: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    expectancy: float
    total_return: float
    n_periods: int

    def __str__(self) -> str:
        return (
            f"CAGR:            {self.cagr:.2%}\n"
            f"Ann. Volatility: {self.annualized_volatility:.2%}\n"
            f"Sharpe:          {self.sharpe:.2f}\n"
            f"Sortino:         {self.sortino:.2f}\n"
            f"Max Drawdown:    {self.max_drawdown:.2%}\n"
            f"Win Rate:        {self.win_rate:.2%}\n"
            f"Expectancy/day:  {self.expectancy:.4%}\n"
            f"Total Return:    {self.total_return:.2%}\n"
            f"Periods:         {self.n_periods}"
        )


def build_report(
    equity_curve: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> PerformanceReport:
    return PerformanceReport(
        cagr=cagr(equity_curve, periods_per_year),
        annualized_volatility=annualized_volatility(returns, periods_per_year),
        sharpe=sharpe_ratio(returns, risk_free_rate, periods_per_year),
        sortino=sortino_ratio(returns, risk_free_rate, periods_per_year),
        max_drawdown=max_drawdown(equity_curve),
        win_rate=win_rate(returns),
        expectancy=expectancy(returns),
        total_return=equity_curve.iloc[-1] / equity_curve.iloc[0] - 1,
        n_periods=len(returns),
    )
