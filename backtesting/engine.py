"""
Backtest engine: the single entry point that wires together a data source,
a strategy, execution costs, and performance reporting.

Usage:
    from data.loaders import synthetic_gbm_series
    from backtesting.strategy import MovingAverageCrossover
    from backtesting.engine import BacktestEngine

    data = synthetic_gbm_series("TEST", start="2020-01-01", periods=756)
    engine = BacktestEngine(strategy=MovingAverageCrossover(20, 50))
    result = engine.run(data)
    print(result.report)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.execution import ExecutionCosts
from backtesting.metrics import PerformanceReport, build_report
from backtesting.portfolio import PortfolioResult, simulate_portfolio
from backtesting.strategy import Strategy


@dataclass
class BacktestResult:
    strategy_name: str
    portfolio: PortfolioResult
    report: PerformanceReport
    signals: pd.Series


class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        starting_capital: float = 10_000.0,
        execution_costs: ExecutionCosts | None = None,
        risk_free_rate: float = 0.0,
    ):
        self.strategy = strategy
        self.starting_capital = starting_capital
        self.execution_costs = execution_costs or ExecutionCosts()
        self.risk_free_rate = risk_free_rate

    def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Run the full pipeline on a single symbol's OHLCV data:
        generate signals -> simulate portfolio (with a 1-bar signal lag
        and transaction costs) -> compute performance metrics.
        """
        signals = self.strategy.generate_signals(data)

        if not signals.index.equals(data.index):
            raise ValueError(
                "Strategy.generate_signals must return a Series indexed "
                "identically to the input data."
            )

        portfolio = simulate_portfolio(
            prices=data["close"],
            signals=signals,
            starting_capital=self.starting_capital,
            costs=self.execution_costs,
        )

        report = build_report(
            equity_curve=portfolio.equity_curve,
            returns=portfolio.returns,
            risk_free_rate=self.risk_free_rate,
        )

        return BacktestResult(
            strategy_name=self.strategy.name,
            portfolio=portfolio,
            report=report,
            signals=signals,
        )
