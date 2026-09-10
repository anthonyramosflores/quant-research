import numpy as np
import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from backtesting.execution import ExecutionCosts
from backtesting.portfolio import simulate_portfolio
from backtesting.strategy import BuyAndHold, MovingAverageCrossover, Strategy
from data.loaders import synthetic_gbm_series


@pytest.fixture
def sample_data():
    return synthetic_gbm_series("TEST", start="2022-01-01", periods=500, seed=7)


def test_buy_and_hold_matches_asset_return_minus_costs(sample_data):
    engine = BacktestEngine(
        strategy=BuyAndHold(),
        starting_capital=10_000,
        execution_costs=ExecutionCosts(commission_per_trade=0, slippage_bps=0),
    )
    result = engine.run(sample_data)

    asset_total_return = sample_data["close"].iloc[-1] / sample_data["close"].iloc[0] - 1
    portfolio_total_return = (
        result.portfolio.equity_curve.iloc[-1] / result.portfolio.starting_capital - 1
    )

    # With zero costs and a 1-bar lag, buy-and-hold should closely track
    # the underlying's return (off by the first day, which is unavoidable
    # since day 0 has no prior signal to act on).
    assert portfolio_total_return == pytest.approx(asset_total_return, rel=0.01)


def test_signal_is_lagged_before_use(sample_data):
    """
    The single most important correctness property in this whole engine:
    a signal computed using bar t's close must not earn bar t's return.
    We build a strategy that goes long exactly when tomorrow's return
    will be positive (impossible in reality — this is a "cheating"
    strategy) and confirm the engine does NOT let it capture that return
    on the same bar it "knew" about it.
    """

    class CheatingStrategy(Strategy):
        name = "cheater"

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            future_return = data["close"].shift(-1) / data["close"] - 1
            signal = (future_return > 0).astype(float)
            signal.iloc[-1] = 0.0
            return signal

    engine = BacktestEngine(
        strategy=CheatingStrategy(),
        execution_costs=ExecutionCosts(commission_per_trade=0, slippage_bps=0),
    )
    result = engine.run(sample_data)

    # positions held == signals shifted by 1, i.e. NOT the raw cheating signal
    raw_signal = CheatingStrategy().generate_signals(sample_data)
    assert not result.portfolio.positions.equals(raw_signal)
    pd.testing.assert_series_equal(
        result.portfolio.positions,
        raw_signal.shift(1).fillna(0.0),
        check_names=False,
    )


def test_transaction_costs_reduce_equity_vs_zero_cost(sample_data):
    strategy = MovingAverageCrossover(fast=5, slow=20)

    no_cost = BacktestEngine(
        strategy=strategy, execution_costs=ExecutionCosts(0, 0, 0)
    ).run(sample_data)
    with_cost = BacktestEngine(
        strategy=strategy, execution_costs=ExecutionCosts(0, slippage_bps=25)
    ).run(sample_data)

    assert with_cost.portfolio.equity_curve.iloc[-1] <= no_cost.portfolio.equity_curve.iloc[-1]


def test_strategy_index_mismatch_raises(sample_data):
    class BrokenStrategy(Strategy):
        name = "broken"

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=data.index[:-1])  # wrong length

    engine = BacktestEngine(strategy=BrokenStrategy())
    with pytest.raises(ValueError):
        engine.run(sample_data)


def test_flat_signal_produces_flat_returns(sample_data):
    class FlatStrategy(Strategy):
        name = "flat"

        def generate_signals(self, data: pd.DataFrame) -> pd.Series:
            return pd.Series(0.0, index=data.index)

    engine = BacktestEngine(strategy=FlatStrategy())
    result = engine.run(sample_data)
    assert result.portfolio.equity_curve.iloc[-1] == pytest.approx(
        result.portfolio.starting_capital
    )
