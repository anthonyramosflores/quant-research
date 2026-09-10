"""
Minimal end-to-end example: generate data, run two strategies, compare.

Run with:  python research/example_run.py
"""

from backtesting.engine import BacktestEngine
from backtesting.execution import ExecutionCosts
from backtesting.strategy import BuyAndHold, MovingAverageCrossover
from data.loaders import synthetic_gbm_series
from risk.drawdown import find_drawdown_episodes


def main():
    data = synthetic_gbm_series(
        "SYNTH", start="2020-01-01", periods=1000, annual_drift=0.08, annual_vol=0.22, seed=3
    )

    costs = ExecutionCosts(commission_per_trade=0.0, slippage_bps=5.0)

    strategies = [BuyAndHold(), MovingAverageCrossover(fast=20, slow=50)]

    for strat in strategies:
        engine = BacktestEngine(strategy=strat, starting_capital=10_000, execution_costs=costs)
        result = engine.run(data)

        print(f"\n=== {result.strategy_name} ===")
        print(result.report)

        episodes = find_drawdown_episodes(result.portfolio.equity_curve)
        print(f"Drawdown episodes: {len(episodes)}")
        if episodes:
            worst = min(episodes, key=lambda e: e.depth)
            print(f"Worst: {worst.depth:.2%} over {worst.duration_days} days "
                  f"({'recovered' if worst.recovered else 'still underwater'})")


if __name__ == "__main__":
    main()
