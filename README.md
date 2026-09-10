# quant-research

An independent quantitative research and systematic trading platform —
built as a learning/research project, not a live trading operation.

**Status: Phase 1/2 — foundations.** Data ingestion, a vectorized
backtesting engine, performance metrics, and testing infrastructure are
in place. No live trading, no options/derivatives pricing, and no ML yet
— those come later, deliberately, once the foundations are solid.

## Why this exists

A CS degree + genuine interest in markets, derivatives, and systematic
strategy research, combined into one project that's both a serious
learning exercise and a software-engineering portfolio piece. The
central goal is **not** "turn $100 into $10,000 fast" — it's to find out,
rigorously, whether a repeatable trading edge is achievable, using sound
statistics and honest evaluation. See `docs/philosophy.md`-style
principles baked into the design choices below.

## What's built so far

```
quant-research/
├── data/
│   ├── schema.py       # canonical OHLCV schema + validation (catches
│   │                     dupes, unsorted timestamps, bad OHLC relationships)
│   └── loaders.py       # DataSource interface, CSVDataSource, synthetic
│                         GBM generator (for offline testing/dev)
├── backtesting/
│   ├── strategy.py       # Strategy interface + BuyAndHold, MovingAverageCrossover
│   ├── portfolio.py       # simulates equity curve; ENFORCES 1-bar signal lag
│   ├── execution.py       # commission + slippage cost model
│   ├── metrics.py         # CAGR, Sharpe, Sortino, max DD, win rate, expectancy
│   └── engine.py           # wires data + strategy + portfolio + metrics together
├── risk/
│   ├── drawdown.py         # drawdown series + episode detection
│   ├── position_sizing.py   # fixed-fractional, vol-targeting, capped Kelly
│   └── var.py                # historical VaR/CVaR, parametric VaR
├── strategies/               # momentum / volatility / mean_reversion — empty,
│                              for strategy implementations as they're researched
├── options/                  # NOT YET BUILT — see options/README.md
├── research/
│   └── example_run.py        # end-to-end example script
└── tests/                    # 25 tests, including explicit look-ahead-bias checks
```

## Design principles this codebase tries to enforce structurally

- **Look-ahead bias**: `simulate_portfolio` shifts every signal forward
  one bar before applying it, so a strategy can never earn the return of
  the bar it used to make its decision. This is directly tested in
  `tests/test_backtesting.py::test_signal_is_lagged_before_use` with a
  deliberately "cheating" strategy that would otherwise show an
  impossible edge.
- **Data integrity**: `data/schema.py` rejects duplicate timestamps,
  non-monotonic indices, and inconsistent OHLC relationships before any
  of it reaches a strategy or the backtester.
- **Transaction costs are never optional** — `ExecutionCosts` defaults to
  a nonzero slippage assumption; you have to explicitly zero it out to
  see the (unrealistic) frictionless case, which the tests use only to
  verify the engine's math, not as a strategy's real expected result.
- **Sizing is decoupled from signal generation** so the same directional
  view can be tested under different risk postures without touching
  strategy code.

## Quickstart

```bash
pip install -r requirements.txt
pytest                              # run the test suite (25 tests)
PYTHONPATH=. python research/example_run.py   # run the example pipeline
```

## What's explicitly NOT here yet (by design)

- Options/derivatives pricing, Greeks, IV/RV analysis (`options/`)
- Statistical arbitrage, ML-based strategies, automated execution
- A live/paper broker connection
- Walk-forward / out-of-sample test harness (currently only in-sample)

These come in later phases, once the core platform has proven itself on
simpler strategies first.

## Development path

1. **Learn** — options mechanics, Greeks, IV/RV, position sizing, EV, drawdown
2. **Build** — this repo: data, backtesting engine, metrics, tests *(current)*
3. **Research** — momentum, mean reversion, volatility strategies as hypotheses
4. **Paper trade** — compare backtest vs. paper results, investigate gaps
5. **Tiny live capital** — test execution, psychology, slippage with real (small) money
6. **Gradual scaling** — only with a track record and demonstrated risk control

## Honest framing

This is proprietary research/trading with personal capital — not asset
management, not a hedge fund, and not currently taking outside capital.

