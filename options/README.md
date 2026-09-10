# Options / Derivatives Module — Not Yet Built

Per the project's phased plan, derivatives functionality (Black-Scholes
pricing, Greeks, implied/realized volatility analysis) is intentionally
deferred until the core research platform (data ingestion, backtesting
engine, performance metrics, testing infrastructure) is solid.

Planned modules here, once Phase 1/2 foundations are done:
- `black_scholes.py` — European option pricing
- `greeks.py` — delta, gamma, theta, vega, rho
- `volatility.py` — implied vol solving, realized vol estimators

See the root README's "Development Path" section for the full phase plan.
