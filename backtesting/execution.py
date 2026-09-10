"""
Execution cost model.

Kept deliberately simple and explicit for Phase 1: a flat commission per
trade plus a slippage cost proportional to the size of the position
change, expressed in basis points of trade notional. Swap this out for a
more realistic model (spread-based, volume-dependent, etc.) once you have
real fill data to calibrate against — don't over-engineer this before you
have evidence about what actually matters for your strategies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCosts:
    commission_per_trade: float = 0.0  # flat $ per trade (e.g. per rebalance)
    slippage_bps: float = 5.0  # basis points of notional traded
    spread_bps: float = 0.0  # optional extra cost for bid/ask spread

    def cost_for_trade(self, trade_notional: float) -> float:
        """
        Total $ cost for a single trade of the given absolute notional
        value (i.e. abs(position_change) * price * portfolio_value).
        """
        if trade_notional == 0:
            return 0.0
        bps_cost = trade_notional * (self.slippage_bps + self.spread_bps) / 10_000
        return self.commission_per_trade + bps_cost
