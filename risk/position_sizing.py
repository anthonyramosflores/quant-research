"""
Position sizing methods.

Sizing is kept separate from signal generation on purpose: a strategy
decides *direction/conviction*, sizing decides *how much capital* to put
behind it. Keeping them decoupled makes it possible to test the same
signal under different risk postures without touching strategy code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_fractional(signal: pd.Series, fraction: float = 1.0) -> pd.Series:
    """Scale a signal by a constant fraction of capital (simplest sizing)."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be in [0, 1]")
    return signal * fraction


def volatility_target(
    signal: pd.Series,
    returns: pd.Series,
    target_annual_vol: float = 0.15,
    lookback: int = 20,
    max_leverage: float = 1.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Scale position size inversely to trailing realized volatility, so the
    strategy targets a roughly constant annualized volatility regardless
    of the underlying's current vol regime. Leverage is capped at
    `max_leverage` to avoid the sizing formula blowing up during very
    low-vol periods.
    """
    realized_vol = returns.rolling(lookback).std(ddof=1) * np.sqrt(periods_per_year)
    scale = (target_annual_vol / realized_vol).clip(upper=max_leverage)
    scale = scale.fillna(0.0)
    return signal * scale


def capped_kelly(
    win_prob: float,
    win_loss_ratio: float,
    kelly_fraction: float = 0.5,
    max_position: float = 1.0,
) -> float:
    """
    Kelly criterion position size, scaled down by `kelly_fraction` (using
    "half-Kelly" or smaller is standard practice since full Kelly is
    extremely sensitive to estimation error in win_prob/win_loss_ratio)
    and hard-capped at `max_position`.

    f* = win_prob - (1 - win_prob) / win_loss_ratio
    """
    if not 0 < win_prob < 1:
        raise ValueError("win_prob must be in (0, 1)")
    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio must be positive")

    kelly = win_prob - (1 - win_prob) / win_loss_ratio
    scaled = kelly * kelly_fraction
    return float(np.clip(scaled, -max_position, max_position))
