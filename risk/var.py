"""Value-at-Risk (VaR) and Conditional VaR (Expected Shortfall)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical (empirical) VaR: the loss threshold that returns exceed
    with probability (1 - confidence), based purely on the empirical
    distribution of past returns. No distributional assumption, but
    sensitive to sample size and doesn't extrapolate beyond observed
    history.

    Returns a negative number (e.g. -0.032 = a 3.2% loss at this
    confidence level).
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR / Expected Shortfall: average loss in the tail beyond
    the VaR threshold. More informative than VaR alone since it captures
    how bad the tail actually is, not just where it starts.
    """
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    if tail.empty:
        return var
    return float(tail.mean())


def parametric_var(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """
    Parametric (variance-covariance) VaR assuming normally distributed
    returns. Fast and simple, but real return distributions have fatter
    tails than normal — treat this as a lower bound on tail risk, not the
    full picture. Prefer historical_var/historical_cvar when you have
    enough data.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)
    return float(mu + z * sigma)
