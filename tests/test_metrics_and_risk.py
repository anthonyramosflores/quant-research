import numpy as np
import pandas as pd
import pytest

from backtesting.metrics import (
    build_report,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)
from risk.drawdown import compute_drawdown_series, find_drawdown_episodes
from risk.position_sizing import capped_kelly, fixed_fractional, volatility_target
from risk.var import historical_cvar, historical_var, parametric_var


@pytest.fixture
def up_then_down_equity():
    # rises 20%, then falls 25% from the peak, then flat
    values = [100, 110, 120, 108, 90, 90, 95]
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_max_drawdown_correct(up_then_down_equity):
    dd = max_drawdown(up_then_down_equity)
    assert dd == pytest.approx(90 / 120 - 1)


def test_drawdown_episode_detection(up_then_down_equity):
    # Fixture only recovers to 95, still below the 120 peak, so this
    # drawdown episode should still be open (not recovered) at series end.
    episodes = find_drawdown_episodes(up_then_down_equity)
    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.recovered is False
    assert ep.depth == pytest.approx(90 / 120 - 1)


def test_drawdown_episode_marked_recovered_once_new_peak_hit():
    values = [100, 120, 90, 130]  # drops then makes a new high
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    equity = pd.Series(values, index=idx, dtype=float)
    episodes = find_drawdown_episodes(equity)
    assert len(episodes) == 1
    assert episodes[0].recovered is True


def test_cagr_matches_manual_calc():
    idx = pd.bdate_range("2024-01-01", periods=253)  # ~1 year
    equity = pd.Series(np.linspace(100, 121, len(idx)), index=idx)
    result = cagr(equity, periods_per_year=252)
    expected = (121 / 100) ** (252 / len(idx)) - 1
    assert result == pytest.approx(expected, rel=1e-6)


def test_sharpe_zero_for_zero_vol_returns():
    idx = pd.bdate_range("2024-01-01", periods=50)
    returns = pd.Series(0.0, index=idx)
    assert np.isnan(sharpe_ratio(returns))  # zero std -> undefined, not 0


def test_win_rate_basic():
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01])
    assert win_rate(returns) == pytest.approx(2 / 4)  # zero excluded


def test_historical_var_and_cvar_ordering():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.02, 2000))
    var = historical_var(returns, confidence=0.95)
    cvar = historical_cvar(returns, confidence=0.95)
    # CVaR (average of tail) should be worse than or equal to VaR (the threshold)
    assert cvar <= var


def test_parametric_var_reasonable_for_normal_data():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.02, 5000))
    pvar = parametric_var(returns, confidence=0.95)
    hvar = historical_var(returns, confidence=0.95)
    assert pvar == pytest.approx(hvar, abs=0.005)


def test_fixed_fractional_scaling():
    signal = pd.Series([1.0, -1.0, 0.5])
    scaled = fixed_fractional(signal, fraction=0.5)
    pd.testing.assert_series_equal(scaled, signal * 0.5)


def test_fixed_fractional_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        fixed_fractional(pd.Series([1.0]), fraction=1.5)


def test_volatility_target_caps_leverage():
    idx = pd.bdate_range("2024-01-01", periods=60)
    # very low realized vol -> would otherwise imply huge leverage
    returns = pd.Series(0.0001, index=idx)
    signal = pd.Series(1.0, index=idx)
    sized = volatility_target(signal, returns, target_annual_vol=0.5, max_leverage=2.0)
    assert sized.max() <= 2.0


def test_capped_kelly_bounds():
    f = capped_kelly(win_prob=0.55, win_loss_ratio=1.5, kelly_fraction=0.5, max_position=1.0)
    assert -1.0 <= f <= 1.0


def test_capped_kelly_rejects_bad_inputs():
    with pytest.raises(ValueError):
        capped_kelly(win_prob=1.5, win_loss_ratio=1.0)
    with pytest.raises(ValueError):
        capped_kelly(win_prob=0.5, win_loss_ratio=-1.0)
