import pandas as pd
import pytest

from data.schema import DataValidationError, validate_ohlcv
from data.loaders import synthetic_gbm_series


def _valid_df():
    idx = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14],
            "high": [11, 12, 13, 14, 15],
            "low": [9, 10, 11, 12, 13],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "volume": [100, 200, 300, 400, 500],
        },
        index=idx,
    )


def test_valid_dataframe_passes():
    df = validate_ohlcv(_valid_df(), symbol="TEST")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing


def test_missing_column_raises():
    df = _valid_df().drop(columns=["volume"])
    with pytest.raises(DataValidationError):
        validate_ohlcv(df, symbol="TEST")


def test_duplicate_timestamps_raise():
    df = _valid_df()
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(DataValidationError):
        validate_ohlcv(df, symbol="TEST")


def test_unsorted_index_gets_sorted_not_dropped():
    df = _valid_df().iloc[::-1]  # reverse order
    out = validate_ohlcv(df, symbol="TEST")
    assert out.index.is_monotonic_increasing
    assert len(out) == len(df)


def test_negative_price_raises():
    df = _valid_df()
    df.loc[df.index[0], "close"] = -5.0
    with pytest.raises(DataValidationError):
        validate_ohlcv(df, symbol="TEST")


def test_inconsistent_high_low_raises():
    df = _valid_df()
    df.loc[df.index[0], "high"] = 1.0  # lower than open/close/low
    with pytest.raises(DataValidationError):
        validate_ohlcv(df, symbol="TEST")


def test_synthetic_series_is_valid_and_deterministic():
    a = synthetic_gbm_series("TEST", start="2024-01-01", periods=100, seed=1)
    b = synthetic_gbm_series("TEST", start="2024-01-01", periods=100, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 100
