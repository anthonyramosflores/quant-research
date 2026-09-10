"""
Canonical OHLCV schema + validation.

Every data source in this project must normalize to this schema before
anything downstream (backtester, strategies, risk tools) touches it.
Centralizing validation here is what makes look-ahead bias and survivorship
bias *structurally* harder to introduce by accident later.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataValidationError(ValueError):
    """Raised when a DataFrame does not conform to the OHLCV schema."""


def validate_ohlcv(df: pd.DataFrame, symbol: str = "<unknown>") -> pd.DataFrame:
    """
    Validate and normalize an OHLCV DataFrame.

    Enforces:
      - DatetimeIndex, sorted ascending, no duplicate timestamps
      - required columns present and numeric
      - no negative prices/volume
      - high >= low, high >= open/close, low <= open/close

    Returns a cleaned copy. Raises DataValidationError on structural
    problems that would silently corrupt a backtest (e.g. duplicate
    timestamps, which can otherwise cause a strategy to "see" the same
    bar twice, or an unsorted index, which can leak future bars in during
    a naive `.shift()`).
    """
    if df.empty:
        raise DataValidationError(f"[{symbol}] DataFrame is empty")

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise DataValidationError(
                f"[{symbol}] index could not be parsed as dates: {e}"
            )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"[{symbol}] missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].astype(float)

    if df.index.duplicated().any():
        dupes = df.index[df.index.duplicated()].tolist()
        raise DataValidationError(f"[{symbol}] duplicate timestamps: {dupes[:5]}")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError(f"[{symbol}] non-positive prices found")

    if (df["volume"] < 0).any():
        raise DataValidationError(f"[{symbol}] negative volume found")

    bad_high = (df["high"] < df[["open", "close", "low"]].max(axis=1)).sum()
    bad_low = (df["low"] > df[["open", "close", "high"]].min(axis=1)).sum()
    if bad_high or bad_low:
        raise DataValidationError(
            f"[{symbol}] {bad_high} bars with high < max(o,c,l), "
            f"{bad_low} bars with low > min(o,c,h)"
        )

    df.index.name = "date"
    return df
