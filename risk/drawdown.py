"""Drawdown analysis: magnitude, duration, and recovery."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DrawdownEpisode:
    start: pd.Timestamp
    trough: pd.Timestamp
    end: pd.Timestamp | None  # None if still in drawdown at end of series
    depth: float  # negative number, e.g. -0.18 for -18%
    duration_days: int
    recovered: bool


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1


def find_drawdown_episodes(equity_curve: pd.Series) -> list[DrawdownEpisode]:
    """
    Identify individual drawdown episodes (peak -> trough -> recovery).
    Useful for understanding not just max drawdown but how often and how
    long the strategy spends underwater.
    """
    dd = compute_drawdown_series(equity_curve)
    episodes: list[DrawdownEpisode] = []

    in_drawdown = False
    peak_idx = equity_curve.index[0]
    trough_idx = None
    trough_val = 0.0

    for i, (idx, val) in enumerate(dd.items()):
        if val < 0 and not in_drawdown:
            in_drawdown = True
            peak_idx = equity_curve.index[i - 1] if i > 0 else idx
            trough_idx = idx
            trough_val = val
        elif val < 0 and in_drawdown:
            if val < trough_val:
                trough_val = val
                trough_idx = idx
        elif val >= 0 and in_drawdown:
            in_drawdown = False
            episodes.append(
                DrawdownEpisode(
                    start=peak_idx,
                    trough=trough_idx,
                    end=idx,
                    depth=trough_val,
                    duration_days=(idx - peak_idx).days,
                    recovered=True,
                )
            )

    if in_drawdown:
        episodes.append(
            DrawdownEpisode(
                start=peak_idx,
                trough=trough_idx,
                end=None,
                depth=trough_val,
                duration_days=(equity_curve.index[-1] - peak_idx).days,
                recovered=False,
            )
        )

    return episodes
