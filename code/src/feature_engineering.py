"""Time-series features used by the forecasting models."""

from __future__ import annotations

import pandas as pd


TARGETS = ("pickups", "returns")
FEATURE_COLUMNS = ["hour", "day_of_week", "is_weekend", "month", "lag_1h", "lag_2h", "lag_24h", "lag_168h", "rolling_3h"]


def make_features(hourly: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    frame = hourly.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values(["station_id", "timestamp"])
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["month"] = frame["timestamp"].dt.month
    for target in TARGETS:
        grouped = frame.groupby("station_id", sort=False)[target]
        frame[f"{target}_lag_1h"] = grouped.shift(1)
        frame[f"{target}_lag_2h"] = grouped.shift(2)
        frame[f"{target}_lag_24h"] = grouped.shift(24)
        frame[f"{target}_lag_168h"] = grouped.shift(168)
        frame[f"{target}_rolling_3h"] = grouped.transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    if drop_missing:
        frame = frame.dropna().reset_index(drop=True)
    return frame


def feature_columns(target: str) -> list[str]:
    return FEATURE_COLUMNS + [f"{target}_{suffix}" for suffix in ("lag_1h", "lag_2h", "lag_24h", "lag_168h", "rolling_3h")]
