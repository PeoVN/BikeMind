from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_weekend", "is_peak",
    "lag_1h", "lag_2h", "lag_24h", "lag_168h", "rolling_3h",
]


def make_features(grid: pd.DataFrame, target: str = "pickups") -> pd.DataFrame:
    """Add calendar, lag, and rolling features without using future observations."""
    if target not in {"pickups", "returns"}:
        raise ValueError("target must be 'pickups' or 'returns'")
    if grid.empty:
        return grid.copy()
    frame = grid.sort_values(["station_id", "timestamp"]).copy()
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype("int8")
    frame["is_peak"] = frame["hour"].isin([7, 8, 9, 17, 18, 19]).astype("int8")
    grouped = frame.groupby("station_id", sort=False)[target]
    for lag in (1, 2, 24, 168):
        frame[f"lag_{lag}h"] = grouped.shift(lag)
    frame["rolling_3h"] = grouped.shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    return frame.dropna(subset=FEATURE_COLUMNS + [target]).reset_index(drop=True)


@dataclass
class ForecastModel:
    target: str
    estimator: object
    backend: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = self.estimator.predict(frame[FEATURE_COLUMNS])
        return np.maximum(np.asarray(values, dtype=float), 0)


def train_forecaster(features: pd.DataFrame, target: str = "pickups") -> tuple[ForecastModel, dict[str, float]]:
    """Train LightGBM when available, otherwise use sklearn's histogram gradient boosting."""
    if features.empty:
        raise ValueError("Cannot train a model from an empty feature frame")
    if "timestamp" not in features.columns:
        raise ValueError("features must contain timestamp for chronological splitting")
    ordered = features.sort_values("timestamp").reset_index(drop=True)
    split_1 = ordered["timestamp"].quantile(0.70)
    split_2 = ordered["timestamp"].quantile(0.85)
    train = ordered[ordered["timestamp"] < split_1]
    validation = ordered[(ordered["timestamp"] >= split_1) & (ordered["timestamp"] < split_2)]
    test = ordered[ordered["timestamp"] >= split_2]
    if train.empty or validation.empty or test.empty:
        raise ValueError("chronological split produced an empty train, validation, or test partition")
    backend = "LightGBM"
    try:
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(n_estimators=250, learning_rate=0.05, max_depth=8, verbosity=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        estimator = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=42)
        backend = "HistGradientBoosting (LightGBM fallback)"
    estimator.fit(train[FEATURE_COLUMNS], train[target])
    predictions = np.maximum(np.asarray(estimator.predict(test[FEATURE_COLUMNS]), dtype=float), 0)
    actual = test[target].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(actual - predictions)))
    rmse = float(np.sqrt(np.mean((actual - predictions) ** 2)))
    denominator = float(np.abs(actual).sum())
    wape = float(np.abs(actual - predictions).sum() / denominator * 100) if denominator else 0.0
    return ForecastModel(target, estimator, backend), {"MAE": mae, "RMSE": rmse, "WAPE": wape}
