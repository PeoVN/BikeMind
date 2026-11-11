"""Chronological forecasting training and benchmark utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .feature_engineering import feature_columns, make_features


def _regressor() -> object:
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)
    except ImportError:
        return HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=42)


def train_models(hourly: pd.DataFrame, output_dir: str | Path) -> dict[str, dict[str, float]]:
    features = make_features(hourly)
    if features.empty:
        raise ValueError("Không đủ dữ liệu sau khi tạo lag để huấn luyện mô hình.")
    features = features.sort_values("timestamp")
    n = len(features)
    train_end, valid_end = max(1, int(n * 0.7)), max(2, int(n * 0.85))
    metrics: dict[str, dict[str, float]] = {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for target in ("pickups", "returns"):
        cols = feature_columns(target)
        model = _regressor()
        model.fit(features.iloc[:train_end][cols], features.iloc[:train_end][target])
        validation = features.iloc[train_end:valid_end]
        predictions = np.maximum(0, model.predict(validation[cols]))
        actual = validation[target].to_numpy()
        metrics[target] = {
            "MAE": float(mean_absolute_error(actual, predictions)),
            "RMSE": float(mean_squared_error(actual, predictions, squared=False)),
            "WAPE": float(np.abs(actual - predictions).sum() / max(np.abs(actual).sum(), 1)),
        }
        with (output / f"lgb_{target}.pkl").open("wb") as stream:
            pickle.dump({"model": model, "features": cols}, stream)
    return metrics


def load_model(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)
