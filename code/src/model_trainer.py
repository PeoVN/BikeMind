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
        return LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            random_state=42,
            verbosity=-1,
        )
    except ImportError:
        return HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42,
        )


def _fit_model(model: object, train_x: pd.DataFrame, train_y: pd.Series,
               valid_x: pd.DataFrame, valid_y: pd.Series) -> object:
    if model.__class__.__module__.startswith("lightgbm"):
        from lightgbm import early_stopping
        model.fit(
            train_x,
            train_y,
            eval_X=[valid_x],
            eval_y=[valid_y],
            callbacks=[early_stopping(50, verbose=False)],
        )
    else:
        model.fit(train_x, train_y)
    return model


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "WAPE": float(np.abs(actual - predicted).sum() / max(np.abs(actual).sum(), 1)),
    }


def _historical_average(train: pd.DataFrame, test: pd.DataFrame, target: str) -> np.ndarray:
    keys = ["station_id", "day_of_week", "hour"]
    averages = train.groupby(keys)[target].mean()
    prediction = test.set_index(keys).index.map(averages)
    fallback = float(train[target].mean())
    return np.asarray(pd.Series(prediction).fillna(fallback), dtype=float)


def train_models(hourly: pd.DataFrame, output_dir: str | Path) -> dict[str, dict[str, float]]:
    features = make_features(hourly)
    if features.empty:
        raise ValueError("Không đủ dữ liệu sau khi tạo lag để huấn luyện mô hình.")
    features = features.sort_values("datetime_hour")
    n = len(features)
    train_end = max(1, int(n * 0.7))
    valid_end = max(train_end + 1, int(n * 0.85))
    valid_end = min(valid_end, n - 1) if n > 2 else n
    train = features.iloc[:train_end]
    validation = features.iloc[train_end:valid_end]
    test = features.iloc[valid_end:]
    if validation.empty or test.empty:
        raise ValueError("Không đủ dữ liệu cho split Train/Validation/Test 70/15/15.")
    metrics: dict[str, dict[str, float]] = {}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for target in ("pickups", "returns"):
        cols = feature_columns(target)
        model = _regressor()
        _fit_model(model, train[cols], train[target], validation[cols], validation[target])
        predictions = np.maximum(0, model.predict(test[cols]))
        metrics[target] = _metrics(test[target].to_numpy(), predictions)
        with (output / f"lgb_{target}.pkl").open("wb") as stream:
            pickle.dump(
                {
                    "model": model,
                    "features": cols,
                    "target": target,
                    "trained_until": str(train["datetime_hour"].max()),
                },
                stream,
            )
    metrics["_split"] = {
        "train_rows": float(len(train)),
        "validation_rows": float(len(validation)),
        "test_rows": float(len(test)),
    }
    return metrics


def benchmark_models(
    hourly: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compare LightGBM predictions with historical and seasonal-naive baselines."""
    features = make_features(hourly)
    features = features.sort_values("datetime_hour")
    n = len(features)
    train_end = max(1, int(n * 0.7))
    valid_end = max(train_end + 1, int(n * 0.85))
    valid_end = min(valid_end, n - 1) if n > 2 else n
    train, validation, test = features.iloc[:train_end], features.iloc[train_end:valid_end], features.iloc[valid_end:]
    if validation.empty or test.empty:
        raise ValueError("Không đủ dữ liệu để benchmark theo split 70/15/15.")
    rows: list[dict[str, object]] = []
    for target in ("pickups", "returns"):
        model_path = Path(output_dir) / f"lgb_{target}.pkl" if output_dir else None
        if model_path is not None and model_path.exists():
            bundle = load_model(model_path)
            model_prediction = np.maximum(0, bundle["model"].predict(test[bundle["features"]]))
        else:
            model = _regressor()
            _fit_model(model, train[feature_columns(target)], train[target],
                       validation[feature_columns(target)], validation[target])
            model_prediction = np.maximum(0, model.predict(test[feature_columns(target)]))
        actual = test[target].to_numpy()
        historical_prediction = np.maximum(0, _historical_average(train, test, target))
        seasonal_prediction = np.maximum(0, test[f"{target}_lag_168h"].fillna(train[target].mean()).to_numpy())
        predictions = {
            "Historical Average": historical_prediction,
            "Seasonal Naive": seasonal_prediction,
            "BikeMind LightGBM": model_prediction,
        }
        lightgbm_wape = _metrics(actual, model_prediction)["WAPE"]
        for name, prediction in predictions.items():
            values = _metrics(actual, prediction)
            rows.append(
                {
                    "Target": target,
                    "Model": name,
                    "MAE": values["MAE"],
                    "RMSE": values["RMSE"],
                    "WAPE": values["WAPE"],
                    "Delta WAPE (%)": (
                        (values["WAPE"] - lightgbm_wape) / max(values["WAPE"], 1e-9) * 100
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_model(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)
