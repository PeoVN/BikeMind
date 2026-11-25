"""Chronological forecasting training and benchmark utilities."""

from __future__ import annotations

import os
import pickle
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .feature_engineering import feature_columns, make_features


_TARGETS = ("pickups", "returns")
_MAX_ROWS_PER_SPLIT = 100_000
_SPLIT_FRACTIONS = (0.70, 0.85)


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


def _fit_model(
    model: object,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    valid_y: pd.Series,
) -> object:
    if model.__class__.__module__.startswith("lightgbm"):
        from lightgbm import early_stopping
        from lightgbm.basic import LightGBMError

        try:
            model.fit(
                train_x,
                train_y,
                eval_X=valid_x,
                eval_y=valid_y,
                callbacks=[early_stopping(50, verbose=False)],
            )
        except LightGBMError as exc:
            raise RuntimeError(f"LightGBM không thể huấn luyện mô hình: {exc}") from exc
    else:
        model.fit(train_x, train_y)
    return model


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("Số lượng giá trị thực tế và dự báo không khớp hoặc rỗng.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Không thể tính metric trên giá trị NaN hoặc vô hạn.")
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "WAPE": float(np.abs(actual - predicted).sum() / max(np.abs(actual).sum(), 1)),
    }


def _split_by_time(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if "datetime_hour" not in frame.columns:
        raise ValueError("Tập feature thiếu cột datetime_hour.")
    frame = frame.dropna(subset=["datetime_hour"]).sort_values(
        ["datetime_hour", "station_id"], kind="mergesort"
    )
    timestamps = pd.Index(frame["datetime_hour"].drop_duplicates().sort_values())
    if len(timestamps) < 3:
        raise ValueError("Cần ít nhất 3 mốc thời gian để chia Train/Validation/Test.")

    train_end = max(1, int(len(timestamps) * _SPLIT_FRACTIONS[0]))
    valid_end = max(train_end + 1, int(len(timestamps) * _SPLIT_FRACTIONS[1]))
    if valid_end >= len(timestamps):
        valid_end = len(timestamps) - 1
        train_end = min(train_end, valid_end - 1)
    if train_end <= 0 or valid_end <= train_end:
        raise ValueError("Không đủ mốc thời gian cho split Train/Validation/Test 70/15/15.")

    train_cutoff = timestamps[train_end]
    test_cutoff = timestamps[valid_end]
    train = frame.loc[frame["datetime_hour"] < train_cutoff]
    validation = frame.loc[
        frame["datetime_hour"].ge(train_cutoff)
        & frame["datetime_hour"].lt(test_cutoff)
    ]
    test = frame.loc[frame["datetime_hour"].ge(test_cutoff)]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Một tập Train/Validation/Test bị rỗng sau khi chia theo thời gian.")
    return train, validation, test, {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }


def _sample_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(frame) <= _MAX_ROWS_PER_SPLIT:
        return frame.reset_index(drop=True)
    return frame.sample(
        n=_MAX_ROWS_PER_SPLIT,
        random_state=seed,
    ).reset_index(drop=True)


def _prepare_dataframe_splits(
    source: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    features = make_features(source)
    if features.empty:
        raise ValueError("Không đủ dữ liệu sau khi tạo lag để huấn luyện mô hình.")
    train, validation, test, source_rows = _split_by_time(features)
    samples = {
        "train": _sample_frame(train, 42),
        "validation": _sample_frame(validation, 43),
        "test": _sample_frame(test, 44),
    }
    return samples, source_rows


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def _duckdb_connection(temp_directory: Path):
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB chưa được cài đặt để đọc feature Parquet theo lô.") from exc

    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET memory_limit='2GB'")
    connection.execute(f"SET threads={max(1, min(4, os.cpu_count() or 2))}")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute("SET max_temp_directory_size='100GB'")
    escaped_temp = str(temp_directory.resolve()).replace("\\", "/").replace("'", "''")
    connection.execute(f"SET temp_directory='{escaped_temp}'")
    return connection


def _prepare_parquet_splits(
    source: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, str]]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy tập feature Parquet: {source}")
    if source.suffix.lower() != ".parquet":
        raise ValueError("train_models chỉ nhận DataFrame hoặc tệp feature .parquet.")

    feature_columns_by_target = {
        target: feature_columns(target)
        for target in _TARGETS
    }
    required_features = set().union(*feature_columns_by_target.values())
    temp_directory = Path(
        os.environ.get(
            "BIKEMIND_DUCKDB_TEMP_DIR",
            str(source.parent / "duckdb_tmp"),
        )
    ).expanduser().resolve()
    connection = _duckdb_connection(temp_directory)
    path_expression = _sql_path(source)
    try:
        described = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({path_expression})"
        ).fetchall()
        available_columns = {str(row[0]) for row in described}
        timestamp_column = (
            "datetime_hour" if "datetime_hour" in available_columns else "timestamp"
        )
        required = required_features | {"station_id", "pickups", "returns"}
        missing = required.difference(available_columns)
        if timestamp_column not in available_columns:
            missing.add("datetime_hour hoặc timestamp")
        if missing:
            raise ValueError(
                "Tệp feature thiếu cột bắt buộc: " + ", ".join(sorted(missing))
            )

        quoted_timestamp = '"' + timestamp_column.replace('"', '""') + '"'
        timestamp_expression = f"try_cast({quoted_timestamp} AS TIMESTAMP)"
        base_query = f"""
            SELECT
                CAST(station_id AS VARCHAR) AS station_id,
                {timestamp_expression} AS datetime_hour,
                {", ".join('"' + name.replace('"', '""') + '"' for name in sorted(required_features | {'pickups', 'returns'}))}
            FROM read_parquet({path_expression})
            WHERE station_id IS NOT NULL
              AND {timestamp_expression} IS NOT NULL
        """
        source_relation = f"({base_query}) AS feature_source"
        timestamp_count = int(
            connection.execute(
                f"SELECT count(DISTINCT datetime_hour) FROM {source_relation}"
            ).fetchone()[0]
        )
        if timestamp_count < 3:
            raise ValueError("Cần ít nhất 3 mốc thời gian trong feature Parquet.")

        train_end = max(1, int(timestamp_count * _SPLIT_FRACTIONS[0]))
        valid_end = max(train_end + 1, int(timestamp_count * _SPLIT_FRACTIONS[1]))
        if valid_end >= timestamp_count:
            valid_end = timestamp_count - 1
            train_end = min(train_end, valid_end - 1)
        if train_end <= 0 or valid_end <= train_end:
            raise ValueError("Không đủ mốc thời gian cho split Train/Validation/Test 70/15/15.")

        cutoffs = connection.execute(
            f"""
            SELECT
                (SELECT datetime_hour FROM (
                    SELECT DISTINCT datetime_hour FROM {source_relation}
                    ORDER BY datetime_hour
                ) LIMIT 1 OFFSET ?) AS validation_start,
                (SELECT datetime_hour FROM (
                    SELECT DISTINCT datetime_hour FROM {source_relation}
                    ORDER BY datetime_hour
                ) LIMIT 1 OFFSET ?) AS test_start
            """,
            [train_end, valid_end],
        ).fetchone()
        validation_start, test_start = cutoffs
        if validation_start is None or test_start is None:
            raise ValueError("Không xác định được mốc chia Train/Validation/Test.")

        counts = connection.execute(
            f"""
            SELECT
                count(*) FILTER (WHERE datetime_hour < ?) AS train_rows,
                count(*) FILTER (
                    WHERE datetime_hour >= ? AND datetime_hour < ?
                ) AS validation_rows,
                count(*) FILTER (WHERE datetime_hour >= ?) AS test_rows,
                max(datetime_hour) FILTER (WHERE datetime_hour < ?) AS trained_until
            FROM {source_relation}
            """,
            [
                validation_start,
                validation_start,
                test_start,
                test_start,
                validation_start,
            ],
        ).fetchone()
        source_rows = {
            "train_rows": int(counts[0]),
            "validation_rows": int(counts[1]),
            "test_rows": int(counts[2]),
        }
        if any(source_rows[name] < 1 for name in source_rows):
            raise ValueError("Một tập Train/Validation/Test bị rỗng sau khi chia theo thời gian.")

        split_conditions = {
            "train": ("datetime_hour < ?", [validation_start]),
            "validation": (
                "datetime_hour >= ? AND datetime_hour < ?",
                [validation_start, test_start],
            ),
            "test": ("datetime_hour >= ?", [test_start]),
        }
        samples: dict[str, pd.DataFrame] = {}
        selected_columns = [
            "station_id",
            "datetime_hour",
            *sorted(required_features),
            "pickups",
            "returns",
        ]
        select_sql = ", ".join(
            '"' + name.replace('"', '""') + '"'
            for name in dict.fromkeys(selected_columns)
        )
        for seed, (split, (condition, parameters)) in enumerate(
            split_conditions.items(), start=42
        ):
            source_count = source_rows[f"{split}_rows"]
            if source_count <= _MAX_ROWS_PER_SPLIT:
                query = (
                    f"SELECT {select_sql} FROM {source_relation} "
                    f"WHERE {condition} ORDER BY datetime_hour, station_id"
                )
            else:
                query = (
                    f"SELECT {select_sql} FROM ("
                    f"SELECT {select_sql} FROM {source_relation} "
                    f"WHERE {condition}"
                    f") AS split_sample "
                    f"USING SAMPLE reservoir({_MAX_ROWS_PER_SPLIT} ROWS) "
                    f"REPEATABLE ({seed})"
                )
            samples[split] = connection.execute(query, parameters).df()
    finally:
        connection.close()

    return samples, source_rows, {
        "trained_until": str(counts[3]),
        "validation_start": str(validation_start),
        "test_start": str(test_start),
    }


def _prepare_training_splits(
    source: pd.DataFrame | str | Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[str, str]]:
    if isinstance(source, pd.DataFrame):
        samples, row_counts = _prepare_dataframe_splits(source)
        train = samples["train"]
        return samples, row_counts, {
            "trained_until": str(train["datetime_hour"].max()),
            "validation_start": str(samples["validation"]["datetime_hour"].min()),
            "test_start": str(samples["test"]["datetime_hour"].min()),
        }
    samples, row_counts, split_metadata = _prepare_parquet_splits(Path(source))
    return samples, row_counts, split_metadata


def _save_bundles_atomically(
    output: Path,
    bundles: dict[str, dict[str, object]],
    validation_samples: dict[str, pd.DataFrame],
) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for target, bundle in bundles.items():
            destination = output / f"lgb_{target}.pkl"
            temporary = output / f".lgb_{target}.{uuid.uuid4().hex}.tmp"
            staged.append((temporary, destination))
            with temporary.open("xb") as stream:
                pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())

            with temporary.open("rb") as stream:
                restored = pickle.load(stream)
            if (
                not isinstance(restored, dict)
                or restored.get("target") != target
                or restored.get("features") != bundle["features"]
                or "model" not in restored
            ):
                raise ValueError(f"Tệp model tạm không hợp lệ cho target {target}.")
            sample = validation_samples[target].loc[
                :, list(bundle["features"])
            ].head(32)
            predictions = np.asarray(restored["model"].predict(sample), dtype=float).reshape(-1)
            if predictions.size != len(sample) or not np.isfinite(predictions).all():
                raise ValueError(f"Model tạm không dự báo hợp lệ cho target {target}.")

        for temporary, destination in staged:
            temporary.replace(destination)
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()


def train_models(
    hourly: pd.DataFrame | str | Path,
    output_dir: str | Path,
) -> dict[str, dict[str, float]]:
    """Train and evaluate two chronological models, sampling large Parquet inputs."""
    samples, source_rows, split_metadata = _prepare_training_splits(hourly)
    train = samples["train"]
    validation = samples["validation"]
    test = samples["test"]
    metrics: dict[str, dict[str, float]] = {}
    bundles: dict[str, dict[str, object]] = {}
    validation_samples: dict[str, pd.DataFrame] = {}
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    for target in _TARGETS:
        columns = feature_columns(target)
        train_x, train_y = train[columns], train[target]
        valid_x, valid_y = validation[columns], validation[target]
        test_x, test_y = test[columns], test[target]
        model = _regressor()
        _fit_model(model, train_x, train_y, valid_x, valid_y)
        predictions = np.asarray(model.predict(test_x), dtype=float).reshape(-1)
        if predictions.size != len(test_y):
            raise ValueError(f"Model {target} trả số dự báo không khớp tập Test.")
        predictions = np.maximum(0, predictions)
        metrics[target] = _metrics(test_y.to_numpy(), predictions)
        validation_samples[target] = valid_x
        bundles[target] = {
            "model": model,
            "features": columns,
            "target": target,
            "trained_until": split_metadata["trained_until"],
            "validation_start": split_metadata["validation_start"],
            "test_start": split_metadata["test_start"],
            "metrics": metrics[target],
            "source_rows": source_rows,
            "sampled_rows": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test),
            },
            "model_type": type(model).__name__,
            "python_version": sys.version.split()[0],
        }

    _save_bundles_atomically(output, bundles, validation_samples)
    metrics["_split"] = {
        **{name: float(value) for name, value in source_rows.items()},
        "train_sampled_rows": float(len(train)),
        "validation_sampled_rows": float(len(validation)),
        "test_sampled_rows": float(len(test)),
        "max_rows_per_split": float(_MAX_ROWS_PER_SPLIT),
    }
    return metrics


def _historical_average(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> np.ndarray:
    keys = ["station_id", "day_of_week", "hour"]
    averages = train.groupby(keys)[target].mean()
    prediction = test.set_index(keys).index.map(averages)
    fallback = float(train[target].mean())
    return np.asarray(pd.Series(prediction).fillna(fallback), dtype=float)


def benchmark_models(
    hourly: pd.DataFrame | str | Path,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compare LightGBM predictions with historical and seasonal-naive baselines."""
    samples, _, _ = _prepare_training_splits(hourly)
    train = samples["train"]
    validation = samples["validation"]
    test = samples["test"]
    rows: list[dict[str, object]] = []
    for target in _TARGETS:
        model_path = Path(output_dir).expanduser().resolve() / f"lgb_{target}.pkl" if output_dir else None
        if model_path is not None and model_path.exists():
            bundle = load_model(model_path)
            model_prediction = np.maximum(
                0,
                bundle["model"].predict(test[bundle["features"]]),
            )
        else:
            columns = feature_columns(target)
            model = _regressor()
            _fit_model(
                model,
                train[columns],
                train[target],
                validation[columns],
                validation[target],
            )
            model_prediction = np.maximum(0, model.predict(test[columns]))
        actual = test[target].to_numpy(dtype=float)
        historical_prediction = np.maximum(
            0,
            _historical_average(train, test, target),
        )
        seasonal_prediction = np.maximum(
            0,
            test[f"{target}_lag_168h"]
            .fillna(train[target].mean())
            .to_numpy(dtype=float),
        )
        predictions = {
            "Historical Average": historical_prediction,
            "Seasonal Naive": seasonal_prediction,
            "BikeMind LightGBM": model_prediction,
        }
        lightgbm_wape = _metrics(actual, model_prediction)["WAPE"]
        for name, prediction in predictions.items():
            values = _metrics(actual, prediction)
            reference_wape = values["WAPE"]
            if reference_wape == 0:
                wape_improvement = (
                    0.0 if lightgbm_wape == 0 else float("nan")
                )
            else:
                wape_improvement = (
                    (reference_wape - lightgbm_wape)
                    / reference_wape
                    * 100
                )
            rows.append(
                {
                    "Target": target,
                    "Model": name,
                    "MAE": values["MAE"],
                    "RMSE": values["RMSE"],
                    "WAPE": values["WAPE"],
                    "Delta WAPE (%)": wape_improvement,
                }
            )
    return pd.DataFrame(rows)


def load_model(path: str | Path) -> dict:
    model_path = Path(path).expanduser().resolve()
    with model_path.open("rb") as stream:
        return pickle.load(stream)
