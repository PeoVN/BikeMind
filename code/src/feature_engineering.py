"""Time-series features used by the forecasting models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ("pickups", "returns")
FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "is_peak_hour",
]


def _normalise_timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "datetime_hour" not in result.columns:
        if "timestamp" in result.columns:
            result = result.rename(columns={"timestamp": "datetime_hour"})
        else:
            raise ValueError("Dữ liệu station-hour thiếu cột datetime_hour hoặc timestamp.")
    result["datetime_hour"] = pd.to_datetime(result["datetime_hour"], errors="coerce")
    result = result.dropna(subset=["datetime_hour"])
    if "station_id" not in result.columns:
        raise ValueError("Dữ liệu station-hour thiếu cột station_id.")
    for target in TARGETS:
        if target not in result.columns:
            raise ValueError(f"Dữ liệu station-hour thiếu cột {target}.")
        result[target] = pd.to_numeric(result[target], errors="coerce")
    return result


def _station_lag(frame: pd.DataFrame, target: str, periods: int) -> pd.Series:
    """Shift a station series without pandas reindexing the full input."""
    values = frame[target].to_numpy()
    station_ids = frame["station_id"].to_numpy()
    source_positions = np.arange(len(frame), dtype=np.int64) - periods
    valid = source_positions >= 0
    safe_positions = np.maximum(source_positions, 0)
    valid &= station_ids == station_ids[safe_positions]
    shifted = np.full(len(frame), np.nan, dtype=float)
    shifted[valid] = values[safe_positions[valid]]
    return pd.Series(shifted, index=frame.index)


def make_features(hourly: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    frame = _normalise_timestamp_column(hourly)
    frame = frame.sort_values(["station_id", "datetime_hour"]).reset_index(drop=True)
    frame["timestamp"] = frame["datetime_hour"]
    frame["hour"] = frame["datetime_hour"].dt.hour
    frame["day_of_week"] = frame["datetime_hour"].dt.dayofweek
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["month"] = frame["datetime_hour"].dt.month
    frame["is_peak_hour"] = frame["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    for target in TARGETS:
        grouped = frame.groupby("station_id", sort=False)[target]
        frame[f"{target}_lag_1h"] = _station_lag(frame, target, 1)
        frame[f"{target}_lag_2h"] = _station_lag(frame, target, 2)
        frame[f"{target}_lag_24h"] = _station_lag(frame, target, 24)
        frame[f"{target}_lag_168h"] = _station_lag(frame, target, 168)
        frame[f"{target}_rolling_3h"] = grouped.transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    if drop_missing:
        frame = frame.dropna().reset_index(drop=True)
    return frame


def feature_columns(target: str) -> list[str]:
    if target not in TARGETS:
        raise ValueError(f"Target không hợp lệ: {target}")
    return FEATURE_COLUMNS + [
        f"{target}_{suffix}"
        for suffix in ("lag_1h", "lag_2h", "lag_24h", "lag_168h", "rolling_3h")
    ]


def build_features_pipeline(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build and persist model features from a station-hour Parquet table."""
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu station-hour: {source}")
    destination = Path(output_path) if output_path is not None else source.with_name("station_features.parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB chưa được cài đặt để tạo features ngoài RAM.") from exc

    def sql_path(path: Path) -> str:
        return "'" + str(path.resolve()).replace("\\", "/").replace("'", "''") + "'"

    if destination.exists():
        destination.unlink()
    connection = duckdb.connect()
    try:
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(
            f"""
            COPY (
                WITH source AS (
                    SELECT *,
                           try_cast(timestamp AS TIMESTAMP) AS datetime_hour
                    FROM read_parquet({sql_path(source)})
                ),
                features AS (
                    SELECT
                        source.*,
                        date_part('hour', datetime_hour)::INTEGER AS hour,
                        (date_part('isodow', datetime_hour)::INTEGER - 1) AS day_of_week,
                        (date_part('isodow', datetime_hour)::INTEGER >= 6)::INTEGER AS is_weekend,
                        date_part('month', datetime_hour)::INTEGER AS month,
                        (
                            date_part('hour', datetime_hour)::INTEGER
                            IN (7, 8, 9, 17, 18, 19)
                        )::INTEGER AS is_peak_hour,
                        lag(pickups, 1) OVER station_window AS pickups_lag_1h,
                        lag(pickups, 2) OVER station_window AS pickups_lag_2h,
                        lag(pickups, 24) OVER station_window AS pickups_lag_24h,
                        lag(pickups, 168) OVER station_window AS pickups_lag_168h,
                        avg(pickups) OVER (
                            PARTITION BY station_id
                            ORDER BY datetime_hour
                            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                        ) AS pickups_rolling_3h,
                        lag(returns, 1) OVER station_window AS returns_lag_1h,
                        lag(returns, 2) OVER station_window AS returns_lag_2h,
                        lag(returns, 24) OVER station_window AS returns_lag_24h,
                        lag(returns, 168) OVER station_window AS returns_lag_168h,
                        avg(returns) OVER (
                            PARTITION BY station_id
                            ORDER BY datetime_hour
                            ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
                        ) AS returns_rolling_3h
                    FROM source
                    WHERE datetime_hour IS NOT NULL
                      AND station_id IS NOT NULL
                      AND pickups IS NOT NULL
                      AND returns IS NOT NULL
                    WINDOW station_window AS (
                        PARTITION BY station_id
                        ORDER BY datetime_hour
                    )
                )
                SELECT *
                FROM features
                WHERE pickups_lag_168h IS NOT NULL
                  AND returns_lag_168h IS NOT NULL
                ORDER BY station_id, datetime_hour
            ) TO {sql_path(destination)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        row_count = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({sql_path(destination)})"
            ).fetchone()[0]
        )
        preview = connection.execute(
            f"SELECT * FROM read_parquet({sql_path(destination)}) LIMIT 5"
        ).df()
    except (duckdb.Error, OSError, ValueError) as exc:
        raise RuntimeError(f"Không thể tạo feature table: {destination}") from exc
    finally:
        connection.close()
    preview.attrs["row_count"] = row_count
    return preview
