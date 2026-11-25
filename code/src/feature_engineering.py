"""Time-series features used by the forecasting models."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

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


def _normalise_timestamp_column(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = frame.copy()
    if "datetime_hour" not in result.columns:
        if "timestamp" in result.columns:
            result = result.rename(columns={"timestamp": "datetime_hour"})
        else:
            raise ValueError("Dữ liệu station-hour thiếu cột datetime_hour hoặc timestamp.")
    if "station_id" not in result.columns:
        raise ValueError("Dữ liệu station-hour thiếu cột station_id.")
    for target in TARGETS:
        if target not in result.columns:
            raise ValueError(f"Dữ liệu station-hour thiếu cột {target}.")
        result[target] = pd.to_numeric(result[target], errors="coerce")
    result["datetime_hour"] = pd.to_datetime(result["datetime_hour"], errors="coerce")
    invalid_timestamps = int(result["datetime_hour"].isna().sum())
    missing_station_ids = int(result["station_id"].isna().sum())
    result = result.dropna(subset=["datetime_hour", "station_id"]).copy()
    result["datetime_hour"] = result["datetime_hour"].dt.floor("h")
    return result, {
        "input_rows": len(frame),
        "invalid_timestamp_rows": invalid_timestamps,
        "missing_station_id_rows": missing_station_ids,
    }


def _station_lag(frame: pd.DataFrame, target: str, periods: int) -> pd.Series:
    """Return values only where the matching station-hour exists."""
    values = frame[target].to_numpy(dtype=float, na_value=np.nan)
    timestamps = _timestamp_ns(frame)
    shifted = np.full(len(frame), np.nan, dtype=float)
    offset_ns = pd.Timedelta(hours=periods).value
    for positions in frame.groupby("station_id", sort=False).indices.values():
        station_timestamps = timestamps[positions]
        source_timestamps = station_timestamps - offset_ns
        source_positions = np.searchsorted(station_timestamps, source_timestamps)
        valid = source_positions < len(positions)
        valid_positions = np.flatnonzero(valid)
        valid[valid_positions] = (
            station_timestamps[source_positions[valid_positions]]
            == source_timestamps[valid_positions]
        )
        destination_positions = positions[valid]
        shifted[destination_positions] = values[positions[source_positions[valid]]]
    return pd.Series(shifted, index=frame.index)


def _station_rolling_mean(frame: pd.DataFrame, target: str) -> pd.Series:
    """Average available values in the three calendar hours before each row."""
    result = np.full(len(frame), np.nan, dtype=float)
    for positions in frame.groupby("station_id", sort=False).indices.values():
        group = frame.iloc[positions]
        values = pd.Series(
            group[target].to_numpy(dtype=float, na_value=np.nan),
            index=pd.DatetimeIndex(group["datetime_hour"]),
        )
        result[positions] = (
            values.rolling("3h", closed="left", min_periods=1).mean().to_numpy()
        )
    return pd.Series(result, index=frame.index)


def _hour_gap_count(frame: pd.DataFrame) -> int:
    hour_ns = pd.Timedelta(hours=1).value
    gaps = 0
    timestamps = _timestamp_ns(frame)
    for positions in frame.groupby("station_id", sort=False).indices.values():
        station_timestamps = timestamps[positions]
        expected_hours = (
            (station_timestamps[-1] - station_timestamps[0]) // hour_ns
        ) + 1
        gaps += int(expected_hours - len(station_timestamps))
    return gaps


def _timestamp_ns(frame: pd.DataFrame) -> np.ndarray:
    timestamps = pd.to_datetime(frame["datetime_hour"], utc=True).dt.tz_localize(None)
    return timestamps.astype("datetime64[ns]").astype("int64").to_numpy()


def _history_feature_columns() -> list[str]:
    return [
        f"{target}_{suffix}"
        for target in TARGETS
        for suffix in ("lag_1h", "lag_2h", "lag_24h", "lag_168h", "rolling_3h")
    ]


def make_features(hourly: pd.DataFrame, drop_missing: bool = True) -> pd.DataFrame:
    """Create hourly features and attach cleaning/history counts in ``attrs``."""
    frame, stats = _normalise_timestamp_column(hourly)
    frame = frame.sort_values(["station_id", "datetime_hour"]).reset_index(drop=True)
    duplicate_hours = frame.duplicated(["station_id", "datetime_hour"]).sum()
    if duplicate_hours:
        raise ValueError(
            "Dữ liệu station-hour có nhiều dòng cho cùng station_id và giờ "
            "sau khi chuẩn hóa."
        )
    frame["timestamp"] = frame["datetime_hour"]
    frame["hour"] = frame["datetime_hour"].dt.hour
    frame["day_of_week"] = frame["datetime_hour"].dt.day_of_week
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["month"] = frame["datetime_hour"].dt.month
    frame["is_peak_hour"] = frame["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    for target in TARGETS:
        frame[f"{target}_lag_1h"] = _station_lag(frame, target, 1)
        frame[f"{target}_lag_2h"] = _station_lag(frame, target, 2)
        frame[f"{target}_lag_24h"] = _station_lag(frame, target, 24)
        frame[f"{target}_lag_168h"] = _station_lag(frame, target, 168)
        frame[f"{target}_rolling_3h"] = _station_rolling_mean(frame, target)

    history_columns = _history_feature_columns()
    missing_history = frame[history_columns].isna()
    valid_targets = frame[list(TARGETS)].notna().all(axis=1)
    stats.update(
        {
            "normalized_rows": len(frame),
            "feature_input_rows": int(valid_targets.sum()),
            "hour_gaps": _hour_gap_count(frame),
            "missing_target_rows": int((~valid_targets).sum()),
            "rows_missing_history": int(
                (missing_history.any(axis=1) & valid_targets).sum()
            ),
            "missing_history_values": int(
                missing_history.loc[valid_targets].sum().sum()
            ),
        }
    )
    if drop_missing:
        rows_before_drop = len(frame)
        frame = frame.dropna().reset_index(drop=True)
        stats["rows_dropped"] = rows_before_drop - len(frame)
    else:
        stats["rows_dropped"] = 0
    stats["rows_dropped_missing_history"] = (
        stats["rows_missing_history"] if drop_missing else 0
    )
    stats["output_rows"] = len(frame)
    stats["drop_missing"] = drop_missing
    frame.attrs["feature_stats"] = stats
    frame.attrs.update(stats)
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
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu station-hour: {source}")
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name("station_features.parquet")
    )
    if source == destination:
        raise ValueError("Tệp đầu ra feature không được trùng với tệp station-hour đầu vào.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB chưa được cài đặt để tạo features ngoài RAM.") from exc

    def sql_path(path: Path) -> str:
        return "'" + str(path.resolve()).replace("\\", "/").replace("'", "''") + "'"

    temporary_destination = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    connection = duckdb.connect()
    try:
        connection.execute("SET preserve_insertion_order=false")
        input_columns = {
            str(column[0])
            for column in connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({sql_path(source)})"
            ).fetchall()
        }
        missing_columns = {"timestamp", "station_id", *TARGETS}.difference(input_columns)
        if missing_columns:
            raise ValueError(
                "Dữ liệu station-hour thiếu cột bắt buộc: "
                + ", ".join(sorted(missing_columns))
            )
        source_cte = f"""
            source AS (
                SELECT
                    * REPLACE (
                        date_trunc('hour', try_cast(timestamp AS TIMESTAMP)) AS timestamp,
                        try_cast(pickups AS DOUBLE) AS pickups,
                        try_cast(returns AS DOUBLE) AS returns
                    ),
                    date_trunc('hour', try_cast(timestamp AS TIMESTAMP)) AS datetime_hour
                FROM read_parquet({sql_path(source)})
            )
        """
        source_stats = connection.execute(
            f"""
            WITH {source_cte},
            valid_source AS (
                SELECT *
                FROM source
                WHERE datetime_hour IS NOT NULL
                  AND station_id IS NOT NULL
                  AND pickups IS NOT NULL
                  AND returns IS NOT NULL
            ),
            station_source AS (
                SELECT *
                FROM source
                WHERE datetime_hour IS NOT NULL
                  AND station_id IS NOT NULL
            ),
            station_gaps AS (
                SELECT
                    date_diff('hour', min(datetime_hour), max(datetime_hour))
                    + 1 - count(*) AS missing_hours
                FROM station_source
                GROUP BY station_id
            ),
            duplicate_hours AS (
                SELECT station_id, datetime_hour
                FROM station_source
                GROUP BY station_id, datetime_hour
                HAVING count(*) > 1
            )
            SELECT
                (SELECT count(*) FROM source) AS input_rows,
                (SELECT count(*) FROM source WHERE datetime_hour IS NOT NULL)
                    AS valid_timestamp_rows,
                (SELECT count(*) FROM source
                 WHERE datetime_hour IS NOT NULL AND station_id IS NOT NULL)
                    AS valid_station_rows,
                (SELECT count(*) FROM valid_source) AS valid_rows,
                coalesce((SELECT sum(missing_hours) FROM station_gaps), 0)
                    AS hour_gaps,
                (SELECT count(*) FROM duplicate_hours) AS duplicate_hours
            """
        ).fetchone()
        (
            input_rows,
            valid_timestamp_rows,
            valid_station_rows,
            valid_rows,
            hour_gaps,
            duplicate_hours,
        ) = (int(value) for value in source_stats)
        if duplicate_hours:
            raise ValueError(
                "Parquet station-hour có nhiều dòng cho cùng station_id và giờ "
                "sau khi chuẩn hóa."
            )

        connection.execute(
            f"""
            COPY (
                WITH {source_cte},
                valid_source AS (
                    SELECT *
                    FROM source
                    WHERE datetime_hour IS NOT NULL
                      AND station_id IS NOT NULL
                      AND pickups IS NOT NULL
                      AND returns IS NOT NULL
                ),
                windowed AS (
                    SELECT
                        valid_source.*,
                        avg(pickups) OVER (
                            PARTITION BY station_id
                            ORDER BY datetime_hour
                            RANGE BETWEEN INTERVAL '3 hours' PRECEDING
                                      AND INTERVAL '1 hour' PRECEDING
                        ) AS pickups_rolling_3h,
                        avg(returns) OVER (
                            PARTITION BY station_id
                            ORDER BY datetime_hour
                            RANGE BETWEEN INTERVAL '3 hours' PRECEDING
                                      AND INTERVAL '1 hour' PRECEDING
                        ) AS returns_rolling_3h
                    FROM valid_source
                ),
                features AS (
                    SELECT
                        windowed.*,
                        date_part('hour', windowed.datetime_hour)::INTEGER AS hour,
                        (date_part('isodow', windowed.datetime_hour)::INTEGER - 1) AS day_of_week,
                        (date_part('isodow', windowed.datetime_hour)::INTEGER >= 6)::INTEGER AS is_weekend,
                        date_part('month', windowed.datetime_hour)::INTEGER AS month,
                        (
                            date_part('hour', windowed.datetime_hour)::INTEGER
                            IN (7, 8, 9, 17, 18, 19)
                        )::INTEGER AS is_peak_hour,
                        lag_1.pickups AS pickups_lag_1h,
                        lag_2.pickups AS pickups_lag_2h,
                        lag_24.pickups AS pickups_lag_24h,
                        lag_168.pickups AS pickups_lag_168h,
                        lag_1.returns AS returns_lag_1h,
                        lag_2.returns AS returns_lag_2h,
                        lag_24.returns AS returns_lag_24h,
                        lag_168.returns AS returns_lag_168h
                    FROM windowed
                    LEFT JOIN valid_source AS lag_1
                      ON lag_1.station_id = windowed.station_id
                     AND lag_1.datetime_hour = windowed.datetime_hour - INTERVAL '1 hour'
                    LEFT JOIN valid_source AS lag_2
                      ON lag_2.station_id = windowed.station_id
                     AND lag_2.datetime_hour = windowed.datetime_hour - INTERVAL '2 hours'
                    LEFT JOIN valid_source AS lag_24
                      ON lag_24.station_id = windowed.station_id
                     AND lag_24.datetime_hour = windowed.datetime_hour - INTERVAL '24 hours'
                    LEFT JOIN valid_source AS lag_168
                      ON lag_168.station_id = windowed.station_id
                     AND lag_168.datetime_hour = windowed.datetime_hour - INTERVAL '168 hours'
                )
                SELECT *
                FROM features
                WHERE pickups IS NOT NULL
                  AND returns IS NOT NULL
                  AND pickups_lag_1h IS NOT NULL
                  AND pickups_lag_2h IS NOT NULL
                  AND pickups_lag_24h IS NOT NULL
                  AND pickups_lag_168h IS NOT NULL
                  AND pickups_rolling_3h IS NOT NULL
                  AND returns_lag_1h IS NOT NULL
                  AND returns_lag_2h IS NOT NULL
                  AND returns_lag_24h IS NOT NULL
                  AND returns_lag_168h IS NOT NULL
                  AND returns_rolling_3h IS NOT NULL
                ORDER BY station_id, datetime_hour
            ) TO {sql_path(temporary_destination)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        row_count = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet({sql_path(temporary_destination)})"
            ).fetchone()[0]
        )
        preview = connection.execute(
            f"SELECT * FROM read_parquet({sql_path(temporary_destination)}) LIMIT 5"
        ).df()
        temporary_destination.replace(destination)
    except ValueError:
        raise
    except (duckdb.Error, OSError) as exc:
        raise RuntimeError(f"Không thể tạo feature table: {destination}") from exc
    finally:
        connection.close()
        if temporary_destination.exists():
            temporary_destination.unlink()
    stats = {
        "input_rows": input_rows,
        "invalid_timestamp_rows": input_rows - valid_timestamp_rows,
        "missing_station_id_rows": valid_timestamp_rows - valid_station_rows,
        "missing_target_rows": valid_station_rows - valid_rows,
        "normalized_rows": valid_station_rows,
        "feature_input_rows": valid_rows,
        "hour_gaps": hour_gaps,
        "rows_missing_history": valid_rows - row_count,
        "rows_dropped_missing_history": valid_rows - row_count,
        "rows_dropped": valid_station_rows - row_count,
        "input_rows_dropped": input_rows - row_count,
        "output_rows": row_count,
        "drop_missing": True,
    }
    preview.attrs["row_count"] = row_count
    preview.attrs["feature_stats"] = stats
    preview.attrs.update(stats)
    return preview
