"""Trip validation, hourly aggregation, and 24/7 zero filling."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _normalise_trip_columns(trips: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "tripduration": "trip_duration",
        "trip_duration": "trip_duration",
        "starttime": "start_time",
        "start_time": "start_time",
        "stoptime": "end_time",
        "end_time": "end_time",
        "start station id": "start_station_id",
        "start_station_id": "start_station_id",
        "start station name": "start_station_name",
        "start_station_name": "start_station_name",
        "start station latitude": "start_lat",
        "start_lat": "start_lat",
        "start station longitude": "start_lon",
        "start_lon": "start_lon",
        "end station id": "end_station_id",
        "end_station_id": "end_station_id",
        "end station name": "end_station_name",
        "end_station_name": "end_station_name",
        "end station latitude": "end_lat",
        "end_lat": "end_lat",
        "end station longitude": "end_lon",
        "end_lon": "end_lon",
    }
    rename = {
        column: aliases[str(column).strip().lower()]
        for column in trips.columns
        if str(column).strip().lower() in aliases
    }
    return trips.rename(columns=rename)


def _clean_trips_with_stats(
    trips: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "input_rows": len(trips),
        "removed_invalid_timestamp": 0,
        "removed_missing_station_id": 0,
        "removed_invalid_coordinates": 0,
        "removed_time_order": 0,
        "removed_duration": 0,
        "removed_same_station_short": 0,
        "removed_duplicate_rows": 0,
    }
    if trips.empty:
        return trips.copy(), stats
    frame = _normalise_trip_columns(trips.copy())
    required = [
        "start_time", "end_time", "trip_duration",
        "start_station_id", "end_station_id",
        "start_lat", "start_lon", "end_lat", "end_lon",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Thiếu cột bắt buộc cho UC02: " + ", ".join(missing))

    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce")
    frame["trip_duration"] = pd.to_numeric(frame["trip_duration"], errors="coerce")
    frame["start_station_id"] = pd.to_numeric(frame["start_station_id"], errors="coerce")
    frame["end_station_id"] = pd.to_numeric(frame["end_station_id"], errors="coerce")
    for column in ("start_lat", "start_lon", "end_lat", "end_lon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    def retain(mask: pd.Series, stat_key: str) -> None:
        nonlocal frame
        stats[stat_key] = int((~mask).sum())
        frame = frame.loc[mask].copy()

    valid_timestamps = frame["start_time"].notna() & frame["end_time"].notna()
    retain(valid_timestamps, "removed_invalid_timestamp")

    int32_info = np.iinfo(np.int32)
    valid_station_ids = pd.Series(True, index=frame.index)
    for column in ("start_station_id", "end_station_id"):
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        valid_station_ids &= (
            pd.Series(
                np.isfinite(values)
                & (values >= int32_info.min)
                & (values <= int32_info.max)
                & (values == np.floor(values)),
                index=frame.index,
            )
        )
    retain(valid_station_ids, "removed_missing_station_id")

    valid_coordinates = pd.Series(True, index=frame.index)
    for column, lower, upper in (
        ("start_lat", -90, 90),
        ("end_lat", -90, 90),
        ("start_lon", -180, 180),
        ("end_lon", -180, 180),
    ):
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        valid_coordinates &= pd.Series(
            np.isfinite(values) & (values >= lower) & (values <= upper),
            index=frame.index,
        )
    retain(valid_coordinates, "removed_invalid_coordinates")

    valid_time_order = frame["end_time"] > frame["start_time"]
    retain(valid_time_order, "removed_time_order")

    valid_duration = frame["trip_duration"].between(60, 7200, inclusive="both")
    retain(valid_duration, "removed_duration")

    frame["start_station_id"] = frame["start_station_id"].astype("int32")
    frame["end_station_id"] = frame["end_station_id"].astype("int32")
    valid_station_visit = ~(
        (frame["start_station_id"] == frame["end_station_id"])
        & (frame["trip_duration"] < 180)
    )
    retain(valid_station_visit, "removed_same_station_short")

    duplicate_count = len(frame) - len(frame.drop_duplicates())
    stats["removed_duplicate_rows"] = duplicate_count
    frame = frame.drop_duplicates().reset_index(drop=True)
    stats["clean_rows"] = len(frame)
    stats["removed_rows"] = stats["input_rows"] - len(frame)
    return frame, stats


def clean_trips(trips: pd.DataFrame) -> pd.DataFrame:
    cleaned, _ = _clean_trips_with_stats(trips)
    return cleaned


def build_station_metadata(trips: pd.DataFrame) -> pd.DataFrame:
    """Derive station coordinates and capacities when no static table exists."""
    if trips.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "lat",
                "lon",
                "capacity",
                "capacity_is_estimated",
                "capacity_source",
            ]
        )
    starts = trips.rename(columns={"start_station_id": "station_id", "start_station_name": "station_name",
                                   "start_lat": "lat", "start_lon": "lon"})
    ends = trips.rename(columns={"end_station_id": "station_id", "end_station_name": "station_name",
                                 "end_lat": "lat", "end_lon": "lon"})
    stations = pd.concat([starts[["station_id", "station_name", "lat", "lon"]],
                          ends[["station_id", "station_name", "lat", "lon"]]])
    stations["lat"] = pd.to_numeric(stations["lat"], errors="coerce")
    stations["lon"] = pd.to_numeric(stations["lon"], errors="coerce")
    valid_coordinates = (
        np.isfinite(stations[["lat", "lon"]].to_numpy(dtype=float, na_value=np.nan)).all(axis=1)
        & stations["lat"].between(-90, 90).to_numpy()
        & stations["lon"].between(-180, 180).to_numpy()
    )
    stations = stations.loc[valid_coordinates].dropna(subset=["station_id"]).copy()
    if stations.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "lat",
                "lon",
                "capacity",
                "capacity_is_estimated",
                "capacity_source",
            ]
        )
    stations = (
        stations.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            lat=("lat", "median"),
            lon=("lon", "median"),
        )
    )
    stations["station_id"] = stations["station_id"].astype("int32")
    stations["capacity"] = 25
    stations["capacity_is_estimated"] = True
    stations["capacity_source"] = "estimated_default"
    return stations[
        [
            "station_id",
            "station_name",
            "lat",
            "lon",
            "capacity",
            "capacity_is_estimated",
            "capacity_source",
        ]
    ]


def aggregate_hourly(trips: pd.DataFrame, station_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate pickups/returns and fill every station-hour in the observed period."""
    clean = clean_trips(trips)
    if clean.empty:
        return pd.DataFrame(columns=["timestamp", "station_id", "pickups", "returns"])
    starts = clean.groupby(
        [clean["start_time"].dt.floor("h"), "start_station_id"]
    ).size().rename("pickups")
    returns = clean.groupby(
        [clean["end_time"].dt.floor("h"), "end_station_id"]
    ).size().rename("returns")
    hourly = pd.concat([starts, returns], axis=1).fillna(0).rename_axis(["timestamp", "station_id"]).reset_index()
    station_ids = (
        station_metadata["station_id"].tolist()
        if station_metadata is not None and not station_metadata.empty
        else hourly["station_id"].drop_duplicates().tolist()
    )
    timestamps = pd.date_range(
        hourly["timestamp"].min(),
        hourly["timestamp"].max(),
        freq="h",
    )
    full = pd.MultiIndex.from_product(
        [station_ids, timestamps], names=["station_id", "timestamp"]
    ).to_frame(index=False)
    hourly = full.merge(hourly, on=["timestamp", "station_id"], how="left").fillna({"pickups": 0, "returns": 0})
    hourly["pickups"] = hourly["pickups"].astype(int)
    hourly["returns"] = hourly["returns"].astype(int)
    hourly["net_flow"] = hourly["returns"] - hourly["pickups"]
    return hourly.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def process_and_build_parquet_grid(
    input_directory: str | Path,
    output_directory: str | Path,
    input_parquet: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run UC02 cleaning, station metadata extraction, and hourly zero filling."""
    output_path = Path(output_directory).expanduser().resolve()
    full_parquet = output_path / "uc1_trips_deduplicated.parquet"
    input_path = (
        Path(input_parquet).expanduser().resolve()
        if input_parquet is not None
        else full_parquet
    )
    if full_parquet.exists():
        input_path = full_parquet

    if input_path.exists() and input_path.suffix.lower() == ".parquet":
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError("DuckDB chưa được cài đặt để xử lý Parquet UC01.") from exc

        output_path.mkdir(parents=True, exist_ok=True)

        def sql_path(path: Path) -> str:
            return "'" + str(path.resolve()).replace("\\", "/").replace("'", "''") + "'"

        connection = duckdb.connect()
        try:
            connection.execute("SET memory_limit='80%'")
            connection.execute(f"SET threads={max(1, min(8, os.cpu_count() or 4))}")
            connection.execute("SET preserve_insertion_order=false")
            source_columns = {
                str(column[0])
                for column in connection.execute(
                    f"DESCRIBE SELECT * FROM read_parquet({sql_path(input_path)})"
                ).fetchall()
            }
            required_columns = {
                "tripduration",
                "starttime",
                "stoptime",
                "start station id",
                "start station name",
                "start station latitude",
                "start station longitude",
                "end station id",
                "end station name",
                "end station latitude",
                "end station longitude",
            }
            missing_columns = required_columns.difference(source_columns)
            if missing_columns:
                raise ValueError(
                    "Parquet UC02 thiếu cột bắt buộc: "
                    + ", ".join(sorted(missing_columns))
                )
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE uc2_typed_trips AS
                SELECT
                    try_cast(tripduration AS INTEGER) AS tripduration,
                    try_cast(starttime AS TIMESTAMP) AS starttime,
                    try_cast(stoptime AS TIMESTAMP) AS stoptime,
                    try_cast("start station id" AS INTEGER) AS "start station id",
                    CAST("start station name" AS VARCHAR) AS "start station name",
                    try_cast("start station latitude" AS DOUBLE) AS "start station latitude",
                    try_cast("start station longitude" AS DOUBLE) AS "start station longitude",
                    try_cast("end station id" AS INTEGER) AS "end station id",
                    CAST("end station name" AS VARCHAR) AS "end station name",
                    try_cast("end station latitude" AS DOUBLE) AS "end station latitude",
                    try_cast("end station longitude" AS DOUBLE) AS "end station longitude"
                FROM read_parquet({sql_path(input_path)})
                """
            )
            source_rows = int(
                connection.execute(
                    "SELECT count(*) FROM uc2_typed_trips"
                ).fetchone()[0]
            )
            audit_counts = connection.execute(
                """
                SELECT
                    count(*) FILTER (
                        WHERE starttime IS NOT NULL AND stoptime IS NOT NULL
                    ) AS valid_timestamps,
                    count(*) FILTER (
                        WHERE starttime IS NOT NULL AND stoptime IS NOT NULL
                          AND "start station id" IS NOT NULL
                          AND "end station id" IS NOT NULL
                    ) AS valid_station_ids,
                    count(*) FILTER (
                        WHERE starttime IS NOT NULL AND stoptime IS NOT NULL
                          AND "start station id" IS NOT NULL
                          AND "end station id" IS NOT NULL
                          AND "start station latitude" BETWEEN -90 AND 90
                          AND "end station latitude" BETWEEN -90 AND 90
                          AND "start station longitude" BETWEEN -180 AND 180
                          AND "end station longitude" BETWEEN -180 AND 180
                    ) AS valid_coordinates,
                    count(*) FILTER (
                        WHERE starttime IS NOT NULL AND stoptime IS NOT NULL
                          AND "start station id" IS NOT NULL
                          AND "end station id" IS NOT NULL
                          AND "start station latitude" BETWEEN -90 AND 90
                          AND "end station latitude" BETWEEN -90 AND 90
                          AND "start station longitude" BETWEEN -180 AND 180
                          AND "end station longitude" BETWEEN -180 AND 180
                          AND stoptime > starttime
                    ) AS valid_time_order,
                    count(*) FILTER (
                        WHERE starttime IS NOT NULL AND stoptime IS NOT NULL
                          AND "start station id" IS NOT NULL
                          AND "end station id" IS NOT NULL
                          AND "start station latitude" BETWEEN -90 AND 90
                          AND "end station latitude" BETWEEN -90 AND 90
                          AND "start station longitude" BETWEEN -180 AND 180
                          AND "end station longitude" BETWEEN -180 AND 180
                          AND stoptime > starttime
                          AND tripduration BETWEEN 60 AND 7200
                    ) AS valid_duration,
                    count(*) FILTER (
                        WHERE starttime IS NOT NULL AND stoptime IS NOT NULL
                          AND "start station id" IS NOT NULL
                          AND "end station id" IS NOT NULL
                          AND "start station latitude" BETWEEN -90 AND 90
                          AND "end station latitude" BETWEEN -90 AND 90
                          AND "start station longitude" BETWEEN -180 AND 180
                          AND "end station longitude" BETWEEN -180 AND 180
                          AND stoptime > starttime
                          AND tripduration BETWEEN 60 AND 7200
                          AND NOT (
                              "start station id" = "end station id"
                              AND tripduration < 180
                          )
                    ) AS valid_station_visits
                FROM uc2_typed_trips
                """
            ).fetchone()
            (
                valid_timestamps,
                valid_station_ids,
                valid_coordinates,
                valid_time_order,
                valid_duration,
                valid_station_visits,
            ) = (int(value) for value in audit_counts)
            connection.execute(
                """
                CREATE OR REPLACE TEMP TABLE cleaned_trips AS
                SELECT DISTINCT *
                FROM uc2_typed_trips
                WHERE starttime IS NOT NULL
                  AND stoptime IS NOT NULL
                  AND "start station id" IS NOT NULL
                  AND "end station id" IS NOT NULL
                  AND "start station latitude" BETWEEN -90 AND 90
                  AND "end station latitude" BETWEEN -90 AND 90
                  AND "start station longitude" BETWEEN -180 AND 180
                  AND "end station longitude" BETWEEN -180 AND 180
                  AND stoptime > starttime
                  AND tripduration BETWEEN 60 AND 7200
                  AND NOT (
                      "start station id" = "end station id"
                      AND tripduration < 180
                  )
                """
            )
            clean_rows = int(
                connection.execute("SELECT count(*) FROM cleaned_trips").fetchone()[0]
            )
            if clean_rows == 0:
                raise ValueError(
                    "Không còn chuyến đi hợp lệ sau khi kiểm tra thời gian, "
                    "station_id, tọa độ GPS và quy tắc thời lượng."
                )
            connection.execute(
                f"""
                COPY (
                    SELECT
                        station_id,
                        any_value(station_name) AS station_name,
                        median(lat) AS lat,
                        median(lon) AS lon,
                        25 AS capacity,
                        TRUE AS capacity_is_estimated,
                        'estimated_default' AS capacity_source
                    FROM (
                        SELECT
                            "start station id" AS station_id,
                            "start station name" AS station_name,
                            "start station latitude" AS lat,
                            "start station longitude" AS lon
                        FROM cleaned_trips
                        UNION ALL
                        SELECT
                            "end station id",
                            "end station name",
                            "end station latitude",
                            "end station longitude"
                        FROM cleaned_trips
                    )
                    GROUP BY station_id
                ) TO {sql_path(output_path / "station_metadata.parquet")}
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            station_count = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({sql_path(output_path / 'station_metadata.parquet')})"
                ).fetchone()[0]
            )
            connection.execute(
                f"""
                COPY (
                    WITH events AS (
                        SELECT
                            date_trunc('hour', starttime) AS timestamp,
                            "start station id" AS station_id,
                            count(*)::INTEGER AS pickups,
                            0::INTEGER AS returns
                        FROM cleaned_trips
                        GROUP BY 1, 2
                        UNION ALL
                        SELECT
                            date_trunc('hour', stoptime),
                            "end station id",
                            0::INTEGER,
                            count(*)::INTEGER
                        FROM cleaned_trips
                        GROUP BY 1, 2
                    ),
                    event_totals AS (
                        SELECT timestamp, station_id,
                               sum(pickups)::INTEGER AS pickups,
                               sum(returns)::INTEGER AS returns
                        FROM events
                        GROUP BY 1, 2
                    ),
                    bounds AS (
                        SELECT min(timestamp) AS first_hour, max(timestamp) AS last_hour
                        FROM event_totals
                    ),
                    hours AS (
                        SELECT unnest(generate_series(
                            first_hour, last_hour, INTERVAL '1 hour'
                        )) AS timestamp
                        FROM bounds
                    ),
                    station_list AS (
                        SELECT station_id
                        FROM read_parquet({sql_path(output_path / "station_metadata.parquet")})
                    )
                    SELECT
                        station_list.station_id,
                        hours.timestamp,
                        coalesce(event_totals.pickups, 0)::INTEGER AS pickups,
                        coalesce(event_totals.returns, 0)::INTEGER AS returns,
                        (
                            coalesce(event_totals.returns, 0)
                            - coalesce(event_totals.pickups, 0)
                        )::INTEGER AS net_flow
                    FROM station_list
                    CROSS JOIN hours
                    LEFT JOIN event_totals
                      ON event_totals.timestamp = hours.timestamp
                     AND event_totals.station_id = station_list.station_id
                ) TO {sql_path(output_path / "station_hour_cleaned.parquet")}
                (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            grid_stats = connection.execute(
                f"""
                SELECT
                    count(*) AS station_hour_rows,
                    count(*) FILTER (
                        WHERE pickups > 0 OR returns > 0
                    ) AS observed_station_hours,
                    count(*) FILTER (
                        WHERE pickups = 0 AND returns = 0
                    ) AS zero_filled_cells,
                    min(timestamp) AS t_min,
                    max(timestamp) AS t_max
                FROM read_parquet({sql_path(output_path / 'station_hour_cleaned.parquet')})
                """
            ).fetchone()
            station_hour_rows = int(grid_stats[0])
            observed_station_hours = int(grid_stats[1])
            zero_filled_cells = int(grid_stats[2])
            preview = connection.execute(
                f"SELECT * FROM read_parquet({sql_path(output_path / 'station_hour_cleaned.parquet')}) LIMIT 5"
            ).df()
            stations = connection.execute(
                f"SELECT * FROM read_parquet({sql_path(output_path / 'station_metadata.parquet')})"
            ).df()
        finally:
            connection.close()
        return preview, stations, {
            "source_rows": source_rows,
            "input_rows": source_rows,
            "removed_invalid_timestamp": source_rows - valid_timestamps,
            "removed_missing_station_id": valid_timestamps - valid_station_ids,
            "removed_invalid_coordinates": valid_station_ids - valid_coordinates,
            "removed_time_order": valid_coordinates - valid_time_order,
            "removed_duration": valid_time_order - valid_duration,
            "removed_same_station_short": valid_duration - valid_station_visits,
            "removed_duplicate_rows": valid_station_visits - clean_rows,
            "clean_rows": clean_rows,
            "removed_rows": source_rows - clean_rows,
            "station_count": station_count,
            "station_hour_rows": station_hour_rows,
            "observed_station_hours": observed_station_hours,
            "zero_filled_cells": zero_filled_cells,
            "t_min": str(grid_stats[3]),
            "t_max": str(grid_stats[4]),
            "capacity": 25,
            "capacity_estimated": True,
            "capacity_source": "estimated_default",
        }

    from .data_loader import load_trip_data

    trips = load_trip_data(Path(input_directory).expanduser().resolve())
    if trips.empty:
        raise ValueError("Không tìm thấy dữ liệu chuyến đi để thực hiện UC02.")
    source_rows = len(trips)
    cleaned, cleaning_stats = _clean_trips_with_stats(trips)
    if cleaned.empty:
        raise ValueError("Không còn bản ghi hợp lệ sau khi làm sạch UC02.")
    station_metadata = build_station_metadata(cleaned)
    if station_metadata.empty:
        raise ValueError("Không thể tạo metadata trạm từ dữ liệu hợp lệ.")
    hourly = aggregate_hourly(cleaned, station_metadata)
    save_processed(hourly, station_metadata, output_path)
    observed_station_hours = int(
        (hourly["pickups"].gt(0) | hourly["returns"].gt(0)).sum()
    )
    return hourly.head(5), station_metadata, {
        **cleaning_stats,
        "source_rows": source_rows,
        "clean_rows": len(cleaned),
        "removed_rows": source_rows - len(cleaned),
        "station_count": len(station_metadata),
        "station_hour_rows": len(hourly),
        "observed_station_hours": observed_station_hours,
        "zero_filled_cells": len(hourly) - observed_station_hours,
        "t_min": str(hourly["timestamp"].min()),
        "t_max": str(hourly["timestamp"].max()),
        "capacity": 25,
        "capacity_estimated": True,
        "capacity_source": "estimated_default",
    }


def save_processed(
    hourly: pd.DataFrame,
    station_metadata: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    path = Path(output_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in (("station_hour_cleaned", hourly), ("station_metadata", station_metadata)):
        try:
            frame.to_parquet(path / f"{name}.parquet", index=False)
        except (ImportError, ValueError):
            frame.to_csv(path / f"{name}.csv", index=False)
