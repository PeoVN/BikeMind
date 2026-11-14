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


def clean_trips(trips: pd.DataFrame) -> pd.DataFrame:
    if trips.empty:
        return trips.copy()
    frame = _normalise_trip_columns(trips.copy())
    required = [
        "start_time", "end_time", "trip_duration",
        "start_station_id", "end_station_id",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("Thiếu cột bắt buộc cho UC02: " + ", ".join(missing))
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce")
    frame["trip_duration"] = pd.to_numeric(frame["trip_duration"], errors="coerce")
    frame["start_station_id"] = pd.to_numeric(frame["start_station_id"], errors="coerce")
    frame["end_station_id"] = pd.to_numeric(frame["end_station_id"], errors="coerce")
    frame = frame.dropna(subset=required)
    frame["start_station_id"] = frame["start_station_id"].astype("int32")
    frame["end_station_id"] = frame["end_station_id"].astype("int32")
    frame = frame[frame["trip_duration"].between(60, 7200)]
    frame = frame[frame["end_time"] > frame["start_time"]]
    frame = frame[
        ~(
            (frame["start_station_id"] == frame["end_station_id"])
            & (frame["trip_duration"] < 180)
        )
    ]
    return frame.drop_duplicates().reset_index(drop=True)


def build_station_metadata(trips: pd.DataFrame) -> pd.DataFrame:
    """Derive station coordinates and capacities when no static table exists."""
    if trips.empty:
        return pd.DataFrame(columns=["station_id", "station_name", "lat", "lon", "capacity"])
    starts = trips.rename(columns={"start_station_id": "station_id", "start_station_name": "station_name",
                                   "start_lat": "lat", "start_lon": "lon"})
    ends = trips.rename(columns={"end_station_id": "station_id", "end_station_name": "station_name",
                                 "end_lat": "lat", "end_lon": "lon"})
    stations = pd.concat([starts[["station_id", "station_name", "lat", "lon"]],
                          ends[["station_id", "station_name", "lat", "lon"]]])
    stations = stations.dropna(subset=["station_id", "lat", "lon"])
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
    return stations[["station_id", "station_name", "lat", "lon", "capacity"]]


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
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Run UC02 cleaning, station metadata extraction, and hourly zero filling."""
    output_path = Path(output_directory)
    full_parquet = output_path / "uc1_trips_deduplicated.parquet"
    input_path = Path(input_parquet) if input_parquet is not None else full_parquet
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
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE cleaned_trips AS
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
                WHERE tripduration BETWEEN 60 AND 7200
                  AND starttime IS NOT NULL
                  AND stoptime IS NOT NULL
                  AND try_cast("start station id" AS INTEGER) IS NOT NULL
                  AND try_cast("end station id" AS INTEGER) IS NOT NULL
                  AND NOT (
                      try_cast("start station id" AS INTEGER)
                      = try_cast("end station id" AS INTEGER)
                      AND tripduration < 180
                  )
                  AND stoptime > starttime
                GROUP BY ALL
                """
            )
            source_rows = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({sql_path(input_path)})"
                ).fetchone()[0]
            )
            clean_rows = int(
                connection.execute("SELECT count(*) FROM cleaned_trips").fetchone()[0]
            )
            connection.execute(
                f"""
                COPY (
                    SELECT
                        station_id,
                        any_value(station_name) AS station_name,
                        median(lat) AS lat,
                        median(lon) AS lon,
                        25 AS capacity
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
                    WHERE lat IS NOT NULL AND lon IS NOT NULL
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
            station_hour_rows = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({sql_path(output_path / 'station_hour_cleaned.parquet')})"
                ).fetchone()[0]
            )
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
            "clean_rows": clean_rows,
            "removed_rows": source_rows - clean_rows,
            "station_count": station_count,
            "station_hour_rows": station_hour_rows,
        }

    from .data_loader import load_trip_data

    trips = load_trip_data(input_directory)
    if trips.empty:
        raise ValueError("Không tìm thấy dữ liệu chuyến đi để thực hiện UC02.")
    source_rows = len(trips)
    cleaned = clean_trips(trips)
    if cleaned.empty:
        raise ValueError("Không còn bản ghi hợp lệ sau khi làm sạch UC02.")
    station_metadata = build_station_metadata(cleaned)
    if station_metadata.empty:
        raise ValueError("Không thể tạo metadata trạm từ dữ liệu hợp lệ.")
    hourly = aggregate_hourly(cleaned, station_metadata)
    save_processed(hourly, station_metadata, str(output_directory))
    return hourly.head(5), station_metadata, {
        "source_rows": source_rows,
        "clean_rows": len(cleaned),
        "removed_rows": source_rows - len(cleaned),
        "station_count": len(station_metadata),
        "station_hour_rows": len(hourly),
    }


def save_processed(hourly: pd.DataFrame, station_metadata: pd.DataFrame, output_dir: str) -> None:
    from pathlib import Path
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in (("station_hour_cleaned", hourly), ("station_metadata", station_metadata)):
        try:
            frame.to_parquet(path / f"{name}.parquet", index=False)
        except (ImportError, ValueError):
            frame.to_csv(path / f"{name}.csv", index=False)
