"""GBFS station-status collection and SQLite persistence for Use Case 3."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


DEFAULT_STATION_STATUS_URL = (
    "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
)


def fetch_gbfs_station_status(
    url: str = DEFAULT_STATION_STATUS_URL,
    timeout: float = 10.0,
) -> tuple[pd.DataFrame, str | None]:
    """Fetch the current GBFS station status without fabricating fallback data."""
    columns = [
        "station_id",
        "bikes_available",
        "docks_available",
        "last_reported",
    ]
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        return pd.DataFrame(columns=columns), f"Không thể lấy dữ liệu GBFS: {exc}"

    stations = payload.get("data", {}).get("stations")
    if not isinstance(stations, list):
        return pd.DataFrame(columns=columns), (
            "Phản hồi GBFS không có trường data.stations hợp lệ."
        )

    frame = pd.DataFrame(stations)
    if frame.empty:
        return pd.DataFrame(columns=columns), "API GBFS không trả về trạm nào."
    missing = [
        column
        for column in ("station_id", "num_bikes_available", "num_docks_available")
        if column not in frame.columns
    ]
    if missing:
        return pd.DataFrame(columns=columns), (
            "Phản hồi GBFS thiếu trường bắt buộc: " + ", ".join(missing)
        )

    result = pd.DataFrame(
        {
            "station_id": frame["station_id"].astype("string").str.strip(),
            "bikes_available": pd.to_numeric(
                frame["num_bikes_available"], errors="coerce"
            ),
            "docks_available": pd.to_numeric(
                frame["num_docks_available"], errors="coerce"
            ),
            "last_reported": pd.to_numeric(
                frame.get("last_reported", pd.Series(pd.NA, index=frame.index)),
                errors="coerce",
            ),
        }
    )
    result = result.dropna(
        subset=["station_id", "bikes_available", "docks_available"]
    )
    if result.empty:
        return pd.DataFrame(columns=columns), (
            "Không có bản ghi trạng thái trạm hợp lệ trong phản hồi GBFS."
        )
    result["bikes_available"] = result["bikes_available"].astype("int64")
    result["docks_available"] = result["docks_available"].astype("int64")
    result["last_reported"] = result["last_reported"].astype("Int64")
    return result.reset_index(drop=True), None


def save_station_status_to_db(
    db_path: str | Path,
    status_df: pd.DataFrame,
    collected_at: datetime | None = None,
) -> str:
    """Persist one station-status snapshot and return its collection timestamp."""
    required = {"station_id", "bikes_available", "docks_available"}
    missing = required.difference(status_df.columns)
    if missing:
        raise ValueError(
            "Dữ liệu trạng thái trạm thiếu cột: " + ", ".join(sorted(missing))
        )
    if status_df.empty:
        raise ValueError("Không thể lưu snapshot rỗng vào SQLite.")

    timestamp = (collected_at or datetime.now(timezone.utc)).isoformat(
        timespec="seconds"
    )
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            timestamp,
            str(row.station_id),
            int(row.bikes_available),
            int(row.docks_available),
        )
        for row in status_df.itertuples(index=False)
    ]
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS station_status (
                timestamp TEXT NOT NULL,
                station_id TEXT NOT NULL,
                bikes_available INTEGER NOT NULL,
                docks_available INTEGER NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO station_status
                (timestamp, station_id, bikes_available, docks_available)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return timestamp


def run_gbfs_collector_job(
    db_path: str | Path,
    url: str = DEFAULT_STATION_STATUS_URL,
    timeout: float = 10.0,
) -> dict[str, object]:
    """Fetch and persist one GBFS snapshot."""
    status_df, fetch_error = fetch_gbfs_station_status(url=url, timeout=timeout)
    if fetch_error is not None:
        return {
            "success": False,
            "records_count": 0,
            "message": fetch_error,
            "timestamp": None,
            "status_df": status_df,
        }
    try:
        timestamp = save_station_status_to_db(db_path, status_df)
        connection = sqlite3.connect(db_path)
        try:
            cursor = connection.execute(
                """
                SELECT timestamp, station_id, bikes_available, docks_available
                FROM station_status
                WHERE timestamp = ?
                ORDER BY station_id
                """,
                (timestamp,),
            )
            stored_status = pd.DataFrame(
                cursor.fetchall(),
                columns=["timestamp", "station_id", "bikes_available", "docks_available"],
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError, pd.errors.DatabaseError) as exc:
        return {
            "success": False,
            "records_count": 0,
            "message": f"Không thể lưu snapshot GBFS vào SQLite: {exc}",
            "timestamp": None,
            "status_df": status_df,
        }
    return {
        "success": True,
        "records_count": len(status_df),
        "message": "Đã thu thập và lưu snapshot trạng thái trạm.",
        "timestamp": timestamp,
        "status_df": stored_status,
    }

