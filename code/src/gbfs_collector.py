"""GBFS station-status collection and SQLite persistence for Use Case 3."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests


logger = logging.getLogger(__name__)

DEFAULT_STATION_INFORMATION_URL = (
    "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
)
DEFAULT_STATION_STATUS_URL = (
    "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
)
DEFAULT_MAX_AGE_SECONDS = 15 * 60
MAX_FUTURE_SKEW_SECONDS = 5 * 60

STATUS_COLUMNS = [
    "station_id",
    "bikes_available",
    "docks_available",
    "last_reported",
    "station_name",
    "capacity",
    "missing_fields",
    "identity_status",
    "is_fresh",
]


def _empty_status_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": pd.Series(dtype="string"),
            "bikes_available": pd.Series(dtype="Int64"),
            "docks_available": pd.Series(dtype="Int64"),
            "last_reported": pd.Series(dtype="Int64"),
            "station_name": pd.Series(dtype="string"),
            "capacity": pd.Series(dtype="Int64"),
            "missing_fields": pd.Series(dtype="string"),
            "identity_status": pd.Series(dtype="string"),
            "is_fresh": pd.Series(dtype="bool"),
        }
    )


def _utc_datetime(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _information_url_for(status_url: str) -> str:
    if status_url == DEFAULT_STATION_STATUS_URL:
        return DEFAULT_STATION_INFORMATION_URL
    parts = urlsplit(status_url)
    filename = Path(parts.path).name
    if filename == "station_status.json":
        path = parts.path[: -len(filename)] + "station_information.json"
        return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))
    return status_url


def _fetch_feed(
    url: str,
    timeout: float,
    feed_name: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        message = f"Không thể lấy GBFS {feed_name}: {exc}"
        logger.warning("%s (endpoint=%s)", message, url)
        return None, message

    data = payload.get("data") if isinstance(payload, dict) else None
    stations = data.get("stations") if isinstance(data, dict) else None
    if not isinstance(stations, list):
        message = f"Phản hồi GBFS {feed_name} không có trường data.stations hợp lệ."
        logger.warning("%s (endpoint=%s)", message, url)
        return None, message
    return stations, None


def _station_id(value: object) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    station_id = str(value).strip()
    if not station_id:
        return None
    if station_id.endswith(".0"):
        station_id = station_id[:-2]
    if station_id.startswith("+"):
        station_id = station_id[1:]
    try:
        number = float(station_id)
        if math.isfinite(number) and number.is_integer():
            station_id = str(int(number))
    except (TypeError, ValueError, OverflowError):
        pass
    return station_id or None


def _optional_nonnegative_int(value: object) -> int | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _quality_for_reported_time(
    value: object,
    now: datetime,
    max_age_seconds: int,
) -> tuple[int | None, bool]:
    reported_at = _optional_nonnegative_int(value)
    if reported_at is None or reported_at == 0:
        return None, False
    age_seconds = now.timestamp() - reported_at
    if age_seconds < -MAX_FUTURE_SKEW_SECONDS or age_seconds > max_age_seconds:
        return reported_at, False
    return reported_at, True


def fetch_gbfs_station_status(
    url: str = DEFAULT_STATION_STATUS_URL,
    timeout: float = 10.0,
    information_url: str | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: datetime | None = None,
    station_aliases: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Fetch, validate, and freshness-check station information and status."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout phải lớn hơn 0.")
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds phải lớn hơn 0.")

    normalized_aliases: dict[str, str] = {}
    if isinstance(station_aliases, dict):
        for alias, canonical in station_aliases.items():
            normalized_alias = _station_id(alias)
            normalized_canonical = _station_id(canonical)
            if normalized_alias is not None and normalized_canonical is not None:
                normalized_aliases[normalized_alias] = normalized_canonical

    current_time = _utc_datetime(now)
    info_url = information_url or _information_url_for(url)
    information_records, information_error = _fetch_feed(
        info_url, timeout, "station_information"
    )
    info_by_id: dict[str, dict[str, Any]] = {}
    duplicate_info_ids: set[str] = set()
    if information_records is not None:
        for record in information_records:
            if not isinstance(record, dict):
                continue
            station_id = _station_id(record.get("station_id"))
            if station_id is not None:
                station_id = normalized_aliases.get(station_id, station_id)
            if station_id is not None:
                if station_id in info_by_id:
                    duplicate_info_ids.add(station_id)
                else:
                    info_by_id[station_id] = record

    status_records, status_error = _fetch_feed(url, timeout, "station_status")
    if status_error is not None or status_records is None:
        return _empty_status_frame(), status_error
    if not status_records:
        message = "API GBFS station_status không trả về trạm nào."
        logger.warning(message)
        return _empty_status_frame(), message

    rows: list[dict[str, object]] = []
    invalid_ids = 0
    invalid_records = 0
    duplicate_ids: set[str] = set()
    seen_ids: set[str] = set()
    stale_count = 0
    mismatched_count = 0
    for record in status_records:
        if not isinstance(record, dict):
            invalid_records += 1
            continue
        station_id = _station_id(record.get("station_id"))
        if station_id is not None:
            station_id = normalized_aliases.get(station_id, station_id)
        if station_id is None:
            invalid_ids += 1
            continue
        if station_id in seen_ids:
            duplicate_ids.add(station_id)
        seen_ids.add(station_id)

        missing_fields: list[str] = []
        bikes = _optional_nonnegative_int(record.get("num_bikes_available"))
        docks = _optional_nonnegative_int(record.get("num_docks_available"))
        reported_at, is_fresh = _quality_for_reported_time(
            record.get("last_reported"), current_time, max_age_seconds
        )
        if bikes is None:
            missing_fields.append("num_bikes_available")
        if docks is None:
            missing_fields.append("num_docks_available")
        if reported_at is None:
            missing_fields.append("last_reported")
        elif not is_fresh:
            missing_fields.append("last_reported_stale_or_future")
            stale_count += 1

        information = info_by_id.get(station_id)
        if information_records is None:
            identity_status = "unverified"
            missing_fields.append("station_information")
            station_name = None
            capacity = None
        elif station_id in duplicate_info_ids:
            identity_status = "ambiguous"
            missing_fields.append("duplicate_station_information")
            station_name = None
            capacity = None
        elif information is None:
            identity_status = "mismatch"
            missing_fields.append("station_information_mismatch")
            station_name = None
            capacity = None
            mismatched_count += 1
        else:
            identity_status = "matched"
            station_name = information.get("name")
            if not isinstance(station_name, str) or not station_name.strip():
                station_name = None
                missing_fields.append("station_name")
            capacity = _optional_nonnegative_int(information.get("capacity"))
            if capacity is None or capacity == 0:
                capacity = None
                missing_fields.append("capacity")
        if (
            capacity is not None
            and bikes is not None
            and docks is not None
            and bikes + docks > capacity
        ):
            missing_fields.append("counts_exceed_capacity")

        if station_id in duplicate_ids:
            missing_fields.append("duplicate_station_id")
        rows.append(
            {
                "station_id": station_id,
                "bikes_available": bikes,
                "docks_available": docks,
                "last_reported": reported_at,
                "station_name": station_name,
                "capacity": capacity,
                "missing_fields": json.dumps(
                    sorted(set(missing_fields)), ensure_ascii=False
                ),
                "identity_status": identity_status,
                "is_fresh": is_fresh,
            }
        )

    if not rows:
        message = "Không có station_id hợp lệ trong phản hồi GBFS."
        logger.warning(message)
        return _empty_status_frame(), message

    for row in rows:
        if str(row["station_id"]) in duplicate_ids:
            missing_fields = json.loads(str(row["missing_fields"]))
            missing_fields.append("duplicate_station_id")
            row["missing_fields"] = json.dumps(
                sorted(set(missing_fields)), ensure_ascii=False
            )
            row["identity_status"] = "ambiguous"

    result = pd.DataFrame(rows, columns=STATUS_COLUMNS)
    for column in ("bikes_available", "docks_available", "last_reported", "capacity"):
        result[column] = pd.array(result[column], dtype="Int64")
    result["station_id"] = result["station_id"].astype("string")
    result["station_name"] = result["station_name"].astype("string")
    result["missing_fields"] = result["missing_fields"].astype("string")
    result["identity_status"] = result["identity_status"].astype("string")
    result["is_fresh"] = result["is_fresh"].astype(bool)

    warnings: list[str] = []
    if information_error is not None:
        warnings.append(information_error)
    if invalid_ids:
        warnings.append(f"Bỏ qua {invalid_ids} bản ghi thiếu station_id hợp lệ.")
    if invalid_records:
        warnings.append(f"Bỏ qua {invalid_records} bản ghi không đúng định dạng object.")
    if duplicate_ids:
        warnings.append(
            f"Có {len(duplicate_ids)} station_id bị lặp; các bản ghi đó không được dùng làm realtime."
        )
    if stale_count:
        warnings.append(
            f"Có {stale_count} bản ghi có last_reported quá cũ hoặc nằm trong tương lai."
        )
    if mismatched_count:
        warnings.append(
            f"Có {mismatched_count} station_id không khớp station_information."
        )
    partial_count = int(
        (
            result["missing_fields"].astype(str).ne("[]")
            | result["identity_status"].ne("matched")
        ).sum()
    )
    if partial_count:
        warnings.append(
            f"Có {partial_count} bản ghi thiếu trường tùy chọn hoặc chưa xác thực đầy đủ."
        )
    if not result["is_fresh"].any():
        warnings.append("Không có trạng thái nào đủ mới để hiển thị là thời gian thực.")
    warning_message = " ".join(warnings) or None
    if warning_message:
        logger.warning("GBFS snapshot có cảnh báo: %s", warning_message)
    return result.reset_index(drop=True), warning_message


def save_station_status_to_db(
    db_path: str | Path,
    status_df: pd.DataFrame,
    collected_at: datetime | None = None,
) -> str:
    """Persist one snapshot and return its UTC collection timestamp."""
    required = {"station_id", "bikes_available", "docks_available"}
    missing = required.difference(status_df.columns)
    if missing:
        raise ValueError(
            "Dữ liệu trạng thái trạm thiếu cột: " + ", ".join(sorted(missing))
        )
    if status_df.empty:
        raise ValueError("Không thể lưu snapshot rỗng vào SQLite.")

    timestamp = _utc_datetime(collected_at).isoformat(timespec="seconds")
    database_path = Path(db_path).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[object, ...]] = []
    for row in status_df.to_dict("records"):
        station_id = _station_id(row.get("station_id"))
        if station_id is None:
            raise ValueError("Không thể lưu bản ghi GBFS thiếu station_id.")
        missing_fields = row.get("missing_fields", "[]")
        if isinstance(missing_fields, (list, tuple, set)):
            missing_fields = json.dumps(
                sorted(str(item) for item in missing_fields), ensure_ascii=False
            )
        elif missing_fields is None or pd.isna(missing_fields):
            missing_fields = "[]"
        else:
            missing_fields = str(missing_fields)

        def nullable_integer(value: object, field: str) -> int | None:
            if value is None or pd.isna(value):
                return None
            integer = _optional_nonnegative_int(value)
            if integer is None:
                raise ValueError(f"{field} phải là số nguyên không âm hoặc rỗng.")
            return integer

        rows.append(
            (
                timestamp,
                station_id,
                nullable_integer(row.get("bikes_available"), "bikes_available"),
                nullable_integer(row.get("docks_available"), "docks_available"),
                nullable_integer(row.get("last_reported"), "last_reported"),
                None if pd.isna(row.get("station_name")) else str(row.get("station_name")),
                nullable_integer(row.get("capacity"), "capacity"),
                str(missing_fields),
                str(row.get("identity_status", "unverified")),
                int(bool(row.get("is_fresh", False))),
            )
        )

    connection = sqlite3.connect(database_path, timeout=10)
    try:
        _ensure_station_status_table(connection)
        connection.executemany(
            """
            INSERT INTO station_status (
                timestamp, station_id, bikes_available, docks_available,
                last_reported, station_name, capacity, missing_fields,
                identity_status, is_fresh
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        connection.close()
    return timestamp


def _ensure_station_status_table(connection: sqlite3.Connection) -> None:
    """Create or safely migrate the UC03 table while preserving old snapshots."""
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'station_status'"
    ).fetchone()
    columns = (
        connection.execute("PRAGMA table_info(station_status)").fetchall()
        if table_exists
        else []
    )
    column_names = {str(column[1]) for column in columns}
    not_null = {str(column[1]): bool(column[3]) for column in columns}
    desired = {
        "timestamp",
        "station_id",
        "bikes_available",
        "docks_available",
        "last_reported",
        "station_name",
        "capacity",
        "missing_fields",
        "identity_status",
        "is_fresh",
    }
    needs_migration = bool(table_exists) and (
        not desired.issubset(column_names)
        or not_null.get("bikes_available", False)
        or not_null.get("docks_available", False)
    )
    if not table_exists or needs_migration:
        if needs_migration:
            connection.execute("DROP TABLE IF EXISTS station_status_migration")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS station_status_migration (
                timestamp TEXT NOT NULL,
                station_id TEXT NOT NULL,
                bikes_available INTEGER,
                docks_available INTEGER,
                last_reported INTEGER,
                station_name TEXT,
                capacity INTEGER,
                missing_fields TEXT NOT NULL DEFAULT '[]',
                identity_status TEXT NOT NULL DEFAULT 'unverified',
                is_fresh INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        if needs_migration:
            def source(column: str, fallback: str) -> str:
                return column if column in column_names else fallback

            connection.execute(
                f"""
                INSERT INTO station_status_migration (
                    timestamp, station_id, bikes_available, docks_available,
                    last_reported, station_name, capacity, missing_fields,
                    identity_status, is_fresh
                )
                SELECT
                    {source("timestamp", "''")},
                    {source("station_id", "''")},
                    {source("bikes_available", "NULL")},
                    {source("docks_available", "NULL")},
                    {source("last_reported", "NULL")},
                    {source("station_name", "NULL")},
                    {source("capacity", "NULL")},
                    {source("missing_fields", "'[]'")},
                    {source("identity_status", "'unverified'")},
                    {source("is_fresh", "0")}
                FROM station_status
                """
            )
            connection.execute("DROP TABLE station_status")
            connection.execute(
                "ALTER TABLE station_status_migration RENAME TO station_status"
            )
        else:
            connection.execute(
                "ALTER TABLE station_status_migration RENAME TO station_status"
            )


def run_gbfs_collector_job(
    db_path: str | Path,
    url: str = DEFAULT_STATION_STATUS_URL,
    timeout: float = 10.0,
    information_url: str | None = None,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    station_aliases: dict[str, str] | None = None,
) -> dict[str, object]:
    """Fetch and persist one GBFS snapshot, preserving partial quality warnings."""
    status_df, fetch_warning = fetch_gbfs_station_status(
        url=url,
        timeout=timeout,
        information_url=information_url,
        max_age_seconds=max_age_seconds,
        station_aliases=station_aliases,
    )
    if status_df.empty:
        return {
            "success": False,
            "records_count": 0,
            "message": fetch_warning or "GBFS không trả về bản ghi trạng thái hợp lệ.",
            "timestamp": None,
            "status_df": status_df,
        }
    try:
        timestamp = save_station_status_to_db(db_path, status_df)
        stored_status = status_df.copy()
        stored_status.insert(0, "timestamp", timestamp)
    except (OSError, sqlite3.Error, TypeError, ValueError, pd.errors.DatabaseError) as exc:
        logger.exception("Không thể lưu snapshot GBFS vào SQLite: %s", exc)
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
        "message": fetch_warning or "Đã thu thập và lưu snapshot trạng thái trạm.",
        "timestamp": timestamp,
        "status_df": stored_status,
        "warnings": fetch_warning,
    }


def load_latest_station_status(
    db_path: str | Path,
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the newest snapshot and mark only verified, fresh rows as usable."""
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds phải lớn hơn 0.")
    empty = _empty_status_frame()
    empty["timestamp"] = pd.Series(dtype="string")
    database_path = Path(db_path).expanduser().resolve()
    missing_state: dict[str, object] = {
        "state": "missing",
        "collected_at": None,
        "fresh_count": 0,
        "total_count": 0,
        "stale_count": 0,
        "unverified_count": 0,
        "mismatched_count": 0,
        "message": "Chưa có snapshot GBFS được lưu.",
    }
    if not database_path.exists():
        return empty, missing_state

    try:
        connection = sqlite3.connect(database_path, timeout=5)
    except sqlite3.Error as exc:
        logger.exception("Không thể mở cơ sở dữ liệu GBFS: %s", exc)
        return empty, {
            **missing_state,
            "state": "error",
            "message": f"Không thể mở cơ sở dữ liệu GBFS: {exc}",
        }
    try:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'station_status'"
        ).fetchone()
        if not table_exists:
            return empty, missing_state
        columns = {
            str(item[1])
            for item in connection.execute("PRAGMA table_info(station_status)")
        }
        required_columns = {"timestamp", "station_id"}
        if not required_columns.issubset(columns):
            return empty, {
                **missing_state,
                "state": "error",
                "message": "Bảng station_status không có các cột bắt buộc.",
            }
        optional_columns = [
            "bikes_available",
            "docks_available",
            "last_reported",
            "station_name",
            "capacity",
            "missing_fields",
            "identity_status",
            "is_fresh",
        ]
        select_columns = ["timestamp", "station_id"] + [
            column for column in optional_columns if column in columns
        ]
        latest_timestamp = connection.execute(
            "SELECT MAX(timestamp) FROM station_status"
        ).fetchone()[0]
        if latest_timestamp is None:
            return empty, missing_state
        selected = ", ".join(f'"{column}"' for column in select_columns)
        frame = pd.read_sql_query(
            f'SELECT {selected} FROM station_status WHERE timestamp = ? ORDER BY station_id',
            connection,
            params=(latest_timestamp,),
        )
    except sqlite3.Error as exc:
        logger.exception("Không thể đọc snapshot GBFS từ SQLite: %s", exc)
        return empty, {
            **missing_state,
            "state": "error",
            "message": f"Không thể đọc snapshot GBFS: {exc}",
        }
    finally:
        connection.close()

    current_time = _utc_datetime(now)
    try:
        collected_at = pd.Timestamp(latest_timestamp)
        if collected_at.tzinfo is None:
            collected_at = collected_at.tz_localize("UTC")
        else:
            collected_at = collected_at.tz_convert("UTC")
        collection_age = (pd.Timestamp(current_time) - collected_at).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return empty, {
            **missing_state,
            "state": "error",
            "message": "Thời điểm thu thập trong SQLite không hợp lệ.",
        }
    collection_fresh = (
        0 <= collection_age <= max_age_seconds
        and collection_age >= -MAX_FUTURE_SKEW_SECONDS
    )

    for column, default in (
        ("bikes_available", pd.NA),
        ("docks_available", pd.NA),
        ("last_reported", pd.NA),
        ("station_name", pd.NA),
        ("capacity", pd.NA),
        ("missing_fields", "[]"),
        ("identity_status", "unverified"),
    ):
        if column not in frame.columns:
            frame[column] = default
    usable: list[bool] = []
    fresh_flags: list[bool] = []
    current_epoch = current_time.timestamp()
    for row in frame.to_dict("records"):
        reported = _optional_nonnegative_int(row.get("last_reported"))
        row_fresh = False
        if reported is not None and reported > 0:
            age = current_epoch - reported
            row_fresh = (
                -MAX_FUTURE_SKEW_SECONDS <= age <= max_age_seconds
            )
        complete_counts = all(
            _optional_nonnegative_int(row.get(column)) is not None
            for column in ("bikes_available", "docks_available")
        )
        try:
            row_warnings = json.loads(str(row.get("missing_fields", "[]")))
        except (TypeError, ValueError):
            row_warnings = ["invalid_missing_fields"]
        counts_consistent = "counts_exceed_capacity" not in row_warnings
        fresh_flags.append(bool(collection_fresh and row_fresh))
        usable.append(
            bool(
                collection_fresh
                and row_fresh
                and complete_counts
                and counts_consistent
                and row.get("identity_status") == "matched"
            )
        )
    frame["is_fresh"] = fresh_flags
    frame["is_usable"] = usable
    frame["timestamp"] = frame["timestamp"].astype("string")
    for column in ("bikes_available", "docks_available", "last_reported", "capacity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    usable_count = int(sum(usable))
    total_count = len(frame)
    mismatch_count = int((frame["identity_status"] == "mismatch").sum())
    unverified_count = int((frame["identity_status"] == "unverified").sum())
    stale_count = total_count - int(sum(fresh_flags))
    optional_missing_count = int(frame["missing_fields"].astype(str).ne("[]").sum())
    if usable_count:
        state = "fresh" if usable_count == total_count else "partial"
        message = (
            f"{usable_count:,}/{total_count:,} bản ghi đủ điều kiện GBFS thời gian thực."
        )
    elif not collection_fresh:
        state = "stale"
        message = "Snapshot GBFS đã quá hạn; không dùng làm tồn kho thời gian thực."
    else:
        state = "partial"
        reasons = []
        if stale_count:
            reasons.append(f"{stale_count:,} bản ghi quá hạn")
        if mismatch_count:
            reasons.append(f"{mismatch_count:,} station_id không khớp")
        if unverified_count:
            reasons.append(f"{unverified_count:,} bản ghi chưa xác thực")
        message = (
            "Snapshot không có bản ghi nào đủ điều kiện xác thực và cập nhật"
            + (": " + ", ".join(reasons) if reasons else ".")
        )
    return frame, {
        "state": state,
        "collected_at": collected_at.isoformat(timespec="seconds"),
        "age_seconds": max(0, int(collection_age)),
        "fresh_count": usable_count,
        "total_count": total_count,
        "stale_count": stale_count,
        "unverified_count": unverified_count,
        "mismatched_count": mismatch_count,
        "optional_missing_count": optional_missing_count,
        "collection_fresh": collection_fresh,
        "message": message,
    }

