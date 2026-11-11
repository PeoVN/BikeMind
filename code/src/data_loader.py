"""Reliable readers for raw Citi Bike files and processed BikeMind tables."""

from __future__ import annotations

import hashlib
import gc
import json
import logging
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

logger = logging.getLogger(__name__)

CORE_TRIP_COLUMNS = [
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
]

CORE_TRIP_DTYPES = {
    "tripduration": "int32",
    "start station id": "float32",
    "start station name": "category",
    "start station latitude": "float32",
    "start station longitude": "float32",
    "end station id": "float32",
    "end station name": "category",
    "end station latitude": "float32",
    "end station longitude": "float32",
}


TRIP_COLUMN_ALIASES = {
    "start station id": "start_station_id",
    "start station name": "start_station_name",
    "start station latitude": "start_lat",
    "start station longitude": "start_lon",
    "end station id": "end_station_id",
    "end station name": "end_station_name",
    "end station latitude": "end_lat",
    "end station longitude": "end_lon",
    "starttime": "start_time",
    "stoptime": "end_time",
    "tripduration": "trip_duration",
}


def calculate_file_hash(file_path: str | Path) -> str:
    """Return an MD5 digest calculated in 64 KiB chunks.

    An empty string indicates that the file could not be read. The caller can
    then skip the file while keeping the rest of the ingestion run alive.
    """
    digest = hashlib.md5()
    path = Path(file_path)
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
    except (OSError, PermissionError) as exc:
        logger.warning("Không thể đọc file để tính MD5: %s (%s)", path, exc)
        return ""
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    """Backward-compatible alias used by the generic CSV scanner."""
    return calculate_file_hash(path)


def _default_data_directory() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data"


def _is_ignored_path(path: Path) -> bool:
    return any(part.lower() == "__macosx" or part.startswith("._") for part in path.parts)


def scan_and_load_unique_csv(
    base_directory: str | Path | None = None,
    progress_callback: Callable[[int, int, str, int], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recursively load unique Citi Bike CSV files for Use Case 1.

    Files are deduplicated by MD5 before loading and rows are deduplicated
    after concatenation. Invalid or unreadable files are skipped with a log.
    """
    base_path = Path(base_directory) if base_directory is not None else _default_data_directory()
    candidates = [
        path for path in base_path.rglob("*.csv")
        if path.is_file() and path.suffix.lower() == ".csv"
        and not _is_ignored_path(path) and path.stat().st_size > 0
    ] if base_path.exists() else []
    stats = {"total": len(candidates), "unique": 0, "dup_files": 0, "dup_rows": 0}
    seen_hashes: set[str] = set()
    frames: list[pd.DataFrame] = []

    if progress_callback is not None:
        progress_callback(0, stats["total"], "Đang chuẩn bị danh sách file CSV...", 0)

    for processed_files, path in enumerate(sorted(candidates), start=1):
        file_hash = calculate_file_hash(path)
        if not file_hash:
            if progress_callback is not None:
                progress_callback(processed_files, stats["total"], "Không thể đọc file, đang chuyển sang file tiếp theo...", 0)
            continue
        if file_hash in seen_hashes:
            stats["dup_files"] += 1
            logger.info("Bỏ qua file trùng MD5: %s", path)
            print(f"[UC1] Duplicate file skipped (MD5): {path}")
            if progress_callback is not None:
                progress_callback(processed_files, stats["total"], f"Bỏ qua file trùng: {path.name}", sum(len(frame) for frame in frames))
            continue
        seen_hashes.add(file_hash)
        if progress_callback is not None:
            progress_callback(processed_files, stats["total"], f"Đang đọc dữ liệu: {path.name}", sum(len(frame) for frame in frames))
        try:
            frame = pd.read_csv(
                path,
                usecols=lambda column: str(column).strip().lower() in {item.lower() for item in CORE_TRIP_COLUMNS},
                dtype=CORE_TRIP_DTYPES,
                low_memory=False,
            )
        except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, TypeError) as exc:
            logger.warning("Bỏ qua file CSV không hợp lệ %s: %s", path, exc)
            if progress_callback is not None:
                progress_callback(processed_files, stats["total"], f"File không hợp lệ: {path.name}", sum(len(item) for item in frames))
            continue
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        missing = [column for column in CORE_TRIP_COLUMNS if column not in frame.columns]
        if missing:
            logger.warning("Bỏ qua %s vì thiếu cột bắt buộc: %s", path, ", ".join(missing))
            continue
        # A category is efficient while reading each file, but concatenating
        # many independent category dictionaries causes a large temporary
        # union allocation inside pandas. Normalize before batching.
        for name_column in ("start station name", "end station name"):
            frame[name_column] = frame[name_column].astype("string")
        frames.append(frame[CORE_TRIP_COLUMNS])
        stats["unique"] += 1
        if progress_callback is not None:
            progress_callback(processed_files, stats["total"], f"Đã nạp: {path.name}", sum(len(item) for item in frames))

    if not frames:
        if progress_callback is not None:
            progress_callback(stats["total"], stats["total"], "Không tìm thấy dữ liệu CSV hợp lệ.", 0)
        return pd.DataFrame(columns=CORE_TRIP_COLUMNS), stats
    merged_parts: list[pd.DataFrame] = []
    batch_size = 8
    for start in range(0, len(frames), batch_size):
        merged_parts.append(pd.concat(frames[start:start + batch_size], ignore_index=True, copy=False))
    del frames
    gc.collect()
    merged = pd.concat(merged_parts, ignore_index=True, copy=False)
    del merged_parts
    gc.collect()
    before_dedup = len(merged)
    merged = merged.drop_duplicates(ignore_index=True)
    for name_column in ("start station name", "end station name"):
        merged[name_column] = merged[name_column].astype("category")
    stats["dup_rows"] = before_dedup - len(merged)
    if progress_callback is not None:
        progress_callback(stats["total"], stats["total"], "Đã khử trùng bản ghi và hoàn tất nạp dữ liệu.", len(merged))
    return merged, stats


def scan_and_prepare_unique_csv(
    base_directory: str | Path | None = None,
    output_directory: str | Path | None = None,
    progress_callback: Callable[[int, int, str, int], None] | None = None,
) -> tuple[pd.DataFrame, dict[str, int], int, int]:
    """Process very large CSV collections with DuckDB spill-to-disk execution.

    Only a five-row preview is returned to Python. The complete deduplicated
    trips remain on disk, while dashboard aggregates are written to Parquet.
    """
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB chưa được cài đặt. Chạy: pip install duckdb") from exc

    base_path = Path(base_directory) if base_directory is not None else _default_data_directory()
    output_path = Path(output_directory) if output_directory is not None else base_path.parent / "code" / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    candidates = [
        path for path in base_path.rglob("*.csv")
        if path.is_file() and not _is_ignored_path(path) and path.stat().st_size > 0
    ] if base_path.exists() else []
    stats = {"total": len(candidates), "unique": 0, "dup_files": 0, "dup_rows": 0}
    seen_hashes: set[str] = set()
    unique_paths: list[Path] = []
    file_hashes: dict[Path, str] = {}
    for index, path in enumerate(sorted(candidates), start=1):
        digest = calculate_file_hash(path)
        if digest and digest not in seen_hashes:
            seen_hashes.add(digest)
            file_hashes[path] = digest
            unique_paths.append(path)
            stats["unique"] += 1
        elif digest:
            stats["dup_files"] += 1
        if progress_callback is not None:
            progress_callback(index, len(candidates), f"Đã kiểm tra MD5: {path.name}", 0)
    if not unique_paths:
        return pd.DataFrame(columns=CORE_TRIP_COLUMNS), stats, 0, 0

    manifest_path = output_path / "uc1_loaded_csv_files.json"
    previous_manifest: list[dict[str, object]] = []
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            previous_manifest = []
    previous_hashes = {
        str(item["md5"])
        for item in previous_manifest
        if isinstance(item, dict) and item.get("md5")
    }
    current_hashes = set(file_hashes.values())
    trip_output = output_path / "uc1_trips_deduplicated.parquet"
    can_increment = (
        bool(previous_hashes)
        and trip_output.exists()
        and previous_hashes.issubset(current_hashes)
    )
    incremental_paths = [
        path for path in unique_paths
        if file_hashes.get(path) not in previous_hashes
    ] if can_increment else unique_paths
    if can_increment and not incremental_paths and (
        not (output_path / "station_hour_cleaned.parquet").exists()
        or not (output_path / "station_metadata.parquet").exists()
    ):
        can_increment = False
        incremental_paths = unique_paths
    if can_increment and not incremental_paths:
        connection = duckdb.connect()
        try:
            preview = connection.execute(
                "SELECT * FROM read_parquet(?) LIMIT 5", [str(trip_output)]
            ).df()
            unique_count = int(connection.execute(
                "SELECT count(*) FROM read_parquet(?)", [str(trip_output)]
            ).fetchone()[0])
            hourly_count = int(connection.execute(
                "SELECT count(*) FROM read_parquet(?)",
                [str(output_path / "station_hour_cleaned.parquet")],
            ).fetchone()[0])
            station_count = int(connection.execute(
                "SELECT count(*) FROM read_parquet(?)",
                [str(output_path / "station_metadata.parquet")],
            ).fetchone()[0])
        finally:
            connection.close()
        stats["dup_rows"] = 0
        if progress_callback is not None:
            progress_callback(len(candidates), len(candidates), "Không có file CSV mới, sử dụng dữ liệu đã xử lý.", unique_count)
        return preview, {**stats, "raw_rows": unique_count, "row_count": unique_count}, hourly_count, station_count

    def sql_path(path: Path) -> str:
        return "'" + str(path.resolve()).replace("\\", "/").replace("'", "''") + "'"

    file_list = "[" + ", ".join(sql_path(path) for path in incremental_paths) + "]"
    normalized_csv_sql = f"""
        SELECT
            try_cast(tripduration AS INTEGER) AS tripduration,
            try_cast(starttime AS TIMESTAMP) AS starttime,
            try_cast(stoptime AS TIMESTAMP) AS stoptime,
            try_cast("start station id" AS DOUBLE) AS "start station id",
            CAST("start station name" AS VARCHAR) AS "start station name",
            try_cast("start station latitude" AS DOUBLE) AS "start station latitude",
            try_cast("start station longitude" AS DOUBLE) AS "start station longitude",
            try_cast("end station id" AS DOUBLE) AS "end station id",
            CAST("end station name" AS VARCHAR) AS "end station name",
            try_cast("end station latitude" AS DOUBLE) AS "end station latitude",
            try_cast("end station longitude" AS DOUBLE) AS "end station longitude"
        FROM read_csv_auto({file_list}, union_by_name=true, ignore_errors=true, null_padding=true)
    """
    if can_increment:
        source_sql = f"""
            SELECT * FROM read_parquet({sql_path(trip_output)})
            UNION ALL
            {normalized_csv_sql}
        """
    else:
        source_sql = normalized_csv_sql
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit='4GB'")
        connection.execute("SET threads=2")
        connection.execute("SET max_temp_directory_size='200GB'")
        temp_directory = output_path / "duckdb_tmp"
        temp_directory.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory='{str(temp_directory).replace(chr(92), '/').replace(chr(39), chr(39) * 2)}'")
        connection.execute("SET preserve_insertion_order=false")
        if progress_callback is not None:
            progress_callback(len(candidates), len(candidates), "DuckDB đang đọc CSV ngoài RAM...", 0)
        connection.execute(f"CREATE OR REPLACE TEMP TABLE raw_trips AS {source_sql}")
        raw_count = int(connection.execute("SELECT count(*) FROM raw_trips").fetchone()[0])
        connection.execute("""
            CREATE OR REPLACE TEMP TABLE unique_trips AS
            SELECT DISTINCT *
            FROM raw_trips
            WHERE tripduration IS NOT NULL
              AND starttime IS NOT NULL
              AND stoptime IS NOT NULL
              AND "start station id" IS NOT NULL
              AND "end station id" IS NOT NULL
        """)
        unique_count = int(connection.execute("SELECT count(*) FROM unique_trips").fetchone()[0])
        stats["dup_rows"] = max(0, raw_count - unique_count)
        connection.execute("DROP TABLE raw_trips")
        connection.execute(f"COPY unique_trips TO '{str(trip_output).replace(chr(92), '/').replace(chr(39), chr(39) * 2)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        connection.execute("""
            COPY (
                SELECT
                    CAST(station_id AS VARCHAR) AS station_id,
                    any_value(station_name) AS station_name,
                    any_value(lat) AS lat,
                    any_value(lon) AS lon,
                    40 AS capacity
                FROM (
                    SELECT "start station id" AS station_id, "start station name" AS station_name,
                           "start station latitude" AS lat, "start station longitude" AS lon
                    FROM unique_trips
                    UNION ALL
                    SELECT "end station id", "end station name",
                           "end station latitude", "end station longitude"
                    FROM unique_trips
                )
                WHERE station_id IS NOT NULL AND lat IS NOT NULL AND lon IS NOT NULL
                GROUP BY station_id
            ) TO '""" + str(output_path / "station_metadata.parquet").replace("\\", "/").replace("'", "''") + """' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        connection.execute("""
            COPY (
                SELECT date_trunc('hour', starttime) AS timestamp,
                       CAST("start station id" AS VARCHAR) AS station_id,
                       count(*)::INTEGER AS pickups,
                       0::INTEGER AS returns
                FROM unique_trips GROUP BY 1, 2
                UNION ALL
                SELECT date_trunc('hour', stoptime),
                       CAST("end station id" AS VARCHAR),
                       0::INTEGER, count(*)::INTEGER
                FROM unique_trips GROUP BY 1, 2
            ) TO '""" + str(output_path / "station_hour_events.parquet").replace("\\", "/").replace("'", "''") + """' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        connection.execute("""
            COPY (
                WITH event_totals AS (
                    SELECT timestamp, station_id,
                           sum(pickups)::INTEGER AS pickups,
                           sum(returns)::INTEGER AS returns
                    FROM read_parquet('""" + str(output_path / "station_hour_events.parquet").replace("\\", "/").replace("'", "''") + """')
                    GROUP BY 1, 2
                ),
                bounds AS (
                    SELECT min(timestamp) AS first_hour, max(timestamp) AS last_hour
                    FROM event_totals
                ),
                hours AS (
                    SELECT unnest(generate_series(first_hour, last_hour, INTERVAL '1 hour')) AS timestamp
                    FROM bounds
                ),
                station_list AS (
                    SELECT CAST(station_id AS VARCHAR) AS station_id
                    FROM read_parquet('""" + str(output_path / "station_metadata.parquet").replace("\\", "/").replace("'", "''") + """')
                )
                SELECT hours.timestamp, station_list.station_id,
                       coalesce(event_totals.pickups, 0)::INTEGER AS pickups,
                       coalesce(event_totals.returns, 0)::INTEGER AS returns
                FROM hours
                CROSS JOIN station_list
                LEFT JOIN event_totals
                  ON event_totals.timestamp = hours.timestamp
                 AND event_totals.station_id = station_list.station_id
            ) TO '""" + str(output_path / "station_hour_cleaned.parquet").replace("\\", "/").replace("'", "''") + """' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        preview = connection.execute("SELECT * FROM unique_trips LIMIT 5").df()
        station_count = int(connection.execute("SELECT count(*) FROM read_parquet(?)", [str(output_path / "station_metadata.parquet")]).fetchone()[0])
        hourly_count = int(connection.execute("SELECT count(*) FROM read_parquet(?)", [str(output_path / "station_hour_cleaned.parquet")]).fetchone()[0])
        manifest = []
        for path in unique_paths:
            file_stat = path.stat()
            manifest.append(
                {
                    "file": path.name,
                    "relative_path": str(path.relative_to(base_path)),
                    "md5": file_hashes[path],
                    "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                    "modified_at": pd.Timestamp(file_stat.st_mtime, unit="s").isoformat(timespec="seconds"),
                }
            )
        (output_path / "uc1_loaded_csv_files.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if progress_callback is not None:
            progress_callback(len(candidates), len(candidates), "Đã dedup ngoài RAM và ghi Parquet dashboard.", unique_count)
        return preview, {**stats, "raw_rows": raw_count, "row_count": unique_count}, hourly_count, station_count
    except duckdb.OutOfMemoryException as exc:
        raise RuntimeError(
            "Dữ liệu vượt quá bộ nhớ khả dụng khi tạo station-hour. "
            "DuckDB đã dùng chế độ spill-to-disk; hãy giải phóng thêm dung lượng RAM/ổ đĩa tạm rồi thử lại."
        ) from exc
    finally:
        connection.close()


def scan_csv_files(roots: str | Path | Iterable[str | Path]) -> list[Path]:
    """Recursively find CSVs, ignoring macOS metadata and duplicate file bytes."""
    root_list = [roots] if isinstance(roots, (str, Path)) else list(roots)
    files: list[Path] = []
    seen: set[str] = set()
    for root in root_list:
        root_path = Path(root)
        if root_path.is_file() and root_path.suffix.lower() == ".csv":
            candidates = [root_path]
        else:
            candidates = root_path.rglob("*.csv") if root_path.exists() else []
        for path in candidates:
            if "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            digest = _file_digest(path)
            if not digest:
                continue
            if digest not in seen:
                files.append(path)
                seen.add(digest)
    return sorted(files)


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: TRIP_COLUMN_ALIASES.get(str(col).strip().lower(), str(col).strip()) for col in frame.columns}
    return frame.rename(columns=renamed)


def load_trip_data(roots: str | Path | Iterable[str | Path], limit_files: int | None = None) -> pd.DataFrame:
    """Read raw files, remove duplicate rows, and standardise column names."""
    files = scan_csv_files(roots)
    if limit_files is not None:
        files = files[:limit_files]
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_csv(path, low_memory=False)
            frame = _normalise_columns(frame)
            frame["_source_file"] = str(path)
            frames.append(frame)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            continue
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    dedup_cols = [col for col in result.columns if col != "_source_file"]
    return result.drop_duplicates(subset=dedup_cols).reset_index(drop=True)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except (ImportError, ValueError, OSError):
            csv_path = path.with_suffix(".csv")
            return pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_station_metadata(processed_dir: str | Path) -> pd.DataFrame:
    """Load station metadata from Parquet/CSV and provide a stable schema."""
    directory = Path(processed_dir)
    for name in ("station_metadata.parquet", "station_metadata.csv"):
        path = directory / name
        if path.exists():
            frame = _read_table(path)
            break
    else:
        frame = pd.DataFrame()
    if frame.empty:
        return frame
    frame = _normalise_columns(frame)
    aliases = {
        "station_id": "station_id",
        "id": "station_id",
        "station name": "station_name",
        "latitude": "lat",
        "longitude": "lon",
        "capacity": "capacity",
        "available bikes": "bikes_available",
    }
    return frame.rename(columns={col: aliases.get(str(col).lower(), col) for col in frame.columns})


def _dashboard_parquet(processed_dir: str | Path) -> Path:
    directory = Path(processed_dir)
    parquet_path = directory / "station_hour_cleaned.parquet"
    if parquet_path.exists():
        return parquet_path
    return directory / "station_hour_cleaned.csv"


def query_hourly_metadata(processed_dir: str | Path) -> dict[str, object]:
    """Read station-hour counts and bounds without materialising the table."""
    path = _dashboard_parquet(processed_dir)
    if not path.exists():
        return {"row_count": 0, "first_timestamp": None, "last_timestamp": None}
    try:
        import duckdb

        connection = duckdb.connect()
        try:
            row = connection.execute(
                "SELECT count(*), min(timestamp), max(timestamp) FROM read_parquet(?)",
                [str(path)],
            ).fetchone()
        finally:
            connection.close()
    except (ImportError, OSError, ValueError):
        frame = pd.read_csv(path, usecols=["timestamp"])
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
        return {
            "row_count": len(timestamps),
            "first_timestamp": timestamps.min() if not timestamps.empty else None,
            "last_timestamp": timestamps.max() if not timestamps.empty else None,
        }
    return {"row_count": int(row[0]), "first_timestamp": row[1], "last_timestamp": row[2]}


def query_station_snapshot(
    processed_dir: str | Path,
    timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Return the latest row per station at or before a dashboard timestamp."""
    path = _dashboard_parquet(processed_dir)
    if not path.exists():
        return pd.DataFrame()
    import duckdb

    connection = duckdb.connect()
    try:
        if path.suffix.lower() == ".parquet":
            source = "read_parquet(?)"
            parameters: list[object] = [str(path), timestamp.to_pydatetime()]
        else:
            source = "read_csv_auto(?)"
            parameters = [str(path), timestamp.to_pydatetime()]
        return connection.execute(
            f"""
            SELECT station_id, pickups, returns, timestamp
            FROM (
                SELECT *,
                       row_number() OVER (
                           PARTITION BY station_id
                           ORDER BY timestamp DESC
                       ) AS row_number
                FROM {source}
                WHERE timestamp <= ?
            )
            WHERE row_number = 1
            """,
            parameters,
        ).df()
    finally:
        connection.close()


def query_station_history(
    processed_dir: str | Path,
    station_id: str,
    end_time: pd.Timestamp | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Return a bounded station history for forecast features and charts."""
    path = _dashboard_parquet(processed_dir)
    if not path.exists():
        return pd.DataFrame()
    import duckdb

    connection = duckdb.connect()
    try:
        source = "read_parquet(?)" if path.suffix.lower() == ".parquet" else "read_csv_auto(?)"
        parameters: list[object] = [str(path), str(station_id)]
        time_clause = ""
        if end_time is not None:
            time_clause = "AND timestamp <= ?"
            parameters.append(end_time.to_pydatetime())
        parameters.append(limit)
        return connection.execute(
            f"""
            SELECT timestamp, CAST(station_id AS VARCHAR) AS station_id, pickups, returns
            FROM {source}
            WHERE CAST(station_id AS VARCHAR) = ?
              {time_clause}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            parameters,
        ).df().sort_values("timestamp")
    finally:
        connection.close()


def query_hourly_total(processed_dir: str | Path) -> float:
    """Return total pickups without loading the complete station-hour table."""
    path = _dashboard_parquet(processed_dir)
    if not path.exists():
        return 0.0
    import duckdb

    connection = duckdb.connect()
    try:
        source = "read_parquet(?)" if path.suffix.lower() == ".parquet" else "read_csv_auto(?)"
        return float(connection.execute(f"SELECT coalesce(sum(pickups), 0) FROM {source}", [str(path)]).fetchone()[0])
    finally:
        connection.close()
