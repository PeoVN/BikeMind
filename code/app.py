"""BikeMind - Smart Mobility Dashboard & Decision Support System."""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import sqlite3
import shutil
import tempfile
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import folium
import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Fullscreen
from streamlit_folium import st_folium

from src.data_cleaner import (
    aggregate_hourly,
    build_station_metadata,
    clean_trips,
    process_and_build_parquet_grid,
    save_processed,
)
from src.data_loader import (
    load_station_metadata,
    load_trip_data,
    query_hourly_metadata,
    query_hourly_total,
    query_station_history,
    query_station_snapshot,
    scan_and_prepare_unique_csv,
    scan_and_load_unique_csv,
    scan_csv_files,
)
from src.feature_engineering import build_features_pipeline, make_features, feature_columns
from src.gbfs_collector import load_latest_station_status, run_gbfs_collector_job
from src.model_trainer import benchmark_models, load_model, train_models
from src.rebalancing_engine import (
    build_incentive_recommendations,
    calculate_expected_inventory,
    clean_incentive_executions,
    generate_rebalancing_plan,
    run_greedy_rebalancing,
)
from src.system_health import collect_system_health


ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = ROOT.parent / "data"
DATA_LOAD_DIRECTORY = ROOT.parent / "data_load"
PROCESSED = ROOT / "processed"
MODELS = ROOT / "models"
DB_PATH = (ROOT / "bikemind_decisions.sqlite").resolve()
UC1_DATA_PATH = PROCESSED / "uc1_loaded_trips.parquet"
UC1_FULL_DATA_PATH = PROCESSED / "uc1_trips_deduplicated.parquet"
UC1_DATA_CSV_PATH = PROCESSED / "uc1_loaded_trips.csv"
UC1_STATS_PATH = PROCESSED / "uc1_load_stats.json"
UC1_CSV_MANIFEST_PATH = PROCESSED / "uc1_loaded_csv_files.json"
UC1_REJECTED_CSV_PATH = PROCESSED / "uc1_rejected_csv_files.json"
USE_CASE_STATE_PATH = PROCESSED / "use_case_state.json"
FEATURES_PATH = PROCESSED / "station_features.parquet"
BENCHMARK_PATH = PROCESSED / "bikemind_benchmark.csv"
BACKTESTING_PATH = PROCESSED / "bikemind_backtesting_report.csv"
_UC013_EXECUTION_CSV_REQUIRED_COLUMNS = [
    "event_id",
    "recommendation_id",
    "station_id",
    "status",
    "reconciliation_status",
    "executed_at",
    "bikes_returned",
]
_UC013_EXECUTION_CSV_COLUMNS = [
    *_UC013_EXECUTION_CSV_REQUIRED_COLUMNS,
    "source",
    "is_estimated",
    "updated_at",
]


def _file_signature(path: Path | None) -> tuple[int, int]:
    if path is None:
        return 0, 0
    try:
        stat = path.stat()
    except FileNotFoundError:
        return 0, 0
    return stat.st_mtime_ns, stat.st_size


def _demo_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    names = ["Central Park South", "W 42 St", "Broadway & 14 St", "Hudson River", "East Village",
             "City Hall", "Union Square", "SoHo", "Brooklyn Bridge", "Chelsea"]
    coords = [(40.767, -73.971), (40.758, -73.989), (40.735, -73.991), (40.735, -74.011),
              (40.727, -73.985), (40.713, -74.006), (40.736, -73.99), (40.724, -74.0),
              (40.706, -73.997), (40.746, -74.001)]
    stations = pd.DataFrame({"station_id": range(1, 11), "station_name": names,
                             "lat": [item[0] for item in coords], "lon": [item[1] for item in coords],
                             "capacity": [40, 50, 35, 45, 30, 55, 42, 38, 48, 36]})
    end = pd.Timestamp.now().floor("h")
    timestamps = pd.date_range(end - pd.Timedelta(days=14), end, freq="h")
    rows = []
    for timestamp in timestamps:
        for station in stations.itertuples():
            rush = 1.8 if timestamp.hour in (8, 9, 17, 18) else 0.7
            weekend = 0.8 if timestamp.dayofweek >= 5 else 1.0
            rows.append((timestamp, station.station_id,
                         int(rng.poisson(3.2 * rush * weekend)),
                         int(rng.poisson(2.9 * (2.1 - rush / 2) * weekend))))
    hourly = pd.DataFrame(rows, columns=["timestamp", "station_id", "pickups", "returns"])
    return hourly, stations


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    stations = load_station_metadata(PROCESSED)
    hourly_path = PROCESSED / "station_hour_cleaned.parquet"
    if hourly_path.exists():
        try:
            metadata = query_hourly_metadata(PROCESSED)
            last_timestamp = metadata["last_timestamp"]
            hourly = query_station_snapshot(PROCESSED, pd.Timestamp(last_timestamp))
            hourly.attrs.update(metadata)
        except (ImportError, ValueError, OSError):
            hourly = pd.read_csv(PROCESSED / "station_hour_cleaned.csv") if (PROCESSED / "station_hour_cleaned.csv").exists() else pd.DataFrame()
    elif (PROCESSED / "station_hour_cleaned.csv").exists():
        hourly = pd.read_csv(PROCESSED / "station_hour_cleaned.csv")
    else:
        hourly = pd.DataFrame()
    if hourly.empty or stations.empty:
        return (
            pd.DataFrame(columns=["timestamp", "station_id", "pickups", "returns"]),
            pd.DataFrame(columns=["station_id", "station_name", "lat", "lon", "capacity"]),
            "Không có dữ liệu",
        )
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    stations["station_id"] = stations["station_id"].astype(str)
    hourly["station_id"] = hourly["station_id"].astype(str)
    return hourly, stations, "Processed Parquet/CSV"


@st.cache_data(show_spinner=False, ttl=60)
def _cached_latest_station_status(
    database_path: str,
    modified_ns: int,
    file_size: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    del modified_ns, file_size
    return load_latest_station_status(Path(database_path))


def _load_latest_station_status(database_path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = Path(database_path).expanduser().resolve()
    modified_ns, file_size = _file_signature(path)
    return _cached_latest_station_status(str(path), modified_ns, file_size)


def build_data_status(hourly: pd.DataFrame, stations: pd.DataFrame, data_source: str) -> dict[str, str]:
    """Build a transparent, human-readable description of dashboard inputs."""
    is_demo = data_source.startswith("Demo")
    if data_source == "Không có dữ liệu":
        return {
            "mode": "KHÔNG CÓ DỮ LIỆU",
            "source": "Không có dữ liệu",
            "hourly_rows": "Không có dữ liệu",
            "station_rows": "Không có dữ liệu",
            "time_range": "Không có dữ liệu",
            "hourly_files": "Không có dữ liệu",
            "station_files": "Không có dữ liệu",
            "models": "Không có dữ liệu",
        }
    hourly_files = [
        path for path in (PROCESSED / "station_hour_cleaned.parquet", PROCESSED / "station_hour_cleaned.csv")
        if path.exists()
    ]
    station_files = [
        path for path in (PROCESSED / "station_metadata.parquet", PROCESSED / "station_metadata.csv")
        if path.exists()
    ]
    model_files = [path for path in (MODELS / "lgb_pickups.pkl", MODELS / "lgb_returns.pkl") if path.exists()]
    first_timestamp = hourly.attrs.get("first_timestamp")
    last_timestamp = hourly.attrs.get("last_timestamp")
    if first_timestamp is not None and last_timestamp is not None:
        time_range = f"{pd.Timestamp(first_timestamp):%d/%m/%Y %H:%M} → {pd.Timestamp(last_timestamp):%d/%m/%Y %H:%M}"
    else:
        timestamps = pd.to_datetime(hourly["timestamp"], errors="coerce").dropna()
        time_range = "Không xác định" if timestamps.empty else f"{timestamps.min():%d/%m/%Y %H:%M} → {timestamps.max():%d/%m/%Y %H:%M}"
    hourly_count = hourly.attrs.get("row_count", len(hourly))
    if hourly_count == 0:
        time_range = "Không xác định"
    return {
        "mode": "DEMO / dữ liệu mô phỏng" if is_demo else "DỮ LIỆU ĐÃ XỬ LÝ / dữ liệu thật",
        "source": data_source,
        "hourly_rows": f"{hourly_count:,}",
        "station_rows": f"{len(stations):,}",
        "time_range": time_range,
        "hourly_files": ", ".join(path.name for path in hourly_files) or "Không tìm thấy",
        "station_files": ", ".join(path.name for path in station_files) or "Không tìm thấy",
        "models": f"{len(model_files)}/2 model đã lưu"
        if model_files else "Chưa có model đã lưu (forecast dùng fallback)",
    }


@st.cache_data(show_spinner=False)
def list_project_csv_files(loaded_only: bool = False) -> pd.DataFrame:
    """List eligible raw CSV files, optionally limited to the latest load."""
    columns = ["File", "Đường dẫn tương đối", "Dung lượng (MB)", "Cập nhật lần cuối"]
    data_directory = DATA_LOAD_DIRECTORY if loaded_only else DATA_DIRECTORY
    manifest_by_path: dict[str, dict[str, object]] = {}
    if loaded_only and UC1_CSV_MANIFEST_PATH.exists():
        manifest = json.loads(UC1_CSV_MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest_by_path = {
            str(item["relative_path"]): item
            for item in manifest
            if isinstance(item, dict) and item.get("relative_path")
        }
    rows: list[dict[str, object]] = []
    if not data_directory.exists():
        return pd.DataFrame(columns=columns)
    for path in sorted(data_directory.rglob("*.csv")):
        if not path.is_file() or "__macosx" in {part.lower() for part in path.parts} or path.name.startswith("._"):
            continue
        stat = path.stat()
        relative_path = str(path.relative_to(data_directory))
        manifest_item = manifest_by_path.get(relative_path)
        rows.append(
            {
                "File": path.name,
                "Đường dẫn tương đối": relative_path,
                "Dung lượng (MB)": manifest_item.get("size_mb", round(stat.st_size / (1024 * 1024), 2))
                if manifest_item else round(stat.st_size / (1024 * 1024), 2),
                "Cập nhật lần cuối": manifest_item.get("modified_at")
                if manifest_item
                else datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m/%Y %H:%M:%S"),
            }
        )
    return pd.DataFrame(rows)


def copy_csv_files_to_load_directory() -> int:
    """Copy source CSVs to data_load while preserving their relative paths."""
    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu gốc: {DATA_DIRECTORY}")
    DATA_LOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    for source_path in sorted(DATA_DIRECTORY.rglob("*.csv")):
        if (
            not source_path.is_file()
            or "__macosx" in {part.lower() for part in source_path.parts}
            or source_path.name.startswith("._")
        ):
            continue
        relative_path = source_path.relative_to(DATA_DIRECTORY)
        destination = DATA_LOAD_DIRECTORY / relative_path
        source_stat = source_path.stat()
        destination_stat = destination.stat() if destination.exists() else None
        if (
            destination_stat is not None
            and destination_stat.st_size == source_stat.st_size
            and destination_stat.st_mtime_ns == source_stat.st_mtime_ns
        ):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied_count += 1
    return copied_count


def delete_loaded_csv_files(relative_paths: list[str]) -> tuple[int, int, list[str]]:
    """Delete selected loaded CSVs from data_load/ and update the load manifest."""
    if not relative_paths:
        return 0, 0, []
    manifest: list[dict[str, object]] = []
    if UC1_CSV_MANIFEST_PATH.exists():
        manifest = json.loads(UC1_CSV_MANIFEST_PATH.read_text(encoding="utf-8"))
    selected_paths = set(relative_paths)
    manifest_paths = {
        str(item.get("relative_path", ""))
        for item in manifest
        if isinstance(item, dict)
    }
    manifest.extend(
        {"relative_path": relative_path}
        for relative_path in selected_paths - manifest_paths
    )
    data_directory = DATA_LOAD_DIRECTORY.resolve()
    if not data_directory.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu đã load: {data_directory}")

    remaining: list[dict[str, object]] = []
    deleted_count = 0
    missing_count = 0
    failed_paths: list[str] = []
    for item in manifest:
        relative_path = str(item.get("relative_path", ""))
        if relative_path not in selected_paths:
            remaining.append(item)
            continue
        candidate = (data_directory / relative_path).resolve()
        try:
            candidate.relative_to(data_directory)
        except ValueError as exc:
            raise ValueError(f"Đường dẫn file không hợp lệ: {relative_path}") from exc
        if candidate.suffix.lower() != ".csv":
            raise ValueError(f"Chỉ được xóa file CSV: {relative_path}")
        if candidate.exists():
            deleted = False
            for attempt in range(3):
                try:
                    candidate.unlink()
                    deleted = True
                    deleted_count += 1
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(0.5)
            if not deleted:
                remaining.append(item)
                failed_paths.append(relative_path)
        else:
            missing_count += 1

    UC1_CSV_MANIFEST_PATH.write_text(
        json.dumps(remaining, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if UC1_REJECTED_CSV_PATH.exists():
        rejected = json.loads(UC1_REJECTED_CSV_PATH.read_text(encoding="utf-8"))
        rejected = [
            item for item in rejected
            if str(item.get("Đường dẫn tương đối", "")) not in selected_paths
        ]
        UC1_REJECTED_CSV_PATH.write_text(
            json.dumps(rejected, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return deleted_count, missing_count, failed_paths


def clear_loaded_data_artifacts() -> None:
    """Invalidate derived data after loaded source files are removed."""
    artifact_paths = [
        PROCESSED / "station_hour_cleaned.parquet",
        PROCESSED / "station_hour_cleaned.csv",
        PROCESSED / "station_hour_events.parquet",
        PROCESSED / "station_metadata.parquet",
        PROCESSED / "station_metadata.csv",
        FEATURES_PATH,
        PROCESSED / "uc1_trips_deduplicated.parquet",
        UC1_DATA_PATH,
        UC1_DATA_CSV_PATH,
        UC1_STATS_PATH,
        MODELS / "lgb_pickups.pkl",
        MODELS / "lgb_returns.pkl",
        BENCHMARK_PATH,
        BACKTESTING_PATH,
        USE_CASE_STATE_PATH,
    ]
    for path in artifact_paths:
        if path.exists():
            path.unlink()
    st.session_state.pop("use_case_state", None)
    load_dashboard_data.clear()


def render_data_status(status: dict[str, str], location: str) -> None:
    """Render the current data provenance without changing dashboard data."""
    title = "📦 Dữ liệu đang được sử dụng"
    if location == "sidebar":
        with st.sidebar.expander("🔎 App đang sử dụng dữ liệu gì?", expanded=False):
            st.markdown(f"**{title}**")
            st.caption(f"Nguồn: {status['source']}")
            st.write(f"Hourly records: **{status['hourly_rows']}**")
            st.write(f"Stations: **{status['station_rows']}**")
            st.write(f"Khoảng thời gian: **{status['time_range']}**")
            st.write(f"Hourly file: `{status['hourly_files']}`")
            st.write(f"Station file: `{status['station_files']}`")
            st.write(f"Model: **{status['models']}**")
    else:
        st.markdown(f"<div class='data-status-title'>{title}</div>", unsafe_allow_html=True)
        details = pd.DataFrame(
            [
                ["Chế độ dữ liệu", status["mode"]],
                ["Nguồn", status["source"]],
                ["Hourly records", status["hourly_rows"]],
                ["Stations", status["station_rows"]],
                ["Khoảng thời gian", status["time_range"]],
                ["Hourly file", status["hourly_files"]],
                ["Station file", status["station_files"]],
                ["Forecast model", status["models"]],
            ],
            columns=["Thông tin", "Giá trị"],
        )
        st.dataframe(details, hide_index=True, width="stretch")


@st.cache_data(show_spinner=False, ttl=60)
def _cached_system_health(
    processed_directory: str,
    models_directory: str,
    database_path: str,
) -> pd.DataFrame:
    return collect_system_health(
        processed_directory,
        models_directory,
        database_path,
    )


def render_system_health() -> None:
    st.subheader("Use Case 14: System health & provenance")
    st.caption(
        "Kiểm tra đọc mẫu artifact processed/, khả năng nạp hai model và tính toàn vẹn/schema SQLite. "
        "Không quét toàn bộ dữ liệu; kết quả được cache 60 giây."
    )
    if st.button("Làm mới kiểm tra sức khỏe", key="uc014_refresh_health"):
        _cached_system_health.clear()
    try:
        health = _cached_system_health(
            str(PROCESSED.resolve()),
            str(MODELS.resolve()),
            str(DB_PATH.resolve()),
        )
    except (
        OSError,
        sqlite3.Error,
        duckdb.Error,
        ImportError,
        ValueError,
        TypeError,
        RuntimeError,
        MemoryError,
    ) as exc:
        st.error(f"Không thể hoàn tất kiểm tra sức khỏe hệ thống: {exc}")
        return
    if health.empty:
        st.warning("Không có kết quả kiểm tra thành phần.")
        return

    counts = health["status"].value_counts()
    error_count = int(counts.get("ERROR", 0))
    warning_count = int(counts.get("WARNING", 0))
    ok_count = int(counts.get("OK", 0))
    info_count = int(counts.get("INFO", 0))
    overall_status = (
        "ERROR"
        if error_count
        else "WARNING"
        if warning_count
        else "INFO"
        if info_count
        else "OK"
    )
    metric_columns = st.columns(4)
    with metric_columns[0]:
        st.metric("Trạng thái tổng thể", overall_status)
    with metric_columns[1]:
        st.metric("Lỗi", f"{error_count:,}")
    with metric_columns[2]:
        st.metric("Cảnh báo", f"{warning_count:,}")
    with metric_columns[3]:
        st.metric("Đạt / thông tin", f"{ok_count + info_count:,}")

    if overall_status == "ERROR":
        st.error("Có thành phần bắt buộc không đọc được hoặc thiếu schema.")
    elif overall_status == "WARNING":
        st.warning(
            "Có cảnh báo về độ mới của dữ liệu nguồn hoặc thành phần tùy chọn; "
            "chưa phát hiện lỗi mức ERROR. Xem chi tiết bên dưới."
        )
    elif overall_status == "INFO":
        st.info(
            "Một số bảng chưa phát sinh bản ghi vận hành; đây không phải lỗi hệ thống. "
            "Xem chi tiết bên dưới."
        )
    else:
        st.success("Các thành phần được kiểm tra đang khả dụng.")
    checked_at = str(health["checked_at"].iloc[0])
    st.caption(f"Kiểm tra lúc: {checked_at} · UTC. Nhấn Làm mới để bỏ cache ngay.")
    st.dataframe(
        health.rename(
            columns={
                "component": "Thành phần",
                "status": "Trạng thái",
                "detail": "Chi tiết",
                "checked_at": "Kiểm tra lúc (UTC)",
                "updated_at": "Cập nhật gần nhất (UTC/DB)",
                "estimated_flag": "Cờ ước tính / provenance",
                "path": "Đường dẫn",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def _safety_thresholds(safety_buffer: float) -> tuple[float, float, float]:
    try:
        buffer = float(safety_buffer)
    except (TypeError, ValueError, OverflowError):
        buffer = 0.15
    if not math.isfinite(buffer):
        buffer = 0.15
    buffer = float(np.clip(buffer, 0.0, 0.5))
    return buffer, max(0.0, 0.5 - buffer), min(1.0, 0.5 + buffer)


def _classify_station_status(
    snapshot: pd.DataFrame,
    safety_buffer: float,
) -> pd.DataFrame:
    result = snapshot.copy()
    _, lower_ratio, upper_ratio = _safety_thresholds(safety_buffer)
    capacity = pd.to_numeric(
        result.get("capacity", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    inventory = pd.to_numeric(
        result.get("bikes_available", pd.Series(np.nan, index=result.index)),
        errors="coerce",
    )
    capacity_values = capacity.to_numpy(dtype=float, na_value=np.nan)
    inventory_values = inventory.to_numpy(dtype=float, na_value=np.nan)
    valid_capacity = np.isfinite(capacity_values) & (capacity_values > 0)
    valid_inventory = np.isfinite(inventory_values) & (inventory_values >= 0)
    valid = valid_capacity & valid_inventory & (inventory_values <= capacity_values)
    safe_min = np.full(len(result), np.nan, dtype=float)
    safe_max = np.full(len(result), np.nan, dtype=float)
    safe_min[valid_capacity] = capacity_values[valid_capacity] * lower_ratio
    safe_max[valid_capacity] = capacity_values[valid_capacity] * upper_ratio
    utilization = np.full(len(result), np.nan, dtype=float)
    utilization[valid] = inventory_values[valid] / capacity_values[valid]
    status = np.full(len(result), "Thiếu dữ liệu", dtype=object)
    status[valid & (inventory_values < safe_min)] = "Thiếu xe"
    status[valid & (inventory_values > safe_max)] = "Đầy xe"
    status[valid & (inventory_values >= safe_min) & (inventory_values <= safe_max)] = "Cân bằng"
    result["utilization"] = utilization
    result["safe_min_bikes"] = safe_min
    result["safe_max_bikes"] = safe_max
    result["safety_buffer_pct"] = _safety_thresholds(safety_buffer)[0] * 100
    result["status"] = status
    inventory_source = result.get(
        "inventory_source", pd.Series("", index=result.index)
    ).fillna("").astype(str)
    capacity_source = result.get(
        "capacity_source", pd.Series("", index=result.index)
    ).fillna("").astype(str)
    data_status = np.full(len(result), "Thiếu dữ liệu", dtype=object)
    valid_rows = status != "Thiếu dữ liệu"
    data_status[valid_rows & inventory_source.eq("Ước tính từ station-hour").to_numpy()] = (
        "Ước tính từ station-hour"
    )
    live = inventory_source.eq("GBFS thời gian thực").to_numpy()
    live_capacity = capacity_source.eq("GBFS station_information").to_numpy()
    data_status[valid_rows & live & live_capacity] = "GBFS thực tế"
    data_status[valid_rows & live & ~live_capacity] = (
        "GBFS tồn kho · sức chứa từ metadata"
    )
    result["data_status"] = data_status
    return result


def _station_snapshot(
    hourly: pd.DataFrame,
    stations: pd.DataFrame,
    timestamp: pd.Timestamp,
    safety_buffer: float = 0.15,
) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame(
            columns=[
                "station_id", "station_name", "lat", "lon", "capacity",
                "pickups", "returns", "bikes_available", "utilization", "status",
                "safe_min_bikes", "safe_max_bikes", "safety_buffer_pct",
                "inventory_source", "capacity_source", "data_status",
                "inventory_updated_at", "capacity_updated_at",
            ]
        )
    latest = hourly.copy() if isinstance(hourly, pd.DataFrame) else pd.DataFrame()
    required = {"timestamp", "station_id", "pickups", "returns"}
    if not required.issubset(latest.columns):
        latest = pd.DataFrame(columns=["station_id", "pickups", "returns", "timestamp"])
    else:
        latest["station_id"] = latest["station_id"].map(_normalize_station_id)
        latest["timestamp"] = pd.to_datetime(latest["timestamp"], errors="coerce")
        latest = latest.dropna(subset=["timestamp"])
        latest = latest.loc[latest["timestamp"] <= pd.Timestamp(timestamp)]
        latest = latest.sort_values("timestamp").groupby("station_id").tail(1)
    result = stations.copy()
    result["station_id"] = result["station_id"].map(_normalize_station_id)
    result["capacity"] = pd.to_numeric(result["capacity"], errors="coerce")
    result = result.merge(
        latest[["station_id", "pickups", "returns", "timestamp"]],
        on="station_id",
        how="left",
        validate="many_to_one",
    ).rename(columns={"timestamp": "inventory_updated_at"})
    result["pickups"] = pd.to_numeric(result["pickups"], errors="coerce")
    result["returns"] = pd.to_numeric(result["returns"], errors="coerce")
    result["inventory_updated_at"] = pd.to_datetime(
        result["inventory_updated_at"], errors="coerce", utc=True
    )
    result["capacity_updated_at"] = pd.NaT
    valid_capacity = (
        np.isfinite(result["capacity"].to_numpy(dtype=float, na_value=np.nan))
        & result["capacity"].gt(0).to_numpy()
    )
    valid_flows = np.isfinite(
        result[["pickups", "returns"]].to_numpy(dtype=float, na_value=np.nan)
    ).all(axis=1) & result["pickups"].ge(0).to_numpy() & result["returns"].ge(0).to_numpy()
    valid_estimate = valid_capacity & valid_flows
    result["bikes_available"] = np.nan
    result.loc[valid_estimate, "bikes_available"] = (
        result.loc[valid_estimate, "capacity"] * 0.55
        + result.loc[valid_estimate, "returns"]
        - result.loc[valid_estimate, "pickups"]
    ).clip(lower=0, upper=result.loc[valid_estimate, "capacity"])
    return _classify_station_status(result, safety_buffer)


def _query_station_snapshot(
    timestamp: pd.Timestamp,
    stations: pd.DataFrame,
    safety_buffer: float = 0.15,
) -> pd.DataFrame:
    source_path = next(
        (
            path
            for path in (
                PROCESSED / "station_hour_cleaned.parquet",
                PROCESSED / "station_hour_cleaned.csv",
            )
            if path.is_file()
        ),
        None,
    )
    if source_path is None:
        source_signature = ("", 0, 0)
    else:
        source_stat = source_path.stat()
        source_signature = (
            str(source_path.resolve()),
            source_stat.st_mtime_ns,
            source_stat.st_size,
        )
    latest = _cached_station_snapshot(
        str(pd.Timestamp(timestamp)),
        str(PROCESSED.resolve()),
        source_signature,
    )
    return _station_snapshot(latest, stations, timestamp, safety_buffer)


@st.cache_data(show_spinner=False)
def _cached_station_snapshot(
    timestamp: str,
    processed_directory: str,
    source_signature: tuple[str, int, int],
) -> pd.DataFrame:
    del source_signature
    return query_station_snapshot(
        Path(processed_directory),
        pd.Timestamp(timestamp),
    )


def _normalize_station_id(value: object) -> str:
    """Normalize numeric/text station identifiers without changing source data."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return ""
    station_id = str(value).strip()
    if not station_id:
        return ""
    if station_id.startswith("+"):
        station_id = station_id[1:]
    try:
        number = Decimal(station_id)
    except InvalidOperation:
        return station_id
    if not number.is_finite():
        return ""
    if number == number.to_integral_value():
        return str(int(number))
    return station_id


def _load_station_aliases() -> dict[str, str]:
    """Load optional station-id aliases from processed/station_id_aliases.json."""
    alias_path = PROCESSED / "station_id_aliases.json"
    if not alias_path.exists():
        return {}
    try:
        payload = json.loads(alias_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    aliases: dict[str, str] = {}
    for alias, canonical in payload.items():
        normalized_alias = _normalize_station_id(alias)
        normalized_canonical = _normalize_station_id(canonical)
        if normalized_alias and normalized_canonical:
            aliases[normalized_alias] = normalized_canonical
    return aliases


def _overlay_gbfs_snapshot(
    estimated: pd.DataFrame,
    gbfs_status: pd.DataFrame,
    safety_buffer: float = 0.15,
) -> pd.DataFrame:
    """Use only fresh, identity-matched GBFS rows; label all estimated rows."""
    result = estimated.copy()
    if result.empty:
        return _classify_station_status(result, safety_buffer)
    result["inventory_source"] = "Ước tính từ station-hour"
    result["capacity_source"] = "Ước tính từ station_metadata"
    if "inventory_updated_at" not in result.columns:
        result["inventory_updated_at"] = pd.NaT
    if "capacity_updated_at" not in result.columns:
        result["capacity_updated_at"] = pd.NaT
    if gbfs_status.empty or "is_usable" not in gbfs_status.columns:
        return _classify_station_status(result, safety_buffer)

    live = gbfs_status.loc[gbfs_status["is_usable"]].copy()
    if live.empty:
        return _classify_station_status(result, safety_buffer)
    aliases = _load_station_aliases()
    if "station_id_alias" in result.columns:
        for alias, canonical in result[["station_id_alias", "station_id"]].dropna().astype(str).itertuples(index=False, name=None):
            normalized_alias = _normalize_station_id(alias)
            normalized_canonical = _normalize_station_id(canonical)
            if normalized_alias and normalized_canonical:
                aliases[normalized_alias] = normalized_canonical

    result["station_id"] = result["station_id"].map(_normalize_station_id)
    live["station_id"] = live["station_id"].map(_normalize_station_id)
    if aliases:
        result["station_id"] = result["station_id"].map(
            lambda station_id: aliases.get(station_id, station_id)
        )
        live["station_id"] = live["station_id"].map(
            lambda station_id: aliases.get(station_id, station_id)
        )
    live = live.drop_duplicates("station_id", keep="last").set_index("station_id")
    live_ids = result["station_id"].isin(live.index)
    if not live_ids.any():
        return _classify_station_status(result, safety_buffer)

    live_bikes = result["station_id"].map(live["bikes_available"])
    result.loc[live_ids, "bikes_available"] = live_bikes.loc[live_ids].astype(float)
    if "capacity" in live:
        live_capacity = result["station_id"].map(live["capacity"])
        use_live_capacity = live_ids & live_capacity.notna()
        result.loc[use_live_capacity, "capacity"] = live_capacity.loc[
            use_live_capacity
        ].astype(float)
        result.loc[use_live_capacity, "capacity_source"] = "GBFS station_information"
        if "timestamp" in live:
            result.loc[use_live_capacity, "capacity_updated_at"] = pd.to_datetime(
                result.loc[use_live_capacity, "station_id"].map(live["timestamp"]),
                errors="coerce",
                utc=True,
            )
    if "station_name" in live:
        live_names = result["station_id"].map(live["station_name"])
        use_live_name = live_ids & live_names.notna() & live_names.astype(str).str.strip().ne("")
        result.loc[use_live_name, "station_name"] = live_names.loc[use_live_name]
    result.loc[live_ids, "pickups"] = np.nan
    result.loc[live_ids, "returns"] = np.nan
    result.loc[live_ids, "inventory_source"] = "GBFS thời gian thực"
    if "timestamp" in live:
        result.loc[live_ids, "inventory_updated_at"] = pd.to_datetime(
            result.loc[live_ids, "station_id"].map(live["timestamp"]),
            errors="coerce",
            utc=True,
        )
    return _classify_station_status(result, safety_buffer)


def _status_color(status: str) -> str:
    return {"Thiếu xe": "#EF4444", "Đầy xe": "#A855F7", "Cân bằng": "#22C55E"}.get(status, "#94A3B8")


def _map_timestamp(value: object) -> str:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return "Chưa có"
    return timestamp.strftime("%d/%m/%Y %H:%M UTC")


def _build_folium_map(
    snapshot_records: str,
    plan_records: str,
) -> folium.Map:
    valid_snapshot = pd.DataFrame(json.loads(snapshot_records))
    plan = pd.DataFrame(json.loads(plan_records))
    if valid_snapshot.empty:
        center = [40.74, -73.99]
    else:
        center = [
            float(valid_snapshot["lat"].mean()),
            float(valid_snapshot["lon"].mean()),
        ]
    fmap = folium.Map(
        location=center,
        zoom_start=12,
        tiles=None,
        control_scale=True,
        attributionControl=False,
    )
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
        name="Ảnh vệ tinh",
        overlay=False,
        control=False,
        show=True,
        max_zoom=19,
        z_index=1,
    ).add_to(fmap)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Sources: Esri, HERE, Garmin, OpenStreetMap "
            "contributors, and the GIS user community"
        ),
        name="Tên đường và giao thông",
        overlay=True,
        control=False,
        show=True,
        max_zoom=19,
        z_index=400,
    ).add_to(fmap)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Sources: Esri, HERE, Garmin, OpenStreetMap "
            "contributors, and the GIS user community"
        ),
        name="Địa danh và ranh giới",
        overlay=True,
        control=False,
        show=True,
        max_zoom=19,
        z_index=450,
    ).add_to(fmap)
    Fullscreen(position="topleft").add_to(fmap)

    for row in valid_snapshot.to_dict("records"):
        station_name = html.escape(str(row.get("station_name", "Không rõ tên")))
        station_id = html.escape(str(row.get("station_id", "Không rõ mã")))
        status = str(row.get("status", "Thiếu dữ liệu"))
        color = _status_color(status)
        bikes = pd.to_numeric(row.get("bikes_available"), errors="coerce")
        capacity = pd.to_numeric(row.get("capacity"), errors="coerce")
        bikes_text = "Thiếu dữ liệu" if pd.isna(bikes) else f"{bikes:.0f}"
        capacity_text = "Không rõ" if pd.isna(capacity) else f"{capacity:.0f}"
        pickups = pd.to_numeric(row.get("pickups"), errors="coerce")
        returns = pd.to_numeric(row.get("returns"), errors="coerce")
        safe_min = pd.to_numeric(row.get("safe_min_bikes"), errors="coerce")
        safe_max = pd.to_numeric(row.get("safe_max_bikes"), errors="coerce")
        buffer_pct = pd.to_numeric(row.get("safety_buffer_pct"), errors="coerce")
        pickups_text = "—" if pd.isna(pickups) else f"{pickups:.0f}"
        returns_text = "—" if pd.isna(returns) else f"{returns:.0f}"
        safe_min_text = "Chưa xác định" if pd.isna(safe_min) else f"{safe_min:.1f}"
        safe_max_text = "Chưa xác định" if pd.isna(safe_max) else f"{safe_max:.1f}"
        buffer_text = "Chưa xác định" if pd.isna(buffer_pct) else f"{buffer_pct:.0f}%"
        inventory_source = html.escape(str(row.get("inventory_source", "Không rõ nguồn")))
        capacity_source = html.escape(str(row.get("capacity_source", "Không rõ nguồn")))
        data_status = html.escape(str(row.get("data_status", "Thiếu dữ liệu")))
        popup = (
            f"<b>{station_name}</b><br>Mã trạm: {station_id}"
            f"<br>Tồn kho: {bikes_text}/{capacity_text}"
            f"<br>Pickups / Returns: {pickups_text} / {returns_text}"
            f"<br>Trạng thái: {html.escape(status)}"
            f"<br>Ngưỡng an toàn thấp / cao: {safe_min_text} / {safe_max_text} xe"
            f"<br>Safety Buffer: {buffer_text}"
            f"<br>Chất lượng dữ liệu: {data_status}"
            f"<br>Nguồn tồn kho: {inventory_source}"
            f"<br>Nguồn sức chứa: {capacity_source}"
            f"<br>Cập nhật tồn kho: {_map_timestamp(row.get('inventory_updated_at'))}"
            f"<br>Cập nhật sức chứa: {_map_timestamp(row.get('capacity_updated_at'))}"
        )
        estimated = data_status != "GBFS thực tế"
        folium.CircleMarker(
            [row["lat"], row["lon"]],
            radius=6,
            color=color,
            weight=1.5,
            dash_array="3 3" if estimated else None,
            fill=True,
            fill_color=color,
            fill_opacity=0.72 if estimated else 0.9,
            tooltip=(
                f"{station_name} · {station_id} · {html.escape(status)} · "
                f"{bikes_text}/{capacity_text} xe"
            ),
            popup=folium.Popup(popup, max_width=340),
        ).add_to(fmap)

    if not plan.empty:
        lookup_by_id = (
            valid_snapshot.drop_duplicates(subset=["station_id"], keep="last")
            .set_index("station_id")[["lat", "lon"]]
            .to_dict("index")
        )
        lookup_by_name = (
            valid_snapshot.drop_duplicates(subset=["station_name"], keep="last")
            .set_index("station_name")[["lat", "lon"]]
            .to_dict("index")
        )
        for row in plan.to_dict("records"):
            from_location = lookup_by_id.get(row.get("from_station_id"))
            to_location = lookup_by_id.get(row.get("to_station_id"))
            if from_location is None:
                from_location = lookup_by_name.get(row.get("from_station"))
            if to_location is None:
                to_location = lookup_by_name.get(row.get("to_station"))
            if from_location is not None and to_location is not None:
                folium.PolyLine(
                    [
                        (from_location["lat"], from_location["lon"]),
                        (to_location["lat"], to_location["lon"]),
                    ],
                    color="#F59E0B",
                    weight=3,
                    opacity=0.8,
                    tooltip=f"{row.get('bikes', 0)} xe · {row.get('distance_km', 0)} km",
                ).add_to(fmap)
    return fmap


def render_map(
    snapshot: pd.DataFrame,
    plan: pd.DataFrame | None = None,
    height: int = 570,
) -> str | None:
    valid_snapshot = snapshot.copy()
    if {"lat", "lon"}.issubset(valid_snapshot.columns):
        valid_snapshot["lat"] = pd.to_numeric(valid_snapshot["lat"], errors="coerce")
        valid_snapshot["lon"] = pd.to_numeric(valid_snapshot["lon"], errors="coerce")
        valid_coordinates = (
            np.isfinite(
                valid_snapshot[["lat", "lon"]].to_numpy(
                    dtype=float, na_value=np.nan
                )
            ).all(axis=1)
            & valid_snapshot["lat"].between(-90, 90).to_numpy()
            & valid_snapshot["lon"].between(-180, 180).to_numpy()
        )
        invalid_count = int((~valid_coordinates).sum())
        valid_snapshot = valid_snapshot.loc[valid_coordinates].copy()
    else:
        invalid_count = len(valid_snapshot)
        valid_snapshot = valid_snapshot.iloc[0:0].copy()
    if invalid_count:
        st.warning(
            f"Bỏ qua {invalid_count:,} trạm có tọa độ thiếu hoặc ngoài giới hạn hợp lệ."
        )
    if valid_snapshot.empty:
        st.info("Không có trạm với tọa độ hợp lệ để hiển thị trên bản đồ.")
        return None

    valid_snapshot = valid_snapshot.sort_values("station_id").reset_index(drop=True)
    plan = plan if isinstance(plan, pd.DataFrame) else pd.DataFrame()
    snapshot_records = valid_snapshot.to_json(
        orient="records", date_format="iso", default_handler=str
    )
    plan_records = plan.to_json(
        orient="records", date_format="iso", default_handler=str
    )
    map_key = hashlib.sha256(
        f"uc007-map-v3|{height}|{snapshot_records}|{plan_records}".encode("utf-8")
    ).hexdigest()
    cache = st.session_state.setdefault("uc007_map_cache", {})
    fmap = cache.get(map_key)
    if fmap is None:
        fmap = _build_folium_map(snapshot_records, plan_records)
        cache[map_key] = fmap
        while len(cache) > 4:
            cache.pop(next(iter(cache)))

    click_result = st_folium(
        fmap,
        height=height,
        width=None,
        returned_objects=["last_object_clicked"],
        key=f"uc007-map-{map_key[:16]}",
    )
    st.caption(
        "Nền vệ tinh và lớp nhãn: Esri; ảnh vệ tinh: Esri, Maxar, Earthstar Geographics; "
        "đường/địa danh: Esri, HERE, Garmin, OpenStreetMap contributors."
    )
    clicked = click_result.get("last_object_clicked") if isinstance(click_result, dict) else None
    if isinstance(clicked, dict) and {"lat", "lng"}.issubset(clicked):
        click_lat = float(clicked["lat"])
        click_lon = float(clicked["lng"])
        distances = np.hypot(
            (valid_snapshot["lat"].to_numpy(dtype=float) - click_lat) * 111_000,
            (
                (valid_snapshot["lon"].to_numpy(dtype=float) - click_lon)
                * 111_000
                * math.cos(math.radians(click_lat))
            ),
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] <= 150:
            return str(valid_snapshot.iloc[nearest]["station_id"])
    return None


_DECISION_HISTORY_COLUMNS = [
    "decision_id",
    "action",
    "created_at",
    "updated_at",
    "route_count",
    "scheduled_bikes",
    "distance_km",
    "uses_estimated_data",
]


def _decision_flag(value: object) -> bool:
    if value is None:
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "có"}
    return bool(value)


def _decision_plan_payload(
    action: str,
    plan: pd.DataFrame,
) -> tuple[str, dict[str, object]]:
    if not isinstance(plan, pd.DataFrame):
        raise TypeError("Lịch điều phối phải là pandas.DataFrame.")
    if plan.empty:
        raise ValueError("Không thể ghi quyết định khi chưa có tuyến điều phối hợp lệ.")
    normalized_action = str(action).strip().upper()
    if normalized_action not in {"APPROVED", "REJECTED"}:
        raise ValueError("Trạng thái quyết định chỉ được là APPROVED hoặc REJECTED.")

    required = {
        "route_id",
        "from_station_id",
        "to_station_id",
        "bikes",
        "distance_km",
        "priority",
    }
    missing = required.difference(plan.columns)
    if missing:
        raise ValueError(
            "Lịch điều phối thiếu cột bắt buộc: " + ", ".join(sorted(missing))
        )
    generation = plan.attrs.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("Thông tin cấu hình lập lịch không hợp lệ.")
    try:
        truck_capacity = float(generation.get("truck_capacity"))
        safety_buffer_pct = float(generation.get("safety_buffer_pct"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Thiếu sức chứa xe tải hoặc Safety Buffer hợp lệ.") from exc
    if (
        not math.isfinite(truck_capacity)
        or truck_capacity <= 0
        or not truck_capacity.is_integer()
        or not math.isfinite(safety_buffer_pct)
        or not 0 <= safety_buffer_pct <= 50
    ):
        raise ValueError("Sức chứa xe tải hoặc Safety Buffer không hợp lệ.")

    routes = json.loads(
        plan.to_json(orient="records", date_format="iso", default_handler=str)
    )
    route_ids: set[str] = set()
    for index, route in enumerate(routes, start=1):
        if not isinstance(route, dict):
            raise ValueError(f"Tuyến thứ {index} không có định dạng bản ghi hợp lệ.")
        route_id = str(route.get("route_id", "")).strip()
        from_id = _normalize_station_id(route.get("from_station_id"))
        to_id = _normalize_station_id(route.get("to_station_id"))
        if not route_id or route_id in route_ids:
            raise ValueError("Mã tuyến bị thiếu hoặc trùng lặp.")
        if not from_id or not to_id or from_id == to_id:
            raise ValueError(f"Tuyến {route_id} có mã trạm nguồn/đích không hợp lệ.")
        try:
            bikes = float(route.get("bikes"))
            distance_km = float(route.get("distance_km"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Tuyến {route_id} có số xe hoặc khoảng cách không hợp lệ.") from exc
        if (
            not math.isfinite(bikes)
            or bikes <= 0
            or not bikes.is_integer()
            or bikes > truck_capacity
            or bikes > np.iinfo(np.int64).max
            or not math.isfinite(distance_km)
            or distance_km < 0
        ):
            raise ValueError(f"Tuyến {route_id} có số xe hoặc khoảng cách không hợp lệ.")
        if str(route.get("priority", "")).strip() not in {"Cao", "Vừa"}:
            raise ValueError(f"Tuyến {route_id} có trạng thái ưu tiên không hợp lệ.")
        route_ids.add(route_id)
        route["route_id"] = route_id
        route["from_station_id"] = from_id
        route["to_station_id"] = to_id
        route["bikes"] = int(bikes)
        route["distance_km"] = distance_km
        route["uses_estimated_data"] = _decision_flag(
            route.get("uses_estimated_data", False)
        )

    unmatched_value = plan.attrs.get("unmatched_stations", [])
    if isinstance(unmatched_value, pd.DataFrame):
        unmatched = json.loads(
            unmatched_value.to_json(
                orient="records", date_format="iso", default_handler=str
            )
        )
    elif isinstance(unmatched_value, list):
        unmatched = unmatched_value
    else:
        raise ValueError("Danh sách trạm chưa ghép có định dạng không hợp lệ.")
    if any(not isinstance(record, dict) for record in unmatched):
        raise ValueError("Bản ghi trạm chưa ghép không hợp lệ.")

    unmatched_ids = [
        _normalize_station_id(record.get("station_id"))
        for record in unmatched
        if record.get("station_id") is not None
    ]
    nonempty_unmatched_ids = [station_id for station_id in unmatched_ids if station_id]
    if len(nonempty_unmatched_ids) != len(set(nonempty_unmatched_ids)):
        raise ValueError("Mã trạm trong danh sách chưa ghép bị trùng lặp.")

    unmatched_frame = pd.DataFrame(unmatched)
    bikes = pd.to_numeric(
        pd.Series([route["bikes"] for route in routes]), errors="coerce"
    )
    distances = pd.to_numeric(
        pd.Series([route["distance_km"] for route in routes]), errors="coerce"
    )
    summary = {
        "route_count": len(routes),
        "scheduled_bikes": int(bikes.sum()),
        "distance_km": float(distances.sum()),
        "estimated_route_count": sum(
            bool(route["uses_estimated_data"]) for route in routes
        ),
        "unmatched_station_count": len(unmatched),
        "unmatched_surplus": int(
            pd.to_numeric(
                unmatched_frame.get(
                    "remaining_surplus", pd.Series(dtype="float64")
                ),
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .sum()
        ),
        "unmatched_deficit": int(
            pd.to_numeric(
                unmatched_frame.get(
                    "remaining_deficit", pd.Series(dtype="float64")
                ),
                errors="coerce",
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .sum()
        ),
    }
    payload: dict[str, object] = {
        "schema_version": 3,
        "routes": routes,
        "summary": summary,
        "unmatched_stations": unmatched,
        "generation": generation,
    }
    canonical_payload = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        allow_nan=False,
        separators=(",", ":"),
    )
    decision_id = hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
    payload["action"] = normalized_action
    return decision_id, payload


def _ensure_decision_log_schema(connection: sqlite3.Connection) -> None:
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'decision_log'"
    ).fetchone()
    if not table_exists:
        connection.execute(
            """
            CREATE TABLE decision_log (
                decision_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('APPROVED', 'REJECTED')),
                payload TEXT NOT NULL
            )
            """
        )
        return

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(decision_log)").fetchall()
    }
    required = {"created_at", "action", "payload"}
    if not required.issubset(columns):
        raise sqlite3.DatabaseError(
            "Bảng decision_log thiếu cột legacy bắt buộc; không thể di trú an toàn."
        )
    if "decision_id" not in columns:
        connection.execute("ALTER TABLE decision_log ADD COLUMN decision_id TEXT")
    if "updated_at" not in columns:
        connection.execute("ALTER TABLE decision_log ADD COLUMN updated_at TEXT")

    rows = connection.execute(
        "SELECT rowid, decision_id, created_at, updated_at FROM decision_log ORDER BY rowid"
    ).fetchall()
    seen_ids: set[str] = set()
    for rowid, existing_id, created_at, updated_at in rows:
        decision_id = str(existing_id or "").strip()
        if not decision_id or decision_id in seen_ids:
            decision_id = f"legacy-{rowid}"
            suffix = 1
            while decision_id in seen_ids:
                decision_id = f"legacy-{rowid}-{suffix}"
                suffix += 1
        seen_ids.add(decision_id)
        connection.execute(
            """
            UPDATE decision_log
            SET decision_id = ?, updated_at = ?
            WHERE rowid = ?
            """,
            (decision_id, updated_at or created_at or "", rowid),
        )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_decision_log_decision_id "
        "ON decision_log(decision_id)"
    )


def log_decision(action: str, plan: pd.DataFrame) -> str:
    decision_id, payload = _decision_plan_payload(action, plan)
    database_path = Path(DB_PATH).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        default=str,
        allow_nan=False,
        separators=(",", ":"),
    )
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_decision_log_schema(connection)
        connection.execute(
            """
            INSERT INTO decision_log (
                decision_id, created_at, updated_at, action, payload
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(decision_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                action = excluded.action,
                payload = excluded.payload
            """,
            (decision_id, timestamp, timestamp, payload["action"], serialized_payload),
        )
        connection.commit()
    _cached_decision_history.clear()
    return decision_id


@st.cache_data(show_spinner=False)
def _cached_decision_history(
    database_path: str,
    modified_ns: int,
    file_size: int,
    limit: int,
) -> pd.DataFrame:
    del modified_ns, file_size
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        return pd.DataFrame(columns=_DECISION_HISTORY_COLUMNS)

    uri = path.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'decision_log'"
        ).fetchone()
        if not exists:
            return pd.DataFrame(columns=_DECISION_HISTORY_COLUMNS)
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(decision_log)").fetchall()
        }
        if not {"created_at", "action", "payload"}.issubset(columns):
            raise sqlite3.DatabaseError(
                "Bảng decision_log thiếu cột cần thiết để đọc lịch sử."
            )
        decision_id = '"decision_id"' if "decision_id" in columns else "CAST(rowid AS TEXT)"
        updated_at = '"updated_at"' if "updated_at" in columns else '"created_at"'
        rows = connection.execute(
            f"""
            SELECT {decision_id}, action, created_at, {updated_at}, payload
            FROM decision_log
            ORDER BY {updated_at} DESC, created_at DESC, rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    history: list[dict[str, object]] = []
    for record_id, action, created_at, updated, raw_payload in rows:
        try:
            payload = json.loads(raw_payload) if raw_payload else {}
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        routes = payload.get("routes", [])
        if not isinstance(routes, list):
            routes = []
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        def finite_number(value: object, default: float = 0.0) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                return default
            return number if math.isfinite(number) else default

        history.append(
            {
                "decision_id": str(record_id or ""),
                "action": (
                    str(action).upper()
                    if str(action).upper() in {"APPROVED", "REJECTED"}
                    else "KHÔNG HỢP LỆ"
                ),
                "created_at": str(created_at or ""),
                "updated_at": str(updated or ""),
                "route_count": int(
                    finite_number(summary.get("route_count", len(routes)))
                ),
                "scheduled_bikes": int(
                    finite_number(
                        summary.get(
                            "scheduled_bikes",
                            sum(
                                finite_number(route.get("bikes"))
                                for route in routes
                                if isinstance(route, dict)
                            ),
                        )
                    )
                ),
                "distance_km": finite_number(
                    summary.get(
                        "distance_km",
                        sum(
                            finite_number(route.get("distance_km"))
                            for route in routes
                            if isinstance(route, dict)
                        ),
                    )
                ),
                "uses_estimated_data": any(
                    isinstance(route, dict)
                    and _decision_flag(route.get("uses_estimated_data"))
                    for route in routes
                ),
            }
        )
    return pd.DataFrame(history, columns=_DECISION_HISTORY_COLUMNS)


def load_decision_history(
    db_path: str | Path,
    limit: int = 10,
) -> pd.DataFrame:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit phải là số nguyên lớn hơn 0.")
    path = Path(db_path).expanduser().resolve()
    signature = _file_signature(path)
    return _cached_decision_history(str(path), signature[0], signature[1], limit)


def render_decision_history() -> None:
    st.subheader("Use Case 12: Lịch sử quyết định")
    try:
        history = load_decision_history(DB_PATH)
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(f"Không thể đọc lịch sử quyết định UC012: {exc}")
        return
    if history.empty:
        st.info("Chưa có quyết định điều phối nào được lưu.")
        return
    display = history.rename(
        columns={
            "decision_id": "Mã quyết định",
            "action": "Trạng thái",
            "created_at": "Tạo lúc (UTC)",
            "updated_at": "Cập nhật lúc (UTC)",
            "route_count": "Số tuyến",
            "scheduled_bikes": "Số xe",
            "distance_km": "Quãng đường (km)",
            "uses_estimated_data": "Có dữ liệu ước tính",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(
        "Mã quyết định gắn với nội dung kế hoạch; thao tác lại trên cùng kế hoạch "
        "sẽ cập nhật trạng thái thay vì tạo bản ghi trùng."
    )


def _ensure_uc013_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS incentive_recommendations (
            recommendation_id TEXT PRIMARY KEY,
            station_id TEXT NOT NULL,
            station_name TEXT NOT NULL,
            forecast_for TEXT NOT NULL,
            published_at TEXT NOT NULL,
            valid_from TEXT NOT NULL,
            valid_until TEXT NOT NULL,
            reward_points INTEGER NOT NULL CHECK(reward_points > 0),
            is_estimated INTEGER,
            source_summary TEXT NOT NULL,
            provenance_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('PUBLISHED', 'CANCELLED')),
            UNIQUE(station_id, forecast_for)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS incentive_executions (
            event_id TEXT PRIMARY KEY,
            recommendation_id TEXT NOT NULL,
            station_id TEXT NOT NULL,
            status TEXT NOT NULL,
            reconciliation_status TEXT NOT NULL,
            executed_at TEXT,
            bikes_returned INTEGER,
            source TEXT NOT NULL,
            is_estimated INTEGER,
            updated_at TEXT,
            imported_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_uc013_executions_recommendation "
        "ON incentive_executions(recommendation_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_uc013_executions_executed_at "
        "ON incentive_executions(executed_at)"
    )
    expected_columns = {
        "incentive_recommendations": {
            "recommendation_id", "station_id", "station_name", "forecast_for",
            "published_at", "valid_from", "valid_until", "reward_points",
            "is_estimated", "source_summary", "provenance_json", "status",
        },
        "incentive_executions": {
            "event_id", "recommendation_id", "station_id", "status",
            "reconciliation_status", "executed_at", "bikes_returned", "source",
            "is_estimated", "updated_at", "imported_at",
        },
    }
    for table, expected in expected_columns.items():
        actual = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if not expected.issubset(actual):
            raise sqlite3.DatabaseError(
                f"Bảng {table} hiện có không tương thích với schema UC013."
            )


@st.cache_data(show_spinner=False)
def _cached_uc013_records(
    database_path: str,
    modified_ns: int,
    file_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del modified_ns, file_size
    empty_recommendations = pd.DataFrame(
        columns=[
            "recommendation_id", "station_id", "station_name", "forecast_for",
            "published_at", "valid_from", "valid_until", "reward_points",
            "is_estimated", "source_summary", "provenance_json", "status",
        ]
    )
    empty_executions = pd.DataFrame(
        columns=[
            "event_id", "recommendation_id", "station_id", "status",
            "reconciliation_status", "executed_at", "bikes_returned", "source",
            "is_estimated", "updated_at", "imported_at",
        ]
    )
    path = Path(database_path).expanduser().resolve()
    if not path.is_file():
        return empty_recommendations, empty_executions
    uri = path.as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        recommendations = (
            pd.read_sql_query(
                """
                SELECT recommendation_id, station_id, station_name, forecast_for,
                       published_at, valid_from, valid_until, reward_points,
                       is_estimated, source_summary, provenance_json, status
                FROM incentive_recommendations
                ORDER BY published_at DESC
                """,
                connection,
            )
            if "incentive_recommendations" in tables
            else empty_recommendations
        )
        executions = (
            pd.read_sql_query(
                """
                SELECT event_id, recommendation_id, station_id, status,
                       reconciliation_status, executed_at, bikes_returned, source,
                       is_estimated, updated_at, imported_at
                FROM incentive_executions
                ORDER BY COALESCE(executed_at, imported_at) DESC
                """,
                connection,
            )
            if "incentive_executions" in tables
            else empty_executions
        )
    return recommendations, executions


def _load_uc013_records(
    database_path: str | Path = DB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(database_path).expanduser().resolve()
    signature = _file_signature(path)
    return _cached_uc013_records(str(path), signature[0], signature[1])


@st.cache_data(show_spinner=False)
def _cached_uc013_clean_records(
    database_path: str,
    modified_ns: int,
    file_size: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    recommendations, executions = _cached_uc013_records(
        database_path,
        modified_ns,
        file_size,
    )
    return clean_incentive_executions(executions, recommendations)


def _load_uc013_clean_records(
    database_path: str | Path = DB_PATH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    path = Path(database_path).expanduser().resolve()
    signature = _file_signature(path)
    return _cached_uc013_clean_records(str(path), signature[0], signature[1])


def _uc013_to_utc_iso(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Mốc thời gian UC013 không hợp lệ.")
    if timestamp.tzinfo is None:
        local_timezone = datetime.now().astimezone().tzinfo
        timestamp = timestamp.tz_localize(local_timezone)
    return timestamp.tz_convert("UTC").isoformat(timespec="seconds")


def _persist_uc013_recommendations(
    recommendations: pd.DataFrame,
    database_path: str | Path = DB_PATH,
) -> int:
    if not isinstance(recommendations, pd.DataFrame) or recommendations.empty:
        raise ValueError("Không có khuyến nghị UC013 để phát hành.")
    required = {
        "recommendation_id", "station_id", "station_name", "forecast_for",
        "valid_from", "valid_until", "reward_points", "is_estimated",
        "publishable", "source_summary", "provenance_json",
    }
    missing = required.difference(recommendations.columns)
    if missing:
        raise ValueError(
            "Khuyến nghị UC013 thiếu cột: " + ", ".join(sorted(missing))
        )
    eligible = recommendations.loc[
        recommendations["publishable"].map(_decision_flag)
    ].copy()
    if eligible.empty:
        raise ValueError("Không có khuyến nghị đủ mới để phát hành.")
    if eligible["recommendation_id"].astype(str).duplicated().any():
        raise ValueError("Mã khuyến nghị UC013 bị trùng.")

    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    published_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    values: list[tuple[object, ...]] = []
    for _, row in eligible.iterrows():
        recommendation_id = str(row["recommendation_id"]).strip()
        station_id = _normalize_station_id(row["station_id"])
        if not recommendation_id or not station_id:
            raise ValueError("Mã khuyến nghị hoặc mã trạm không hợp lệ.")
        try:
            reward_value = float(row["reward_points"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Điểm thưởng UC013 phải là số nguyên dương.") from exc
        if (
            not math.isfinite(reward_value)
            or reward_value <= 0
            or not reward_value.is_integer()
            or reward_value >= 2**63
        ):
            raise ValueError("Điểm thưởng UC013 phải là số nguyên dương hữu hạn.")
        reward_points = int(reward_value)
        provenance = json.loads(str(row["provenance_json"]))
        if not isinstance(provenance, dict):
            raise ValueError("Provenance khuyến nghị UC013 không hợp lệ.")
        valid_from = _uc013_to_utc_iso(row["valid_from"])
        valid_until = _uc013_to_utc_iso(row["valid_until"])
        if pd.Timestamp(valid_until) <= pd.Timestamp(published_at):
            raise ValueError("Khuyến nghị UC013 đã hết hạn trước khi được ghi nhận.")
        if pd.Timestamp(valid_until) <= pd.Timestamp(valid_from):
            raise ValueError("Khoảng thời hạn khuyến nghị UC013 không hợp lệ.")
        station_name = row["station_name"]
        if station_name is None or pd.isna(station_name):
            station_name = ""
        source_summary = row["source_summary"]
        if source_summary is None or pd.isna(source_summary):
            source_summary = ""
        values.append(
            (
                recommendation_id,
                station_id,
                str(station_name).strip(),
                _uc013_to_utc_iso(row["forecast_for"]),
                published_at,
                valid_from,
                valid_until,
                reward_points,
                int(_decision_flag(row["is_estimated"])),
                str(source_summary),
                json.dumps(
                    provenance,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
                "PUBLISHED",
            )
        )
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_uc013_schema(connection)
        cursor = connection.executemany(
            """
            INSERT OR IGNORE INTO incentive_recommendations (
                recommendation_id, station_id, station_name, forecast_for,
                published_at, valid_from, valid_until, reward_points,
                is_estimated, source_summary, provenance_json, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        inserted = max(0, cursor.rowcount)
        connection.commit()
    _cached_uc013_records.clear()
    _cached_uc013_clean_records.clear()
    return inserted


def _prepare_uc013_execution_import(
    csv_bytes: bytes,
    recommendations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not csv_bytes:
        raise ValueError("Tệp CSV UC013 đang rỗng.")
    if len(csv_bytes) > 25 * 1024 * 1024:
        raise ValueError("Tệp CSV UC013 vượt giới hạn 25 MB; hãy nhập theo từng lô.")
    try:
        raw = pd.read_csv(
            io.BytesIO(csv_bytes),
            dtype={
                "event_id": "string",
                "recommendation_id": "string",
                "station_id": "string",
            },
        )
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        raise ValueError(f"Không đọc được CSV thực thi UC013: {exc}") from exc
    if len(raw) > 100_000:
        raise ValueError("CSV UC013 vượt giới hạn 100.000 dòng; hãy nhập theo từng lô.")
    missing = set(_UC013_EXECUTION_CSV_REQUIRED_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(
            "CSV thiếu cột bắt buộc: " + ", ".join(sorted(missing))
        )
    report: dict[str, object] = {
        "input_rows": len(raw),
        "accepted_rows": 0,
        "rejected_rows": 0,
        "reasons": {},
    }
    if raw.empty:
        return pd.DataFrame(columns=_UC013_EXECUTION_CSV_COLUMNS), report
    if recommendations.empty:
        report["rejected_rows"] = len(raw)
        report["reasons"] = {"chưa có khuyến nghị đã phát hành": len(raw)}
        return pd.DataFrame(columns=_UC013_EXECUTION_CSV_COLUMNS), report

    recs = recommendations.copy()
    recs["recommendation_id"] = (
        recs["recommendation_id"].fillna("").astype(str).str.strip()
    )
    recs["station_id"] = recs["station_id"].map(_normalize_station_id)
    recs = recs.loc[
        recs["status"].astype(str).str.upper().eq("PUBLISHED")
        & recs["recommendation_id"].ne("")
    ].drop_duplicates("recommendation_id", keep=False)
    rec_lookup = {
        str(row["recommendation_id"]): row
        for _, row in recs.iterrows()
    }

    event_ids = raw["event_id"].fillna("").astype(str).str.strip()
    duplicate_ids = event_ids.ne("") & event_ids.duplicated(keep=False)
    status_aliases = {
        "completed": "completed",
        "executed": "completed",
        "cancelled": "cancelled",
        "canceled": "cancelled",
        "not_run": "not_run",
        "not run": "not_run",
        "pending": "pending",
    }
    reason_counts: dict[str, int] = {}
    accepted: list[dict[str, object]] = []

    def reject(reason: str) -> None:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    def optional_timestamp(value: object) -> str | None:
        if value is None or pd.isna(value) or not str(value).strip():
            return None
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return ""
        timestamp = pd.Timestamp(parsed)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(datetime.now().astimezone().tzinfo)
        return timestamp.tz_convert("UTC").isoformat(timespec="seconds")

    for position, (_, row) in enumerate(raw.iterrows()):
        event_id = str(row["event_id"] if pd.notna(row["event_id"]) else "").strip()
        if not event_id or len(event_id) > 200:
            reject("event_id thiếu hoặc quá dài")
            continue
        if duplicate_ids.iloc[position]:
            reject("event_id trùng trong tệp")
            continue
        recommendation_id = str(
            row["recommendation_id"]
            if pd.notna(row["recommendation_id"])
            else ""
        ).strip()
        station_id = _normalize_station_id(row["station_id"])
        recommendation = rec_lookup.get(recommendation_id)
        if not recommendation_id or not station_id:
            reject("thiếu định danh khuyến nghị/trạm")
            continue
        if recommendation is None:
            reject("không tìm thấy khuyến nghị đã phát hành")
            continue
        if station_id != _normalize_station_id(recommendation["station_id"]):
            reject("mã trạm không khớp khuyến nghị")
            continue

        raw_status = str(row["status"] if pd.notna(row["status"]) else "")
        status = status_aliases.get(raw_status.strip().lower())
        if status is None:
            reject("trạng thái không hợp lệ")
            continue
        reconciliation_status = str(
            row["reconciliation_status"]
            if pd.notna(row["reconciliation_status"])
            else ""
        ).strip().lower()
        if reconciliation_status not in {"verified", "unverified"}:
            reject("trạng thái đối chiếu không hợp lệ")
            continue

        executed_at = optional_timestamp(row["executed_at"])
        if executed_at == "":
            reject("thời điểm thực thi không hợp lệ")
            continue
        bikes_value = row["bikes_returned"]
        bikes_returned: int | None = None
        if pd.notna(bikes_value) and str(bikes_value).strip():
            try:
                bikes_number = float(bikes_value)
            except (TypeError, ValueError, OverflowError):
                reject("số xe trả không hợp lệ")
                continue
            if (
                not math.isfinite(bikes_number)
                or not bikes_number.is_integer()
                or bikes_number < 0
                or bikes_number >= 2**63
            ):
                reject("số xe trả không hợp lệ")
                continue
            bikes_returned = int(bikes_number)
        if status == "completed":
            if executed_at is None or bikes_returned is None or bikes_returned <= 0:
                reject("sự kiện hoàn thành cần thời gian và số xe trả dương")
                continue
            published_at = pd.to_datetime(
                recommendation["published_at"], errors="coerce", utc=True
            )
            valid_until = pd.to_datetime(
                recommendation["valid_until"], errors="coerce", utc=True
            )
            executed_timestamp = pd.Timestamp(executed_at)
            if (
                pd.isna(published_at)
                or pd.isna(valid_until)
                or executed_timestamp < published_at
                or executed_timestamp > valid_until
            ):
                reject("thực thi nằm ngoài thời hạn khuyến nghị")
                continue

        source = ""
        if "source" in raw.columns and pd.notna(row.get("source")):
            source = str(row.get("source")).strip()
        source = source or "UC013 CSV import"
        estimated: int | None = None
        estimated_value = row.get("is_estimated")
        if estimated_value is not None and pd.notna(estimated_value) and str(estimated_value).strip():
            normalized_flag = str(estimated_value).strip().lower()
            if normalized_flag in {"true", "1", "yes", "y", "có"}:
                estimated = 1
            elif normalized_flag in {"false", "0", "no", "n", "không"}:
                estimated = 0
            else:
                reject("cờ ước tính không hợp lệ")
                continue
        updated_at = None
        if "updated_at" in raw.columns:
            updated_at = optional_timestamp(row.get("updated_at"))
            if updated_at == "":
                reject("updated_at không hợp lệ")
                continue
        accepted.append(
            {
                "event_id": event_id,
                "recommendation_id": recommendation_id,
                "station_id": station_id,
                "status": status,
                "reconciliation_status": reconciliation_status,
                "executed_at": executed_at,
                "bikes_returned": bikes_returned,
                "source": source,
                "is_estimated": estimated,
                "updated_at": updated_at,
            }
        )

    report["accepted_rows"] = len(accepted)
    report["reasons"] = reason_counts
    report["rejected_rows"] = len(raw) - len(accepted)
    return pd.DataFrame(accepted, columns=_UC013_EXECUTION_CSV_COLUMNS), report


def _persist_uc013_execution_import(
    events: pd.DataFrame,
    database_path: str | Path = DB_PATH,
) -> tuple[int, int]:
    if events.empty:
        return 0, 0
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    inserted_or_updated = 0
    identity_conflicts = 0
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_uc013_schema(connection)
        for _, row in events.iterrows():
            cursor = connection.execute(
                """
                INSERT INTO incentive_executions (
                    event_id, recommendation_id, station_id, status,
                    reconciliation_status, executed_at, bikes_returned, source,
                    is_estimated, updated_at, imported_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    status = excluded.status,
                    reconciliation_status = excluded.reconciliation_status,
                    executed_at = excluded.executed_at,
                    bikes_returned = excluded.bikes_returned,
                    source = excluded.source,
                    is_estimated = excluded.is_estimated,
                    updated_at = excluded.updated_at,
                    imported_at = excluded.imported_at
                WHERE incentive_executions.recommendation_id = excluded.recommendation_id
                  AND incentive_executions.station_id = excluded.station_id
                """,
                (
                    str(row["event_id"]),
                    str(row["recommendation_id"]),
                    str(row["station_id"]),
                    str(row["status"]),
                    str(row["reconciliation_status"]),
                    row["executed_at"],
                    row["bikes_returned"],
                    str(row["source"]),
                    row["is_estimated"],
                    row["updated_at"],
                    imported_at,
                ),
            )
            if cursor.rowcount:
                inserted_or_updated += 1
            else:
                identity_conflicts += 1
        connection.commit()
    _cached_uc013_records.clear()
    _cached_uc013_clean_records.clear()
    return inserted_or_updated, identity_conflicts


def render_uc013_panel(
    projected: pd.DataFrame,
    snapshot_time: pd.Timestamp,
) -> None:
    st.subheader("Use Case 13: Ưu đãi trả xe và đối chiếu thực thi")
    st.caption(
        "Đề xuất +15 điểm tại trạm thiếu xe theo dự báo t+1; thời hạn phát hành 60 phút. "
        "Chỉ phát hành khi mốc tồn kho/dự báo không quá 90 phút và provenance đầy đủ. "
        "Chỉ dùng forecast, nguồn và mốc cập nhật sẵn có. MAE/WAPE hiển thị khi model "
        "có metric hold-out; dữ liệu thời tiết/thiết bị hiện không có nên không được suy diễn."
    )
    try:
        forecast_for = pd.Timestamp(snapshot_time) + pd.Timedelta(hours=1)
        candidates = build_incentive_recommendations(
            projected,
            forecast_for=forecast_for,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        candidates = pd.DataFrame()
        st.error(f"Không thể tạo khuyến nghị UC013: {exc}")

    if not candidates.empty:
        candidate_count = len(candidates)
        publishable = candidates.loc[candidates["publishable"].map(_decision_flag)]
        estimated_count = int(
            publishable["is_estimated"].map(_decision_flag).sum()
        )
        st.metric(
            "Trạm thiếu xe có dự báo hợp lệ",
            f"{candidate_count:,}",
            f"{len(publishable):,} có thể phát hành lúc này",
        )
        candidate_columns = [
            "station_id",
            "station_name",
            "forecast_for",
            "expected_stock",
            "safe_min_bikes",
            "forecast_pickups",
            "forecast_returns",
            "forecast_pickups_mae",
            "forecast_returns_mae",
            "forecast_pickups_wape",
            "forecast_returns_wape",
            "is_peak_hour",
            "is_estimated",
            "publishable",
            "not_publishable_reason",
            "source_summary",
        ]
        candidate_columns = [
            column for column in candidate_columns if column in candidates.columns
        ]
        candidate_labels = {
            "station_id": "Mã trạm",
            "station_name": "Trạm",
            "forecast_for": "Mốc dự báo t+1",
            "expected_stock": "Tồn dự kiến",
            "safe_min_bikes": "Ngưỡng thiếu",
            "forecast_pickups": "Dự báo lấy",
            "forecast_returns": "Dự báo trả",
            "forecast_pickups_mae": "MAE lấy (hold-out)",
            "forecast_returns_mae": "MAE trả (hold-out)",
            "forecast_pickups_wape": "WAPE lấy",
            "forecast_returns_wape": "WAPE trả",
            "is_peak_hour": "Giờ cao điểm",
            "is_estimated": "Có dữ liệu ước tính",
            "publishable": "Có thể phát hành",
            "not_publishable_reason": "Lý do chưa thể phát hành",
            "source_summary": "Nguồn dữ liệu",
        }
        st.dataframe(
            candidates[candidate_columns].rename(columns=candidate_labels),
            hide_index=True,
            width="stretch",
        )
        if candidates["is_estimated"].map(_decision_flag).any():
            st.warning(
                "Có đề xuất dựa trên ít nhất một đầu vào ước tính/fallback. "
                "Cờ ước tính và provenance được lưu cùng khuyến nghị."
            )
        if estimated_count:
            confirmation_data = publishable[
                [
                    "recommendation_id",
                    "expected_stock",
                    "forecast_pickups",
                    "forecast_returns",
                    "is_estimated",
                    "source_summary",
                    "provenance_json",
                ]
            ].to_json(orient="records", date_format="iso")
            confirmation_key = (
                "uc013_confirm_estimated_"
                + hashlib.sha256(confirmation_data.encode("utf-8")).hexdigest()[:12]
            )
            confirm_estimates = st.checkbox(
                "Xác nhận phát hành các đề xuất có dữ liệu ước tính/fallback",
                key=confirmation_key,
            )
        else:
            confirm_estimates = True
        if st.button(
            "Use Case 13: Phát hành khuyến nghị đủ điều kiện",
            key="uc013_publish",
            type="primary",
            disabled=publishable.empty or (estimated_count > 0 and not confirm_estimates),
        ):
            try:
                inserted = _persist_uc013_recommendations(publishable)
            except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
                st.error(f"Không thể ghi khuyến nghị UC013 vào SQLite: {exc}")
            else:
                if inserted:
                    st.success(f"Đã ghi nhận {inserted:,} khuyến nghị đã phát hành.")
                else:
                    st.info(
                        "Các khuyến nghị này đã được ghi nhận trước đó cho cùng trạm/khung giờ."
                    )
    else:
        reason = candidates.attrs.get(
            "unavailable_reason",
            "Chưa có trạm thiếu xe với dự báo t+1 hợp lệ.",
        )
        st.info(str(reason))

    template = pd.DataFrame(columns=_UC013_EXECUTION_CSV_COLUMNS)
    st.download_button(
        "Tải mẫu CSV thực thi/đối chiếu UC013",
        template.to_csv(index=False).encode("utf-8-sig"),
        file_name="uc013_execution_template.csv",
        mime="text/csv",
        key="uc013_template_download",
    )
    st.caption(
        "Cột bắt buộc: event_id, recommendation_id, station_id, status "
        "(completed/cancelled/not_run/pending), reconciliation_status "
        "(verified/unverified), executed_at (ISO-8601; mốc không kèm múi giờ "
        "được hiểu theo múi giờ máy chủ), bikes_returned. "
        "source, is_estimated và updated_at được giữ lại nếu có."
    )
    try:
        stored_recommendations, stored_events = _load_uc013_records()
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(f"Không thể đọc dữ liệu UC013 từ SQLite: {exc}")
        return

    published_for_download = stored_recommendations.loc[
        stored_recommendations.get(
            "status", pd.Series(dtype="object")
        ).astype(str).str.upper().eq("PUBLISHED")
    ] if not stored_recommendations.empty else stored_recommendations
    if not published_for_download.empty:
        export_columns = [
            "recommendation_id",
            "station_id",
            "station_name",
            "forecast_for",
            "valid_until",
            "reward_points",
            "is_estimated",
            "source_summary",
        ]
        st.download_button(
            "Tải danh sách khuyến nghị đã phát hành",
            published_for_download[export_columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="uc013_published_recommendations.csv",
            mime="text/csv",
            key="uc013_recommendations_download",
        )

    upload = st.file_uploader(
        "Nhập CSV thực thi và đối chiếu",
        type=["csv"],
        key="uc013_execution_upload",
    )
    if upload is not None and st.button(
        "Use Case 13: Nhập và đối chiếu sự kiện",
        key="uc013_import",
    ):
        try:
            prepared, report = _prepare_uc013_execution_import(
                upload.getvalue(),
                stored_recommendations,
            )
            saved, conflicts = _persist_uc013_execution_import(prepared)
        except (
            OSError,
            sqlite3.Error,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            st.error(f"Không thể nhập/đối chiếu CSV UC013: {exc}")
        else:
            st.success(
                f"Đã thêm/cập nhật {saved:,} sự kiện; "
                f"{conflicts:,} bản ghi có event_id gắn với định danh khác bị bỏ qua."
            )
            st.caption(
                f"{report['accepted_rows']:,}/{report['input_rows']:,} dòng hợp lệ; "
                f"{report['rejected_rows']:,} dòng bị loại."
            )
            if report["reasons"]:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {"Lý do loại": reason, "Số dòng": count}
                            for reason, count in report["reasons"].items()
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
            stored_recommendations, stored_events = _load_uc013_records()

    try:
        completed_events, audit = _load_uc013_clean_records()
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        st.error(f"Không thể làm sạch/đối chiếu sự kiện UC013: {exc}")
        return
    if completed_events.empty:
        if stored_events.empty:
            st.info(
                "Chưa có dữ liệu thực thi để đánh giá hiệu quả; KPI và biểu đồ "
                "chỉ xuất hiện sau khi nhập sự kiện hoàn thành đã xác minh."
            )
        else:
            st.warning(
                "Chưa có sự kiện nào đủ điều kiện tổng hợp. Sự kiện pending, "
                "not_run, cancelled, chưa xác minh, sai mã hoặc ngoài hạn không được tính."
            )
            st.caption(
                f"Đã loại {audit['excluded_statuses']:,} sự kiện chưa hoàn thành, "
                f"{audit['unverified_records']:,} sự kiện chưa xác minh, "
                f"{audit['unmatched_recommendations'] + audit['station_mismatches']:,} "
                "sự kiện không khớp định danh."
            )
        return

    st.subheader("Kết quả thực thi đã đối chiếu")
    metric_columns = st.columns(3)
    total_bikes_returned = sum(
        int(value) for value in completed_events["bikes_returned"]
    )
    total_reward_points = sum(
        int(value) for value in completed_events["reward_points"]
    )
    with metric_columns[0]:
        st.metric("Sự kiện hoàn thành đã xác minh", f"{len(completed_events):,}")
    with metric_columns[1]:
        st.metric("Số xe thực trả", f"{total_bikes_returned:,}")
    with metric_columns[2]:
        st.metric("Điểm thưởng ghi nhận", f"{total_reward_points:,}")

    chart_data = (
        completed_events.groupby("station_name", dropna=False)["bikes_returned"]
        .agg(lambda values: sum(int(value) for value in values))
        .sort_values(ascending=False)
        .rename("Số xe trả")
    )
    st.bar_chart(chart_data.to_frame())
    event_columns = [
        "event_id",
        "recommendation_id",
        "station_id",
        "station_name",
        "executed_at",
        "bikes_returned",
        "reward_points",
        "source",
        "source_summary",
        "provenance_json",
        "event_is_estimated",
        "recommendation_is_estimated",
        "is_estimated",
        "updated_at",
        "imported_at",
    ]
    st.dataframe(
        completed_events[event_columns].rename(
            columns={
                "event_id": "Mã sự kiện",
                "recommendation_id": "Mã khuyến nghị",
                "station_id": "Mã trạm",
                "station_name": "Trạm",
                "executed_at": "Thực thi lúc (UTC)",
                "bikes_returned": "Số xe trả",
                "reward_points": "Điểm thưởng",
                "source": "Nguồn sự kiện",
                "source_summary": "Nguồn khuyến nghị",
                "provenance_json": "Provenance dự báo/tồn kho",
                "event_is_estimated": "Cờ ước tính sự kiện",
                "recommendation_is_estimated": "Cờ ước tính khuyến nghị",
                "is_estimated": "Có dữ liệu ước tính",
                "updated_at": "Cập nhật nguồn (UTC)",
                "imported_at": "Nhập vào hệ thống (UTC)",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    estimated_events = completed_events["is_estimated"].map(
        lambda value: _decision_flag(value) if not pd.isna(value) else False
    )
    unknown_estimate_count = int(completed_events["is_estimated"].isna().sum())
    if estimated_events.any() or unknown_estimate_count:
        st.warning(
            f"{int(estimated_events.sum()):,} sự kiện có cờ ước tính; "
            f"{unknown_estimate_count:,} sự kiện chưa rõ cờ ước tính."
        )
    st.caption(
        "KPI, bảng và biểu đồ cùng dùng tập sự kiện khớp recommendation_id/station_id, "
        "hoàn thành, đã xác minh và nằm trong thời hạn phát hành. "
        "Bản ghi lặp hoặc trạng thái không hợp lệ bị loại."
    )


def _station_hour_source_path() -> Path | None:
    return next(
        (
            path.resolve()
            for path in (
                PROCESSED / "station_hour_cleaned.parquet",
                PROCESSED / "station_hour_cleaned.csv",
            )
            if path.is_file()
        ),
        None,
    )


@st.cache_data(show_spinner=False)
def _cached_uc010_forecasts(
    fallback_history_json: str,
    source_path: str,
    selected_time: str,
    station_ids_json: str,
    source_mtime_ns: int,
    source_size: int,
    alias_mtime_ns: int,
    alias_size: int,
    pickups_model_mtime_ns: int,
    pickups_model_size: int,
    returns_model_mtime_ns: int,
    returns_model_size: int,
) -> pd.DataFrame:
    del (
        source_mtime_ns,
        source_size,
        alias_mtime_ns,
        alias_size,
        pickups_model_mtime_ns,
        pickups_model_size,
        returns_model_mtime_ns,
        returns_model_size,
    )
    selected = pd.Timestamp(selected_time)
    forecast_time = selected + pd.Timedelta(hours=1)
    station_ids = [str(value) for value in json.loads(station_ids_json)]
    result = pd.DataFrame({"station_id": station_ids})
    for column in ("forecast_pickups", "forecast_returns"):
        result[column] = np.nan
    for column in (
        "forecast_pickups_mae",
        "forecast_returns_mae",
        "forecast_pickups_wape",
        "forecast_returns_wape",
    ):
        result[column] = np.nan
    for column in ("forecast_pickups_source", "forecast_returns_source"):
        result[column] = "Thiếu dữ liệu tại mốc gốc"
    result["forecast_updated_at"] = forecast_time
    result["forecast_data_updated_at"] = pd.NaT
    if not station_ids:
        return result

    if source_path:
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu station-hour: {source}")
        source_reader = (
            "read_parquet(?)"
            if source.suffix.lower() == ".parquet"
            else "read_csv_auto(?)"
        )
        history_start = selected - pd.Timedelta(hours=240)
        connection = duckdb.connect()
        try:
            history = connection.execute(
                f"""
                SELECT timestamp, CAST(station_id AS VARCHAR) AS station_id,
                       pickups, returns
                FROM (
                    SELECT timestamp, station_id, pickups, returns,
                           row_number() OVER (
                               PARTITION BY CAST(station_id AS VARCHAR)
                               ORDER BY timestamp DESC
                           ) AS history_rank
                    FROM {source_reader}
                    WHERE timestamp >= ? AND timestamp <= ?
                )
                WHERE history_rank <= 240
                ORDER BY station_id, timestamp
                """,
                [
                    str(source),
                    history_start.to_pydatetime(),
                    selected.to_pydatetime(),
                ],
            ).df()
        finally:
            connection.close()
    else:
        history = pd.DataFrame.from_records(
            json.loads(fallback_history_json or "[]")
        )
        if "datetime_hour" in history.columns and "timestamp" not in history.columns:
            history = history.rename(columns={"datetime_hour": "timestamp"})

    required = {"timestamp", "station_id", "pickups", "returns"}
    if history.empty or not required.issubset(history.columns):
        return result

    history = history.loc[:, ["timestamp", "station_id", "pickups", "returns"]].copy()
    history["station_id"] = history["station_id"].map(_normalize_station_id)
    aliases = _load_station_aliases()
    if aliases:
        history["station_id"] = history["station_id"].map(
            lambda value: aliases.get(value, value)
        )
    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
    try:
        source_timezone = history["timestamp"].dt.tz
    except (AttributeError, TypeError, ValueError):
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce", utc=True)
        source_timezone = history["timestamp"].dt.tz
    if source_timezone is None:
        selected = selected.tz_localize(None) if selected.tzinfo else selected
    elif selected.tzinfo is None:
        selected = selected.tz_localize(source_timezone)
    else:
        selected = selected.tz_convert(source_timezone)
    forecast_time = selected + pd.Timedelta(hours=1)
    result["forecast_updated_at"] = forecast_time

    history["pickups"] = pd.to_numeric(history["pickups"], errors="coerce")
    history["returns"] = pd.to_numeric(history["returns"], errors="coerce")
    finite_flows = np.isfinite(
        history[["pickups", "returns"]].to_numpy(dtype=float, na_value=np.nan)
    ).all(axis=1)
    valid_history = history.loc[
        history["station_id"].isin(station_ids)
        & history["timestamp"].notna()
        & history["timestamp"].le(selected)
        & finite_flows
        & history["pickups"].ge(0)
        & history["returns"].ge(0)
    ].copy()
    valid_history = (
        valid_history.sort_values("timestamp")
        .drop_duplicates(["station_id", "timestamp"], keep="last")
        .groupby("station_id", sort=False, group_keys=False)
        .tail(240)
    )
    origins = valid_history.loc[valid_history["timestamp"].eq(selected)].copy()
    if origins.empty:
        for column in ("forecast_pickups_source", "forecast_returns_source"):
            result[column] = "Không có dữ liệu gốc hợp lệ"
        return result

    origin_ids = origins["station_id"].drop_duplicates().tolist()
    future_rows = origins.copy()
    future_rows["timestamp"] = forecast_time
    feature_rows = make_features(
        pd.concat(
            [
                valid_history.loc[valid_history["station_id"].isin(origin_ids)],
                future_rows,
            ],
            ignore_index=True,
        ),
        drop_missing=False,
    )
    future_features = (
        feature_rows.loc[feature_rows["timestamp"].eq(forecast_time)]
        .drop_duplicates("station_id", keep="last")
        .set_index("station_id")
    )
    if future_features.empty:
        return result

    output = result.set_index("station_id")
    output.loc[origins["station_id"], "forecast_data_updated_at"] = selected
    for target in ("pickups", "returns"):
        output_column = f"forecast_{target}"
        source_column = f"forecast_{target}_source"
        model_path = MODELS / f"lgb_{target}.pkl"
        fallback_source = (
            "Ước tính heuristic (model không khả dụng)"
            if model_path.is_file()
            else "Ước tính heuristic (model chưa có)"
        )
        predictions_by_id = pd.Series(dtype="float64")
        prediction_source = fallback_source
        if model_path.is_file():
            try:
                bundle = load_model(model_path)
                features = bundle.get("features") if isinstance(bundle, dict) else None
                if (
                    not isinstance(features, (list, tuple))
                    or not features
                    or any(column not in future_features.columns for column in features)
                ):
                    raise ValueError("Bundle model thiếu danh sách feature hợp lệ.")
                sample = future_features.loc[:, list(features)]
                sample_values = sample.to_numpy(dtype=float)
                valid_sample = np.isfinite(sample_values).all(axis=1)
                if valid_sample.any():
                    predictions = np.asarray(
                        bundle["model"].predict(sample.loc[valid_sample]),
                        dtype=float,
                    ).reshape(-1)
                    if predictions.size != int(valid_sample.sum()):
                        raise ValueError("Model trả về số dự báo không khớp số trạm.")
                    predictions[~np.isfinite(predictions)] = np.nan
                    predictions_by_id = pd.Series(
                        np.maximum(predictions, 0),
                        index=sample.index[valid_sample],
                        dtype="float64",
                    ).dropna()
                    prediction_source = "LightGBM · t+1"
                    model_metrics = bundle.get("metrics", {})
                    if isinstance(model_metrics, dict):
                        target_metrics = model_metrics.get(target, model_metrics)
                        if isinstance(target_metrics, dict):
                            for metric_key in ("MAE", "WAPE"):
                                try:
                                    metric_value = float(target_metrics.get(metric_key))
                                except (TypeError, ValueError, OverflowError):
                                    continue
                                if math.isfinite(metric_value) and metric_value >= 0:
                                    output_column_metric = (
                                        f"forecast_{target}_{metric_key.lower()}"
                                    )
                                    output.loc[
                                        predictions_by_id.index, output_column_metric
                                    ] = metric_value
            except (
                OSError,
                ValueError,
                KeyError,
                AttributeError,
                TypeError,
                ImportError,
            ) as exc:
                fallback_source = (
                    "Ước tính heuristic (model lỗi: "
                    f"{type(exc).__name__})"
                )
                prediction_source = fallback_source

        rolling = pd.to_numeric(
            future_features.get(
                f"{target}_rolling_3h",
                pd.Series(np.nan, index=future_features.index),
            ),
            errors="coerce",
        )
        multiplier = 1.05 if target == "pickups" else 0.95
        for station_id in future_features.index:
            prediction = predictions_by_id.get(station_id, np.nan)
            source_label = prediction_source
            if not np.isfinite(prediction):
                rolling_value = rolling.get(station_id, np.nan)
                if np.isfinite(rolling_value):
                    prediction = max(0.0, float(rolling_value) * multiplier)
                    source_label = fallback_source
                else:
                    source_label = "Không đủ dữ liệu lịch sử"
            output.loc[station_id, output_column] = prediction
            output.loc[station_id, source_column] = source_label
    return output.reset_index()


def _uc010_forecasts(
    hourly: pd.DataFrame,
    snapshot: pd.DataFrame,
    selected_time: pd.Timestamp,
) -> pd.DataFrame:
    if snapshot.empty or "station_id" not in snapshot.columns:
        return pd.DataFrame(
            columns=[
                "station_id", "forecast_pickups", "forecast_returns",
                "forecast_pickups_source", "forecast_returns_source",
                "forecast_updated_at", "forecast_data_updated_at",
                "forecast_pickups_mae", "forecast_returns_mae",
                "forecast_pickups_wape", "forecast_returns_wape",
            ]
        )
    station_ids = sorted({
        _normalize_station_id(value)
        for value in snapshot["station_id"]
        if _normalize_station_id(value)
    })
    source = next(
        (
            path.resolve()
            for path in (
                PROCESSED / "station_hour_cleaned.parquet",
                PROCESSED / "station_hour_cleaned.csv",
            )
            if path.is_file()
        ),
        None,
    )
    fallback_json = (
        "[]"
        if source is not None
        else hourly.to_json(orient="records", date_format="iso", default_handler=str)
    )
    alias_path = PROCESSED / "station_id_aliases.json"
    pickups_model = MODELS / "lgb_pickups.pkl"
    returns_model = MODELS / "lgb_returns.pkl"
    return _cached_uc010_forecasts(
        fallback_json,
        str(source) if source is not None else "",
        pd.Timestamp(selected_time).isoformat(),
        json.dumps(station_ids),
        *_file_signature(source),
        *_file_signature(alias_path),
        *_file_signature(pickups_model),
        *_file_signature(returns_model),
    )


@st.cache_data(show_spinner=False)
def _cached_uc010_analysis(
    planning_snapshot: pd.DataFrame,
    safety_buffer: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    projected = calculate_expected_inventory(planning_snapshot, safety_buffer)
    plan = run_greedy_rebalancing(planning_snapshot, buffer_pct=safety_buffer)
    return projected, plan


def _align_station_times(
    timestamps: pd.Series,
    selected_time: pd.Timestamp,
) -> tuple[pd.Series, pd.Timestamp]:
    parsed = pd.to_datetime(timestamps, errors="coerce")
    try:
        timezone = parsed.dt.tz
    except (AttributeError, TypeError, ValueError):
        parsed = pd.to_datetime(timestamps, errors="coerce", utc=True)
        timezone = parsed.dt.tz
    selected = pd.Timestamp(selected_time)
    if timezone is None:
        if selected.tzinfo is not None:
            selected = selected.tz_localize(None)
    elif selected.tzinfo is None:
        selected = selected.tz_localize(timezone)
    else:
        selected = selected.tz_convert(timezone)
    return parsed.dt.floor("h"), selected.floor("h")


def build_forecast_table(
    hourly: pd.DataFrame,
    station_id: str,
    selected_time: pd.Timestamp,
    data_source: str | None = None,
) -> pd.DataFrame:
    """Return 24 hourly observations and at most one forecast at t+1."""
    source_path = _station_hour_source_path()
    selected_for_query = pd.Timestamp(selected_time)
    if selected_for_query.tzinfo is not None:
        selected_for_query = selected_for_query.tz_localize(None)
    source_label = data_source or "Dữ liệu station-hour"
    source_updated_at: pd.Timestamp | pd.NaT = pd.NaT
    query_error = None
    if source_path is not None:
        try:
            series = query_station_history(
                PROCESSED,
                str(station_id),
                selected_for_query,
                240,
            )
            source_label = source_path.name
            source_updated_at = pd.Timestamp(
                source_path.stat().st_mtime,
                unit="s",
                tz="UTC",
            )
        except (duckdb.Error, ImportError, OSError, ValueError) as exc:
            query_error = type(exc).__name__
            series = pd.DataFrame()
    else:
        series = pd.DataFrame()

    if series.empty and isinstance(hourly, pd.DataFrame) and not hourly.empty:
        required = {"timestamp", "station_id", "pickups", "returns"}
        if required.issubset(hourly.columns):
            series = hourly.loc[
                hourly["station_id"].map(_normalize_station_id)
                == _normalize_station_id(station_id)
            ].copy()
            source_label = data_source or "Dữ liệu station-hour trong bộ nhớ"
            if query_error:
                source_label += f" · fallback RAM ({query_error})"

    required = {"timestamp", "station_id", "pickups", "returns"}
    if series.empty or not required.issubset(series.columns):
        return pd.DataFrame()
    series = series.loc[:, ["timestamp", "station_id", "pickups", "returns"]].copy()
    series["station_id"] = series["station_id"].map(_normalize_station_id)
    series["timestamp"], selected = _align_station_times(
        series["timestamp"],
        pd.Timestamp(selected_time),
    )
    series["pickups"] = pd.to_numeric(series["pickups"], errors="coerce")
    series["returns"] = pd.to_numeric(series["returns"], errors="coerce")
    invalid_flows = (
        series["pickups"].lt(0)
        | series["returns"].lt(0)
    )
    series.loc[invalid_flows, ["pickups", "returns"]] = np.nan
    series = (
        series.loc[
            series["station_id"].eq(_normalize_station_id(station_id))
            & series["timestamp"].notna()
            & series["timestamp"].le(selected)
        ]
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
    )
    if series.empty:
        return pd.DataFrame()

    latest_source_time = series["timestamp"].max()
    if pd.isna(source_updated_at):
        source_updated_at = latest_source_time
    history_start = selected - pd.Timedelta(hours=23)
    history_index = pd.date_range(history_start, selected, freq="h")
    observed = series.set_index("timestamp").reindex(history_index)
    observed.index.name = "timestamp"
    observed = observed.reset_index()
    observed["station_id"] = _normalize_station_id(station_id)
    observed["net_flow"] = observed["returns"] - observed["pickups"]
    observed["pickups_pred"] = np.nan
    observed["returns_pred"] = np.nan
    observed["net_flow_pred"] = np.nan
    observed["record_type"] = "Lịch sử 24 giờ"
    observed["forecast_source"] = ""
    observed["is_estimated"] = bool(
        str(data_source or "").lower().startswith(("demo", "dữ liệu demo"))
    )
    observed["data_source"] = source_label
    observed["updated_at"] = observed["timestamp"]
    observed["source_updated_at"] = source_updated_at
    observed["forecast_generated_at"] = pd.NaT

    forecast_time = selected + pd.Timedelta(hours=1)
    forecast_values: dict[str, float] = {"pickups": np.nan, "returns": np.nan}
    forecast_sources: dict[str, str] = {}
    exact_origin = series.loc[series["timestamp"].eq(selected)]
    if not exact_origin.empty:
        history_for_features = series.tail(240).copy()
        future_row = pd.DataFrame(
            {
                "timestamp": [forecast_time],
                "station_id": [_normalize_station_id(station_id)],
                "pickups": [np.nan],
                "returns": [np.nan],
            }
        )
        feature_rows = make_features(
            pd.concat([history_for_features, future_row], ignore_index=True),
            drop_missing=False,
        )
        future_features = feature_rows.loc[
            feature_rows["timestamp"].eq(forecast_time)
        ]
        if not future_features.empty:
            future_features = future_features.iloc[[-1]]
            origin_has_flows = bool(
                np.isfinite(
                    exact_origin[["pickups", "returns"]]
                    .to_numpy(dtype=float, na_value=np.nan)
                ).all()
            )
            if origin_has_flows:
                for target in ("pickups", "returns"):
                    fallback_multiplier = 1.05 if target == "pickups" else 0.95
                    rolling_value = pd.to_numeric(
                        future_features.iloc[0].get(f"{target}_rolling_3h"),
                        errors="coerce",
                    )
                    prediction = np.nan
                    prediction_source = "Không đủ dữ liệu lịch sử"
                    model_path = MODELS / f"lgb_{target}.pkl"
                    if model_path.is_file():
                        try:
                            bundle = load_model(model_path)
                            model_features = (
                                bundle.get("features")
                                if isinstance(bundle, dict)
                                else None
                            )
                            if (
                                not isinstance(model_features, (list, tuple))
                                or not model_features
                                or any(
                                    column not in future_features.columns
                                    for column in model_features
                                )
                            ):
                                raise ValueError("Bundle model thiếu feature hợp lệ.")
                            sample = future_features.loc[:, list(model_features)]
                            sample_values = sample.to_numpy(dtype=float)
                            if np.isfinite(sample_values).all():
                                prediction_values = np.asarray(
                                    bundle["model"].predict(sample),
                                    dtype=float,
                                ).reshape(-1)
                                if prediction_values.size != 1:
                                    raise ValueError(
                                        "Model phải trả đúng một dự báo cho mốc t+1."
                                    )
                                if np.isfinite(prediction_values[0]):
                                    prediction = float(prediction_values[0])
                                    prediction_source = "LightGBM · t+1"
                        except (
                            OSError,
                            ValueError,
                            KeyError,
                            AttributeError,
                            TypeError,
                            ImportError,
                            RuntimeError,
                        ) as exc:
                            prediction_source = (
                                f"Fallback rolling 3h (model lỗi: {type(exc).__name__})"
                            )
                    if not np.isfinite(prediction) and np.isfinite(rolling_value):
                        prediction = float(rolling_value) * fallback_multiplier
                        if prediction_source == "Không đủ dữ liệu lịch sử":
                            prediction_source = "Fallback trung bình rolling 3h · t+1"
                    if np.isfinite(prediction):
                        forecast_values[target] = max(0.0, prediction)
                    forecast_sources[target] = prediction_source
            else:
                forecast_sources = {
                    target: "Không dự báo: thiếu tồn tại tại mốc t"
                    for target in ("pickups", "returns")
                }
    else:
        forecast_sources = {
            target: "Không dự báo: thiếu dữ liệu tại mốc t"
            for target in ("pickups", "returns")
        }

    forecast_net_flow = (
        forecast_values["returns"] - forecast_values["pickups"]
        if np.isfinite(forecast_values["returns"])
        and np.isfinite(forecast_values["pickups"])
        else np.nan
    )
    forecast_generated_at = pd.Timestamp.now(tz="UTC")
    forecast_row = pd.DataFrame(
        [
            {
                "timestamp": forecast_time,
                "station_id": _normalize_station_id(station_id),
                "pickups": np.nan,
                "returns": np.nan,
                "net_flow": np.nan,
                "pickups_pred": forecast_values["pickups"],
                "returns_pred": forecast_values["returns"],
                "net_flow_pred": forecast_net_flow,
                "record_type": "Dự báo t+1",
                "forecast_source": (
                    f"Pickups: {forecast_sources.get('pickups', 'Thiếu dữ liệu')}; "
                    f"Returns: {forecast_sources.get('returns', 'Thiếu dữ liệu')}"
                ),
                "pickups_forecast_source": forecast_sources.get("pickups", ""),
                "returns_forecast_source": forecast_sources.get("returns", ""),
                "is_estimated": True,
                "data_source": source_label,
                "updated_at": forecast_generated_at,
                "source_updated_at": source_updated_at,
                "forecast_generated_at": forecast_generated_at,
            }
        ]
    )
    observed["pickups_forecast_source"] = ""
    observed["returns_forecast_source"] = ""
    return pd.concat([observed, forecast_row], ignore_index=True)


_BACKTESTING_ESTIMATE_COLUMNS = {
    "is_estimated",
    "projection_is_estimated",
    "inventory_is_estimated",
    "capacity_is_estimated",
}


def _backtesting_input_error(
    hourly: pd.DataFrame,
    stations: pd.DataFrame,
) -> str | None:
    if not isinstance(hourly, pd.DataFrame) or hourly.empty:
        return "Không có dữ liệu hourly để chạy backtesting."
    if not isinstance(stations, pd.DataFrame) or stations.empty:
        return "Không có metadata trạm để chạy backtesting."
    required_hourly = {"timestamp", "station_id", "pickups"}
    required_stations = {"station_id", "capacity", "lat", "lon"}
    missing_hourly = required_hourly.difference(hourly.columns)
    missing_stations = required_stations.difference(stations.columns)
    if missing_hourly or missing_stations:
        messages = []
        if missing_hourly:
            messages.append(
                "Hourly thiếu cột: " + ", ".join(sorted(missing_hourly))
            )
        if missing_stations:
            messages.append(
                "Metadata trạm thiếu cột: " + ", ".join(sorted(missing_stations))
            )
        return "; ".join(messages)
    return None


def _validate_backtesting_ratios(
    passive_capacity_ratio: float,
    dss_capacity_ratio: float,
) -> tuple[float, float]:
    validated: list[float] = []
    for name, value in (
        ("Baseline capacity ratio", passive_capacity_ratio),
        ("DSS capacity ratio", dss_capacity_ratio),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} phải là số hữu hạn trong khoảng [0, 1].")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} phải là số hữu hạn trong khoảng [0, 1]."
            ) from exc
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"{name} phải nằm trong khoảng [0, 1].")
        validated.append(number)
    if validated[1] <= validated[0]:
        raise ValueError("DSS capacity ratio phải lớn hơn Baseline capacity ratio.")
    return validated[0], validated[1]


def _backtesting_estimated_rows(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    columns = sorted(
        column
        for column in frame.columns
        if str(column).lower() in _BACKTESTING_ESTIMATE_COLUMNS
    )
    mask = pd.Series(False, index=frame.index, dtype=bool)
    true_values = {"true", "1", "1.0", "yes", "y", "có"}
    for column in columns:
        values = frame[column].astype("string").str.strip().str.lower()
        mask |= values.isin(true_values).fillna(False)
    return mask, columns


def _backtesting_input_updated_at(data_source: str) -> str:
    if not data_source.startswith("Processed"):
        return (
            "Không áp dụng cho dữ liệu demo"
            if data_source.startswith("Demo")
            else "Không xác định"
        )
    candidates = (
        PROCESSED / "station_hour_cleaned.parquet",
        PROCESSED / "station_hour_cleaned.csv",
        PROCESSED / "station_metadata.parquet",
        PROCESSED / "station_metadata.csv",
    )
    latest_mtime = 0.0
    latest_names: list[str] = []
    for path in candidates:
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            continue
        if modified_at > latest_mtime:
            latest_mtime = modified_at
            latest_names = [path.name]
        elif modified_at == latest_mtime:
            latest_names.append(path.name)
    if not latest_mtime:
        return "Không đọc được metadata thời gian của input"
    updated_at = datetime.fromtimestamp(latest_mtime, timezone.utc).isoformat(
        timespec="seconds"
    )
    return f"{updated_at} ({', '.join(latest_names)})"


def run_historical_simulation(
    hourly: pd.DataFrame,
    stations: pd.DataFrame,
    passive_capacity_ratio: float = 0.55,
    dss_capacity_ratio: float = 0.75,
    data_source: str = "Không xác định",
    input_updated_at: str = "",
) -> pd.DataFrame:
    """Compare passive and DSS-style service outcomes on observed station-hour data."""
    passive_capacity_ratio, dss_capacity_ratio = _validate_backtesting_ratios(
        passive_capacity_ratio,
        dss_capacity_ratio,
    )
    input_error = _backtesting_input_error(hourly, stations)
    if input_error:
        raise ValueError(input_error)

    frame = hourly.copy()
    metadata = stations.copy()
    frame["station_id"] = frame["station_id"].astype("string").str.strip()
    metadata["station_id"] = metadata["station_id"].astype("string").str.strip()
    hourly_estimated, hourly_estimate_columns = _backtesting_estimated_rows(frame)
    station_estimated, station_estimate_columns = _backtesting_estimated_rows(metadata)
    frame["_uc15_hour_estimated"] = hourly_estimated.to_numpy(dtype=bool)
    metadata["_uc15_station_estimated"] = station_estimated.to_numpy(dtype=bool)
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
        utc=True,
    )
    pickups = pd.to_numeric(frame["pickups"], errors="coerce")
    finite_pickups = np.isfinite(pickups.fillna(np.nan).to_numpy(dtype=float))
    valid_hourly = (
        frame["station_id"].notna()
        & frame["station_id"].ne("")
        & frame["timestamp"].notna()
        & finite_pickups
    )
    excluded_hourly_rows = int((~valid_hourly).sum())
    negative_pickup_rows = int((pickups.lt(0) & valid_hourly).sum())
    frame = frame.loc[valid_hourly].copy()
    frame["pickups"] = pickups.loc[valid_hourly].astype(float)
    if frame.empty:
        raise ValueError("Không còn dòng hourly hợp lệ sau khi kiểm tra dữ liệu.")
    duplicate_hourly_rows = int(
        frame.duplicated(subset=["station_id", "timestamp"]).sum()
    )
    if duplicate_hourly_rows:
        raise ValueError(
            f"Hourly có {duplicate_hourly_rows:,} dòng trùng station_id/timestamp."
        )

    metadata = metadata.loc[
        metadata["station_id"].notna() & metadata["station_id"].ne("")
    ].copy()
    if metadata.empty:
        raise ValueError("Metadata không có station_id hợp lệ.")
    if metadata["station_id"].duplicated().any():
        raise ValueError("Metadata có station_id trùng lặp; không thể ghép an toàn.")

    metadata["capacity"] = pd.to_numeric(metadata["capacity"], errors="coerce")
    capacities = metadata["capacity"].fillna(np.nan).to_numpy(dtype=float)
    invalid_capacity = ~np.isfinite(capacities) | (capacities <= 0)
    metadata.loc[invalid_capacity, "capacity"] = 25.0
    metadata["_uc15_station_estimated"] = (
        metadata["_uc15_station_estimated"].astype(bool) | invalid_capacity
    )
    if "station_name" not in metadata.columns:
        metadata["station_name"] = metadata["station_id"]
    else:
        metadata["station_name"] = (
            metadata["station_name"].fillna(metadata["station_id"]).astype(str)
        )
    for coordinate in ("lat", "lon"):
        metadata[coordinate] = pd.to_numeric(metadata[coordinate], errors="coerce")
    coordinates = metadata[["lat", "lon"]].to_numpy(dtype=float)
    valid_coordinates = (
        np.isfinite(coordinates).all(axis=1)
        & metadata["lat"].between(-90, 90).to_numpy()
        & metadata["lon"].between(-180, 180).to_numpy()
    )
    excluded_routing_stations = int((~valid_coordinates).sum())

    merged = frame.merge(
        metadata[["station_id", "capacity", "_uc15_station_estimated"]],
        on="station_id",
        how="left",
        validate="many_to_one",
    )
    unmatched_station_rows = int(merged["capacity"].isna().sum())
    station_flags = (
        merged["_uc15_station_estimated"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    estimated_hourly_rows = int(
        (
            merged["_uc15_hour_estimated"].to_numpy(dtype=bool)
            | station_flags
            | merged["capacity"].isna().to_numpy()
        ).sum()
    )
    merged["capacity"] = merged["capacity"].fillna(25.0)
    merged["demand"] = merged["pickups"].clip(lower=0)
    merged["passive_unmet"] = (
        merged["demand"] - merged["capacity"] * passive_capacity_ratio
    ).clip(lower=0)
    merged["dss_unmet"] = (
        merged["demand"] - merged["capacity"] * dss_capacity_ratio
    ).clip(lower=0)
    total_demand = float(merged["demand"].sum())
    passive_unmet = float(merged["passive_unmet"].sum())
    dss_unmet = float(merged["dss_unmet"].sum())
    if not all(math.isfinite(value) for value in (total_demand, passive_unmet, dss_unmet)):
        raise ValueError("Tổng hợp nhu cầu tạo ra giá trị không hữu hạn.")

    route_metadata = metadata.loc[valid_coordinates].copy()
    plan = generate_rebalancing_plan(
        route_metadata.assign(
            bikes_available=(route_metadata["capacity"] * passive_capacity_ratio).round()
        )
    )
    distance_values = (
        pd.to_numeric(plan["distance_km"], errors="coerce")
        if not plan.empty and "distance_km" in plan.columns
        else pd.Series(dtype=float)
    )
    if not np.isfinite(distance_values.fillna(np.nan).to_numpy(dtype=float)).all():
        raise ValueError("Khoảng cách điều phối chứa giá trị không hữu hạn.")
    distance = float(distance_values.sum()) if not distance_values.empty else 0.0
    report_generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    input_window_start = frame["timestamp"].min().isoformat()
    input_window_end = frame["timestamp"].max().isoformat()
    estimated_station_count = int(metadata["_uc15_station_estimated"].sum())
    estimate_columns = sorted(
        set(hourly_estimate_columns + station_estimate_columns)
        | ({"capacity_fallback_25"} if invalid_capacity.any() or unmatched_station_rows else set())
    )
    report_metadata = {
        "Input Source": data_source or "Không xác định",
        "Input Updated At (UTC)": input_updated_at or "Không xác định",
        "Input Window Start (UTC)": input_window_start,
        "Input Window End (UTC)": input_window_end,
        "Estimated Hourly Rows": estimated_hourly_rows,
        "Estimated Stations": estimated_station_count,
        "Estimate Flag Columns": ", ".join(estimate_columns) or "Không có cờ",
        "Excluded Invalid Hourly Rows": excluded_hourly_rows,
        "Negative Pickups Clipped": negative_pickup_rows,
        "Unmatched Station Rows": unmatched_station_rows,
        "Capacity Fallback Records": int(invalid_capacity.sum())
        + unmatched_station_rows,
        "Stations Excluded From Routing": excluded_routing_stations,
        "Baseline Capacity Ratio": passive_capacity_ratio,
        "DSS Capacity Ratio": dss_capacity_ratio,
        "Report Generated At (UTC)": report_generated_at,
    }
    rows = [
        {
            "Scenario": "A · Thụ động",
            "Unmet Demand": round(passive_unmet, 2),
            "Service Level (%)": round((1 - passive_unmet / max(total_demand, 1)) * 100, 2),
            "Truck Distance (km)": 0.0,
            **report_metadata,
        },
        {
            "Scenario": "B · BikeMind DSS",
            "Unmet Demand": round(dss_unmet, 2),
            "Service Level (%)": round((1 - dss_unmet / max(total_demand, 1)) * 100, 2),
            "Truck Distance (km)": round(distance, 2),
            **report_metadata,
        },
    ]
    return pd.DataFrame(rows)


def _backtesting_summary(scenario: pd.DataFrame) -> tuple[float, float, float]:
    if not isinstance(scenario, pd.DataFrame) or scenario.empty:
        raise ValueError("Backtesting chưa tạo được bảng kết quả.")
    required = {"Scenario", "Unmet Demand"}
    missing = required.difference(scenario.columns)
    if missing:
        raise ValueError(
            "Báo cáo backtesting thiếu cột: " + ", ".join(sorted(missing))
        )

    values: dict[str, float] = {}
    for label in ("A · Thụ động", "B · BikeMind DSS"):
        matching = scenario.loc[scenario["Scenario"].eq(label), "Unmet Demand"]
        if len(matching) != 1:
            raise ValueError(f"Báo cáo cần đúng một dòng cho kịch bản {label}.")
        try:
            value = float(matching.iloc[0])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Unmet Demand không hợp lệ cho kịch bản {label}.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Unmet Demand không hợp lệ cho kịch bản {label}.")
        values[label] = value
    baseline = values["A · Thụ động"]
    dss = values["B · BikeMind DSS"]
    improvement = max(0.0, (baseline - dss) / max(baseline, 1.0) * 100)
    return baseline, dss, improvement


def _backtesting_csv_bytes(scenario: pd.DataFrame) -> bytes:
    _backtesting_summary(scenario)
    return scenario.to_csv(index=False).encode("utf-8-sig")


def _write_backtesting_report(path: str | Path, report_bytes: bytes) -> Path:
    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            dir=report_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(report_bytes)
            temporary_file.flush()
        temporary_path.replace(report_path)
        temporary_path = None
        return report_path
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


@st.cache_data(show_spinner=False)
def _cached_backtesting_report(
    report_path: str,
    modified_ns: int,
    file_size: int,
) -> pd.DataFrame:
    del modified_ns, file_size
    return pd.read_csv(Path(report_path).expanduser().resolve())


def _load_backtesting_report(report_path: str | Path) -> pd.DataFrame:
    path = Path(report_path).expanduser().resolve()
    signature = _file_signature(path)
    return _cached_backtesting_report(str(path), signature[0], signature[1])


def metric_card(label: str, value: str, hint: str, color: str = "#8B5CF6") -> None:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
    <div class="metric-value" style="color:{color}">{value}</div><div class="metric-hint">{hint}</div></div>""", unsafe_allow_html=True)


def clear_url_fragment() -> None:
    """Remove stale heading fragments without changing query parameters."""
    st.iframe(
        """
        <script>
        (() => {
            const parentWindow = window.parent;
            if (parentWindow && parentWindow.location.hash) {
                parentWindow.history.replaceState(
                    null,
                    document.title,
                    parentWindow.location.pathname + parentWindow.location.search
                );
            }
        })();
        </script>
        """,
        height=1,
        width=1,
        tab_index=-1,
        alt="Clear stale URL fragment",
    )


def forecast_chart(
    hourly: pd.DataFrame,
    station_id: str,
    selected_time: pd.Timestamp,
    data_source: str | None = None,
    series: pd.DataFrame | None = None,
) -> None:
    if series is None:
        series = build_forecast_table(
            hourly,
            station_id,
            selected_time,
            data_source,
        )
    if series.empty:
        st.info("Chưa có dữ liệu theo giờ cho trạm này.")
        return
    historical = series.loc[series["record_type"] == "Lịch sử 24 giờ"]
    forecast = series.loc[series["record_type"] == "Dự báo t+1"]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=historical["timestamp"],
            y=historical["pickups"],
            name="Pickups lịch sử",
            line={"color": "#22C55E"},
            connectgaps=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=historical["timestamp"],
            y=historical["returns"],
            name="Returns lịch sử",
            line={"color": "#A855F7"},
            connectgaps=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=historical["timestamp"],
            y=historical["net_flow"],
            name="Net flow lịch sử",
            line={"color": "#2563EB", "dash": "dash"},
            yaxis="y2",
            connectgaps=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["pickups_pred"],
            name="Pickups dự báo t+1",
            mode="markers",
            marker={"color": "#86EFAC", "size": 11, "symbol": "diamond"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["returns_pred"],
            name="Returns dự báo t+1",
            mode="markers",
            marker={"color": "#D8B4FE", "size": 11, "symbol": "diamond"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast["timestamp"],
            y=forecast["net_flow_pred"],
            name="Net flow dự báo t+1",
            mode="markers",
            marker={"color": "#60A5FA", "size": 11, "symbol": "diamond"},
            yaxis="y2",
        )
    )
    figure.update_layout(
        template="plotly_white",
        height=470,
        margin={"l": 10, "r": 45, "t": 45, "b": 10},
        title="24 giờ lịch sử + dự báo cho đúng mốc t+1",
        hovermode="x unified",
        yaxis={"title": "Pickups / Returns"},
        yaxis2={"title": "Net flow", "overlaying": "y", "side": "right"},
    )
    st.plotly_chart(figure, width="stretch")


def run_pipeline(uploaded_files: list) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    temporary = save_uploaded_directory(uploaded_files)
    preview, stats, activated_hourly_count, activated_station_count = scan_and_prepare_unique_csv(
        base_directory=temporary,
        output_directory=PROCESSED,
    )
    if preview.empty or activated_hourly_count == 0 or activated_station_count == 0:
        raise ValueError("Không tìm thấy dữ liệu CSV hợp lệ để xử lý.")
    load_dashboard_data.clear()
    hourly, stations, _ = load_dashboard_data()
    return (
        hourly,
        stations,
        f"Đã xử lý thành công {stats.get('unique', 0):,} file CSV độc nhất, "
        f"{stats.get('row_count', 0):,} chuyến đi sạch, "
        f"{activated_hourly_count:,} hourly records và {activated_station_count:,} stations.",
    )


def filter_uploaded_csv_files(uploaded_files: list) -> tuple[list, int, int]:
    """Keep only CSV uploads and remove duplicate file contents before loading."""
    filtered = []
    seen_hashes: set[str] = set()
    skipped_extensions = 0
    skipped_duplicates = 0
    for uploaded_file in uploaded_files:
        if Path(str(uploaded_file.name)).suffix.lower() != ".csv":
            skipped_extensions += 1
            continue
        digest = hashlib.md5()
        for start in range(0, uploaded_file.size, 65536):
            digest.update(uploaded_file.getbuffer()[start:start + 65536])
        file_hash = digest.hexdigest()
        if file_hash in seen_hashes:
            skipped_duplicates += 1
            continue
        seen_hashes.add(file_hash)
        filtered.append(uploaded_file)
    return filtered, skipped_extensions, skipped_duplicates


def save_uploaded_directory(uploaded_files: list) -> Path:
    """Persist browser-selected directory files into the existing temp input."""
    temporary = ROOT / ".uploaded_csv"
    temporary.mkdir(exist_ok=True)
    for existing_file in temporary.rglob("*"):
        if existing_file.is_file():
            existing_file.unlink()
    for uploaded_file in uploaded_files:
        relative_name = str(uploaded_file.name).replace("\\", "/")
        parts = [part for part in Path(relative_name).parts if part not in ("", ".")]
        if not parts or ".." in parts or Path(relative_name).is_absolute():
            raise ValueError(f"Tên file không hợp lệ: {uploaded_file.name}")
        destination = temporary.joinpath(*parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(uploaded_file.getbuffer())
    return temporary


def activate_loaded_data(loaded_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the completed UC1 load into the processed dashboard inputs."""
    column_mapping = {
        "tripduration": "trip_duration",
        "starttime": "start_time",
        "stoptime": "end_time",
        "start station id": "start_station_id",
        "start station name": "start_station_name",
        "start station latitude": "start_lat",
        "start station longitude": "start_lon",
        "end station id": "end_station_id",
        "end station name": "end_station_name",
        "end station latitude": "end_lat",
        "end station longitude": "end_lon",
    }
    trips = loaded_data.rename(columns=column_mapping)
    cleaned = clean_trips(trips)
    station_metadata = build_station_metadata(cleaned)
    hourly = aggregate_hourly(cleaned, station_metadata)
    if hourly.empty or station_metadata.empty:
        raise ValueError("Không tạo được dữ liệu station-hour từ dữ liệu vừa nạp.")
    save_processed(hourly, station_metadata, str(PROCESSED))
    load_dashboard_data.clear()
    return hourly, station_metadata


def merge_with_previous_uc1_data(new_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Combine the current load with the previous UC1 result and deduplicate rows."""
    previous_path = None
    for candidate in (UC1_DATA_PATH, UC1_DATA_CSV_PATH):
        if candidate.exists():
            previous_path = candidate
            break
    if previous_path is None:
        return new_data, {
            "previous_rows": 0,
            "new_unique_rows": len(new_data),
            "combined_rows": len(new_data),
        }
    try:
        previous_data = (
            pd.read_parquet(previous_path)
            if previous_path.suffix.lower() == ".parquet"
            else pd.read_csv(previous_path)
        )
        combined = pd.concat([previous_data, new_data], ignore_index=True).drop_duplicates(ignore_index=True)
    except (OSError, ValueError, ImportError, pd.errors.ParserError):
        return new_data, {
            "previous_rows": 0,
            "new_unique_rows": len(new_data),
            "combined_rows": len(new_data),
        }
    previous_rows = len(previous_data)
    combined_rows = len(combined)
    return combined, {
        "previous_rows": previous_rows,
        "new_unique_rows": max(0, combined_rows - previous_rows),
        "combined_rows": combined_rows,
    }


def estimate_load_time_text() -> str:
    """Estimate load duration from the current raw CSV inventory."""
    inventory = list_project_csv_files()
    if inventory.empty:
        return "Thời gian ước tính: chưa xác định (chưa có file CSV)."
    total_mb = float(inventory["Dung lượng (MB)"].sum())
    estimated_seconds = max(10, (total_mb / 30.0) + (len(inventory) * 0.5))
    minutes = math.ceil(estimated_seconds / 60)
    return f"Thời gian ước tính: khoảng {minutes} phút ({len(inventory):,} file · {total_mb:,.0f} MB)"


def persist_uc1_result(
    loaded_data: pd.DataFrame,
    load_stats: dict[str, int],
    activated_rows: int,
    activated_stations: int,
) -> None:
    """Persist the UC1 result so a browser reload does not clear the session."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    try:
        loaded_data.to_parquet(UC1_DATA_PATH, index=False)
        if UC1_DATA_CSV_PATH.exists():
            UC1_DATA_CSV_PATH.unlink()
        data_file = UC1_DATA_PATH.name
    except (ImportError, ValueError, OSError):
        loaded_data.to_csv(UC1_DATA_CSV_PATH, index=False)
        data_file = UC1_DATA_CSV_PATH.name
    metadata = {
        **{key: int(value) for key, value in load_stats.items()},
        "row_count": int(load_stats.get("row_count", len(loaded_data))),
        "activated_hourly_rows": int(activated_rows),
        "activated_station_rows": int(activated_stations),
        "dashboard_activated": True,
        "data_file": data_file,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    UC1_STATS_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def restore_uc1_result() -> None:
    """Restore the latest persisted UC1 result into the current Streamlit session."""
    if "uc1_load_stats" in st.session_state or not UC1_STATS_PATH.exists():
        return
    metadata = json.loads(UC1_STATS_PATH.read_text(encoding="utf-8"))
    data_file = PROCESSED / str(metadata.get("data_file", ""))
    if not data_file.exists():
        return
    if data_file.suffix.lower() == ".parquet":
        loaded_data = pd.read_parquet(data_file)
    else:
        loaded_data = pd.read_csv(data_file)
    stats = {
        key: int(metadata.get(key, 0))
        for key in (
            "total",
            "unique",
            "dup_files",
            "dup_rows",
            "schema_errors",
            "read_errors",
            "invalid_files",
            "raw_rows",
            "row_count",
            "previous_rows",
            "new_unique_rows",
            "combined_rows",
        )
    }
    st.session_state["uc1_loaded_data"] = loaded_data
    st.session_state["uc1_load_stats"] = stats
    st.session_state["uc1_restored_at"] = metadata.get("saved_at", "")
    if metadata.get("dashboard_activated"):
        st.session_state["uc1_activation_message"] = (
            f"Dữ liệu đã load thành công và đang được dashboard sử dụng: "
            f"{int(metadata.get('activated_hourly_rows', 0)):,} hourly records, "
            f"{int(metadata.get('activated_station_rows', 0)):,} stations."
        )


def _frame_preview(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, object]]:
    """Serialize a small result preview without persisting the full dataset."""
    return json.loads(frame.head(limit).to_json(orient="records", date_format="iso"))


def _read_use_case_state() -> dict[str, dict[str, object]]:
    if not USE_CASE_STATE_PATH.exists():
        return {}
    try:
        state = json.loads(USE_CASE_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Không thể đọc trạng thái các Use Case: {USE_CASE_STATE_PATH}") from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"File trạng thái Use Case không hợp lệ: {USE_CASE_STATE_PATH}")
    return {
        str(key): value
        for key, value in state.items()
        if isinstance(value, dict)
    }


def _persist_use_case_result(use_case: str, result: dict[str, object]) -> dict[str, dict[str, object]]:
    state = _read_use_case_state()
    state[use_case] = {
        **result,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    USE_CASE_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    st.session_state["use_case_state"] = state
    return state


def restore_persistent_results() -> None:
    """Restore successful Use Case results after a browser/session reload."""
    if "use_case_state" not in st.session_state:
        st.session_state["use_case_state"] = _read_use_case_state()


def _persisted_preview(state: dict[str, object], key: str) -> pd.DataFrame:
    records = state.get(key, [])
    return pd.DataFrame(records if isinstance(records, list) else [])


def _json_ready(value: object) -> object:
    """Convert UC02 stats to JSON values without coercing timestamps to integers."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _render_uc2_audit(stats: dict[str, object], preview: pd.DataFrame) -> None:
    """Show the latest UC02 audit from memory. This does not rerun cleaning."""
    def as_int(key: str) -> int:
        try:
            return int(stats.get(key, 0) or 0)
        except (TypeError, ValueError):
            return 0

    stat_columns = st.columns(3)
    stat_columns[0].metric("Tổng số trạm đã chuẩn hóa", f"{as_int('station_count'):,}")
    stat_columns[1].metric("Bản ghi Station-Hour", f"{as_int('station_hour_rows'):,}")
    stat_columns[2].metric("Bản ghi bị loại", f"{as_int('removed_rows'):,}")
    if stats.get("t_min") and stats.get("t_max"):
        st.caption(f"Độ phủ thời gian: {stats['t_min']} → {stats['t_max']}")
    st.caption(
        f"Ô giờ không phát sinh giao dịch được phủ 0: {as_int('zero_filled_cells'):,} · "
        f"Ô có giao dịch: {as_int('observed_station_hours'):,}"
    )
    if stats.get("capacity_estimated") in (True, "True", 1, "1"):
        st.caption(
            "Sức chứa trạm là giá trị ước tính thử nghiệm "
            f"({as_int('capacity') or 25}), nhãn `{stats.get('capacity_source', 'estimated_default')}`."
        )
    if "removed_duration" in stats:
        audit = pd.DataFrame(
            [
                ["Đầu vào", as_int("input_rows") or as_int("source_rows")],
                ["Thiếu station_id", as_int("removed_missing_station_id")],
                ["Thời gian không parse được", as_int("removed_invalid_timestamp")],
                ["stoptime <= starttime", as_int("removed_time_order")],
                ["Thời lượng ngoài 60–7200 giây", as_int("removed_duration")],
                ["Cùng trạm và dưới 180 giây", as_int("removed_same_station_short")],
                ["Tọa độ thiếu hoặc ngoài vùng hợp lệ", as_int("removed_invalid_coordinates")],
                ["Bản ghi trùng sau khi lọc", as_int("removed_duplicate_rows")],
                ["Bản ghi sạch", as_int("clean_rows")],
            ],
            columns=["Quy tắc", "Số bản ghi"],
        )
        st.dataframe(audit, hide_index=True, width="stretch")
    if not preview.empty:
        st.dataframe(preview.head(5), hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="BikeMind DSS", page_icon="🚲", layout="wide", initial_sidebar_state="expanded")
    clear_url_fragment()
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: #FFFFFF; color: #172033; } .block-container { padding-top: 1.4rem; }
    [data-testid="stSidebar"] { background: #F6F7FB; border-right: 1px solid #E1E5EF; }
    .metric-card { background:#FFFFFF; border:1px solid #E1E5EF; border-radius:14px; padding:18px; min-height:110px; box-shadow:0 2px 8px rgba(23,32,51,.05); }
    .metric-label,.metric-hint { color:#667085; font-size:.78rem; } .metric-value { font-size:1.75rem; font-weight:800; margin:5px 0; }
    .section-card { background:#FFFFFF; border:1px solid #E1E5EF; border-radius:14px; padding:18px; box-shadow:0 2px 8px rgba(23,32,51,.05); }
    .data-status-title { color:#172033; font-size:1.15rem; font-weight:700; margin:0 0 .6rem; }
    h1,h2,h3 { color:#172033; letter-spacing:-.02em; } .stButton button { border-radius:9px; font-weight:700; }
    [data-testid="stDataFrame"] { border:1px solid #E1E5EF; border-radius:10px; }
    [data-testid="stHeaderActionElements"] { display:none !important; }
    [data-testid="stHeaderActionElements"] button { display:none !important; }
    </style>""", unsafe_allow_html=True)
    hourly, stations, data_source = load_dashboard_data()
    try:
        gbfs_status, gbfs_freshness = _load_latest_station_status(DB_PATH)
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(f"Không thể đọc snapshot GBFS từ SQLite: {exc}")
        gbfs_status = pd.DataFrame()
        gbfs_freshness = {
            "state": "error",
            "message": f"Không thể đọc snapshot GBFS từ SQLite: {exc}",
            "fresh_count": 0,
            "total_count": 0,
            "stale_count": 0,
            "unverified_count": 0,
            "mismatched_count": 0,
            "optional_missing_count": 0,
            "collected_at": None,
        }
    restore_uc1_result()
    try:
        restore_persistent_results()
    except RuntimeError as exc:
        st.error(f"Không thể khôi phục trạng thái Use Case đã lưu: {exc}")
        st.session_state["use_case_state"] = {}
    persistent_state = st.session_state["use_case_state"]
    data_status = build_data_status(hourly, stations, data_source)
    latest = pd.to_datetime(hourly["timestamp"]).max()
    if pd.isna(latest):
        latest = pd.Timestamp.now().floor("h")
    with st.sidebar:
        st.markdown("## 🚲 BikeMind")
        st.caption("Smart Mobility · DSS for Public Bicycle Rebalancing")
        selected_date = st.date_input("Ngày vận hành", value=latest.date())
        selected_hour = st.slider("Giờ quan sát", 0, 23, latest.hour)
        st.selectbox("Loại ngày", ["Tất cả", "Ngày thường", "Cuối tuần"])
        st.selectbox("Thời tiết", ["Tự động", "Nắng", "Mưa", "Nhiều mây"])
        safety = st.slider("Safety Buffer (%)", 0, 50, 15) / 100
        snapshot_time = pd.Timestamp(selected_date) + pd.Timedelta(hours=selected_hour)
        if data_source.startswith("Demo"):
            estimated_snapshot = _station_snapshot(
                hourly, stations, snapshot_time, safety
            )
        else:
            estimated_snapshot = _query_station_snapshot(
                snapshot_time, stations, safety
            )
        snapshot = _overlay_gbfs_snapshot(
            estimated_snapshot, gbfs_status, safety
        )
        live_station_count = (
            int((snapshot["inventory_source"] == "GBFS thời gian thực").sum())
            if "inventory_source" in snapshot
            else 0
        )
        estimated_station_count = max(0, len(snapshot) - live_station_count)
        st.divider()
        st.caption(f"📦 Nguồn dữ liệu: {data_source}")
        if live_station_count:
            st.caption(
                f"GBFS đã xác thực/còn mới: {live_station_count:,} trạm · "
                f"thu thập {gbfs_freshness.get('collected_at', 'không rõ thời điểm')}"
            )
        else:
            st.caption("Tồn kho trên bản đồ đang dùng dữ liệu ước tính station-hour.")
        st.metric("🔴 Thiếu xe", int((snapshot.status == "Thiếu xe").sum()))
        st.metric("🟣 Đầy xe", int((snapshot.status == "Đầy xe").sum()))
        st.metric("🟢 Cân bằng", int((snapshot.status == "Cân bằng").sum()))
    st.title("BikeMind Operations Center")
    st.caption(
        f"Live decision support · Dữ liệu station-hour đến {latest:%d/%m/%Y %H:%M} · "
        f"Safety buffer {safety:.0%}"
    )
    if gbfs_freshness.get("optional_missing_count"):
        st.caption(
            f"GBFS có {int(gbfs_freshness['optional_missing_count']):,} bản ghi thiếu trường tùy chọn; "
            "sức chứa thiếu từ station_information sẽ dùng giá trị station_metadata (có thể là ước tính)."
        )
    if live_station_count and estimated_station_count:
        st.warning(
            f"Nguồn tồn kho hỗn hợp: {live_station_count:,} trạm dùng GBFS đã xác thực "
            f"(thu thập {gbfs_freshness.get('collected_at', 'không rõ thời điểm')}); "
            f"{estimated_station_count:,} trạm còn lại dùng ước tính station-hour. "
            "Xem cột Nguồn tồn kho trên bản đồ trước khi điều phối."
        )
    elif live_station_count:
        st.success(
            f"Nguồn tồn kho: GBFS thời gian thực đã xác thực · "
            f"{live_station_count:,} trạm · thu thập "
            f"{gbfs_freshness.get('collected_at', 'không rõ thời điểm')}."
        )
    elif snapshot.empty:
        st.warning(
            "Không có metadata tọa độ trạm để hiển thị hoặc đối chiếu GBFS; "
            "bản đồ chưa có tồn kho khả dụng."
        )
    elif gbfs_freshness.get("fresh_count", 0):
        st.warning(
            f"GBFS có {int(gbfs_freshness['fresh_count']):,} bản ghi mới và đã xác thực "
            "nhưng không khớp station_id trong metadata bản đồ. "
            "Bản đồ đang dùng tồn kho ước tính từ station-hour."
        )
    elif gbfs_freshness.get("state") == "stale":
        st.warning(
            f"{gbfs_freshness.get('message')} Bản đồ đang dùng tồn kho ước tính "
            "từ station-hour; không dùng snapshot cũ để tính điều phối."
        )
    else:
        st.warning(
            f"{gbfs_freshness.get('message', 'Chưa có snapshot GBFS hợp lệ.')} "
            "Bản đồ đang dùng tồn kho ước tính từ station-hour."
        )
    tabs = st.tabs([
        "Use Case 07/09: 📍 Vận hành Real-time",
        "Use Case 08: 📊 Phân tích Dự báo",
        "Use Case 10-13: 🚛 Đề xuất & Tái phân bổ",
        "Use Case 01-06: ⚙️ Quản trị Backend & AI",
        "Use Case 14/15: 📈 Mô phỏng & Backtesting",
    ])
    with tabs[0]:
        if estimated_station_count and snapshot_time.date() != pd.Timestamp.now().date():
            st.info(
                f"Các chỉ số tại mốc lịch sử {snapshot_time:%d/%m/%Y %H:%M} "
                f"dùng {estimated_station_count:,} trạm fallback/ước tính từ station-hour; "
                "đây không phải tồn kho thực tế của mốc lịch sử đó."
            )
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Tổng trạm", f"{len(snapshot)}", "Mạng lưới đang giám sát")
        source_hint = (
            f"{live_station_count:,} trạm GBFS · {estimated_station_count:,} trạm ước tính"
            if live_station_count and estimated_station_count
            else ("Nguồn GBFS thời gian thực" if live_station_count else "Nguồn ước tính từ station-hour")
        )
        with c2: metric_card("Xe khả dụng", f"{snapshot.bikes_available.sum():,.0f}", source_hint, "#22C55E")
        balance_rate = 0 if snapshot.empty else (snapshot.status == "Cân bằng").mean()
        with c3: metric_card("Tỷ lệ cân bằng", f"{balance_rate:.0%}", "Mục tiêu > 70%", "#38BDF8")
        with c4: metric_card("Cảnh báo", f"{(snapshot.status != 'Cân bằng').sum()}", "Cần hành động", "#F59E0B")
        left, right = st.columns([2.1, 1])
        with left:
            clicked_station_id = render_map(snapshot)
        with right:
            st.subheader("Use Case 07: Stations cần chú ý")
            attention = snapshot[snapshot.status != "Cân bằng"].sort_values("utilization")
            detail_columns = [
                "station_name",
                "bikes_available",
                "capacity",
                "safe_min_bikes",
                "safe_max_bikes",
                "safety_buffer_pct",
                "status",
                "inventory_source",
            ]
            detail_columns = [
                column for column in detail_columns if column in attention.columns
            ]
            st.dataframe(attention[detail_columns].rename(
                columns={
                    "station_name": "Trạm",
                    "bikes_available": "Xe",
                    "capacity": "Sức chứa",
                    "safe_min_bikes": "Ngưỡng thấp",
                    "safe_max_bikes": "Ngưỡng cao",
                    "safety_buffer_pct": "Safety Buffer (%)",
                    "status": "Trạng thái",
                    "inventory_source": "Nguồn tồn kho",
                }),
                hide_index=True, width="stretch")
            st.markdown("#### Chi tiết trạm")
            station_options = (
                snapshot["station_id"].dropna().astype(str).drop_duplicates().tolist()
                if "station_id" in snapshot
                else []
            )
            if station_options:
                if clicked_station_id in station_options:
                    st.session_state["uc007_detail_station"] = clicked_station_id
                if st.session_state.get("uc007_detail_station") not in station_options:
                    st.session_state["uc007_detail_station"] = station_options[0]
                station_names = (
                    snapshot.assign(_station_id=snapshot["station_id"].astype(str))
                    .drop_duplicates("_station_id")
                    .set_index("_station_id")["station_name"]
                    .astype(str)
                    .to_dict()
                    if "station_name" in snapshot
                    else {}
                )
                chosen_station_id = st.selectbox(
                    "Chọn trạm trên bản đồ hoặc danh sách",
                    station_options,
                    format_func=lambda value: (
                        f"{station_names.get(value, 'Trạm')} · {value}"
                    ),
                    key="uc007_detail_station",
                )
                chosen_rows = snapshot.loc[
                    snapshot["station_id"].astype(str).eq(chosen_station_id)
                ]
                if not chosen_rows.empty:
                    chosen = chosen_rows.iloc[0]
                    st.write(
                        f"**{chosen.get('station_name', 'Không rõ tên')}** · "
                        f"Mã trạm `{chosen_station_id}` · "
                        f"Trạng thái **{chosen.get('status', 'Thiếu dữ liệu')}**"
                    )
                    bikes_value = pd.to_numeric(
                        chosen.get("bikes_available"), errors="coerce"
                    )
                    capacity_value = pd.to_numeric(
                        chosen.get("capacity"), errors="coerce"
                    )
                    bikes_display = (
                        "Thiếu dữ liệu"
                        if pd.isna(bikes_value)
                        else f"{bikes_value:.0f}"
                    )
                    capacity_display = (
                        "Thiếu dữ liệu"
                        if pd.isna(capacity_value)
                        else f"{capacity_value:.0f}"
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                ["Tồn kho / sức chứa", f"{bikes_display} / {capacity_display}"],
                                ["Pickups / Returns", f"{chosen.get('pickups', np.nan)} / {chosen.get('returns', np.nan)}"],
                                ["Ngưỡng an toàn", f"{chosen.get('safe_min_bikes', np.nan)} – {chosen.get('safe_max_bikes', np.nan)} xe"],
                                ["Safety Buffer", f"{chosen.get('safety_buffer_pct', np.nan)}%"],
                                ["Nguồn tồn kho", chosen.get("inventory_source", "Không rõ")],
                                ["Nguồn sức chứa", chosen.get("capacity_source", "Không rõ")],
                                ["Chất lượng dữ liệu", chosen.get("data_status", "Thiếu dữ liệu")],
                                ["Cập nhật tồn kho", _map_timestamp(chosen.get("inventory_updated_at"))],
                                ["Cập nhật sức chứa", _map_timestamp(chosen.get("capacity_updated_at"))],
                            ],
                            columns=["Thông tin", "Giá trị"],
                        ),
                        hide_index=True,
                        width="stretch",
                    )
            else:
                st.info("Chưa có trạm để xem chi tiết.")
    with tabs[1]:
        if stations.empty:
            st.info("Không có dữ liệu")
        else:
            station_options = stations.drop_duplicates("station_id", keep="last")
            station_labels = {
                _normalize_station_id(row["station_id"]):
                    f"{row['station_name']} · {_normalize_station_id(row['station_id'])}"
                for _, row in station_options.iterrows()
            }
            if not station_labels:
                st.info("Không có mã trạm hợp lệ để tra cứu dự báo.")
            else:
                station_id = st.selectbox(
                    "Chọn trạm",
                    list(station_labels),
                    format_func=lambda value: station_labels[value],
                )
                forecast_data = build_forecast_table(
                    hourly,
                    station_id,
                    snapshot_time,
                    data_source,
                )
                forecast_chart(
                    hourly,
                    station_id,
                    snapshot_time,
                    data_source,
                    forecast_data,
                )
                if forecast_data.empty:
                    st.info("Chưa có dữ liệu lịch sử phù hợp cho trạm này.")
                else:
                    history_data = forecast_data.loc[
                        forecast_data["record_type"] == "Lịch sử 24 giờ"
                    ]
                    missing_hours = int(
                        (
                            history_data["pickups"].isna()
                            | history_data["returns"].isna()
                        ).sum()
                    )
                    forecast_row = forecast_data.loc[
                        forecast_data["record_type"] == "Dự báo t+1"
                    ]
                    forecast_time = snapshot_time + pd.Timedelta(hours=1)
                    st.caption(
                        f"Lịch sử theo giờ: 24 mốc đến {snapshot_time:%d/%m/%Y %H:%M}; "
                        f"dự báo tại t+1 = {forecast_time:%d/%m/%Y %H:%M}. "
                        f"Nguồn: {forecast_data['data_source'].iloc[0]}."
                    )
                    if missing_hours:
                        st.warning(
                            f"Có {missing_hours} giờ thiếu dữ liệu trong cửa sổ lịch sử; "
                            "các khoảng trống được giữ là NaN, không điền thành 0."
                        )
                    if (
                        not forecast_row.empty
                        and forecast_row[["pickups_pred", "returns_pred"]]
                        .isna()
                        .any()
                        .any()
                    ):
                        st.warning(
                            "Chưa tạo được đầy đủ dự báo t+1: "
                            f"{forecast_row['forecast_source'].iloc[0]}"
                        )
                    display_columns = [
                        "timestamp",
                        "record_type",
                        "pickups",
                        "returns",
                        "net_flow",
                        "pickups_pred",
                        "returns_pred",
                        "net_flow_pred",
                        "forecast_source",
                        "is_estimated",
                        "data_source",
                        "updated_at",
                        "source_updated_at",
                    ]
                    st.dataframe(
                        forecast_data[display_columns],
                        hide_index=True,
                        width="stretch",
                    )
                    st.download_button(
                        "Use Case 08: ⬇️ Tải CSV dự báo 24 giờ",
                        forecast_data.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"forecast_station_{station_id}.csv",
                        mime="text/csv",
                        width="stretch",
                    )
    with tabs[2]:
        st.subheader("Use Case 10: Tồn kho dự kiến & mất cân bằng")
        st.caption(
            "Tồn kho cuối khung giờ = tồn kho đầu kỳ − dự báo lượt lấy + dự báo lượt trả; "
            "dự báo chỉ cho t+1. Vùng an toàn dùng cùng Safety Buffer với UC009."
        )
        projected = pd.DataFrame()
        if snapshot.empty:
            st.info("Không có dữ liệu trạm để tính tồn kho dự kiến.")
        else:
            try:
                planning_base = (
                    estimated_snapshot
                    if snapshot_time < pd.Timestamp.now().floor("h")
                    else snapshot
                )
                forecasts = _uc010_forecasts(hourly, planning_base, snapshot_time)
                planning_snapshot = planning_base.drop(
                    columns=[
                        "forecast_pickups",
                        "forecast_returns",
                        "forecast_pickups_source",
                        "forecast_returns_source",
                    ],
                    errors="ignore",
                ).merge(forecasts, on="station_id", how="left", validate="one_to_one")
                projected, plan = _cached_uc010_analysis(
                    planning_snapshot,
                    safety,
                )
            except (
                duckdb.Error,
                ImportError,
                OSError,
                ValueError,
                RuntimeError,
                MemoryError,
            ) as exc:
                projected = pd.DataFrame()
                plan = pd.DataFrame(columns=[
                    "route", "from_station", "to_station", "bikes",
                        "distance_km", "priority",
                ])
                st.error(f"Không thể tính tồn kho dự kiến hoặc lịch trình: {exc}")

            if not projected.empty:
                projectable = projected["data_quality"].eq("Đủ dữ liệu")
                incomplete_count = int((~projectable).sum())
                estimated_count = int(projected["projection_is_estimated"].sum())
                unbalanced_count = int(
                    projected["imbalance_status"].isin(["Thiếu xe", "Dư xe"]).sum()
                )
                st.caption(
                    f"Mốc tồn kho: {snapshot_time:%d/%m/%Y %H:%M} · "
                    f"mốc dự báo: {snapshot_time + pd.Timedelta(hours=1):%d/%m/%Y %H:%M} · "
                    f"Safety Buffer: {safety:.0%} · "
                    f"{projectable.sum():,}/{len(projected):,} trạm đủ đầu vào."
                )
                if incomplete_count:
                    st.warning(
                        f"{incomplete_count:,} trạm thiếu hoặc có dữ liệu đầu vào không hợp lệ; "
                        "không được phân loại mất cân bằng hay đưa vào lịch điều phối."
                    )
                if estimated_count:
                    st.info(
                        f"{estimated_count:,} trạm có tồn kho đầu kỳ hoặc sức chứa ước tính/fallback "
                        "hoặc chưa xác thực nguồn. Nguồn và thời điểm cập nhật được hiển thị trong bảng."
                    )

                projection_columns = [
                    "station_id",
                    "station_name",
                    "bikes_available",
                    "forecast_pickups",
                    "forecast_returns",
                    "expected_stock",
                    "safe_min_bikes",
                    "safe_max_bikes",
                    "imbalance_status",
                    "data_quality",
                    "projection_is_estimated",
                    "inventory_is_estimated",
                    "capacity_is_estimated",
                    "projection_adjusted",
                    "inventory_source",
                    "capacity_source",
                    "capacity_updated_at",
                    "forecast_pickups_source",
                    "forecast_returns_source",
                    "inventory_updated_at",
                    "forecast_data_updated_at",
                    "forecast_updated_at",
                ]
                projection_columns = [
                    column for column in projection_columns if column in projected.columns
                ]
                projection_labels = {
                    "station_id": "Mã trạm",
                    "station_name": "Trạm",
                    "bikes_available": "Tồn đầu kỳ",
                    "forecast_pickups": "Dự báo lấy",
                    "forecast_returns": "Dự báo trả",
                    "expected_stock": "Tồn cuối kỳ",
                    "safe_min_bikes": "Ngưỡng thiếu",
                    "safe_max_bikes": "Ngưỡng dư",
                    "imbalance_status": "Mất cân bằng",
                    "data_quality": "Chất lượng đầu vào",
                    "projection_is_estimated": "Có dữ liệu ước tính",
                    "inventory_is_estimated": "Tồn đầu kỳ ước tính",
                    "capacity_is_estimated": "Sức chứa ước tính",
                    "projection_adjusted": "Đã chặn theo biên",
                    "inventory_source": "Nguồn tồn đầu kỳ",
                    "capacity_source": "Nguồn sức chứa",
                    "capacity_updated_at": "Cập nhật sức chứa",
                    "forecast_pickups_source": "Nguồn dự báo lấy",
                    "forecast_returns_source": "Nguồn dự báo trả",
                    "inventory_updated_at": "Cập nhật tồn đầu kỳ",
                    "forecast_data_updated_at": "Cập nhật station-hour",
                    "forecast_updated_at": "Mốc dự báo",
                }
                with st.expander(
                    "Use Case 10: Tồn kho dự kiến và trạng thái từng trạm",
                    expanded=True,
                ):
                    st.dataframe(
                        projected[projection_columns].rename(
                            columns=projection_labels
                        ),
                        hide_index=True,
                        width="stretch",
                        height=360,
                    )

                st.subheader("Use Case 10/11: Lịch trình xe tải · Gom → Rải")
                unmatched = pd.DataFrame(
                    plan.attrs.get("unmatched_stations", [])
                )
                if not unmatched.empty:
                    unmatched_labels = {
                        "station_id": "Mã trạm",
                        "station_name": "Trạm",
                        "imbalance_status": "Trạng thái",
                        "remaining_surplus": "Xe dư chưa ghép",
                        "remaining_deficit": "Nhu cầu thiếu chưa ghép",
                        "reason": "Lý do",
                        "data_source": "Nguồn dữ liệu",
                        "updated_at": "Thời điểm cập nhật",
                        "uses_estimated_data": "Có dữ liệu ước tính",
                    }
                    st.warning(
                        f"Còn {len(unmatched):,} trạm chưa ghép hết; lượng dư/thiếu này "
                        "không được tính vào tổng xe đã lập lịch."
                    )
                    st.dataframe(
                        unmatched.rename(columns=unmatched_labels),
                        hide_index=True,
                        width="stretch",
                    )
                if not projectable.any():
                    st.warning(
                        "Chưa đủ dự báo, tồn kho đầu kỳ hoặc sức chứa hợp lệ để lập lịch điều phối."
                    )
                elif plan.empty and unbalanced_count:
                    st.warning(
                        f"Phát hiện {unbalanced_count:,} trạm mất cân bằng nhưng chưa tạo được "
                        "tuyến khả thi; các trạm còn lại được liệt kê ở bảng chưa ghép."
                    )
                elif plan.empty:
                    st.success("Các trạm đủ dữ liệu hiện nằm trong vùng an toàn.")
                else:
                    route_bikes = pd.to_numeric(
                        plan.get("bikes", pd.Series(dtype="float64")),
                        errors="coerce",
                    ).replace([np.inf, -np.inf], np.nan).dropna()
                    route_distances = pd.to_numeric(
                        plan.get("distance_km", pd.Series(dtype="float64")),
                        errors="coerce",
                    ).replace([np.inf, -np.inf], np.nan).dropna()
                    scheduled_bikes = int(route_bikes.sum()) if not route_bikes.empty else 0
                    route_count = len(plan)
                    distance_km = float(route_distances.sum()) if not route_distances.empty else 0.0
                    estimated_route_count = int(
                        plan.get(
                            "uses_estimated_data",
                            pd.Series(False, index=plan.index),
                        )
                        .map(_decision_flag)
                        .sum()
                    )
                    unmatched_surplus = pd.to_numeric(
                        unmatched.get(
                            "remaining_surplus", pd.Series(dtype="float64")
                        ),
                        errors="coerce",
                    ).replace([np.inf, -np.inf], np.nan).fillna(0).sum()
                    unmatched_deficit = pd.to_numeric(
                        unmatched.get(
                            "remaining_deficit", pd.Series(dtype="float64")
                        ),
                        errors="coerce",
                    ).replace([np.inf, -np.inf], np.nan).fillna(0).sum()
                    a, b = st.columns([1.5, 1])
                    with a:
                        plan_labels = {
                            "route_id": "Mã tuyến",
                            "route": "Tuyến",
                            "from_station": "Trạm nguồn",
                            "from_station_id": "Mã trạm nguồn",
                            "to_station": "Trạm đích",
                            "to_station_id": "Mã trạm đích",
                            "bikes": "Số xe",
                            "distance_km": "Khoảng cách (km)",
                            "priority": "Ưu tiên",
                            "from_data_source": "Nguồn dữ liệu trạm nguồn",
                            "to_data_source": "Nguồn dữ liệu trạm đích",
                            "from_updated_at": "Cập nhật trạm nguồn",
                            "to_updated_at": "Cập nhật trạm đích",
                            "uses_estimated_data": "Có dữ liệu ước tính",
                        }
                        st.dataframe(
                            plan.rename(columns=plan_labels),
                            hide_index=True,
                            width="stretch",
                        )
                    with b:
                        metric_card(
                            "Tổng xe điều phối",
                            f"{scheduled_bikes:,}",
                            f"{route_count:,} chặng",
                        )
                        metric_card(
                            "Quãng đường",
                            f"{distance_km:.1f} km",
                            "Greedy Haversine",
                            "#F59E0B",
                        )
                        metric_card(
                            "Xe dư / thiếu chưa ghép",
                            f"{int(unmatched_surplus):,} / "
                            f"{int(unmatched_deficit):,}",
                            f"{len(unmatched):,} trạm",
                            "#EF4444",
                        )
                        if estimated_route_count:
                            st.warning(
                                f"{estimated_route_count:,}/{route_count:,} tuyến có "
                                "ít nhất một đầu vào ước tính/fallback. Hãy kiểm tra "
                                "nguồn dữ liệu và thời điểm cập nhật trước khi quyết định."
                            )
                        if st.button(
                            "Use Case 12: ✅ Phê duyệt Lệnh",
                            type="primary",
                            key="uc012_approve",
                            width="stretch",
                        ):
                            try:
                                decision_id = log_decision("APPROVED", plan)
                            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                                st.error(f"Không thể phê duyệt lệnh UC012: {exc}")
                            else:
                                st.success(
                                    "Đã ghi nhận/cập nhật quyết định phê duyệt "
                                    f"(mã `{decision_id[:12]}`)."
                                )
                        if st.button(
                            "Use Case 12: ❌ Từ chối",
                            key="uc012_reject",
                            width="stretch",
                        ):
                            try:
                                decision_id = log_decision("REJECTED", plan)
                            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                                st.error(f"Không thể từ chối lệnh UC012: {exc}")
                            else:
                                st.warning(
                                    "Đã ghi nhận/cập nhật quyết định từ chối "
                                    f"(mã `{decision_id[:12]}`)."
                                )
                    render_map(planning_base, plan)
                if plan.empty:
                    st.info(
                        "UC012 chưa thể ghi nhận quyết định vì kế hoạch hiện không "
                        "có tuyến điều phối hợp lệ."
                    )
            else:
                st.warning("Không có kết quả dự kiến khả dụng.")
        render_uc013_panel(projected, snapshot_time)
        render_decision_history()
    with tabs[3]:
        st.subheader("Use Case 01-06: ⚙️ Quản trị Backend & AI")
        render_data_status(data_status, "tab")
        pipeline_tabs = st.tabs(["Use Case 01/05/06: Quản trị Pipeline Dữ liệu", "Use Case 02/03: Tiền xử lý dữ liệu"])
        with pipeline_tabs[0]:
            st.markdown("#### Use Case 01/05/06: Quản trị Pipeline Dữ liệu")
            if st.session_state.get("uc1_activation_message"):
                st.success(st.session_state["uc1_activation_message"])
            with st.expander("Use Case 01: 📁 Danh sách toàn bộ file CSV đang sử dụng", expanded=False):
                csv_inventory = list_project_csv_files(loaded_only=True)
                if csv_inventory.empty:
                    st.info("Không tìm thấy file CSV hợp lệ trong thư mục data_load/.")
                else:
                    total_size = csv_inventory["Dung lượng (MB)"].sum()
                    st.caption(
                        f"Tìm thấy **{len(csv_inventory):,} file CSV** · "
                        f"Tổng dung lượng **{total_size:,.2f} MB** · "
                        "Danh sách được đọc từ thư mục data_load/"
                    )
                    st.dataframe(
                        csv_inventory,
                        hide_index=True,
                        width="stretch",
                        height=320,
                    )
            with st.expander("Use Case 01: ⚠️ Danh sách file CSV không sử dụng được", expanded=False):
                if UC1_REJECTED_CSV_PATH.exists():
                    try:
                        rejected_csv = json.loads(UC1_REJECTED_CSV_PATH.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        rejected_csv = []
                    if rejected_csv:
                        st.warning(
                            f"Có **{len(rejected_csv):,} file CSV không được đưa vào xử lý."
                        )
                        rejected_columns = ["File", "Lý do", "Đường dẫn tương đối"]
                        rejected_frame = pd.DataFrame(rejected_csv)
                        if "Lý do" not in rejected_frame.columns:
                            rejected_frame["Lý do"] = (
                                "Chưa được ghi nhận ở lần Load Data trước; "
                                "hãy Load Data lại để xác định nguyên nhân."
                            )
                        for column in rejected_columns:
                            if column not in rejected_frame.columns:
                                rejected_frame[column] = "Không có thông tin"
                        st.dataframe(
                            rejected_frame[rejected_columns],
                            hide_index=True,
                            width="stretch",
                            height=320,
                        )
                    else:
                        st.info("Không có file CSV nào bị loại khỏi quá trình xử lý.")
                else:
                    st.info("Chưa có thông tin file CSV không sử dụng được.")
            with st.expander("Use Case 01: 🗑️ Xóa file CSV đã load", expanded=False):
                loaded_csv_inventory = list_project_csv_files(loaded_only=True)
                if loaded_csv_inventory.empty:
                    st.info("Không có file CSV đã load để xóa.")
                else:
                    loaded_csv_paths = loaded_csv_inventory["Đường dẫn tương đối"].tolist()
                    selected_csv_paths = st.multiselect(
                        "Chọn các file CSV muốn xóa khỏi thư mục data_load/",
                        options=loaded_csv_paths,
                        key="delete_loaded_csv_paths",
                    )
                    if selected_csv_paths:
                        st.warning(
                            f"Bạn đang chọn **{len(selected_csv_paths):,} file**. "
                            "Thao tác này sẽ xóa file khỏi máy và không thể hoàn tác."
                        )
                        confirm_delete = st.checkbox(
                            "Tôi xác nhận muốn xóa các file CSV đã chọn.",
                            key="confirm_delete_loaded_csv",
                        )
                        if st.button(
                            "Use Case 01: 🗑️ Xác nhận xóa file đã chọn",
                            type="secondary",
                            disabled=not confirm_delete,
                            key="delete_selected_loaded_csv",
                            width="stretch",
                        ):
                            try:
                                deleted_count, missing_count, failed_paths = delete_loaded_csv_files(selected_csv_paths)
                                if deleted_count or missing_count:
                                    clear_loaded_data_artifacts()
                                    st.session_state.pop("uc1_loaded_data", None)
                                    st.session_state.pop("uc1_load_stats", None)
                                    st.session_state.pop("uc1_activation_message", None)
                            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                                st.error(f"Không thể xóa file CSV đã chọn: {exc}")
                            else:
                                list_project_csv_files.clear()
                                message = f"Đã xóa {deleted_count:,} file CSV khỏi thư mục data_load/."
                                if missing_count:
                                    message += f" {missing_count:,} file không còn tồn tại nên chỉ được loại khỏi danh sách."
                                if failed_paths:
                                    st.warning(
                                        f"{message} Không thể xóa {len(failed_paths):,} file đang được sử dụng. "
                                        "Hãy đóng tiến trình đang mở file rồi thử lại."
                                    )
                                    st.caption("File chưa xóa: " + ", ".join(failed_paths[:3]))
                                else:
                                    st.success(message)
                                    st.rerun()
            st.caption(
                f"Nguồn gốc: `{DATA_DIRECTORY}` · "
                f"bản sao load: `{DATA_LOAD_DIRECTORY}` · chỉ quét các file `.csv`"
            )
            load_data_clicked = st.button(
                "Use Case 01: 📥 Load Data",
                type="primary",
                key="uc1_load_data",
                width="stretch",
            )
            if load_data_clicked:
                progress_bar = st.progress(0, text="Đang chuẩn bị nạp dữ liệu...")
                progress_text = st.empty()

                def update_load_progress(processed: int, total: int, message: str, loaded_rows: int) -> None:
                    percentage = 1.0 if total == 0 else min(processed / total, 1.0)
                    progress_bar.progress(
                        percentage,
                        text=f"{processed:,}/{total:,} file · {percentage:.0%} · {message}",
                    )
                    progress_text.caption(f"{message} | Đã nạp tạm: {loaded_rows:,} dòng")

                try:
                    copy_csv_files_to_load_directory()
                    with st.spinner("Đang sao chép và quét dữ liệu CSV, tính toán mã MD5, khử trùng file và nạp dữ liệu..."):
                        loaded_data, load_stats, activated_hourly_count, activated_station_count = scan_and_prepare_unique_csv(
                            base_directory=DATA_LOAD_DIRECTORY,
                            output_directory=PROCESSED,
                            progress_callback=update_load_progress,
                        )
                except (OSError, RuntimeError, ValueError, MemoryError) as exc:
                    st.error(f"Không thể sao chép hoặc xử lý dữ liệu CSV: {exc}")
                    st.stop()
                load_dashboard_data.clear()
                list_project_csv_files.clear()
                progress_bar.progress(1.0, text="Hoàn tất Use Case 1 · 100%")
                progress_text.caption(
                    f"Đã xử lý {load_stats['total']:,} file · "
                    f"{load_stats.get('row_count', len(loaded_data)):,} chuyến đi duy nhất."
                )
                st.session_state["uc1_loaded_data"] = loaded_data
                st.session_state["uc1_load_stats"] = load_stats
                persist_uc1_result(
                    loaded_data,
                    load_stats,
                    activated_hourly_count,
                    activated_station_count,
                )
                st.session_state["uc1_activation_message"] = (
                    f"Dữ liệu đã load thành công và đang được dashboard sử dụng: "
                    f"{activated_hourly_count:,} hourly records, "
                    f"{activated_station_count:,} stations; "
                    f"bộ dữ liệu hợp nhất có {load_stats.get('row_count', len(loaded_data)):,} chuyến đi."
                )
                st.rerun()
            if "uc1_load_stats" in st.session_state:
                load_stats = st.session_state["uc1_load_stats"]
                loaded_data = st.session_state["uc1_loaded_data"]
                if not loaded_data.empty:
                    st.success(
                        "Đã xử lý và khử trùng dữ liệu; dashboard đã dùng các bảng Parquet."
                    )
                    stat_columns = st.columns(4)
                    stat_columns[0].metric("CSV đã quét", f"{load_stats['total']:,}")
                    stat_columns[1].metric("CSV độc nhất", f"{load_stats['unique']:,}")
                    stat_columns[2].metric("CSV trùng bỏ qua", f"{load_stats['dup_files']:,}")
                    stat_columns[3].metric(
                        "Bản ghi sạch",
                        f"{load_stats.get('row_count', len(loaded_data)):,}",
                    )
                    st.caption(f"Đã loại {load_stats['dup_rows']:,} bản ghi trùng hoàn toàn.")
                    st.caption(
                        f"File sai schema: {load_stats.get('schema_errors', 0):,} · "
                        f"File đọc/lỗi: {load_stats.get('read_errors', 0):,} · "
                        f"Tổng file không hợp lệ: {load_stats.get('invalid_files', 0):,}"
                    )
                    if "previous_rows" in load_stats:
                        st.caption(
                            f"Hợp nhất dữ liệu: {load_stats['previous_rows']:,} dòng trước đó + "
                            f"{load_stats['new_unique_rows']:,} dòng mới → "
                            f"{load_stats['combined_rows']:,} dòng duy nhất."
                        )
                    st.caption(
                        f"Bảng bên dưới là bản xem trước {min(5, len(loaded_data)):,} dòng; "
                        f"toàn bộ {load_stats.get('row_count', len(loaded_data)):,} chuyến đi duy nhất "
                        "được lưu ngoài RAM trong processed/uc1_trips_deduplicated.parquet."
                    )
                    st.dataframe(loaded_data.head(5), hide_index=True, width="stretch")
                else:
                    st.error("Không tìm thấy dữ liệu hợp lệ trong thư mục data_load/. Vui lòng kiểm tra lại đường dẫn!")
            else:
                st.info("Không có dữ liệu")
        with pipeline_tabs[0]:
            if data_source == "Không có dữ liệu":
                st.info("Không có dữ liệu")
            else:
                build_features_clicked = st.button(
                    "Use Case 04: 🧩 Build Features",
                    width="stretch",
                )
                if build_features_clicked:
                    with st.spinner("Đang tạo temporal, lag và rolling features..."):
                        try:
                            features = build_features_pipeline(
                                PROCESSED / "station_hour_cleaned.parquet",
                                FEATURES_PATH,
                            )
                            feature_rows = int(features.attrs.get("row_count", len(features)))
                            feature_stats = features.attrs.get("feature_stats", {})
                            persistent_state = _persist_use_case_result(
                                "uc4",
                                {
                                    "rows": feature_rows,
                                    "artifact": FEATURES_PATH.name,
                                    "preview": _frame_preview(features),
                                    "feature_stats": feature_stats,
                                },
                            )
                            st.success(f"Đã lưu {feature_rows:,} dòng feature vào station_features.parquet.")
                            st.caption(
                                "Số giờ station-hour bị thiếu: "
                                f"{int(feature_stats.get('hour_gaps', 0)):,} · "
                                "Dòng thiếu lịch sử: "
                                f"{int(feature_stats.get('rows_missing_history', 0)):,} · "
                                "Dòng bị loại do thiếu lịch sử: "
                                f"{int(feature_stats.get('rows_dropped_missing_history', 0)):,}"
                            )
                            st.dataframe(features.head(5), hide_index=True, width="stretch")
                        except (OSError, ImportError, RuntimeError, ValueError) as exc:
                            st.error(f"Không thể tạo features: {exc}")
                retrain_clicked = st.button(
                    "Use Case 05: 🧠 Retrain LightGBM Model",
                    width="stretch",
                )
                if retrain_clicked:
                    with st.spinner("Chronological split 70/15/15 và huấn luyện..."):
                        try:
                            if not FEATURES_PATH.is_file():
                                raise FileNotFoundError(
                                    "Chưa có station_features.parquet. "
                                    "Hãy chạy Use Case 04: Build Features trước."
                                )
                            metrics = train_models(FEATURES_PATH, MODELS)
                            persistent_state = _persist_use_case_result(
                                "uc5",
                                {
                                    "metrics": metrics,
                                    "artifacts": ["lgb_pickups.pkl", "lgb_returns.pkl"],
                                },
                            )
                            st.success("Đã lưu lgb_pickups.pkl và lgb_returns.pkl")
                            split_metrics = metrics.get("_split", {})
                            if split_metrics:
                                st.caption(
                                    "Đánh giá Test theo timestamp, không xáo trộn: "
                                    f"{int(split_metrics['test_rows']):,} dòng nguồn, "
                                    f"{int(split_metrics['test_sampled_rows']):,} dòng mẫu; "
                                    f"Train/Validation tối đa 100.000 dòng mỗi tập."
                                )
                            st.dataframe(pd.DataFrame(metrics).T, width="stretch")
                        except (
                            duckdb.Error,
                            ImportError,
                            OSError,
                            ValueError,
                            RuntimeError,
                            TypeError,
                            MemoryError,
                        ) as exc:
                            st.error(f"Không thể huấn luyện model: {exc}")
                benchmark_clicked = st.button(
                    "Use Case 06: 📏 Run Baseline Benchmark",
                    width="stretch",
                )
                if benchmark_clicked:
                    with st.spinner("Đang so sánh Historical Average, Seasonal Naive và LightGBM..."):
                        try:
                            if not FEATURES_PATH.is_file():
                                raise FileNotFoundError(
                                    "Chưa có station_features.parquet. "
                                    "Hãy chạy Use Case 04: Build Features trước."
                                )
                            benchmark = benchmark_models(FEATURES_PATH, MODELS)
                            benchmark.to_csv(BENCHMARK_PATH, index=False)
                            persistent_state = _persist_use_case_result(
                                "uc6",
                                {
                                    "artifact": BENCHMARK_PATH.name,
                                    "preview": _frame_preview(benchmark),
                                },
                            )
                            st.dataframe(benchmark, hide_index=True, width="stretch")
                            st.download_button(
                                "Use Case 06: ⬇️ Tải benchmark CSV",
                                benchmark.to_csv(index=False).encode("utf-8-sig"),
                                file_name="bikemind_benchmark.csv",
                                mime="text/csv",
                            )
                        except (
                            duckdb.Error,
                            ImportError,
                            OSError,
                            ValueError,
                            RuntimeError,
                            TypeError,
                            MemoryError,
                        ) as exc:
                            st.error(f"Không thể benchmark model: {exc}")
                uc4_state = persistent_state.get("uc4")
                if (
                    not build_features_clicked
                    and isinstance(uc4_state, dict)
                    and FEATURES_PATH.exists()
                ):
                    st.success(
                        f"Use Case 04 đã khôi phục: {int(uc4_state.get('rows', 0)):,} dòng feature."
                    )
                    feature_stats = uc4_state.get("feature_stats", {})
                    if isinstance(feature_stats, dict):
                        st.caption(
                            "Số giờ station-hour bị thiếu: "
                            f"{int(feature_stats.get('hour_gaps', 0)):,} · "
                            "Dòng thiếu lịch sử: "
                            f"{int(feature_stats.get('rows_missing_history', 0)):,} · "
                            "Dòng bị loại do thiếu lịch sử: "
                            f"{int(feature_stats.get('rows_dropped_missing_history', 0)):,}"
                        )
                    st.dataframe(
                        _persisted_preview(uc4_state, "preview"),
                        hide_index=True,
                        width="stretch",
                    )
                uc5_state = persistent_state.get("uc5")
                if (
                    not retrain_clicked
                    and isinstance(uc5_state, dict)
                    and all((MODELS / name).exists() for name in ("lgb_pickups.pkl", "lgb_returns.pkl"))
                ):
                    st.success("Use Case 05 đã khôi phục: lgb_pickups.pkl và lgb_returns.pkl.")
                    st.dataframe(
                        pd.DataFrame(uc5_state.get("metrics", {})).T,
                        width="stretch",
                    )
                uc6_state = persistent_state.get("uc6")
                if (
                    not benchmark_clicked
                    and isinstance(uc6_state, dict)
                    and BENCHMARK_PATH.exists()
                ):
                    benchmark = pd.read_csv(BENCHMARK_PATH)
                    st.success("Use Case 06 đã khôi phục từ benchmark đã lưu.")
                    st.dataframe(benchmark, hide_index=True, width="stretch")
                    st.download_button(
                        "Use Case 06: ⬇️ Tải benchmark CSV",
                        benchmark.to_csv(index=False).encode("utf-8-sig"),
                        file_name="bikemind_benchmark.csv",
                        mime="text/csv",
                        key="restored_benchmark_download",
                    )
            if data_source == "Không có dữ liệu":
                st.info("Không có dữ liệu")
            else:
                st.subheader("Use Case 06: Benchmark")
                st.caption("Nhấn Run Baseline Benchmark để tính lại trên dữ liệu hiện tại.")
        with pipeline_tabs[1]:
            st.markdown("#### Use Case 02: Tiền xử lý dữ liệu")
            clean_data_clicked = st.button(
                "Use Case 02: 🧹 Khởi chạy Data Cleaning & Zero-Filling Pipeline",
                type="primary",
                key="uc2_clean_data",
                width="stretch",
            )
            if clean_data_clicked:
                with st.spinner(
                    "Đang đọc dữ liệu từ data_load/, lọc Outliers, trích xuất metadata "
                    "trạm và phủ kín lưới 24/7 (Zero-Filling)..."
                ):
                    try:
                        cleaned_hourly, cleaned_stations, cleaning_stats = (
                            process_and_build_parquet_grid(
                                input_directory=DATA_LOAD_DIRECTORY,
                                output_directory=PROCESSED,
                                input_parquet=UC1_FULL_DATA_PATH,
                            )
                        )
                    except (OSError, ImportError, ValueError, TypeError) as exc:
                        st.error(f"Không thể thực hiện Use Case 2: {exc}")
                    else:
                        load_dashboard_data.clear()
                        persistent_state = _persist_use_case_result(
                            "uc2",
                            {
                                "stats": _json_ready(cleaning_stats),
                                "preview": _frame_preview(cleaned_hourly),
                                "artifacts": [
                                    "station_hour_cleaned.parquet",
                                    "station_metadata.parquet",
                                ],
                            },
                        )
                        st.success(
                            "Đã hoàn tất làm sạch dữ liệu và tạo lưới thời gian 24/7 thành công!"
                        )
                        _render_uc2_audit(cleaning_stats, cleaned_hourly)
            uc2_state = persistent_state.get("uc2")
            if (
                not clean_data_clicked
                and isinstance(uc2_state, dict)
                and (PROCESSED / "station_hour_cleaned.parquet").exists()
                and (PROCESSED / "station_metadata.parquet").exists()
            ):
                stats = uc2_state.get("stats", {})
                if not isinstance(stats, dict):
                    stats = {}
                st.success("Use Case 02 đã khôi phục từ dữ liệu đã xử lý.")
                _render_uc2_audit(stats, _persisted_preview(uc2_state, "preview"))
            st.markdown("#### Use Case 03: GBFS Station Status Collector")
            collect_gbfs_clicked = st.button(
                "Use Case 03: 🛰️ Kết nối GBFS & Thu thập Snapshot Trạm",
                type="primary",
                key="uc3_collect_gbfs",
                width="stretch",
            )
            if collect_gbfs_clicked:
                with st.spinner(
                    "Đang kết nối tới API GBFS công khai để lấy trạng thái tồn kho trạm thời gian thực..."
                ):
                    collector_result = run_gbfs_collector_job(
                        DB_PATH,
                        station_aliases=_load_station_aliases(),
                    )
                if collector_result["success"]:
                    _cached_latest_station_status.clear()
                    snapshot_data = collector_result["status_df"]
                    persistent_state = _persist_use_case_result(
                        "uc3",
                        {
                            "records_count": int(collector_result["records_count"]),
                            "timestamp": collector_result["timestamp"],
                            "warnings": collector_result.get("warnings"),
                            "preview": _frame_preview(snapshot_data),
                            "artifact": DB_PATH.name,
                        },
                    )
                    st.rerun()
                else:
                    st.error(
                        f"{collector_result['message']} "
                        "Vui lòng kiểm tra kết nối mạng, endpoint GBFS và quyền ghi file SQLite."
                    )
            uc3_state = persistent_state.get("uc3")
            if (
                not collect_gbfs_clicked
                and isinstance(uc3_state, dict)
                and DB_PATH.exists()
            ):
                st.success(
                    "Kết quả thu thập Use Case 03 đã lưu gần nhất: "
                    f"{int(uc3_state.get('records_count', 0)):,} trạm · "
                    f"{uc3_state.get('timestamp', 'Không rõ thời điểm')}."
                )
                if uc3_state.get("warnings"):
                    st.warning(str(uc3_state["warnings"]))
                st.dataframe(
                    _persisted_preview(uc3_state, "preview"),
                    hide_index=True,
                    width="stretch",
                )
    with tabs[4]:
        render_system_health()
        st.divider()
        st.subheader("Use Case 15: Historical Backtesting & Report")
        st.caption(
            "Báo cáo được lưu trong processed/; DSS phải có tỷ lệ sức chứa cao hơn baseline. "
            "Ghi đè chỉ diễn ra sau bước xác nhận."
        )
        input_error = _backtesting_input_error(hourly, stations)
        if input_error:
            st.warning(f"Chưa thể chạy báo cáo: {input_error}")

        ratio_columns = st.columns(2)
        with ratio_columns[0]:
            passive_capacity_percent = st.slider(
                "Baseline: tỷ lệ sức chứa khả dụng",
                min_value=0,
                max_value=100,
                value=55,
                step=5,
                format="%d%%",
                key="uc15_baseline_capacity_ratio",
            )
        with ratio_columns[1]:
            dss_capacity_percent = st.slider(
                "BikeMind DSS: tỷ lệ sức chứa khả dụng",
                min_value=0,
                max_value=100,
                value=75,
                step=5,
                format="%d%%",
                key="uc15_dss_capacity_ratio",
            )
        passive_capacity_ratio = passive_capacity_percent / 100
        dss_capacity_ratio = dss_capacity_percent / 100
        ratio_error = None
        try:
            passive_capacity_ratio, dss_capacity_ratio = _validate_backtesting_ratios(
                passive_capacity_ratio,
                dss_capacity_ratio,
            )
        except ValueError as exc:
            ratio_error = str(exc)
            st.warning(ratio_error)

        pending_key = "uc15_backtesting_pending_confirmation"
        can_run_backtesting = input_error is None and ratio_error is None
        if not st.session_state.get(pending_key, False):
            if st.button(
                "Use Case 15: Chuẩn bị chạy mô phỏng",
                type="primary",
                key="uc15_prepare_backtesting",
                disabled=not can_run_backtesting,
            ):
                st.session_state[pending_key] = True

        pending_confirmation = bool(st.session_state.get(pending_key, False))
        if pending_confirmation:
            if BACKTESTING_PATH.is_file():
                st.warning(
                    f"Xác nhận để thay thế báo cáo hiện có: {BACKTESTING_PATH}."
                )
            else:
                st.warning(
                    "Xác nhận để chạy mô phỏng và ghi báo cáo vào processed/."
                )
            confirm_column, cancel_column = st.columns(2)
            with confirm_column:
                confirm_backtesting = st.button(
                    "Xác nhận chạy và ghi báo cáo",
                    key="uc15_confirm_backtesting",
                    disabled=not can_run_backtesting,
                )
            with cancel_column:
                cancel_backtesting = st.button(
                    "Hủy",
                    key="uc15_cancel_backtesting",
                )
            if cancel_backtesting:
                st.session_state[pending_key] = False
                st.info("Đã hủy thao tác; báo cáo hiện có không bị thay đổi.")
            elif confirm_backtesting:
                st.session_state[pending_key] = False
                try:
                    input_updated_at = _backtesting_input_updated_at(data_source)
                    scenario = run_historical_simulation(
                        hourly,
                        stations,
                        passive_capacity_ratio=passive_capacity_ratio,
                        dss_capacity_ratio=dss_capacity_ratio,
                        data_source=data_source,
                        input_updated_at=input_updated_at,
                    )
                    baseline, dss, improvement = _backtesting_summary(scenario)
                    report_bytes = _backtesting_csv_bytes(scenario)
                    _write_backtesting_report(BACKTESTING_PATH, report_bytes)
                    _cached_backtesting_report.clear()
                except (
                    OSError,
                    duckdb.Error,
                    ImportError,
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                    AttributeError,
                    RuntimeError,
                    MemoryError,
                    OverflowError,
                    FloatingPointError,
                ) as exc:
                    st.error(f"Không thể chạy hoặc lưu báo cáo UC015: {exc}")
                else:
                    try:
                        persistent_state = _persist_use_case_result(
                            "uc14_15",
                            {
                                "artifact": BACKTESTING_PATH.name,
                                "preview": _frame_preview(scenario),
                                "data_source": data_source,
                                "input_updated_at": input_updated_at,
                                "baseline_capacity_ratio": passive_capacity_ratio,
                                "dss_capacity_ratio": dss_capacity_ratio,
                            },
                        )
                    except (OSError, RuntimeError, TypeError, ValueError) as exc:
                        st.error(
                            "Báo cáo đã được ghi nhưng không thể cập nhật trạng thái Use Case: "
                            f"{exc}"
                        )
                    st.dataframe(scenario, hide_index=True, width="stretch")
                    st.success(
                        f"Scenario B giảm unmet demand {improvement:.1f}% so với baseline "
                        f"(baseline {baseline:,.2f}; DSS {dss:,.2f}). "
                        f"Đã lưu: {BACKTESTING_PATH.name}."
                    )
                    st.download_button(
                        "Use Case 15: ⬇️ Tải báo cáo backtesting",
                        report_bytes,
                        file_name="bikemind_backtesting_report.csv",
                        mime="text/csv",
                        key="backtesting_download",
                    )
        else:
            if BACKTESTING_PATH.is_file():
                try:
                    scenario = _load_backtesting_report(BACKTESTING_PATH)
                    _, _, improvement = _backtesting_summary(scenario)
                    report_bytes = _backtesting_csv_bytes(scenario)
                except (
                    OSError,
                    pd.errors.EmptyDataError,
                    pd.errors.ParserError,
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                    AttributeError,
                    RuntimeError,
                ) as exc:
                    st.error(f"Không thể khôi phục báo cáo backtesting UC015: {exc}")
                else:
                    st.success("Use Case 15 đã khôi phục báo cáo backtesting đã lưu.")
                    if {
                        "Baseline Capacity Ratio",
                        "DSS Capacity Ratio",
                    }.issubset(scenario.columns):
                        try:
                            saved_baseline_ratio = float(
                                scenario["Baseline Capacity Ratio"].iloc[0]
                            )
                            saved_dss_ratio = float(
                                scenario["DSS Capacity Ratio"].iloc[0]
                            )
                        except (TypeError, ValueError, OverflowError, IndexError):
                            saved_baseline_ratio = float("nan")
                            saved_dss_ratio = float("nan")
                        if (
                            math.isfinite(saved_baseline_ratio)
                            and math.isfinite(saved_dss_ratio)
                            and (
                                not math.isclose(
                                    saved_baseline_ratio,
                                    passive_capacity_ratio,
                                )
                                or not math.isclose(
                                    saved_dss_ratio,
                                    dss_capacity_ratio,
                                )
                            )
                        ):
                            st.info(
                                "Đang xem báo cáo đã lưu với tham số cũ; thay đổi slider "
                                "chưa được áp dụng cho tới khi xác nhận chạy lại."
                            )
                    st.dataframe(scenario, hide_index=True, width="stretch")
                    st.info(f"Scenario B giảm unmet demand {improvement:.1f}% so với baseline.")
                    st.download_button(
                        "Use Case 15: ⬇️ Tải báo cáo backtesting",
                        report_bytes,
                        file_name="bikemind_backtesting_report.csv",
                        mime="text/csv",
                        key="restored_backtesting_download",
                    )
            elif input_error is None:
                st.info("Chuẩn bị mô phỏng để so sánh hai kịch bản vận hành.")


if __name__ == "__main__":
    main()
