"""BikeMind - Smart Mobility Dashboard & Decision Support System."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import folium
import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
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
    calculate_expected_inventory,
    generate_rebalancing_plan,
    run_greedy_rebalancing,
)


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
        st.dataframe(details, hide_index=True, use_container_width=True)


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
    latest = query_station_snapshot(PROCESSED, timestamp)
    return _station_snapshot(latest, stations, timestamp, safety_buffer)


def _normalize_station_id(value: object) -> str:
    """Normalize numeric/text station identifiers without changing source data."""
    if value is None or pd.isna(value):
        return ""
    station_id = str(value).strip()
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


def render_map(snapshot: pd.DataFrame, plan: pd.DataFrame | None = None, height: int = 570) -> None:
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
        valid_snapshot = valid_snapshot.loc[valid_coordinates].copy()
    else:
        valid_snapshot = valid_snapshot.iloc[0:0].copy()
    center = (
        [valid_snapshot["lat"].mean(), valid_snapshot["lon"].mean()]
        if not valid_snapshot.empty
        else [40.74, -73.99]
    )
    fmap = folium.Map(
        location=center,
        zoom_start=12,
        tiles="OpenStreetMap",
        attr="© OpenStreetMap contributors",
        control_scale=True,
    )
    Fullscreen(position="topright").add_to(fmap)
    for row in valid_snapshot.to_dict("records"):
        pickups = "—" if pd.isna(row["pickups"]) else f"{row['pickups']:.0f}"
        returns = "—" if pd.isna(row["returns"]) else f"{row['returns']:.0f}"
        source = row.get("inventory_source", "Không rõ nguồn")
        popup = (f"<b>{row['station_name']}</b><br>Xe khả dụng: {row['bikes_available']:.0f}/{row['capacity']}<br>"
                 f"Trạng thái: {row['status']}<br>Nguồn tồn kho: {source}<br>"
                 f"Nguồn sức chứa: {row.get('capacity_source', 'Không rõ')}<br>"
                 f"Pickups: {pickups} | Returns: {returns}")
        folium.CircleMarker([row["lat"], row["lon"]], radius=10, color=_status_color(row["status"]),
                            fill=True, fill_color=_status_color(row["status"]), fill_opacity=0.9,
                            tooltip=f"{row['station_name']} · {row['status']}", popup=popup).add_to(fmap)
    if plan is not None and not plan.empty:
        lookup_by_id = {}
        if "station_id" in valid_snapshot.columns:
            lookup_by_id = (
                valid_snapshot
                .drop_duplicates(subset=["station_id"], keep="last")
                .set_index("station_id")[["lat", "lon"]]
                .to_dict("index")
            )
        lookup_by_name = (
            valid_snapshot
            .drop_duplicates(subset=["station_name"], keep="last")
            .set_index("station_name")[["lat", "lon"]]
            .to_dict("index")
        )
        for row in plan.to_dict("records"):
            from_location = lookup_by_id.get(row.get("from_station_id"))
            to_location = lookup_by_id.get(row.get("to_station_id"))
            if from_location is None:
                from_location = lookup_by_name.get(row["from_station"])
            if to_location is None:
                to_location = lookup_by_name.get(row["to_station"])
            if from_location is not None and to_location is not None:
                folium.PolyLine([(from_location["lat"], from_location["lon"]),
                                 (to_location["lat"], to_location["lon"])],
                                color="#F59E0B", weight=4, opacity=0.9,
                                tooltip=f"{row['bikes']} xe · {row['distance_km']} km").add_to(fmap)
    st_folium(fmap, height=height, use_container_width=True, returned_objects=[])


def log_decision(action: str, plan: pd.DataFrame) -> None:
    unmatched = pd.DataFrame(plan.attrs.get("unmatched_stations", []))
    payload = {
        "schema_version": 2,
        "routes": json.loads(plan.to_json(orient="records", date_format="iso")),
        "summary": plan.attrs.get("summary", {}),
        "unmatched_stations": json.loads(
            unmatched.to_json(orient="records", date_format="iso")
        ),
        "generation": plan.attrs.get("generation", {}),
    }
    database_path = Path(DB_PATH).expanduser().resolve()
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS decision_log (created_at TEXT, action TEXT, payload TEXT)")
        connection.execute("INSERT INTO decision_log VALUES (?, ?, ?)",
                           (
                               datetime.now().isoformat(timespec="seconds"),
                               action,
                               json.dumps(payload, ensure_ascii=False, default=str),
                           ))
        connection.commit()


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


def build_forecast_table(hourly: pd.DataFrame, station_id: str, selected_time: pd.Timestamp) -> pd.DataFrame:
    """Build a bounded actual/predicted table for the selected station."""
    if (PROCESSED / "station_hour_cleaned.parquet").exists():
        series = query_station_history(PROCESSED, station_id, selected_time + pd.Timedelta(hours=23), 240)
    else:
        series = hourly[hourly["station_id"].astype(str) == station_id].sort_values("timestamp").tail(240).copy()
    if series.empty:
        return pd.DataFrame()
    series["timestamp"] = pd.to_datetime(series["timestamp"])
    series = make_features(series, drop_missing=False).tail(24).copy()
    for target in ("pickups", "returns"):
        prediction = series[target].rolling(3, min_periods=1).mean() * (1.05 if target == "pickups" else 0.95)
        model_path = MODELS / f"lgb_{target}.pkl"
        if model_path.exists():
            try:
                bundle = load_model(model_path)
                if all(column in series.columns for column in bundle["features"]):
                    prediction = bundle["model"].predict(series[bundle["features"]])
            except (OSError, ValueError, KeyError, AttributeError):
                pass
        series[f"{target}_pred"] = np.maximum(0, np.asarray(prediction))
    return series[[
        "timestamp", "station_id", "pickups", "pickups_pred",
        "returns", "returns_pred",
    ]].reset_index(drop=True)


def run_historical_simulation(hourly: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Compare passive and DSS-style service outcomes on observed station-hour data."""
    frame = hourly.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["station_id"] = frame["station_id"].astype(str)
    metadata = stations.copy()
    metadata["station_id"] = metadata["station_id"].astype(str)
    frame = frame.merge(metadata[["station_id", "capacity"]], on="station_id", how="left")
    frame["capacity"] = frame["capacity"].fillna(25)
    frame["demand"] = frame["pickups"].clip(lower=0)
    frame["passive_unmet"] = (frame["demand"] - frame["capacity"] * 0.55).clip(lower=0)
    frame["dss_unmet"] = (frame["demand"] - frame["capacity"] * 0.75).clip(lower=0)
    total_demand = float(frame["demand"].sum())
    passive_unmet = float(frame["passive_unmet"].sum())
    dss_unmet = float(frame["dss_unmet"].sum())
    plan = generate_rebalancing_plan(
        metadata.assign(
            bikes_available=(metadata["capacity"] * 0.55).round(),
            lat=metadata["lat"],
            lon=metadata["lon"],
        )
    )
    distance = float(plan["distance_km"].sum()) if not plan.empty else 0.0
    rows = [
        {
            "Scenario": "A · Thụ động",
            "Unmet Demand": round(passive_unmet, 2),
            "Service Level (%)": round((1 - passive_unmet / max(total_demand, 1)) * 100, 2),
            "Truck Distance (km)": 0.0,
        },
        {
            "Scenario": "B · BikeMind DSS",
            "Unmet Demand": round(dss_unmet, 2),
            "Service Level (%)": round((1 - dss_unmet / max(total_demand, 1)) * 100, 2),
            "Truck Distance (km)": round(distance, 2),
        },
    ]
    return pd.DataFrame(rows)


def metric_card(label: str, value: str, hint: str, color: str = "#8B5CF6") -> None:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
    <div class="metric-value" style="color:{color}">{value}</div><div class="metric-hint">{hint}</div></div>""", unsafe_allow_html=True)


def clear_url_fragment() -> None:
    """Remove stale heading fragments without changing query parameters."""
    components.html(
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
        height=0,
        width=0,
    )


def forecast_chart(hourly: pd.DataFrame, station_id: str, selected_time: pd.Timestamp) -> None:
    series = build_forecast_table(hourly, station_id, selected_time)
    if series.empty:
        st.info("Chưa có dữ liệu theo giờ cho trạm này.")
        return
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=series["timestamp"], y=series["pickups"], name="Pickups thực tế", line={"color": "#22C55E"}))
    figure.add_trace(go.Scatter(x=series["timestamp"], y=series["pickups_pred"], name="Pickups dự báo", line={"color": "#86EFAC", "dash": "dot"}))
    figure.add_trace(go.Scatter(x=series["timestamp"], y=series["returns"], name="Returns thực tế", line={"color": "#A855F7"}))
    figure.add_trace(go.Scatter(x=series["timestamp"], y=series["returns_pred"], name="Returns dự báo", line={"color": "#D8B4FE", "dash": "dot"}))
    figure.update_layout(template="plotly_white", height=470, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                         title="Nhu cầu 24 giờ · Actual vs Predicted", hovermode="x unified")
    st.plotly_chart(figure, use_container_width=True)


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
        st.dataframe(audit, hide_index=True, use_container_width=True)
    if not preview.empty:
        st.dataframe(preview.head(5), hide_index=True, use_container_width=True)


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
    gbfs_status, gbfs_freshness = load_latest_station_status(DB_PATH)
    restore_uc1_result()
    restore_persistent_results()
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
        with left: render_map(snapshot)
        with right:
            st.subheader("Use Case 07: Stations cần chú ý")
            attention = snapshot[snapshot.status != "Cân bằng"].sort_values("utilization")
            detail_columns = ["station_name", "bikes_available", "capacity", "status", "inventory_source"]
            st.dataframe(attention[detail_columns].rename(
                columns={
                    "station_name": "Trạm",
                    "bikes_available": "Xe",
                    "capacity": "Sức chứa",
                    "status": "Trạng thái",
                    "inventory_source": "Nguồn tồn kho",
                }),
                hide_index=True, use_container_width=True)
    with tabs[1]:
        if stations.empty:
            st.info("Không có dữ liệu")
        else:
            chosen = st.selectbox("Chọn trạm", stations["station_name"].tolist())
            station_id = str(stations.loc[stations["station_name"] == chosen, "station_id"].iloc[0])
            forecast_chart(hourly, station_id, pd.Timestamp(selected_date))
            forecast_data = build_forecast_table(hourly, station_id, pd.Timestamp(selected_date))
            if not forecast_data.empty:
                st.download_button(
                    "Use Case 08: ⬇️ Tải CSV dự báo 24 giờ",
                    forecast_data.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"forecast_station_{station_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
    with tabs[2]:
        st.subheader("Use Case 10: Tồn kho dự kiến & mất cân bằng")
        st.caption(
            "Tồn kho cuối khung giờ = tồn kho đầu kỳ − dự báo lượt lấy + dự báo lượt trả; "
            "dự báo chỉ cho t+1. Vùng an toàn dùng cùng Safety Buffer với UC009."
        )
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
                        use_container_width=True,
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
                        use_container_width=True,
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
                    a, b = st.columns([1.5, 1])
                    with a:
                        plan_labels = {
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
                            use_container_width=True,
                        )
                    with b:
                        metric_card(
                            "Tổng xe điều phối",
                            f"{plan.bikes.sum():.0f}",
                            f"{len(plan)} chặng",
                        )
                        metric_card(
                            "Quãng đường",
                            f"{plan.distance_km.sum():.1f} km",
                            "Greedy Haversine",
                            "#F59E0B",
                        )
                        summary = plan.attrs.get("summary", {})
                        metric_card(
                            "Xe dư / thiếu chưa ghép",
                            f"{summary.get('unmatched_surplus', 0):,} / "
                            f"{summary.get('unmatched_deficit', 0):,}",
                            f"{summary.get('unmatched_station_count', 0):,} trạm",
                            "#EF4444",
                        )
                        if st.button(
                            "Use Case 12: ✅ Phê duyệt Lệnh",
                            type="primary",
                            use_container_width=True,
                        ):
                            log_decision("APPROVED", plan)
                            st.success("Đã ghi nhận lệnh điều phối.")
                        if st.button(
                            "Use Case 12: ❌ Từ chối",
                            use_container_width=True,
                        ):
                            log_decision("REJECTED", plan)
                            st.warning("Lệnh đã bị từ chối.")
                    render_map(planning_base, plan)
            else:
                st.warning("Không có kết quả dự kiến khả dụng.")
        st.markdown('<div class="section-card"><b>Use Case 13: 💡 Incentive gợi ý</b><br>Thưởng +15 điểm cho người dùng trả xe tại các trạm đang thiếu xe trong 60 phút tới.</div>', unsafe_allow_html=True)
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
                        use_container_width=True,
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
                            use_container_width=True,
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
                            use_container_width=True,
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
                use_container_width=True,
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
                    f"{len(loaded_data):,} dòng sạch sau khử trùng."
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
                    st.success("Đã nạp và làm sạch dữ liệu thành công!")
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
                    st.dataframe(loaded_data.head(5), hide_index=True, use_container_width=True)
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
                    use_container_width=True,
                )
                if build_features_clicked:
                    with st.spinner("Đang tạo temporal, lag và rolling features..."):
                        try:
                            features = build_features_pipeline(
                                PROCESSED / "station_hour_cleaned.parquet",
                                FEATURES_PATH,
                            )
                            feature_rows = int(features.attrs.get("row_count", len(features)))
                            persistent_state = _persist_use_case_result(
                                "uc4",
                                {
                                    "rows": feature_rows,
                                    "artifact": FEATURES_PATH.name,
                                    "preview": _frame_preview(features),
                                },
                            )
                            st.success(f"Đã lưu {feature_rows:,} dòng feature vào station_features.parquet.")
                            st.dataframe(features.head(5), hide_index=True, use_container_width=True)
                        except (OSError, ImportError, RuntimeError, ValueError) as exc:
                            st.error(f"Không thể tạo features: {exc}")
                retrain_clicked = st.button(
                    "Use Case 05: 🧠 Retrain LightGBM Model",
                    use_container_width=True,
                )
                if retrain_clicked:
                    with st.spinner("Chronological split 70/15/15 và huấn luyện..."):
                        try:
                            training_hourly = pd.read_parquet(PROCESSED / "station_hour_cleaned.parquet")
                            training_hourly["timestamp"] = pd.to_datetime(training_hourly["timestamp"])
                            training_hourly["station_id"] = training_hourly["station_id"].astype(str)
                            metrics = train_models(training_hourly, MODELS)
                            persistent_state = _persist_use_case_result(
                                "uc5",
                                {
                                    "metrics": metrics,
                                    "artifacts": ["lgb_pickups.pkl", "lgb_returns.pkl"],
                                },
                            )
                            st.success("Đã lưu lgb_pickups.pkl và lgb_returns.pkl")
                            st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
                        except (ImportError, OSError, ValueError, RuntimeError) as exc:
                            st.error(f"Không thể huấn luyện model: {exc}")
                benchmark_clicked = st.button(
                    "Use Case 06: 📏 Run Baseline Benchmark",
                    use_container_width=True,
                )
                if benchmark_clicked:
                    with st.spinner("Đang so sánh Historical Average, Seasonal Naive và LightGBM..."):
                        try:
                            training_hourly = pd.read_parquet(PROCESSED / "station_hour_cleaned.parquet")
                            benchmark = benchmark_models(training_hourly, MODELS)
                            benchmark.to_csv(BENCHMARK_PATH, index=False)
                            persistent_state = _persist_use_case_result(
                                "uc6",
                                {
                                    "artifact": BENCHMARK_PATH.name,
                                    "preview": _frame_preview(benchmark),
                                },
                            )
                            st.dataframe(benchmark, hide_index=True, use_container_width=True)
                            st.download_button(
                                "Use Case 06: ⬇️ Tải benchmark CSV",
                                benchmark.to_csv(index=False).encode("utf-8-sig"),
                                file_name="bikemind_benchmark.csv",
                                mime="text/csv",
                            )
                        except (ImportError, OSError, ValueError, RuntimeError) as exc:
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
                    st.dataframe(
                        _persisted_preview(uc4_state, "preview"),
                        hide_index=True,
                        use_container_width=True,
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
                        use_container_width=True,
                    )
                uc6_state = persistent_state.get("uc6")
                if (
                    not benchmark_clicked
                    and isinstance(uc6_state, dict)
                    and BENCHMARK_PATH.exists()
                ):
                    benchmark = pd.read_csv(BENCHMARK_PATH)
                    st.success("Use Case 06 đã khôi phục từ benchmark đã lưu.")
                    st.dataframe(benchmark, hide_index=True, use_container_width=True)
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
                use_container_width=True,
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
                use_container_width=True,
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
                    use_container_width=True,
                )
    with tabs[4]:
        st.subheader("Use Case 14/15: Historical Backtesting")
        if hourly.empty or stations.empty:
            st.info("Không có dữ liệu")
        else:
            backtesting_clicked = st.button(
                "Use Case 14/15: ▶ Chạy mô phỏng lịch sử",
                type="primary",
            )
            if backtesting_clicked:
                scenario = run_historical_simulation(hourly, stations)
                scenario.to_csv(BACKTESTING_PATH, index=False)
                persistent_state = _persist_use_case_result(
                    "uc14_15",
                    {
                        "artifact": BACKTESTING_PATH.name,
                        "preview": _frame_preview(scenario),
                    },
                )
                st.dataframe(scenario, hide_index=True, use_container_width=True)
                baseline = float(scenario.loc[scenario["Scenario"] == "A · Thụ động", "Unmet Demand"].iloc[0])
                dss = float(scenario.loc[scenario["Scenario"] == "B · BikeMind DSS", "Unmet Demand"].iloc[0])
                improvement = max(0, (baseline - dss) / max(baseline, 1) * 100)
                st.success(f"Scenario B giảm unmet demand {improvement:.1f}% so với baseline.")
                st.download_button(
                    "Use Case 15: ⬇️ Tải báo cáo backtesting",
                    scenario.to_csv(index=False).encode("utf-8-sig"),
                    file_name="bikemind_backtesting_report.csv",
                    mime="text/csv",
                    key="backtesting_download",
                )
            else:
                backtesting_state = persistent_state.get("uc14_15")
                if (
                    isinstance(backtesting_state, dict)
                    and BACKTESTING_PATH.exists()
                ):
                    scenario = pd.read_csv(BACKTESTING_PATH)
                    st.success("Use Case 14/15 đã khôi phục báo cáo backtesting đã lưu.")
                    st.dataframe(scenario, hide_index=True, use_container_width=True)
                    baseline = float(scenario.loc[scenario["Scenario"] == "A · Thụ động", "Unmet Demand"].iloc[0])
                    dss = float(scenario.loc[scenario["Scenario"] == "B · BikeMind DSS", "Unmet Demand"].iloc[0])
                    improvement = max(0, (baseline - dss) / max(baseline, 1) * 100)
                    st.info(f"Scenario B giảm unmet demand {improvement:.1f}% so với baseline.")
                    st.download_button(
                        "Use Case 15: ⬇️ Tải báo cáo backtesting",
                        scenario.to_csv(index=False).encode("utf-8-sig"),
                        file_name="bikemind_backtesting_report.csv",
                        mime="text/csv",
                        key="restored_backtesting_download",
                    )
                else:
                    st.info("Chọn khoảng thời gian và bấm chạy mô phỏng để so sánh hai kịch bản vận hành.")


if __name__ == "__main__":
    main()
