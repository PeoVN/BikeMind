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
from src.gbfs_collector import run_gbfs_collector_job
from src.model_trainer import benchmark_models, load_model, train_models
from src.rebalancing_engine import generate_rebalancing_plan, run_greedy_rebalancing


ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = ROOT.parent / "data"
DATA_LOAD_DIRECTORY = ROOT.parent / "data_load"
PROCESSED = ROOT / "processed"
MODELS = ROOT / "models"
DB_PATH = ROOT / "bikemind_decisions.sqlite"
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


def _station_snapshot(hourly: pd.DataFrame, stations: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    if stations.empty:
        return pd.DataFrame(
            columns=[
                "station_id", "station_name", "lat", "lon", "capacity",
                "pickups", "returns", "bikes_available", "utilization", "status",
            ]
        )
    latest = hourly.copy()
    if "timestamp" not in latest or "station_id" not in latest:
        latest = pd.DataFrame(columns=["station_id", "pickups", "returns", "timestamp"])
    latest["station_id"] = (
        latest["station_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    )
    latest = latest[latest["timestamp"] <= timestamp].sort_values("timestamp").groupby("station_id").tail(1)
    result = stations.copy()
    result["station_id"] = (
        result["station_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    )
    result = result.merge(latest[["station_id", "pickups", "returns"]], on="station_id", how="left").fillna(0)
    result["bikes_available"] = (result["capacity"] * 0.55 + result["returns"] - result["pickups"]).clip(0, result["capacity"])
    result["utilization"] = result["bikes_available"] / result["capacity"]
    result["status"] = np.select([result["utilization"] < 0.25, result["utilization"] > 0.8],
                                  ["Thiếu xe", "Đầy xe"], default="Cân bằng")
    return result


def _query_station_snapshot(timestamp: pd.Timestamp, stations: pd.DataFrame) -> pd.DataFrame:
    latest = query_station_snapshot(PROCESSED, timestamp)
    if latest.empty:
        return _station_snapshot(pd.DataFrame(), stations, timestamp)
    return _station_snapshot(latest, stations, timestamp)


def _status_color(status: str) -> str:
    return {"Thiếu xe": "#EF4444", "Đầy xe": "#A855F7", "Cân bằng": "#22C55E"}.get(status, "#94A3B8")


def render_map(snapshot: pd.DataFrame, plan: pd.DataFrame | None = None, height: int = 570) -> None:
    center = [snapshot["lat"].mean(), snapshot["lon"].mean()] if not snapshot.empty else [40.74, -73.99]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron", control_scale=True)
    Fullscreen(position="topright").add_to(fmap)
    for row in snapshot.to_dict("records"):
        popup = (f"<b>{row['station_name']}</b><br>Xe khả dụng: {row['bikes_available']:.0f}/{row['capacity']}<br>"
                 f"Trạng thái: {row['status']}<br>Pickups: {row['pickups']:.0f} | Returns: {row['returns']:.0f}")
        folium.CircleMarker([row["lat"], row["lon"]], radius=10, color=_status_color(row["status"]),
                            fill=True, fill_color=_status_color(row["status"]), fill_opacity=0.9,
                            tooltip=f"{row['station_name']} · {row['status']}", popup=popup).add_to(fmap)
    if plan is not None and not plan.empty:
        lookup = (
            snapshot
            .drop_duplicates(subset=["station_name"], keep="last")
            .set_index("station_name")[["lat", "lon"]]
            .to_dict("index")
        )
        for row in plan.to_dict("records"):
            if row["from_station"] in lookup and row["to_station"] in lookup:
                folium.PolyLine([(lookup[row["from_station"]]["lat"], lookup[row["from_station"]]["lon"]),
                                 (lookup[row["to_station"]]["lat"], lookup[row["to_station"]]["lon"])],
                                color="#F59E0B", weight=4, opacity=0.9,
                                tooltip=f"{row['bikes']} xe · {row['distance_km']} km").add_to(fmap)
    st_folium(fmap, height=height, use_container_width=True, returned_objects=[])


def log_decision(action: str, plan: pd.DataFrame) -> None:
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS decision_log (created_at TEXT, action TEXT, payload TEXT)")
        connection.execute("INSERT INTO decision_log VALUES (?, ?, ?)",
                           (datetime.now().isoformat(timespec="seconds"), action, plan.to_json(orient="records")))
        connection.commit()


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
            snapshot = _station_snapshot(hourly, stations, snapshot_time)
        else:
            snapshot = _query_station_snapshot(snapshot_time, stations)
        st.divider()
        st.caption(f"📦 Nguồn dữ liệu: {data_source}")
        st.metric("🔴 Thiếu xe", int((snapshot.status == "Thiếu xe").sum()))
        st.metric("🟣 Đầy xe", int((snapshot.status == "Đầy xe").sum()))
        st.metric("🟢 Cân bằng", int((snapshot.status == "Cân bằng").sum()))
    st.title("BikeMind Operations Center")
    st.caption(f"Live decision support · Cập nhật {latest:%d/%m/%Y %H:%M} · Safety buffer {safety:.0%}")
    tabs = st.tabs([
        "Use Case 07/09: 📍 Vận hành Real-time",
        "Use Case 08: 📊 Phân tích Dự báo",
        "Use Case 10-13: 🚛 Đề xuất & Tái phân bổ",
        "Use Case 01-06: ⚙️ Quản trị Backend & AI",
        "Use Case 14/15: 📈 Mô phỏng & Backtesting",
    ])
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Tổng trạm", f"{len(snapshot)}", "Mạng lưới đang giám sát")
        with c2: metric_card("Xe khả dụng", f"{snapshot.bikes_available.sum():,.0f}", "Tổng inventory", "#22C55E")
        balance_rate = 0 if snapshot.empty else (snapshot.status == "Cân bằng").mean()
        with c3: metric_card("Tỷ lệ cân bằng", f"{balance_rate:.0%}", "Mục tiêu > 70%", "#38BDF8")
        with c4: metric_card("Cảnh báo", f"{(snapshot.status != 'Cân bằng').sum()}", "Cần hành động", "#F59E0B")
        left, right = st.columns([2.1, 1])
        with left: render_map(snapshot)
        with right:
            st.subheader("Use Case 07: Stations cần chú ý")
            attention = snapshot[snapshot.status != "Cân bằng"].sort_values("utilization")
            st.dataframe(attention[["station_name", "bikes_available", "capacity", "status"]].rename(
                columns={"station_name": "Trạm", "bikes_available": "Xe", "capacity": "Sức chứa", "status": "Trạng thái"}),
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
        planning_snapshot = snapshot.copy()
        if not planning_snapshot.empty:
            planning_snapshot["forecast_pickups"] = planning_snapshot["pickups"]
            planning_snapshot["forecast_returns"] = planning_snapshot["returns"]
        plan = run_greedy_rebalancing(planning_snapshot, buffer_pct=safety) if not planning_snapshot.empty else pd.DataFrame()
        st.subheader("Use Case 10/11: Lịch trình xe tải · Gom → Rải")
        if snapshot.empty:
            st.info("Không có dữ liệu")
        elif plan.empty:
            st.success("Mạng lưới hiện cân bằng, chưa cần điều phối.")
        else:
            a, b = st.columns([1.5, 1])
            with a: st.dataframe(plan, hide_index=True, use_container_width=True)
            with b:
                metric_card("Tổng xe điều phối", f"{plan.bikes.sum():.0f}", f"{len(plan)} chặng")
                metric_card("Quãng đường", f"{plan.distance_km.sum():.1f} km", "Greedy Haversine", "#F59E0B")
                if st.button("Use Case 12: ✅ Phê duyệt Lệnh", type="primary", use_container_width=True):
                    log_decision("APPROVED", plan); st.success("Đã ghi nhận lệnh điều phối.")
                if st.button("Use Case 12: ❌ Từ chối", use_container_width=True):
                    log_decision("REJECTED", plan); st.warning("Lệnh đã bị từ chối.")
            render_map(snapshot, plan)
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
                                "stats": {
                                    key: int(value)
                                    for key, value in cleaning_stats.items()
                                },
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
                        stat_columns = st.columns(3)
                        stat_columns[0].metric(
                            "Tổng số trạm đã chuẩn hóa",
                            f"{cleaning_stats['station_count']:,}",
                        )
                        stat_columns[1].metric(
                            "Bản ghi Station-Hour",
                            f"{cleaning_stats['station_hour_rows']:,}",
                        )
                        stat_columns[2].metric(
                            "Bản ghi bị loại",
                            f"{cleaning_stats['removed_rows']:,}",
                        )
                        st.dataframe(
                            cleaned_hourly.head(5),
                            hide_index=True,
                            use_container_width=True,
                        )
            uc2_state = persistent_state.get("uc2")
            if (
                not clean_data_clicked
                and isinstance(uc2_state, dict)
                and (PROCESSED / "station_hour_cleaned.parquet").exists()
                and (PROCESSED / "station_metadata.parquet").exists()
            ):
                stats = uc2_state.get("stats", {})
                st.success("Use Case 02 đã khôi phục từ dữ liệu đã xử lý.")
                stat_columns = st.columns(3)
                stat_columns[0].metric(
                    "Tổng số trạm đã chuẩn hóa",
                    f"{int(stats.get('station_count', 0)):,}",
                )
                stat_columns[1].metric(
                    "Bản ghi Station-Hour",
                    f"{int(stats.get('station_hour_rows', 0)):,}",
                )
                stat_columns[2].metric(
                    "Bản ghi bị loại",
                    f"{int(stats.get('removed_rows', 0)):,}",
                )
                st.dataframe(
                    _persisted_preview(uc2_state, "preview"),
                    hide_index=True,
                    use_container_width=True,
                )
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
                    collector_result = run_gbfs_collector_job(DB_PATH)
                if collector_result["success"]:
                    snapshot_data = collector_result["status_df"]
                    persistent_state = _persist_use_case_result(
                        "uc3",
                        {
                            "records_count": int(collector_result["records_count"]),
                            "timestamp": collector_result["timestamp"],
                            "preview": _frame_preview(snapshot_data),
                            "artifact": DB_PATH.name,
                        },
                    )
                    st.success(
                        "Đã thu thập và lưu trữ snapshot trạng thái trạm vào SQLite thành công!"
                    )
                    metric_columns = st.columns(2)
                    metric_columns[0].metric(
                        "Tổng số trạm đã cập nhật",
                        f"{collector_result['records_count']:,}",
                    )
                    metric_columns[1].metric(
                        "Thời điểm ghi nhận",
                        str(collector_result["timestamp"]),
                    )
                    st.dataframe(
                        snapshot_data.head(10),
                        hide_index=True,
                        use_container_width=True,
                    )
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
                    "Use Case 03 đã khôi phục snapshot GBFS gần nhất: "
                    f"{int(uc3_state.get('records_count', 0)):,} trạm · "
                    f"{uc3_state.get('timestamp', 'Không rõ thời điểm')}."
                )
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
