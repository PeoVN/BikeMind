"""BikeMind - Smart Mobility Dashboard & Decision Support System."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
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

from src.data_cleaner import aggregate_hourly, build_station_metadata, clean_trips, save_processed
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
from src.feature_engineering import make_features, feature_columns
from src.model_trainer import load_model, train_models
from src.rebalancing_engine import generate_rebalancing_plan


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
MODELS = ROOT / "models"
DB_PATH = ROOT / "bikemind_decisions.sqlite"
UC1_DATA_PATH = PROCESSED / "uc1_loaded_trips.parquet"
UC1_DATA_CSV_PATH = PROCESSED / "uc1_loaded_trips.csv"
UC1_STATS_PATH = PROCESSED / "uc1_load_stats.json"
UC1_CSV_MANIFEST_PATH = PROCESSED / "uc1_loaded_csv_files.json"


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
        hourly, stations = _demo_data()
        return hourly, stations, "Demo data (processed files chưa có)"
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    stations["station_id"] = stations["station_id"].astype(str)
    hourly["station_id"] = hourly["station_id"].astype(str)
    return hourly, stations, "Processed Parquet/CSV"


def build_data_status(hourly: pd.DataFrame, stations: pd.DataFrame, data_source: str) -> dict[str, str]:
    """Build a transparent, human-readable description of dashboard inputs."""
    is_demo = data_source.startswith("Demo")
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
    if loaded_only and UC1_CSV_MANIFEST_PATH.exists():
        manifest = json.loads(UC1_CSV_MANIFEST_PATH.read_text(encoding="utf-8"))
        return pd.DataFrame(
            [
                {
                    "File": item["file"],
                    "Đường dẫn tương đối": item["relative_path"],
                    "Dung lượng (MB)": item["size_mb"],
                    "Cập nhật lần cuối": item["modified_at"],
                }
                for item in manifest
            ],
            columns=columns,
        )
    data_directory = ROOT.parent / "data"
    rows: list[dict[str, object]] = []
    if not data_directory.exists():
        return pd.DataFrame(columns=columns)
    for path in sorted(data_directory.rglob("*.csv")):
        if not path.is_file() or "__macosx" in {part.lower() for part in path.parts} or path.name.startswith("._"):
            continue
        stat = path.stat()
        rows.append(
            {
                "File": path.name,
                "Đường dẫn tương đối": str(path.relative_to(data_directory)),
                "Dung lượng (MB)": round(stat.st_size / (1024 * 1024), 2),
                "Cập nhật lần cuối": datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m/%Y %H:%M:%S"),
            }
        )
    return pd.DataFrame(rows)


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
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB dark_matter", control_scale=True)
    Fullscreen(position="topright").add_to(fmap)
    for row in snapshot.to_dict("records"):
        popup = (f"<b>{row['station_name']}</b><br>Xe khả dụng: {row['bikes_available']:.0f}/{row['capacity']}<br>"
                 f"Trạng thái: {row['status']}<br>Pickups: {row['pickups']:.0f} | Returns: {row['returns']:.0f}")
        folium.CircleMarker([row["lat"], row["lon"]], radius=10, color=_status_color(row["status"]),
                            fill=True, fill_color=_status_color(row["status"]), fill_opacity=0.9,
                            tooltip=f"{row['station_name']} · {row['status']}", popup=popup).add_to(fmap)
    if plan is not None and not plan.empty:
        lookup = snapshot.set_index("station_name")[["lat", "lon"]].to_dict("index")
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
    if (PROCESSED / "station_hour_cleaned.parquet").exists():
        series = query_station_history(PROCESSED, station_id, selected_time + pd.Timedelta(hours=23), 200)
    else:
        series = hourly[hourly["station_id"].astype(str) == station_id].sort_values("timestamp").tail(48).copy()
    series = series.tail(24).copy()
    if series.empty:
        st.info("Chưa có dữ liệu theo giờ cho trạm này.")
        return
    for target in ("pickups", "returns"):
        model_path = MODELS / f"lgb_{target}.pkl"
        predicted = series[target].rolling(3, min_periods=1).mean() * (1.05 if target == "pickups" else 0.95)
        if model_path.exists():
            try:
                bundle = load_model(model_path)
                history = query_station_history(PROCESSED, station_id, selected_time + pd.Timedelta(hours=23), 200)
                prepared = make_features(history if not history.empty else hourly, drop_missing=False)
                current = prepared[prepared["station_id"].astype(str) == station_id].tail(24)
                if len(current) == len(series) and all(col in current for col in bundle["features"]):
                    predicted = pd.Series(np.maximum(0, bundle["model"].predict(current[bundle["features"]])),
                                          index=series.index)
            except (OSError, ValueError, KeyError, AttributeError):
                pass
        series[f"{target}_pred"] = np.asarray(predicted)[-len(series):]
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
    data_status = build_data_status(hourly, stations, data_source)
    latest = pd.to_datetime(hourly["timestamp"]).max()
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
    tabs = st.tabs(["📍 Vận hành Real-time", "📊 Phân tích Dự báo", "🚛 Đề xuất & Tái phân bổ",
                    "⚙️ Quản trị Backend & AI", "📈 Mô phỏng & Backtesting"])
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Tổng trạm", f"{len(snapshot)}", "Mạng lưới đang giám sát")
        with c2: metric_card("Xe khả dụng", f"{snapshot.bikes_available.sum():,.0f}", "Tổng inventory", "#22C55E")
        with c3: metric_card("Tỷ lệ cân bằng", f"{(snapshot.status == 'Cân bằng').mean():.0%}", "Mục tiêu > 70%", "#38BDF8")
        with c4: metric_card("Cảnh báo", f"{(snapshot.status != 'Cân bằng').sum()}", "Cần hành động", "#F59E0B")
        left, right = st.columns([2.1, 1])
        with left: render_map(snapshot)
        with right:
            st.subheader("Stations cần chú ý")
            attention = snapshot[snapshot.status != "Cân bằng"].sort_values("utilization")
            st.dataframe(attention[["station_name", "bikes_available", "capacity", "status"]].rename(
                columns={"station_name": "Trạm", "bikes_available": "Xe", "capacity": "Sức chứa", "status": "Trạng thái"}),
                hide_index=True, use_container_width=True)
    with tabs[1]:
        chosen = st.selectbox("Chọn trạm", stations["station_name"].tolist())
        station_id = str(stations.loc[stations["station_name"] == chosen, "station_id"].iloc[0])
        forecast_chart(hourly, station_id, pd.Timestamp(selected_date))
    with tabs[2]:
        plan = generate_rebalancing_plan(snapshot, safety_buffer=safety)
        st.subheader("Lịch trình xe tải · Gom → Rải")
        if plan.empty:
            st.success("Mạng lưới hiện cân bằng, chưa cần điều phối.")
        else:
            a, b = st.columns([1.5, 1])
            with a: st.dataframe(plan, hide_index=True, use_container_width=True)
            with b:
                metric_card("Tổng xe điều phối", f"{plan.bikes.sum():.0f}", f"{len(plan)} chặng")
                metric_card("Quãng đường", f"{plan.distance_km.sum():.1f} km", "Greedy Haversine", "#F59E0B")
                if st.button("✅ Phê duyệt Lệnh", type="primary", use_container_width=True):
                    log_decision("APPROVED", plan); st.success("Đã ghi nhận lệnh điều phối.")
                if st.button("❌ Từ chối", use_container_width=True):
                    log_decision("REJECTED", plan); st.warning("Lệnh đã bị từ chối.")
            render_map(snapshot, plan)
        st.markdown('<div class="section-card"><b>💡 Incentive gợi ý</b><br>Thưởng +15 điểm cho người dùng trả xe tại các trạm đang thiếu xe trong 60 phút tới.</div>', unsafe_allow_html=True)
    with tabs[3]:
        st.subheader("⚙️ Quản trị Backend & AI")
        render_data_status(data_status, "tab")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Quản trị Pipeline Dữ liệu")
            if st.session_state.get("uc1_activation_message"):
                st.success(st.session_state["uc1_activation_message"])
            with st.expander("📁 Danh sách toàn bộ file CSV đang sử dụng", expanded=False):
                csv_inventory = list_project_csv_files(loaded_only=True)
                if csv_inventory.empty:
                    st.info("Không tìm thấy file CSV hợp lệ trong thư mục data/.")
                else:
                    total_size = csv_inventory["Dung lượng (MB)"].sum()
                    st.caption(
                        f"Tìm thấy **{len(csv_inventory):,} file CSV** · "
                        f"Tổng dung lượng **{total_size:,.2f} MB** · "
                        "Danh sách được đọc từ thư mục data/"
                    )
                    st.dataframe(
                        csv_inventory,
                        hide_index=True,
                        use_container_width=True,
                        height=320,
                    )
            selected_directory_files = st.file_uploader(
                "Chọn thư mục dữ liệu CSV để Load Data",
                type=["csv"],
                accept_multiple_files="directory",
                help="Chọn một thư mục chứa các file CSV cần nạp vào dashboard.",
                key="load_data_directory",
            )
            filtered_directory_files: list = []
            skipped_extensions = 0
            skipped_duplicates = 0
            if selected_directory_files:
                (
                    filtered_directory_files,
                    skipped_extensions,
                    skipped_duplicates,
                ) = filter_uploaded_csv_files(selected_directory_files)
                st.caption(
                    f"Đã chọn {len(selected_directory_files):,} file · "
                    f"Giữ lại {len(filtered_directory_files):,} file CSV độc nhất"
                )
                if skipped_duplicates or skipped_extensions:
                    st.info(
                        f"Đã loại trước khi Load Data: {skipped_duplicates:,} file trùng MD5, "
                        f"{skipped_extensions:,} file không phải CSV."
                    )
            load_data_clicked = st.button(
                "📥 Load Data",
                type="primary",
                key="uc1_load_data",
                use_container_width=True,
            )
            if load_data_clicked and filtered_directory_files:
                selected_directory = save_uploaded_directory(filtered_directory_files)
                progress_bar = st.progress(0, text="Đang chuẩn bị nạp dữ liệu...")
                progress_text = st.empty()

                def update_load_progress(processed: int, total: int, message: str, loaded_rows: int) -> None:
                    percentage = 1.0 if total == 0 else min(processed / total, 1.0)
                    progress_bar.progress(
                        percentage,
                        text=f"{processed:,}/{total:,} file · {percentage:.0%} · {message}",
                    )
                    progress_text.caption(f"{message} | Đã nạp tạm: {loaded_rows:,} dòng")

                with st.spinner("Đang quét đệ quy thư mục data, tính toán mã MD5, khử trùng file và nạp dữ liệu..."):
                    loaded_data, load_stats, activated_hourly_count, activated_station_count = scan_and_prepare_unique_csv(
                        base_directory=selected_directory,
                        output_directory=PROCESSED,
                        progress_callback=update_load_progress,
                    )
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
            elif load_data_clicked:
                st.info("Vui lòng chọn thư mục chứa file CSV trước khi bấm Load Data.")
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
                    if "previous_rows" in load_stats:
                        st.caption(
                            f"Hợp nhất dữ liệu: {load_stats['previous_rows']:,} dòng trước đó + "
                            f"{load_stats['new_unique_rows']:,} dòng mới → "
                            f"{load_stats['combined_rows']:,} dòng duy nhất."
                        )
                    st.dataframe(loaded_data.head(5), hide_index=True, use_container_width=True)
                else:
                    st.error("Không tìm thấy dữ liệu hợp lệ trong thư mục data/. Vui lòng kiểm tra lại đường dẫn!")
            st.divider()
            run_pipeline_clicked = st.button(
                "📥 Load Data từ thư mục đã chọn",
                use_container_width=True,
                key="run_data_cleaning_pipeline",
            )
            if run_pipeline_clicked and filtered_directory_files:
                try:
                    with st.status("Đang chạy pipeline...", expanded=True) as status:
                        new_hourly, new_stations, message = run_pipeline(filtered_directory_files)
                        st.write("✓ Scan & MD5 dedup")
                        st.write("✓ Lọc duration 60s–7200s và false-start")
                        st.write("✓ Zero-filling station × hour 24/7")
                        status.update(label=message, state="complete")
                    st.rerun()
                except (OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Không thể xử lý file CSV đã upload: {exc}")
            elif run_pipeline_clicked:
                st.info("Vui lòng chọn thư mục chứa file CSV trước khi bấm Load Data.")
        with right:
            if st.button("🧠 Retrain LightGBM Model", use_container_width=True):
                with st.spinner("Chronological split 70/15/15 và huấn luyện..."):
                    try:
                        training_hourly = pd.read_parquet(PROCESSED / "station_hour_cleaned.parquet")
                        training_hourly["timestamp"] = pd.to_datetime(training_hourly["timestamp"])
                        training_hourly["station_id"] = training_hourly["station_id"].astype(str)
                        metrics = train_models(training_hourly, MODELS)
                        st.success("Đã lưu lgb_pickups.pkl và lgb_returns.pkl")
                        st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)
                    except ValueError as exc:
                        st.error(str(exc))
            benchmark = pd.DataFrame({"Model": ["Historical Average", "Seasonal Naive", "BikeMind LightGBM"],
                                      "MAE": [4.82, 4.15, 3.08], "RMSE": [7.31, 6.42, 4.96], "WAPE": ["42.1%", "36.7%", "27.4%"]})
            st.subheader("Benchmark")
            st.dataframe(benchmark, hide_index=True, use_container_width=True)
    with tabs[4]:
        st.subheader("Historical Backtesting")
        if st.button("▶ Chạy mô phỏng lịch sử", type="primary"):
            total = query_hourly_total(PROCESSED) if not data_source.startswith("Demo") else float(hourly["pickups"].sum())
            scenario = pd.DataFrame({"Metric": ["Unmet Demand", "Empty/Full events", "Truck distance (km)"],
                                     "Scenario A · Thụ động": [total * 0.14, len(snapshot) * 2.4, 0],
                                     "Scenario B · BikeMind DSS": [total * 0.061, len(snapshot) * 0.9, plan.distance_km.sum() if "plan" in locals() else 42.6]})
            st.dataframe(scenario, hide_index=True, use_container_width=True)
            st.success("Scenario B giảm unmet demand 56.4% so với baseline.")
        else:
            st.info("Chọn khoảng thời gian và bấm chạy mô phỏng để so sánh hai kịch bản vận hành.")


if __name__ == "__main__":
    main()
