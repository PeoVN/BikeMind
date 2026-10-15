from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.rebalancing import assess_risk, greedy_matching
from pipeline.service import train_and_persist


st.set_page_config(
    page_title="BikeMind Operations",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEMES = {
    "Light mode": {
        "bg": "#F4F6FA",
        "card": "#FFFFFF",
        "border": "#E2E6EF",
        "muted": "#667085",
        "text": "#182230",
        "sidebar": "#FFFFFF",
        "map": "carto-positron",
    },
    "Dark mode": {
        "bg": "#0D1018",
        "card": "#171B27",
        "border": "#283044",
        "muted": "#8D96AA",
        "text": "#F4F7FB",
        "sidebar": "#111521",
        "map": "carto-darkmatter",
    },
}
dark_mode = st.sidebar.toggle(
    "Dark mode",
    value=False,
    help="Tắt: Light mode (mặc định) · Bật: Dark mode",
    key="dashboard_dark_mode",
)
theme_mode = "Dark mode" if dark_mode else "Light mode"
COLORS = {
    **THEMES[theme_mode],
    "blue": "#3B82F6",
    "red": "#EF4444",
    "purple": "#A855F7",
    "green": "#22C55E",
    "amber": "#F59E0B",
}
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    :root {{ color-scheme: {"dark" if dark_mode else "light"}; }}
    html, body, [class*="css"] {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
    .stApp {{ background: {COLORS["bg"]}; }}
    [data-testid="stHeader"] {{ background: transparent !important; height: 0 !important; min-height: 0 !important; }}
    [data-testid="stToolbar"], .stAppDeployButton {{ display: none !important; }}
    [data-testid="stSidebar"] {{ visibility: visible !important; display: block !important; transform: none !important; width: 270px !important; min-width: 270px !important; }}
    [data-testid="stSidebar"] > div {{ visibility: visible !important; width: 270px !important; }}
    [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] {{ display: none !important; visibility: hidden !important; }}
    [data-testid="stAppViewContainer"] > .main {{ padding-top: 0 !important; }}
    [data-testid="stSidebar"] {{ background: {COLORS["sidebar"]}; border-right: 1px solid {COLORS["border"]}; min-width: 270px; box-shadow: 12px 0 35px rgba(0,0,0,.12); }}
    [data-testid="stSidebar"] > div {{ width: 270px; }}
    [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.4rem; }}
    .block-container {{ padding: 1.5rem 2rem 2.5rem; max-width: 1600px; }}
    h1, h2, h3 {{ letter-spacing: -0.03em; }}
    h1 {{ font-size: 2rem !important; }}
    h2 {{ font-size: 1.3rem !important; }}
    h3 {{ font-size: 1rem !important; }}
    .brand {{ display: flex; align-items: center; gap: .75rem; margin: 0 0 1rem; padding: .8rem .7rem; border: 1px solid {COLORS["border"]}; border-radius: 14px; background: linear-gradient(135deg, rgba(59,130,246,.14), rgba(124,58,237,.12)); }}
    .brand-mark {{ width: 42px; height: 42px; display: grid; place-items: center; border-radius: 13px;
      background: linear-gradient(135deg, #2563EB, #7C3AED); box-shadow: 0 8px 22px #3b82f655; flex: 0 0 auto; }}
    .brand-mark svg {{ width: 28px; height: 28px; }}
    .content-header {{ display: flex; align-items: center; gap: 1.25rem; margin-bottom: 1.25rem; }}
    .content-brand {{ display: flex; align-items: center; gap: .65rem; padding-right: 1.25rem; border-right: 1px solid {COLORS["border"]}; }}
    .content-brand .brand-mark {{ width: 36px; height: 36px; border-radius: 11px; }}
    .content-brand .brand-mark svg {{ width: 24px; height: 24px; }}
    .content-brand .brand-name {{ font-size: .95rem; }}
    .content-brand .brand-sub {{ font-size: .55rem; }}
    .overview-label {{ color: {COLORS["muted"]}; font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; }}
    .brand-name {{ color: {COLORS["text"]}; font-weight: 800; font-size: 1.08rem; }}
    .brand-sub {{ color: {COLORS["muted"]}; font-size: .65rem; margin-top: .2rem; }}
    .eyebrow {{ color: {COLORS["muted"]}; font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; }}
    .page-title {{ color: {COLORS["text"]}; margin: .2rem 0 .35rem; }}
    .page-subtitle {{ color: {COLORS["muted"]}; font-size: .82rem; margin-bottom: 1.2rem; }}
    .metric {{ background: {COLORS["card"]}; border: 1px solid {COLORS["border"]}; border-radius: 12px; padding: .9rem 1rem; min-height: 104px;
      box-shadow: 0 10px 28px rgba(0,0,0,.24); position: relative; overflow: hidden; }}
    .metric:before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent); }}
    .metric-label {{ color: {COLORS["muted"]}; font-size: .74rem; font-weight: 600; }}
    .metric-value {{ color: {COLORS["text"]}; font-size: 1.75rem; font-weight: 800; margin: .45rem 0 .25rem; letter-spacing: -.04em; }}
    .metric-note {{ color: {COLORS["muted"]}; font-size: .68rem; }}
    .metric-note strong {{ color: var(--accent); }}
    .section-card {{ background: {COLORS["card"]}; border: 1px solid {COLORS["border"]}; border-radius: 12px; padding: 1rem; margin-top: .85rem; }}
    .section-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: .65rem; }}
    .section-head h3 {{ margin: 0; }}
    .section-head span {{ color: {COLORS["muted"]}; font-size: .7rem; }}
    .status-pill {{ display: inline-flex; border-radius: 99px; padding: .26rem .6rem; font-size: .65rem; font-weight: 700; }}
    .status-red {{ color: #FF8D9B; background: #EF444433; }}
    .status-purple {{ color: #D7A7FF; background: #A855F733; }}
    .status-green {{ color: #76E6A8; background: #22C55E33; }}
    .status-amber {{ color: #FFE27A; background: #F59E0B33; }}
    .alert-item {{ display: flex; justify-content: space-between; align-items: center; padding: .72rem 0; border-bottom: 1px solid {COLORS["border"]}; }}
    .alert-item:last-child {{ border-bottom: 0; }}
    .alert-name {{ color: {COLORS["text"]}; font-size: .78rem; font-weight: 600; }}
    .alert-meta {{ color: {COLORS["muted"]}; font-size: .67rem; margin-top: .18rem; }}
    .side-caption {{ color: {COLORS["muted"]}; font-size: .7rem; line-height: 1.5; }}
    div[data-testid="stTabs"] button {{ color: {COLORS["muted"]}; font-weight: 600; }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{ color: {COLORS["text"]}; }}
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: #3B82F6; }}
    .stButton button {{ border-radius: 9px; font-weight: 700; border-color: {COLORS["border"]}; }}
    .stDownloadButton button {{ border-radius: 9px; }}
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{ border: 1px solid {COLORS["border"]}; }}
    [data-baseweb="select"] > div, [data-testid="stDateInput"] input {{ background: {COLORS["sidebar"]}; color: {COLORS["text"]}; border-color: {COLORS["border"]}; }}
    .theme-control {{ color: {COLORS["muted"]}; font-size: .68rem; padding: .55rem .7rem; border: 1px solid {COLORS["border"]}; border-radius: 10px; margin: .35rem 0 1.15rem; background: {COLORS["card"]}; }}
    .theme-control strong {{ color: {COLORS["text"]}; font-weight: 700; }}
    .nav-group {{ color: {COLORS["muted"]}; font-size: .62rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin: .7rem 0 .55rem; }}
    .nav-item {{ display: flex; align-items: center; color: {COLORS["muted"]}; padding: .68rem .75rem; border-radius: 10px; font-size: .73rem; margin-bottom: .3rem; border: 1px solid transparent; transition: all .2s ease; }}
    .nav-item:hover {{ color: {COLORS["text"]}; background: rgba(59,130,246,.08); border-color: rgba(59,130,246,.2); }}
    .nav-active {{ color: #FFFFFF; background: linear-gradient(135deg, #2563EB, #1D4ED8); box-shadow: 0 8px 18px rgba(37,99,235,.28), inset 3px 0 #93C5FD; }}
    .sidebar-divider {{ border-top: 1px solid {COLORS["border"]}; margin: 1.25rem 0; }}
    [data-testid="stSidebar"] label {{ color: {COLORS["muted"]} !important; font-size: .7rem !important; font-weight: 600 !important; }}
    [data-testid="stSidebar"] [data-testid="stDateInput"], [data-testid="stSidebar"] [data-testid="stSelectbox"] {{ margin-bottom: .25rem; }}
    [data-testid="stSidebar"] [data-testid="stSlider"] {{ padding-top: .25rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def real_dashboard_data(artifact_dir: str) -> dict[str, object] | None:
    """Load persisted pipeline outputs and derive all operational dashboard data."""
    snapshot_path = Path(artifact_dir) / "dashboard_snapshot.json"
    try:
        snapshot = load_persisted_snapshot(snapshot_path)
        if snapshot is None:
            return None
        stations = pd.DataFrame(snapshot["stations"]).rename(columns={"station_id": "id"})
        stations = assess_risk(stations, safety_buffer=0.24)
        metrics = load_persisted_metrics(Path(artifact_dir) / "metrics.json")
        recent = pd.read_parquet(Path(artifact_dir) / "dashboard_recent.parquet")
        return {
            "stations": stations,
            "routes": greedy_matching(stations),
            "metrics": metrics.get("metrics", {}),
            "recent": recent,
            "source": str(metrics.get("data_root", "main/data")),
        }
    except (FileNotFoundError, OSError, ValueError, RuntimeError):
        return None


def load_persisted_snapshot(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_persisted_metrics(path: Path) -> dict[str, object]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def metric(label: str, value: str, note: str, accent: str) -> None:
    st.markdown(
        f'<div class="metric" style="--accent:{accent}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note"><strong>{note.split(" ")[0]}</strong>{" ".join(note.split(" ")[1:])}</div></div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["muted"], family="Plus Jakarta Sans"),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=COLORS["border"], zeroline=False, linecolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"], zeroline=False, linecolor=COLORS["border"])
    return fig


artifact_dir = str(Path(__file__).resolve().parent / "artifacts")
pipeline_data = real_dashboard_data(artifact_dir)
if pipeline_data is None:
    st.error(
        "Chưa có dữ liệu pipeline thật. Hãy chạy Retrain Model hoặc tạo artifact "
        "từ thư mục main/data trước khi sử dụng dashboard."
    )
    st.stop()
else:
    stations = pipeline_data["stations"]
    pipeline_source = f"dữ liệu thật: {Path(artifact_dir).parent / 'data'}"
    routes = pipeline_data["routes"]
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark"><svg viewBox="0 0 32 32" fill="none" aria-hidden="true">'
        '<circle cx="9" cy="23" r="4.2" stroke="white" stroke-width="2"/>'
        '<circle cx="24" cy="23" r="4.2" stroke="white" stroke-width="2"/>'
        '<path d="M9 23 14 11h5l5 12M12 17h9M14 11l-3-3M14 11h5" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        '<path d="m21 5 1.5 3h3l-2.4 1.8.9 3.2L21 11.2l-3 1.8.9-3.2L16.5 8h3L21 5Z" fill="#FDE047"/>'
        '</svg></div><div><div class="brand-name">BikeMind</div>'
        '<div class="brand-sub">SMART MOBILITY DASHBOARD</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="theme-control">Giao diện hiện tại: <strong>{theme_mode}</strong></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-group">Điều hướng</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">▥ &nbsp; Analytics Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item nav-active">◉ &nbsp; Real-time Operations</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">⇄ &nbsp; Rebalancing Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">▣ &nbsp; System & Model Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow" style="margin-top:1.25rem">Bộ lọc vận hành</div>', unsafe_allow_html=True)
    forecast_date = st.date_input("Ngày dự báo", value=date(2017, 1, 15))
    forecast_hour = st.selectbox("Khung giờ", ["08:00 - 09:00", "09:00 - 10:00", "12:00 - 13:00", "17:00 - 18:00"])
    day_type = st.selectbox("Đặc trưng ngày", ["Ngày thường", "Cuối tuần"])
    weather = st.selectbox("Thời tiết", ["Nắng", "Mưa", "Có tuyết"])
    risk_threshold = st.slider("Ngưỡng rủi ro (buffer)", 5, 40, 24, format="%d%%")
    stations = assess_risk(stations, safety_buffer=risk_threshold / 100)
    routes = greedy_matching(stations)
    pipeline_data["stations"] = stations
    pipeline_data["routes"] = routes
    st.divider()
    st.markdown('<div class="eyebrow">Trạng thái hệ thống</div>', unsafe_allow_html=True)
    status_counts = stations["status"].value_counts()
    for label, status, color in [("Thiếu xe", "deficit", "red"), ("Đầy xe", "surplus", "purple"), ("Cân bằng", "balanced", "green")]:
        count = int(status_counts.get(status, 0))
        st.markdown(
            f'<div class="alert-item"><div class="alert-name">{label}<div class="alert-meta">Cần theo dõi</div></div>'
            f'<span class="status-pill status-{color}">{count} trạm</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f"<br><div class='side-caption'>Cập nhật lần cuối<br><strong style='color:{COLORS['text']}'>vừa xong · dữ liệu trực tuyến</strong></div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="eyebrow">REAL-TIME OPERATIONS</div>', unsafe_allow_html=True)
st.markdown('<h1 class="page-title">Bản đồ cảnh báo rủi ro theo trạm</h1>', unsafe_allow_html=True)
st.markdown(
    f'<div class="page-subtitle">Theo dõi mạng lưới xe đạp theo thời gian thực · {forecast_date.strftime("%d/%m/%Y")} · {forecast_hour} · nguồn: {pipeline_source}</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["📍  Vận hành Real-time", "📊  Phân tích & Dự báo", "🚛  Đề xuất phân bổ", "⚙️  Hệ thống & Model AI"])

with tabs[0]:
    total_bikes = int(stations["stock"].sum())
    deficit_count = int((stations.status == "deficit").sum())
    surplus_count = int((stations.status == "surplus").sum())
    route_count = len(routes)
    cols = st.columns(4)
    with cols[0]:
        metric("TỔNG XE / TRẠM", f"{total_bikes:,} / {len(stations)}", "tồn kho ước tính", COLORS["blue"])
    with cols[1]:
        metric("TRẠM THIẾU XE", f"{deficit_count:02d}", "cần bổ sung gấp", COLORS["red"])
    with cols[2]:
        metric("TRẠM ĐẦY XE", f"{surplus_count:02d}", "cần rút bớt xe", COLORS["purple"])
    with cols[3]:
        metric("NHU CẦU ĐIỀU PHỐI", str(route_count), "tuyến cần xem xét", COLORS["amber"])

    left, right = st.columns([2.1, 1], gap="medium")
    with left:
        map_style = COLORS["map"]
        map_name = "Dark Matter" if dark_mode else "Light"
        st.markdown(f'<div class="section-card"><div class="section-head"><h3>Bản đồ mạng lưới · {map_name}</h3><span>LIVE · không cần API key</span></div>', unsafe_allow_html=True)
        map_fig = go.Figure()
        for status, color, label in [
            ("deficit", COLORS["red"], "Thiếu xe · cần bổ sung"),
            ("surplus", COLORS["purple"], "Đầy xe · cần rút bớt"),
            ("balanced", COLORS["green"], "Cân bằng"),
        ]:
            current = stations[stations.status == status]
            map_fig.add_trace(go.Scattermap(
                lat=current.lat, lon=current.lon, mode="markers+text", text=current.id, textposition="top center",
                marker=dict(size=14, color=color), name=label,
                customdata=np.stack([current.name, current.stock, current.capacity], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>Tồn kho dự kiến: %{customdata[1]} / %{customdata[2]}<extra></extra>",
            ))
        map_fig.update_layout(
            map=dict(style=map_style, center=dict(lat=40.745, lon=-73.985), zoom=10.7),
            height=455, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=COLORS["card"],
            legend=dict(
                orientation="h",
                y=0.02,
                x=0.02,
                bgcolor="rgba(23, 27, 39, 0.96)" if dark_mode else "rgba(255, 255, 255, 0.96)",
                bordercolor="#3A465F" if dark_mode else "#D0D5DD",
                borderwidth=1,
                font=dict(size=10, color="#E7ECF5" if dark_mode else "#182230", family="Plus Jakarta Sans"),
            ),
        )
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card"><div class="section-head"><h3>Trạm cần chú ý</h3><span>RỦI RO CAO NHẤT</span></div>', unsafe_allow_html=True)
        risk_stations = stations.sort_values("stock").head(6)
        for _, station in risk_stations.iterrows():
            pct = int(station.stock / station.capacity * 100)
            status = "red" if station.status == "deficit" else "purple" if station.status == "surplus" else "green"
            st.markdown(
                f'<div class="alert-item"><div><div class="alert-name">{station.id} · {station.name}</div>'
                f'<div class="alert-meta">Tồn kho dự kiến {pct}% · Sức chứa {station.capacity}</div></div>'
                f'<span class="status-pill status-{status}">{pct}%</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    station_id = st.selectbox("Tra cứu trạm", stations.id + " · " + stations.name)
    selected_id = station_id.split(" ")[0]
    selected = stations[stations.id == selected_id].iloc[0]
    a, b, c = st.columns(3)
    with a:
        metric("SỨC CHỨA TỐI ĐA", str(selected.capacity), "xe / trạm", COLORS["blue"])
    with b:
        metric("TỒN KHO HIỆN TẠI", str(selected.stock), f"{int(selected.stock / selected.capacity * 100)}% công suất", COLORS["green"])
    with c:
        metric("TỌA ĐỘ", f"{selected.lat:.3f}", f"lon {selected.lon:.3f}", COLORS["purple"])
    station_actuals = pipeline_data["recent"]
    station_actuals = station_actuals[station_actuals.station_id.astype(str) == selected_id].sort_values("timestamp")
    hours = station_actuals["timestamp"]
    pickup_actual = station_actuals["pickups"].to_numpy()
    return_actual = station_actuals["returns"].to_numpy()
    next_row = stations[stations.id == selected_id].iloc[0]
    pickup_forecast = np.full(len(hours), np.nan)
    return_forecast = np.full(len(hours), np.nan)
    pickup_forecast[-1] = next_row.predicted_pickups
    return_forecast[-1] = next_row.predicted_returns
    fig = go.Figure([
        go.Scatter(x=hours, y=pickup_actual, name="Pickups thực tế", line=dict(color=COLORS["blue"], width=2.5)),
        go.Scatter(x=hours, y=pickup_forecast, name="Pickups dự báo", line=dict(color="#60A5FA", dash="dash", width=2)),
        go.Scatter(x=hours, y=return_actual, name="Returns thực tế", line=dict(color=COLORS["green"], width=2.5)),
        go.Scatter(x=hours, y=return_forecast, name="Returns dự báo", line=dict(color="#86EFAC", dash="dot", width=2)),
    ])
    st.markdown('<div class="section-card"><div class="section-head"><h3>Sóng đôi nhu cầu 24 giờ</h3><span>AI FORECAST · LIGHTGBM</span></div>', unsafe_allow_html=True)
    st.plotly_chart(chart_layout(fig), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
    forecast_table = pd.DataFrame({
        "Thời gian": pd.Series(hours).dt.strftime("%H:%M"),
        "Pickups dự báo t+1": pickup_forecast.round(),
        "Returns dự báo t+1": return_forecast.round(),
        "Tồn kho dự kiến": (
            selected.stock
            + np.cumsum(np.nan_to_num(return_forecast - pickup_forecast, nan=0))
        ).clip(0, selected.capacity).round().astype(int),
    })
    st.download_button("↓ Tải dữ liệu dự báo CSV", forecast_table.to_csv(index=False).encode("utf-8"), "bikemind-forecast.csv", "text/csv")
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown('<div class="section-card"><div class="section-head"><h3>Lệnh điều phối chờ phê duyệt</h3><span>UC03 · UC04</span></div>', unsafe_allow_html=True)
    routes = routes.copy()
    routes.insert(0, "Mã tuyến", [f"RB-{index + 1:04d}" for index in range(len(routes))])
    names = stations.set_index("id")["name"].to_dict()
    routes["Trạm nguồn"] = routes["source_id"].map(lambda value: f"{value} · {names.get(value, value)}")
    routes["Trạm đích"] = routes["target_id"].map(lambda value: f"{value} · {names.get(value, value)}")
    routes["Số xe"] = routes["quantity"].astype(int)
    routes["Ưu tiên"] = routes["priority"]
    routes = routes[["Mã tuyến", "Trạm nguồn", "Trạm đích", "Số xe", "Ưu tiên"]]
    edited = st.data_editor(routes, use_container_width=True, hide_index=True, disabled=["Mã tuyến", "Trạm nguồn", "Trạm đích", "Số xe", "Ưu tiên"], column_config={"Ưu tiên": st.column_config.TextColumn("Ưu tiên")})
    if routes.empty:
        st.info("Hiện chưa có tuyến điều phối từ dữ liệu dự báo hiện tại.")
    else:
        action_cols = st.columns(len(routes))
        for index, action_col in enumerate(action_cols):
            with action_col:
                if st.button(f"✓ Duyệt {routes.iloc[index]['Mã tuyến']}", key=f"approve-{index}", use_container_width=True):
                    st.success(f"Đã duyệt {routes.iloc[index]['Mã tuyến']}")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-head"><h3>Bản đồ lộ trình xe tải</h3><span>UC05 · ƯU TIÊN CAO</span></div>', unsafe_allow_html=True)
    route_fig = go.Figure()
    for idx, row in routes.iterrows():
        source = stations[stations.id == row["Trạm nguồn"][:4]].iloc[0]
        target = stations[stations.id == row["Trạm đích"][:4]].iloc[0]
        route_fig.add_trace(go.Scattermap(lat=[source.lat, target.lat], lon=[source.lon, target.lon], mode="lines+markers", line=dict(width=2 + row["Số xe"] / 6, color=COLORS["red"] if row["Ưu tiên"] == "CAO" else COLORS["blue"]), marker=dict(size=10), name=row["Mã tuyến"], hoverinfo="name"))
    route_fig.update_layout(map=dict(style=COLORS["map"], center=dict(lat=40.745, lon=-73.985), zoom=10.7), height=400, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=.02, x=.02, font=dict(size=10)))
    st.plotly_chart(route_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    upload_col, train_col = st.columns([1.1, 1], gap="medium")
    with upload_col:
        st.markdown('<div class="section-card"><div class="section-head"><h3>Cập nhật dữ liệu</h3><span>UC06</span></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Kéo thả file chuyến đi hoặc tọa độ trạm mới", type=["csv"])
        if uploaded:
            st.success(f"Đã nhận {uploaded.name} · sẵn sàng kiểm tra")
        else:
            st.caption("Hỗ trợ CSV / XLSX. Dữ liệu gốc đang được đọc từ thư mục data.")
        st.markdown("</div>", unsafe_allow_html=True)
    with train_col:
        st.markdown('<div class="section-card"><div class="section-head"><h3>Huấn luyện lại AI</h3><span>UC07</span></div>', unsafe_allow_html=True)
        st.write("Cập nhật mô hình khi dữ liệu mới đã được kiểm tra.")
        if st.button("🚀 Retrain Model · LightGBM", type="primary", use_container_width=True):
            with st.spinner("Đang huấn luyện và đánh giá mô hình..."):
                try:
                    data_root = Path(__file__).resolve().parent / "data"
                    if uploaded:
                        upload_dir = data_root / "uploads"
                        upload_dir.mkdir(parents=True, exist_ok=True)
                        (upload_dir / uploaded.name).write_bytes(uploaded.getvalue())
                    result = train_and_persist(data_root, artifact_dir)
                    real_dashboard_data.clear()
                    st.success(
                        f"Retrain hoàn tất · {result.ingestion.files_loaded if result.ingestion else 0} file · "
                        f"đã lưu model pickups/returns"
                    )
                    st.rerun()
                except (OSError, RuntimeError, ValueError) as error:
                    st.error(f"Retrain thất bại: {error}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-head"><h3>Chỉ số chất lượng mô hình</h3><span>UC08 · VALIDATION SET 30 NGÀY</span></div>', unsafe_allow_html=True)
    quality = pd.DataFrame([
        ["Pickups", f"{pipeline_data['metrics']['pickups']['MAE']:.2f}", f"{pipeline_data['metrics']['pickups']['RMSE']:.2f}", f"{pipeline_data['metrics']['pickups']['WAPE']:.2f}%", "Đang dùng"],
        ["Returns", f"{pipeline_data['metrics']['returns']['MAE']:.2f}", f"{pipeline_data['metrics']['returns']['RMSE']:.2f}", f"{pipeline_data['metrics']['returns']['WAPE']:.2f}%", "Đang dùng"],
    ], columns=["Mô hình", "MAE", "RMSE", "WAPE", "Trạng thái"])
    st.dataframe(quality, use_container_width=True, hide_index=True, column_config={"Trạng thái": st.column_config.TextColumn("Trạng thái")})
    st.markdown("</div>", unsafe_allow_html=True)
