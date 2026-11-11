from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import load_metrics, load_model, save_metrics, save_model, save_parquet
from .forecasting import ForecastModel, make_features, train_forecaster
from .ingestion import IngestionReport, build_station_hour_grid, load_trip_data, normalize_station_id


@dataclass
class PipelineResult:
    grid: pd.DataFrame
    stations: pd.DataFrame
    models: dict[str, ForecastModel]
    metrics: dict[str, dict[str, float]]
    ingestion: IngestionReport | None = None


def train_and_persist(
    data_root: str | Path,
    artifact_dir: str | Path,
) -> PipelineResult:
    trips, report = load_trip_data(data_root)
    if trips.empty:
        raise ValueError("No valid trip CSV data was found")
    grid, stations = build_station_hour_grid(trips)
    if grid.empty or stations.empty:
        raise ValueError("The ingestion pipeline produced no station-hour data")

    destination = Path(artifact_dir)
    save_parquet(stations, destination / "station_metadata.parquet")
    save_parquet(grid, destination / "station_hour_cleaned.parquet")

    models: dict[str, ForecastModel] = {}
    metrics: dict[str, dict[str, float]] = {}
    for target in ("pickups", "returns"):
        features = make_features(grid, target=target)
        model, evaluation = train_forecaster(features, target=target)
        models[target] = model
        metrics[target] = evaluation
        save_model(model, destination / f"lgb_{target}.pkl")
    dashboard_predictions = forecast_next_hour(grid, models)
    recent = grid[grid["timestamp"] >= grid["timestamp"].max() - pd.Timedelta(hours=23)]
    save_parquet(recent, destination / "dashboard_recent.parquet")
    snapshot_stations = stations.copy()
    snapshot_stations["station_id"] = snapshot_stations["station_id"].astype(str)
    demand_capacity = recent.groupby("station_id")[["pickups", "returns"]].sum().max(axis=1)
    snapshot_stations["capacity"] = (
        snapshot_stations["station_id"].map(demand_capacity).fillna(1).clip(lower=1) * 4
    ).round().astype(int)
    snapshot_stations["stock"] = (
        recent.groupby("station_id")["returns"].sum()
        - recent.groupby("station_id")["pickups"].sum()
    ).reindex(snapshot_stations["station_id"]).fillna(0).clip(lower=0).round().astype(int).to_numpy()
    dashboard_predictions = dashboard_predictions.rename(columns={"station_id": "station_id"})
    snapshot_stations = snapshot_stations.merge(
        dashboard_predictions[["station_id", "predicted_pickups", "predicted_returns"]],
        on="station_id",
        how="left",
    ).fillna({"predicted_pickups": 0, "predicted_returns": 0})
    save_metrics(
        {"stations": snapshot_stations.to_dict(orient="records")},
        destination / "dashboard_snapshot.json",
    )
    save_metrics(
        {
            "metrics": metrics,
            "backends": {target: models[target].backend for target in models},
            "ingestion": report.__dict__,
            "data_root": str(Path(data_root).resolve()),
            "artifact_version": "real-data-v1",
        },
        destination / "metrics.json",
    )
    return PipelineResult(grid, stations, models, metrics, report)


def load_persisted(artifact_dir: str | Path) -> PipelineResult:
    destination = Path(artifact_dir)
    grid = pd.read_parquet(destination / "station_hour_cleaned.parquet")
    stations = pd.read_parquet(destination / "station_metadata.parquet")
    grid["station_id"] = grid["station_id"].map(normalize_station_id)
    stations["station_id"] = stations["station_id"].map(normalize_station_id)
    models = {
        target: load_model(destination / f"lgb_{target}.pkl")
        for target in ("pickups", "returns")
    }
    raw_metrics = load_metrics(destination / "metrics.json")
    return PipelineResult(grid, stations, models, raw_metrics.get("metrics", {}))


def forecast_next_hour(grid: pd.DataFrame, models: dict[str, ForecastModel]) -> pd.DataFrame:
    """Predict the next hour for every station using only its latest 168 observations."""
    if grid.empty:
        return pd.DataFrame()
    history = grid.copy()
    history["station_id"] = history["station_id"].astype(str)
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    latest = history.groupby("station_id", sort=False).tail(168)
    next_timestamp = history["timestamp"].max() + pd.Timedelta(hours=1)
    rows: list[dict[str, Any]] = []
    feature_rows: dict[str, list[dict[str, Any]]] = {"pickups": [], "returns": []}
    valid_ids: list[str] = []
    for station_id, station in latest.groupby("station_id", sort=False):
        station = station.sort_values("timestamp")
        if len(station) < 168:
            continue
        valid_ids.append(station_id)
        for target in ("pickups", "returns"):
            values = station[target].to_numpy(dtype=float)
            feature_rows[target].append({
                "hour": next_timestamp.hour,
                "day_of_week": next_timestamp.dayofweek,
                "is_weekend": int(next_timestamp.dayofweek >= 5),
                "is_peak": int(next_timestamp.hour in (7, 8, 9, 17, 18, 19)),
                "lag_1h": values[-1],
                "lag_2h": values[-2],
                "lag_24h": values[-24],
                "lag_168h": values[-168],
                "rolling_3h": float(values[-3:].mean()),
            })
    predictions = {
        target: models[target].predict(pd.DataFrame(feature_rows[target]))
        for target in ("pickups", "returns")
    }
    for index, station_id in enumerate(valid_ids):
        rows.append({
            "station_id": station_id,
            "timestamp": next_timestamp,
            "predicted_pickups": float(predictions["pickups"][index]),
            "predicted_returns": float(predictions["returns"][index]),
        })
    return pd.DataFrame(rows)


def _feature_row(history: pd.DataFrame, station_id: str, timestamp: pd.Timestamp, target: str) -> pd.DataFrame:
    values = history[history["station_id"].astype(str) == str(station_id)].copy()
    values = values.sort_values("timestamp")
    if len(values) < 168:
        return pd.DataFrame()
    series = values[target].to_numpy(dtype=float)
    row = {
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "is_weekend": int(timestamp.dayofweek >= 5),
        "is_peak": int(timestamp.hour in (7, 8, 9, 17, 18, 19)),
        "lag_1h": series[-1],
        "lag_2h": series[-2],
        "lag_24h": series[-24],
        "lag_168h": series[-168],
        "rolling_3h": float(series[-3:].mean()),
    }
    return pd.DataFrame([row])


def forecast_horizon(
    grid: pd.DataFrame,
    models: dict[str, ForecastModel],
    horizon: int = 24,
    station_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Recursively forecast both demand streams for selected known stations."""
    if grid.empty:
        return pd.DataFrame()
    history = grid.copy()
    history["station_id"] = history["station_id"].astype(str)
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    next_timestamp = history["timestamp"].max() + pd.Timedelta(hours=1)
    selected_station_ids = (
        [str(value) for value in station_ids]
        if station_ids is not None
        else history["station_id"].unique().tolist()
    )
    histories = {
        station_id: history[history["station_id"] == station_id][
            ["station_id", "timestamp", "pickups", "returns"]
        ].sort_values("timestamp").copy()
        for station_id in selected_station_ids
    }
    output: list[dict[str, Any]] = []
    for offset in range(horizon):
        timestamp = next_timestamp + pd.Timedelta(hours=offset)
        for station_id in selected_station_ids:
            row: dict[str, Any] = {"station_id": station_id, "timestamp": timestamp}
            for target in ("pickups", "returns"):
                feature_row = _feature_row(histories[station_id], station_id, timestamp, target)
                if feature_row.empty:
                    prediction = 0.0
                else:
                    prediction = float(models[target].predict(feature_row)[0])
                row[f"predicted_{target}"] = prediction
            output.append(row)
            histories[station_id] = pd.concat(
                [
                    histories[station_id],
                    pd.DataFrame([{
                        "station_id": station_id,
                        "timestamp": timestamp,
                        "pickups": row["predicted_pickups"],
                        "returns": row["predicted_returns"],
                    }]),
                ],
                ignore_index=True,
            )
    return pd.DataFrame(output)
