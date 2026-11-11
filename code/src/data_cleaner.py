"""Trip validation, hourly aggregation, and 24/7 zero filling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_trips(trips: pd.DataFrame) -> pd.DataFrame:
    if trips.empty:
        return trips.copy()
    frame = trips.copy()
    frame["start_time"] = pd.to_datetime(frame.get("start_time"), errors="coerce")
    frame["end_time"] = pd.to_datetime(frame.get("end_time"), errors="coerce")
    frame["trip_duration"] = pd.to_numeric(frame.get("trip_duration"), errors="coerce")
    required = ["start_time", "end_time", "trip_duration", "start_station_id", "end_station_id"]
    frame = frame.dropna(subset=[col for col in required if col in frame])
    frame = frame[frame["trip_duration"].between(60, 7200)]
    frame = frame[frame["end_time"] > frame["start_time"]]
    frame = frame[frame["start_station_id"].astype(str) != frame["end_station_id"].astype(str)]
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
    stations = stations.dropna(subset=["station_id", "lat", "lon"]).drop_duplicates("station_id")
    stations["capacity"] = 40
    return stations.reset_index(drop=True)


def aggregate_hourly(trips: pd.DataFrame, station_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aggregate pickups/returns and fill every station-hour in the observed period."""
    clean = clean_trips(trips)
    if clean.empty:
        return pd.DataFrame(columns=["timestamp", "station_id", "pickups", "returns"])
    starts = clean.groupby([clean["start_time"].dt.floor("h"), "start_station_id"]).size().rename("pickups")
    returns = clean.groupby([clean["end_time"].dt.floor("h"), "end_station_id"]).size().rename("returns")
    hourly = pd.concat([starts, returns], axis=1).fillna(0).rename_axis(["timestamp", "station_id"]).reset_index()
    station_ids = (station_metadata["station_id"].tolist() if station_metadata is not None and not station_metadata.empty
                   else hourly["station_id"].drop_duplicates().tolist())
    timestamps = pd.date_range(hourly["timestamp"].min().floor("D"), hourly["timestamp"].max().ceil("D") - pd.Timedelta(hours=1), freq="h")
    full = pd.MultiIndex.from_product([timestamps, station_ids], names=["timestamp", "station_id"]).to_frame(index=False)
    hourly = full.merge(hourly, on=["timestamp", "station_id"], how="left").fillna({"pickups": 0, "returns": 0})
    hourly["pickups"] = hourly["pickups"].astype(int)
    hourly["returns"] = hourly["returns"].astype(int)
    return hourly


def save_processed(hourly: pd.DataFrame, station_metadata: pd.DataFrame, output_dir: str) -> None:
    from pathlib import Path
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, frame in (("station_hour_cleaned", hourly), ("station_metadata", station_metadata)):
        try:
            frame.to_parquet(path / f"{name}.parquet", index=False)
        except (ImportError, ValueError):
            frame.to_csv(path / f"{name}.csv", index=False)
