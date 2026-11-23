"""Greedy station pairing using Haversine distance."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    value = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(value))


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two GPS points in kilometres."""
    return haversine_km(lat1, lon1, lat2, lon2)


def generate_rebalancing_plan(stations: pd.DataFrame, safety_buffer: float = 0.15, truck_capacity: int = 20) -> pd.DataFrame:
    """Match surplus stations to deficit stations, nearest first."""
    if stations.empty:
        return pd.DataFrame(columns=["from_station", "to_station", "bikes", "distance_km", "priority"])
    frame = stations.copy()
    frame["target"] = frame["capacity"] * 0.5
    frame["buffer"] = frame["capacity"] * safety_buffer
    frame["surplus"] = (frame["bikes_available"] - frame["target"] - frame["buffer"]).clip(lower=0).round().astype(int)
    frame["deficit"] = (frame["target"] - frame["bikes_available"] + frame["buffer"]).clip(lower=0).round().astype(int)
    donors = frame[frame["surplus"] > 0].to_dict("records")
    receivers = frame[frame["deficit"] > 0].to_dict("records")
    rows: list[dict] = []
    for receiver in receivers:
        while receiver["deficit"] > 0 and donors:
            donor = min(donors, key=lambda item: haversine_km(donor_lat := item["lat"], donor_lon := item["lon"], receiver["lat"], receiver["lon"]))
            amount = min(receiver["deficit"], donor["surplus"], truck_capacity)
            if amount <= 0:
                donors.remove(donor)
                continue
            distance = haversine_km(donor["lat"], donor["lon"], receiver["lat"], receiver["lon"])
            rows.append({"from_station": donor["station_name"], "to_station": receiver["station_name"],
                         "bikes": int(amount), "distance_km": round(distance, 2),
                         "priority": "Cao" if receiver["deficit"] > receiver["capacity"] * 0.3 else "Vừa"})
            donor["surplus"] -= amount
            receiver["deficit"] -= amount
            if donor["surplus"] <= 0:
                donors.remove(donor)
    return pd.DataFrame(rows)


def run_greedy_rebalancing(
    station_df: pd.DataFrame,
    buffer_pct: float = 0.15,
    truck_capacity: int = 20,
) -> pd.DataFrame:
    """Plan transfers from forecasted expected stock using nearest-first matching."""
    required = {"station_id", "station_name", "lat", "lon", "capacity", "bikes_available"}
    missing = required.difference(station_df.columns)
    if missing:
        raise ValueError("Dữ liệu trạm thiếu cột: " + ", ".join(sorted(missing)))
    if not 0 <= buffer_pct <= 1:
        raise ValueError("buffer_pct phải nằm trong khoảng 0 đến 1.")
    if truck_capacity <= 0:
        raise ValueError("truck_capacity phải lớn hơn 0.")
    frame = station_df.copy()
    for column in ("capacity", "bikes_available", "lat", "lon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["capacity", "bikes_available", "lat", "lon"]).copy()
    frame["forecast_pickups"] = pd.to_numeric(
        frame.get("forecast_pickups", frame.get("predicted_pickups", 0)), errors="coerce"
    ).fillna(0)
    frame["forecast_returns"] = pd.to_numeric(
        frame.get("forecast_returns", frame.get("predicted_returns", 0)), errors="coerce"
    ).fillna(0)
    frame["expected_stock"] = (
        frame["bikes_available"] - frame["forecast_pickups"] + frame["forecast_returns"]
    ).clip(lower=0)
    target = frame["capacity"] * 0.5
    buffer = frame["capacity"] * buffer_pct
    frame["surplus"] = (frame["expected_stock"] - target - buffer).clip(lower=0).round().astype(int)
    frame["deficit"] = (target - frame["expected_stock"] + buffer).clip(lower=0).round().astype(int)
    donors = frame[frame["surplus"] > 0].to_dict("records")
    receivers = frame[frame["deficit"] > 0].sort_values("deficit", ascending=False).to_dict("records")
    rows: list[dict[str, object]] = []
    for receiver in receivers:
        while receiver["deficit"] > 0 and donors:
            donor = min(
                donors,
                key=lambda item: calculate_haversine_distance(
                    item["lat"], item["lon"], receiver["lat"], receiver["lon"]
                ),
            )
            amount = min(receiver["deficit"], donor["surplus"], truck_capacity)
            distance = calculate_haversine_distance(
                donor["lat"], donor["lon"], receiver["lat"], receiver["lon"]
            )
            rows.append(
                {
                    "route": f"{donor['station_name']} → {receiver['station_name']}",
                    "from_station": donor["station_name"],
                    "to_station": receiver["station_name"],
                    "bikes": int(amount),
                    "distance_km": round(distance, 2),
                    "priority": "Cao" if receiver["deficit"] > receiver["capacity"] * 0.3 else "Vừa",
                }
            )
            donor["surplus"] -= amount
            receiver["deficit"] -= amount
            if donor["surplus"] <= 0:
                donors.remove(donor)
    return pd.DataFrame(
        rows,
        columns=[
            "route", "from_station", "to_station", "bikes",
            "distance_km", "priority",
        ],
    )
