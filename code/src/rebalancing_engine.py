"""Greedy station pairing using Haversine distance."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

import pandas as pd


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    value = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(value))


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
