from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

import pandas as pd


def haversine_km(lat_1: float, lon_1: float, lat_2: float, lon_2: float) -> float:
    radius = 6371.0
    d_lat = radians(lat_2 - lat_1)
    d_lon = radians(lon_2 - lon_1)
    value = sin(d_lat / 2) ** 2 + cos(radians(lat_1)) * cos(radians(lat_2)) * sin(d_lon / 2) ** 2
    return 2 * radius * atan2(sqrt(value), sqrt(1 - value))


def assess_risk(stations: pd.DataFrame, safety_buffer: float = 0.24) -> pd.DataFrame:
    """Calculate expected stock and classify deficit, surplus, or balanced stations."""
    result = stations.copy()
    result["expected_stock"] = (result["stock"] - result["predicted_pickups"] + result["predicted_returns"]).clip(
        lower=0, upper=result["capacity"]
    )
    result["min_safe"] = result["capacity"] * safety_buffer
    result["max_safe"] = result["capacity"] * (1 - safety_buffer)
    result["status"] = "balanced"
    result.loc[result["expected_stock"] < result["min_safe"], "status"] = "deficit"
    result.loc[result["expected_stock"] > result["max_safe"], "status"] = "surplus"
    return result


def greedy_matching(stations: pd.DataFrame) -> pd.DataFrame:
    """Match nearest surplus and deficit stations using a deterministic greedy strategy."""
    deficits = stations[stations.status == "deficit"].copy()
    surpluses = stations[stations.status == "surplus"].copy()
    routes: list[dict[str, object]] = []
    while not deficits.empty and not surpluses.empty:
        best: tuple[float, int, int] | None = None
        for deficit_index, deficit in deficits.iterrows():
            for surplus_index, surplus in surpluses.iterrows():
                distance = haversine_km(surplus.lat, surplus.lon, deficit.lat, deficit.lon)
                candidate = (distance, surplus_index, deficit_index)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        distance, surplus_index, deficit_index = best
        surplus = surpluses.loc[surplus_index]
        deficit = deficits.loc[deficit_index]
        quantity = int(max(1, min(
            surplus["expected_stock"] - surplus["max_safe"],
            deficit["min_safe"] - deficit["expected_stock"],
        )))
        routes.append({
            "source_id": surplus["id"],
            "target_id": deficit["id"],
            "quantity": quantity,
            "distance_km": round(distance, 2),
            "priority": "CAO" if quantity >= 10 else "TRUNG BÌNH",
        })
        surpluses = surpluses.drop(surplus_index)
        deficits = deficits.drop(deficit_index)
    return pd.DataFrame(routes, columns=["source_id", "target_id", "quantity", "distance_km", "priority"])

