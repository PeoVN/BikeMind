from __future__ import annotations

from time import perf_counter
from typing import Any

from .ingestion import build_station_hour_grid, load_trip_data
from .rebalancing import greedy_matching


def benchmark_ingestion(data_root: str) -> dict[str, Any]:
    started = perf_counter()
    trips, report = load_trip_data(data_root)
    grid, stations = build_station_hour_grid(trips)
    return {
        "duration_seconds": round(perf_counter() - started, 3),
        "report": report.__dict__,
        "grid_rows": len(grid),
        "station_rows": len(stations),
    }


def benchmark_rebalancing(stations) -> dict[str, Any]:
    started = perf_counter()
    routes = greedy_matching(stations)
    return {
        "duration_seconds": round(perf_counter() - started, 6),
        "route_count": len(routes),
    }
