"""Expected-inventory projection and greedy station pairing."""

from __future__ import annotations

from math import asin, cos, floor, isfinite, radians, sin, sqrt

import numpy as np
import pandas as pd


_EXPECTED_COLUMNS = [
    "expected_stock_raw",
    "expected_stock",
    "safe_min_bikes",
    "safe_max_bikes",
    "surplus",
    "deficit",
    "imbalance_status",
    "data_quality",
    "projection_adjusted",
    "inventory_is_estimated",
    "capacity_is_estimated",
    "projection_is_estimated",
    "safety_buffer_pct",
]
_MAX_COUNT = float(np.nextafter(np.float64(np.iinfo(np.int64).max), 0))
_PLAN_COLUMNS = [
    "route",
    "from_station",
    "from_station_id",
    "to_station",
    "to_station_id",
    "bikes",
    "distance_km",
    "priority",
    "from_data_source",
    "to_data_source",
    "from_updated_at",
    "to_updated_at",
    "uses_estimated_data",
]
_LEGACY_PLAN_COLUMNS = [
    "from_station",
    "to_station",
    "bikes",
    "distance_km",
    "priority",
]


def _validate_safety_buffer(value: float) -> float:
    try:
        buffer = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Safety Buffer phải là số hữu hạn từ 0 đến 0.5.") from exc
    if not isfinite(buffer) or not 0 <= buffer <= 0.5:
        raise ValueError("Safety Buffer phải nằm trong khoảng 0 đến 0.5.")
    return buffer


def _validate_truck_capacity(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("truck_capacity phải là số nguyên lớn hơn 0.")
    try:
        capacity = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("truck_capacity phải là số nguyên lớn hơn 0.") from exc
    if not isfinite(capacity) or capacity <= 0 or not capacity.is_integer():
        raise ValueError("truck_capacity phải là số nguyên lớn hơn 0.")
    return int(capacity)


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    alias: str | None = None,
) -> pd.Series:
    if column in frame.columns:
        values = frame[column]
    elif alias is not None and alias in frame.columns:
        values = frame[alias]
    else:
        values = pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(values, errors="coerce")


def _source_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("Không rõ nguồn", index=frame.index, dtype="string")
    return frame[column].fillna("Không rõ nguồn").astype("string").str.strip()


def _is_estimated_source(source: pd.Series) -> pd.Series:
    return source.str.contains(
        r"ước tính|estimated|fallback|demo|metadata|không rõ|unknown",
        case=False,
        regex=True,
        na=False,
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    value = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(min(1.0, max(0.0, value))))


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance between two GPS points in kilometres."""
    return haversine_km(lat1, lon1, lat2, lon2)


def calculate_expected_inventory(
    stations: pd.DataFrame,
    safety_buffer: float = 0.15,
) -> pd.DataFrame:
    """Project end-of-hour stock and classify only rows with complete valid inputs."""
    if not isinstance(stations, pd.DataFrame):
        raise TypeError("stations phải là pandas.DataFrame.")
    buffer = _validate_safety_buffer(safety_buffer)
    result = stations.copy()
    if result.empty:
        for column in _EXPECTED_COLUMNS:
            if column not in result.columns:
                result[column] = pd.Series(dtype="object")
        return result

    required = {"capacity", "bikes_available"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError("Dữ liệu trạm thiếu cột: " + ", ".join(sorted(missing)))

    capacity = _numeric_column(result, "capacity")
    inventory = _numeric_column(result, "bikes_available")
    pickups = _numeric_column(result, "forecast_pickups", "predicted_pickups")
    returns = _numeric_column(result, "forecast_returns", "predicted_returns")
    capacity_values = capacity.to_numpy(dtype=float, na_value=np.nan)
    inventory_values = inventory.to_numpy(dtype=float, na_value=np.nan)
    pickups_values = pickups.to_numpy(dtype=float, na_value=np.nan)
    returns_values = returns.to_numpy(dtype=float, na_value=np.nan)

    valid_capacity = (
        np.isfinite(capacity_values)
        & (capacity_values > 0)
        & (capacity_values <= _MAX_COUNT)
    )
    valid_inventory = (
        np.isfinite(inventory_values)
        & (inventory_values >= 0)
        & (inventory_values <= _MAX_COUNT)
    )
    valid_pickups = (
        np.isfinite(pickups_values)
        & (pickups_values >= 0)
        & (pickups_values <= _MAX_COUNT)
    )
    valid_returns = (
        np.isfinite(returns_values)
        & (returns_values >= 0)
        & (returns_values <= _MAX_COUNT)
    )
    projection_inputs_valid = valid_inventory & valid_pickups & valid_returns
    inventory_within_capacity = (
        ~valid_capacity | (inventory_values <= capacity_values)
    )
    projected_raw = np.full(len(result), np.nan, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        projected_raw[projection_inputs_valid] = (
            inventory_values[projection_inputs_valid]
            - pickups_values[projection_inputs_valid]
            + returns_values[projection_inputs_valid]
        )
    finite_projection = np.isfinite(projected_raw)
    classification_valid = (
        valid_capacity
        & projection_inputs_valid
        & finite_projection
        & inventory_within_capacity
    )

    expected = np.full(len(result), np.nan, dtype=float)
    has_capacity_projection = (
        projection_inputs_valid & finite_projection & valid_capacity
    )
    expected[has_capacity_projection] = np.clip(
        projected_raw[has_capacity_projection],
        0,
        capacity_values[has_capacity_projection],
    )
    without_capacity = (
        projection_inputs_valid & finite_projection & ~valid_capacity
    )
    expected[without_capacity] = np.maximum(projected_raw[without_capacity], 0)

    lower_ratio = 0.5 - buffer
    upper_ratio = 0.5 + buffer
    safe_min = np.full(len(result), np.nan, dtype=float)
    safe_max = np.full(len(result), np.nan, dtype=float)
    safe_min[valid_capacity] = capacity_values[valid_capacity] * lower_ratio
    safe_max[valid_capacity] = capacity_values[valid_capacity] * upper_ratio

    status = np.full(len(result), "Thiếu dữ liệu", dtype=object)
    status[classification_valid & (expected < safe_min)] = "Thiếu xe"
    status[classification_valid & (expected > safe_max)] = "Dư xe"
    status[
        classification_valid
        & (expected >= safe_min)
        & (expected <= safe_max)
    ] = "Cân bằng"

    surplus = np.zeros(len(result), dtype="int64")
    deficit = np.zeros(len(result), dtype="int64")
    surplus_difference = np.maximum(expected - safe_max, 0)
    deficit_difference = np.maximum(safe_min - expected, 0)
    surplus[classification_valid] = np.floor(
        surplus_difference[classification_valid] + 1e-9
    ).astype("int64")
    deficit[classification_valid] = np.ceil(
        deficit_difference[classification_valid] - 1e-9
    ).astype("int64")

    inventory_source = _source_column(result, "inventory_source")
    capacity_source = _source_column(result, "capacity_source")
    inventory_estimated = _is_estimated_source(inventory_source)
    capacity_estimated = _is_estimated_source(capacity_source)
    if "capacity_estimated" in result.columns:
        capacity_estimated |= (
            result["capacity_estimated"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y", "có"])
        )
    if "inventory_estimated" in result.columns:
        inventory_estimated |= (
            result["inventory_estimated"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y", "có"])
        )

    quality: list[str] = []
    for index in range(len(result)):
        issues: list[str] = []
        if not np.isfinite(inventory_values[index]):
            issues.append("Thiếu tồn kho đầu kỳ")
        elif inventory_values[index] < 0 or inventory_values[index] > _MAX_COUNT:
            issues.append("Tồn kho đầu kỳ không hợp lệ")
        elif valid_capacity[index] and inventory_values[index] > capacity_values[index]:
            issues.append("Tồn kho đầu kỳ vượt sức chứa")
        if not valid_capacity[index]:
            issues.append("Thiếu sức chứa hợp lệ")
        if not np.isfinite(pickups_values[index]):
            issues.append("Thiếu dự báo pickups")
        elif pickups_values[index] < 0 or pickups_values[index] > _MAX_COUNT:
            issues.append("Dự báo pickups không hợp lệ")
        if not np.isfinite(returns_values[index]):
            issues.append("Thiếu dự báo returns")
        elif returns_values[index] < 0 or returns_values[index] > _MAX_COUNT:
            issues.append("Dự báo returns không hợp lệ")
        if projection_inputs_valid[index] and not finite_projection[index]:
            issues.append("Tồn kho dự kiến không hữu hạn")
        quality.append("; ".join(issues) if issues else "Đủ dữ liệu")

    result["capacity"] = capacity
    result["bikes_available"] = inventory
    result["forecast_pickups"] = pickups
    result["forecast_returns"] = returns
    result["expected_stock_raw"] = projected_raw
    result["expected_stock"] = expected
    result["safe_min_bikes"] = safe_min
    result["safe_max_bikes"] = safe_max
    result["surplus"] = surplus
    result["deficit"] = deficit
    result["imbalance_status"] = status
    result["data_quality"] = quality
    result["projection_adjusted"] = (
        has_capacity_projection
        & ((projected_raw < 0) | (projected_raw > capacity_values))
    )
    result["inventory_is_estimated"] = inventory_estimated.to_numpy(dtype=bool)
    result["capacity_is_estimated"] = capacity_estimated.to_numpy(dtype=bool)
    result["projection_is_estimated"] = (
        inventory_estimated | capacity_estimated
    ).to_numpy(dtype=bool)
    result["safety_buffer_pct"] = buffer * 100
    return result


def _valid_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("lat", "lon"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    coordinates = result[["lat", "lon"]].to_numpy(dtype=float, na_value=np.nan)
    valid = (
        np.isfinite(coordinates).all(axis=1)
        & result["lat"].between(-90, 90).to_numpy()
        & result["lon"].between(-180, 180).to_numpy()
    )
    return result.loc[valid].copy()


def _is_truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "có"}
    return bool(value)


def _data_source(record: dict[str, object]) -> tuple[str, bool]:
    source_columns = (
        "inventory_source",
        "capacity_source",
        "forecast_pickups_source",
        "forecast_returns_source",
    )
    sources: list[str] = []
    estimated = _is_truthy(record.get("projection_is_estimated", False))
    estimated |= _is_truthy(record.get("inventory_estimated", False))
    estimated |= _is_truthy(record.get("capacity_estimated", False))
    for column in source_columns:
        value = record.get(column)
        if value is None or pd.isna(value):
            continue
        source = str(value).strip()
        if source and source not in sources:
            sources.append(source)
        source_lower = source.lower()
        estimated |= any(
            marker in source_lower
            for marker in (
                "ước tính",
                "estimated",
                "fallback",
                "demo",
                "metadata",
                "không rõ",
                "unknown",
            )
        )
    return ("; ".join(sources) if sources else "Không rõ nguồn", estimated)


def _updated_at(record: dict[str, object]) -> str:
    values = [
        record.get("inventory_updated_at"),
        record.get("capacity_updated_at"),
        record.get("forecast_data_updated_at"),
        record.get("forecast_updated_at"),
    ]
    timestamps = [str(value) for value in values if value is not None and not pd.isna(value)]
    return " · ".join(dict.fromkeys(timestamps)) if timestamps else "Không rõ thời điểm"


def _attach_plan_metadata(
    plan: pd.DataFrame,
    unmatched: list[dict[str, object]],
    safety_buffer: float,
    truck_capacity: int,
) -> pd.DataFrame:
    unmatched_frame = pd.DataFrame(unmatched)
    if unmatched_frame.empty:
        unmatched_frame = pd.DataFrame(
            columns=[
                "station_id",
                "station_name",
                "imbalance_status",
                "remaining_surplus",
                "remaining_deficit",
                "reason",
                "data_source",
                "updated_at",
                "uses_estimated_data",
            ]
        )
    plan.attrs["unmatched_stations"] = unmatched_frame.to_dict("records")
    plan.attrs["summary"] = {
        "scheduled_bikes": int(plan["bikes"].sum()) if "bikes" in plan else 0,
        "route_count": int(len(plan)),
        "distance_km": float(plan["distance_km"].sum()) if "distance_km" in plan else 0.0,
        "unmatched_station_count": int(len(unmatched_frame)),
        "unmatched_surplus": int(unmatched_frame["remaining_surplus"].sum())
        if "remaining_surplus" in unmatched_frame
        else 0,
        "unmatched_deficit": int(unmatched_frame["remaining_deficit"].sum())
        if "remaining_deficit" in unmatched_frame
        else 0,
    }
    plan.attrs["generation"] = {
        "safety_buffer_pct": safety_buffer * 100,
        "truck_capacity": truck_capacity,
    }
    return plan


def _pair_imbalances(
    inventory: pd.DataFrame,
    truck_capacity: int,
    include_route: bool,
    safety_buffer: float,
) -> pd.DataFrame:
    columns = _PLAN_COLUMNS if include_route else _LEGACY_PLAN_COLUMNS
    if inventory.empty:
        return _attach_plan_metadata(
            pd.DataFrame(columns=columns), [], safety_buffer, truck_capacity
        )

    inventory = inventory.reset_index(drop=True).copy()
    inventory["_row_index"] = np.arange(len(inventory))
    coordinates_valid = _valid_coordinates(inventory)
    valid_station_ids = (
        coordinates_valid["station_id"].notna()
        & coordinates_valid["station_id"].astype("string").str.strip().ne("")
    )
    frame = coordinates_valid.loc[valid_station_ids].copy()
    valid_row_indexes = set(frame["_row_index"].tolist())
    frame["station_name"] = frame["station_name"].fillna(
        frame["station_id"]
    ).astype(str)
    donors = frame.loc[frame["surplus"] > 0].to_dict("records")
    receivers = (
        frame.loc[frame["deficit"] > 0]
        .sort_values("deficit", ascending=False)
        .to_dict("records")
    )
    rows: list[dict[str, object]] = []
    remaining = {
        int(record["_row_index"]): {
            "surplus": int(record["surplus"]),
            "deficit": int(record["deficit"]),
        }
        for record in inventory.to_dict("records")
    }
    for receiver in receivers:
        initial_deficit = int(receiver["deficit"])
        while receiver["deficit"] > 0 and donors:
            eligible_donors = [
                donor for donor in donors
                if donor["station_id"] != receiver["station_id"]
                and donor["surplus"] > 0
            ]
            if not eligible_donors:
                break
            donor = min(
                eligible_donors,
                key=lambda item: calculate_haversine_distance(
                    item["lat"], item["lon"], receiver["lat"], receiver["lon"]
                ),
            )
            amount = min(receiver["deficit"], donor["surplus"], truck_capacity)
            if amount <= 0:
                donors.remove(donor)
                continue
            distance = calculate_haversine_distance(
                donor["lat"], donor["lon"], receiver["lat"], receiver["lon"]
            )
            row: dict[str, object] = {
                "from_station": donor["station_name"],
                "from_station_id": donor["station_id"],
                "to_station": receiver["station_name"],
                "to_station_id": receiver["station_id"],
                "bikes": int(amount),
                "distance_km": round(distance, 2),
                "priority": (
                    "Cao"
                    if initial_deficit > receiver["capacity"] * 0.3
                    else "Vừa"
                ),
            }
            from_source, from_estimated = _data_source(donor)
            to_source, to_estimated = _data_source(receiver)
            row.update(
                {
                    "from_data_source": from_source,
                    "to_data_source": to_source,
                    "from_updated_at": _updated_at(donor),
                    "to_updated_at": _updated_at(receiver),
                    "uses_estimated_data": from_estimated or to_estimated,
                }
            )
            if include_route:
                row["route"] = f"{donor['station_name']} → {receiver['station_name']}"
            rows.append(row)
            donor["surplus"] -= amount
            receiver["deficit"] -= amount
            remaining[int(donor["_row_index"])]["surplus"] = int(donor["surplus"])
            remaining[int(receiver["_row_index"])]["deficit"] = int(receiver["deficit"])
            if donor["surplus"] <= 0:
                donors.remove(donor)

    unmatched: list[dict[str, object]] = []
    for record in inventory.to_dict("records"):
        row_index = int(record["_row_index"])
        residual_surplus = remaining[row_index]["surplus"]
        residual_deficit = remaining[row_index]["deficit"]
        if residual_surplus <= 0 and residual_deficit <= 0:
            continue
        data_source, estimated = _data_source(record)
        station_id = record["station_id"]
        if pd.isna(station_id) or not str(station_id).strip():
            reason = "Thiếu mã trạm"
        elif row_index not in valid_row_indexes:
            reason = "Tọa độ thiếu hoặc không hợp lệ"
        elif residual_surplus > 0:
            reason = "Chưa ghép hết lượng xe dư"
        else:
            reason = "Chưa ghép hết nhu cầu thiếu xe"
        unmatched.append(
            {
                "station_id": record["station_id"],
                "station_name": (
                    record["station_name"]
                    if not pd.isna(record["station_name"])
                    else str(record["station_id"])
                ),
                "imbalance_status": record["imbalance_status"],
                "remaining_surplus": residual_surplus,
                "remaining_deficit": residual_deficit,
                "reason": reason,
                "data_source": data_source,
                "updated_at": _updated_at(record),
                "uses_estimated_data": estimated,
            }
        )
    return _attach_plan_metadata(
        pd.DataFrame(rows, columns=columns), unmatched, safety_buffer, truck_capacity
    )


def generate_rebalancing_plan(
    stations: pd.DataFrame,
    safety_buffer: float = 0.15,
    truck_capacity: int = 20,
) -> pd.DataFrame:
    """Match surplus stations to deficit stations, nearest first."""
    if stations.empty:
        return pd.DataFrame(
            columns=["from_station", "to_station", "bikes", "distance_km", "priority"]
        )
    required = {
        "station_id", "station_name", "lat", "lon", "capacity", "bikes_available"
    }
    missing = required.difference(stations.columns)
    if missing:
        raise ValueError("Dữ liệu trạm thiếu cột: " + ", ".join(sorted(missing)))
    buffer = _validate_safety_buffer(safety_buffer)
    truck_capacity = _validate_truck_capacity(truck_capacity)
    frame = stations.copy()
    frame["forecast_pickups"] = 0.0
    frame["forecast_returns"] = 0.0
    inventory = calculate_expected_inventory(frame, buffer)
    return _pair_imbalances(
        inventory, truck_capacity, include_route=False, safety_buffer=buffer
    )


def run_greedy_rebalancing(
    station_df: pd.DataFrame,
    buffer_pct: float = 0.15,
    truck_capacity: int = 20,
) -> pd.DataFrame:
    """Plan transfers from forecasted, safely bounded expected stock."""
    if station_df.empty:
        return pd.DataFrame(columns=_PLAN_COLUMNS)
    required = {"station_id", "station_name", "lat", "lon", "capacity", "bikes_available"}
    missing = required.difference(station_df.columns)
    if missing:
        raise ValueError("Dữ liệu trạm thiếu cột: " + ", ".join(sorted(missing)))
    buffer = _validate_safety_buffer(buffer_pct)
    truck_capacity = _validate_truck_capacity(truck_capacity)
    inventory = calculate_expected_inventory(station_df, buffer)
    return _pair_imbalances(
        inventory, truck_capacity, include_route=True, safety_buffer=buffer
    )
