"""Expected-inventory projection and greedy station pairing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
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
    "route_id",
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
_VALID_IMBALANCE_STATUSES = {"Thiếu xe", "Dư xe", "Cân bằng"}
_INCENTIVE_RECOMMENDATION_COLUMNS = [
    "recommendation_id",
    "station_id",
    "station_name",
    "forecast_for",
    "valid_from",
    "valid_until",
    "reward_points",
    "expected_stock",
    "safe_min_bikes",
    "forecast_pickups",
    "forecast_returns",
    "forecast_pickups_mae",
    "forecast_returns_mae",
    "forecast_pickups_wape",
    "forecast_returns_wape",
    "is_peak_hour",
    "is_estimated",
    "publishable",
    "not_publishable_reason",
    "source_summary",
    "provenance_json",
]
_CLEAN_INCENTIVE_EXECUTION_COLUMNS = [
    "event_id",
    "recommendation_id",
    "station_id",
    "station_name",
    "status",
    "reconciliation_status",
    "executed_at",
    "bikes_returned",
    "reward_points",
    "source",
    "source_summary",
    "event_is_estimated",
    "recommendation_is_estimated",
    "is_estimated",
    "updated_at",
    "imported_at",
    "provenance_json",
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


def _normalise_station_id(value: object) -> str:
    if value is None or isinstance(value, (bool, np.bool_)):
        return ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return ""
    station_id = str(value).strip()
    if not station_id:
        return ""
    if station_id.startswith("+"):
        station_id = station_id[1:]
    try:
        numeric_id = Decimal(station_id)
    except InvalidOperation:
        return station_id
    if not numeric_id.is_finite():
        return ""
    if numeric_id == numeric_id.to_integral_value():
        return str(int(numeric_id))
    return station_id


def _normalise_station_ids(stations: pd.DataFrame) -> pd.DataFrame:
    result = stations.copy()
    result["station_id"] = result["station_id"].map(_normalise_station_id)
    valid_ids = result["station_id"].ne("")
    duplicated = result.loc[valid_ids, "station_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate_ids = sorted(
            result.loc[valid_ids, "station_id"].loc[duplicated].unique().tolist()
        )
        raise ValueError(
            "Mã trạm bị trùng sau khi chuẩn hóa: " + ", ".join(duplicate_ids[:10])
        )
    return result


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
        "scheduled_bikes": int(
            pd.to_numeric(plan["bikes"], errors="coerce").fillna(0).sum()
        )
        if "bikes" in plan
        else 0,
        "route_count": int(len(plan)),
        "distance_km": float(
            pd.to_numeric(plan["distance_km"], errors="coerce").fillna(0).sum()
        )
        if "distance_km" in plan
        else 0.0,
        "estimated_route_count": int(
            plan["uses_estimated_data"].map(_is_truthy).sum()
        )
        if "uses_estimated_data" in plan
        else 0,
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
    required = {
        "station_id",
        "station_name",
        "lat",
        "lon",
        "surplus",
        "deficit",
        "imbalance_status",
    }
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError("Dữ liệu mất cân bằng thiếu cột: " + ", ".join(sorted(missing)))

    inventory = inventory.reset_index(drop=True).copy()
    inventory["_row_index"] = np.arange(len(inventory))
    inventory["surplus"] = pd.to_numeric(inventory["surplus"], errors="coerce")
    inventory["deficit"] = pd.to_numeric(inventory["deficit"], errors="coerce")
    inventory["imbalance_status"] = (
        inventory["imbalance_status"].fillna("").astype(str).str.strip()
    )
    surplus_values = inventory["surplus"].to_numpy(dtype=float, na_value=np.nan)
    deficit_values = inventory["deficit"].to_numpy(dtype=float, na_value=np.nan)
    valid_quantities = (
        np.isfinite(surplus_values)
        & np.isfinite(deficit_values)
        & (surplus_values >= 0)
        & (deficit_values >= 0)
        & (surplus_values == np.floor(surplus_values))
        & (deficit_values == np.floor(deficit_values))
        & (surplus_values <= _MAX_COUNT)
        & (deficit_values <= _MAX_COUNT)
    )
    inventory["surplus"] = np.where(valid_quantities, surplus_values, 0).astype(
        "int64"
    )
    inventory["deficit"] = np.where(valid_quantities, deficit_values, 0).astype(
        "int64"
    )
    status_values = inventory["imbalance_status"].to_numpy(dtype=object)
    valid_status = np.isin(list(status_values), list(_VALID_IMBALANCE_STATUSES))
    consistent_status = (
        ((status_values == "Dư xe") & (surplus_values > 0) & (deficit_values == 0))
        | ((status_values == "Thiếu xe") & (deficit_values > 0) & (surplus_values == 0))
        | (
            (status_values == "Cân bằng")
            & (surplus_values == 0)
            & (deficit_values == 0)
        )
    )
    valid_status &= valid_quantities & consistent_status
    coordinates_valid = _valid_coordinates(inventory)
    valid_station_ids = (
        coordinates_valid["station_id"].notna()
        & coordinates_valid["station_id"].astype("string").str.strip().ne("")
    )
    valid_inventory_rows = pd.Series(valid_status, index=inventory.index)
    frame = coordinates_valid.loc[
        valid_station_ids
        & coordinates_valid["_row_index"].map(valid_inventory_rows).fillna(False)
    ].copy()
    valid_row_indexes = set(frame["_row_index"].tolist())
    frame["station_name"] = (
        frame["station_name"]
        .fillna("")
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna(frame["station_id"])
        .astype(str)
    )
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
                row["route_id"] = f"R{len(rows) + 1:05d}"
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
        elif row_index not in valid_row_indexes and not valid_status[row_index]:
            reason = (
                "Trạng thái mất cân bằng không hợp lệ"
                if valid_quantities[row_index]
                else "Số xe dư/thiếu không hợp lệ"
            )
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
    if not isinstance(stations, pd.DataFrame):
        raise TypeError("stations phải là pandas.DataFrame.")
    buffer = _validate_safety_buffer(safety_buffer)
    truck_capacity = _validate_truck_capacity(truck_capacity)
    if stations.empty:
        return _attach_plan_metadata(
            pd.DataFrame(columns=_LEGACY_PLAN_COLUMNS),
            [],
            buffer,
            truck_capacity,
        )
    required = {
        "station_id", "station_name", "lat", "lon", "capacity", "bikes_available"
    }
    missing = required.difference(stations.columns)
    if missing:
        raise ValueError("Dữ liệu trạm thiếu cột: " + ", ".join(sorted(missing)))
    frame = _normalise_station_ids(stations)
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
    if not isinstance(station_df, pd.DataFrame):
        raise TypeError("station_df phải là pandas.DataFrame.")
    buffer = _validate_safety_buffer(buffer_pct)
    truck_capacity = _validate_truck_capacity(truck_capacity)
    if station_df.empty:
        return _attach_plan_metadata(
            pd.DataFrame(columns=_PLAN_COLUMNS), [], buffer, truck_capacity
        )
    required = {"station_id", "station_name", "lat", "lon", "capacity", "bikes_available"}
    missing = required.difference(station_df.columns)
    if missing:
        raise ValueError("Dữ liệu trạm thiếu cột: " + ", ".join(sorted(missing)))
    frame = _normalise_station_ids(station_df)
    inventory = calculate_expected_inventory(frame, buffer)
    return _pair_imbalances(
        inventory, truck_capacity, include_route=True, safety_buffer=buffer
    )


def build_incentive_recommendations(
    projected: pd.DataFrame,
    forecast_for: object | None = None,
    now: object | None = None,
    reward_points: int = 15,
    validity_minutes: int = 60,
    max_data_age_minutes: int = 90,
) -> pd.DataFrame:
    """Create UC013 candidates from valid next-hour inventory projections."""
    if not isinstance(projected, pd.DataFrame):
        raise TypeError("projected phải là pandas.DataFrame.")
    for name, value in (
        ("reward_points", reward_points),
        ("validity_minutes", validity_minutes),
        ("max_data_age_minutes", max_data_age_minutes),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} phải là số nguyên lớn hơn 0.")

    recommendations = pd.DataFrame(columns=_INCENTIVE_RECOMMENDATION_COLUMNS)
    stats = {
        "input_rows": len(projected),
        "shortage_rows": 0,
        "candidates": 0,
        "excluded_invalid": 0,
        "excluded_duplicate_station_ids": 0,
    }
    recommendations.attrs["stats"] = stats
    if projected.empty:
        recommendations.attrs["unavailable_reason"] = "Không có dữ liệu dự báo."
        return recommendations

    required = {
        "station_id",
        "station_name",
        "expected_stock",
        "safe_min_bikes",
        "imbalance_status",
        "data_quality",
        "forecast_pickups",
        "forecast_returns",
        "forecast_updated_at",
        "forecast_data_updated_at",
    }
    missing = required.difference(projected.columns)
    if missing:
        recommendations.attrs["unavailable_reason"] = (
            "Thiếu cột dự báo cần thiết: " + ", ".join(sorted(missing))
        )
        return recommendations

    local_timezone = datetime.now().astimezone().tzinfo
    current_time = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    if current_time.tzinfo is not None:
        current_time = current_time.tz_convert(local_timezone).tz_localize(None)
    current_time = current_time.floor("s")
    requested_forecast = None
    if forecast_for is not None:
        requested_forecast = pd.Timestamp(forecast_for)
        if requested_forecast.tzinfo is not None:
            requested_forecast = (
                requested_forecast.tz_convert(local_timezone).tz_localize(None)
            )

    frame = projected.copy()
    frame["station_id"] = frame["station_id"].map(_normalise_station_id)
    valid_id = frame["station_id"].ne("")
    duplicate_ids = frame.loc[valid_id, "station_id"].duplicated(keep=False)
    stats["excluded_duplicate_station_ids"] = int(duplicate_ids.sum())
    frame = frame.loc[valid_id & ~duplicate_ids].copy()
    frame["expected_stock"] = pd.to_numeric(frame["expected_stock"], errors="coerce")
    frame["safe_min_bikes"] = pd.to_numeric(frame["safe_min_bikes"], errors="coerce")
    frame["forecast_pickups"] = pd.to_numeric(frame["forecast_pickups"], errors="coerce")
    frame["forecast_returns"] = pd.to_numeric(frame["forecast_returns"], errors="coerce")
    shortage = (
        frame["imbalance_status"].eq("Thiếu xe")
        & frame["data_quality"].eq("Đủ dữ liệu")
        & frame["expected_stock"].lt(frame["safe_min_bikes"])
    )
    stats["shortage_rows"] = int(shortage.sum())
    candidates: list[dict[str, object]] = []

    flag_columns = (
        "projection_is_estimated",
        "inventory_is_estimated",
        "capacity_is_estimated",
    )

    def flag_value(value: object) -> bool:
        if value is None:
            return False
        try:
            missing_value = pd.isna(value)
        except (TypeError, ValueError):
            missing_value = False
        if isinstance(missing_value, (bool, np.bool_)) and missing_value:
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y", "có"}
        return bool(value)

    def finite_optional(row: pd.Series, column: str) -> float | None:
        if column not in row.index:
            return None
        try:
            number = float(row[column])
        except (TypeError, ValueError, OverflowError):
            return None
        return number if isfinite(number) else None

    def text_value(value: object) -> str:
        if value is None:
            return ""
        try:
            missing_value = pd.isna(value)
        except (TypeError, ValueError):
            missing_value = False
        if isinstance(missing_value, (bool, np.bool_)) and missing_value:
            return ""
        return str(value).strip()

    for _, row in frame.loc[shortage].iterrows():
        try:
            expected_stock = float(row["expected_stock"])
            safe_min_bikes = float(row["safe_min_bikes"])
            pickups = float(row["forecast_pickups"])
            returns = float(row["forecast_returns"])
            target_time = pd.Timestamp(row["forecast_updated_at"])
            observed_at = pd.Timestamp(row["forecast_data_updated_at"])
        except (TypeError, ValueError, OverflowError):
            stats["excluded_invalid"] += 1
            continue
        if (
            not all(isfinite(value) for value in (expected_stock, safe_min_bikes, pickups, returns))
            or pickups < 0
            or returns < 0
            or pd.isna(target_time)
            or pd.isna(observed_at)
        ):
            stats["excluded_invalid"] += 1
            continue
        if target_time.tzinfo is not None:
            target_time = target_time.tz_convert(local_timezone).tz_localize(None)
        if observed_at.tzinfo is not None:
            observed_at = observed_at.tz_convert(local_timezone).tz_localize(None)
        target_time = target_time.floor("s")
        observed_at = observed_at.floor("s")
        if requested_forecast is not None and target_time != requested_forecast:
            stats["excluded_invalid"] += 1
            continue

        observed_inventory_at = pd.NaT
        if text_value(row.get("inventory_updated_at", "")):
            try:
                observed_inventory_at = pd.Timestamp(row["inventory_updated_at"])
            except (TypeError, ValueError, OverflowError):
                observed_inventory_at = pd.NaT
            if (
                not pd.isna(observed_inventory_at)
                and observed_inventory_at.tzinfo is not None
            ):
                observed_inventory_at = (
                    observed_inventory_at.tz_convert(local_timezone).tz_localize(None)
                )
        if not pd.isna(observed_inventory_at):
            observed_inventory_at = observed_inventory_at.floor("s")
        forecast_age_minutes = (current_time - observed_at).total_seconds() / 60
        inventory_age_minutes = (
            (current_time - observed_inventory_at).total_seconds() / 60
            if not pd.isna(observed_inventory_at)
            else float("inf")
        )
        age_minutes = max(forecast_age_minutes, inventory_age_minutes)
        in_current_window = (
            current_time.floor("h")
            <= target_time
            <= current_time.floor("h") + pd.Timedelta(hours=1)
        )
        not_publishable_reason = ""
        if pd.isna(observed_inventory_at):
            not_publishable_reason = "Thiếu mốc cập nhật tồn kho."
        elif forecast_age_minutes < 0 or inventory_age_minutes < 0:
            not_publishable_reason = "Mốc dữ liệu nguồn nằm trong tương lai."
        elif age_minutes > max_data_age_minutes:
            not_publishable_reason = (
                f"Dữ liệu nguồn trễ {age_minutes:.0f} phút "
                f"(giới hạn {max_data_age_minutes} phút)."
            )
        elif not in_current_window:
            not_publishable_reason = "Dự báo không thuộc khung giờ hiện tại/kế tiếp."

        source_fields = {
            column: text_value(row.get(column, ""))
            for column in (
                "inventory_source",
                "capacity_source",
                "forecast_pickups_source",
                "forecast_returns_source",
            )
        }
        source_fields = {
            key: value or "Không rõ nguồn"
            for key, value in source_fields.items()
        }
        estimated = any(flag_value(row.get(column)) for column in flag_columns)
        estimated = estimated or any(
            "ước tính" in value.lower()
            or "fallback" in value.lower()
            or value == "Không rõ nguồn"
            for value in source_fields.values()
        )
        source_available = all(
            source_fields[key] != "Không rõ nguồn"
            for key in (
                "inventory_source",
                "capacity_source",
                "forecast_pickups_source",
                "forecast_returns_source",
            )
        )
        if any(
            "thiếu dữ liệu" in value.lower()
            or "không đủ dữ liệu" in value.lower()
            for key, value in source_fields.items()
            if key.startswith("forecast_")
        ):
            stats["excluded_invalid"] += 1
            continue
        if not not_publishable_reason and not source_available:
            not_publishable_reason = "Thiếu provenance của tồn kho hoặc dự báo."
        publishable = not not_publishable_reason and in_current_window

        station_id = str(row["station_id"])
        station_name = text_value(row.get("station_name", ""))
        provenance = {
            **source_fields,
            "inventory_updated_at": text_value(row.get("inventory_updated_at", "")),
            "capacity_updated_at": text_value(row.get("capacity_updated_at", "")),
            "forecast_data_updated_at": observed_at.isoformat(),
            "forecast_updated_at": target_time.isoformat(),
            "data_age_minutes": (
                round(age_minutes, 2) if isfinite(age_minutes) else None
            ),
            "forecast_data_age_minutes": (
                round(forecast_age_minutes, 2)
                if isfinite(forecast_age_minutes)
                else None
            ),
            "inventory_data_age_minutes": (
                round(inventory_age_minutes, 2)
                if isfinite(inventory_age_minutes)
                else None
            ),
            "forecast_pickups_mae": finite_optional(row, "forecast_pickups_mae"),
            "forecast_returns_mae": finite_optional(row, "forecast_returns_mae"),
            "forecast_pickups_wape": finite_optional(row, "forecast_pickups_wape"),
            "forecast_returns_wape": finite_optional(row, "forecast_returns_wape"),
            "is_peak_hour": target_time.hour in {7, 8, 9, 17, 18, 19},
            "weather_data_available": False,
            "device_data_available": False,
        }
        recommendation_id = hashlib.sha256(
            f"UC013|{station_id}|{target_time.isoformat()}".encode("utf-8")
        ).hexdigest()
        candidates.append(
            {
                "recommendation_id": recommendation_id,
                "station_id": station_id,
                "station_name": station_name,
                "forecast_for": target_time,
                "valid_from": current_time,
                "valid_until": current_time + pd.Timedelta(minutes=validity_minutes),
                "reward_points": int(reward_points),
                "expected_stock": expected_stock,
                "safe_min_bikes": safe_min_bikes,
                "forecast_pickups": pickups,
                "forecast_returns": returns,
                "forecast_pickups_mae": provenance["forecast_pickups_mae"],
                "forecast_returns_mae": provenance["forecast_returns_mae"],
                "forecast_pickups_wape": provenance["forecast_pickups_wape"],
                "forecast_returns_wape": provenance["forecast_returns_wape"],
                "is_peak_hour": provenance["is_peak_hour"],
                "is_estimated": estimated,
                "publishable": publishable,
                "not_publishable_reason": not_publishable_reason,
                "source_summary": "; ".join(
                    f"{key}={value}" for key, value in source_fields.items()
                ),
                "provenance_json": json.dumps(
                    provenance,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            }
        )

    recommendations = pd.DataFrame(
        candidates, columns=_INCENTIVE_RECOMMENDATION_COLUMNS
    )
    recommendations.attrs["stats"] = {
        **stats,
        "candidates": len(recommendations),
        "publishable": int(recommendations["publishable"].sum())
        if not recommendations.empty
        else 0,
    }
    if recommendations.empty:
        recommendations.attrs["unavailable_reason"] = (
            "Không có trạm thiếu xe với dự báo t+1 và dữ liệu nguồn hợp lệ."
        )
    return recommendations


def clean_incentive_executions(
    executions: pd.DataFrame,
    recommendations: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return only uniquely matched, completed, verified UC013 events."""
    cleaned = pd.DataFrame(columns=_CLEAN_INCENTIVE_EXECUTION_COLUMNS)
    stats = {
        "input_records": len(executions) if isinstance(executions, pd.DataFrame) else 0,
        "duplicate_event_ids": 0,
        "invalid_identifiers": 0,
        "unmatched_recommendations": 0,
        "station_mismatches": 0,
        "excluded_statuses": 0,
        "unverified_records": 0,
        "outside_validity_window": 0,
        "invalid_execution_values": 0,
        "included_records": 0,
    }
    if not isinstance(executions, pd.DataFrame) or not isinstance(recommendations, pd.DataFrame):
        return cleaned, stats
    required_events = {
        "event_id",
        "recommendation_id",
        "station_id",
        "status",
        "reconciliation_status",
        "executed_at",
        "bikes_returned",
    }
    required_recommendations = {
        "recommendation_id",
        "station_id",
        "station_name",
        "reward_points",
        "published_at",
        "valid_until",
    }
    if (
        executions.empty
        or recommendations.empty
        or not required_events.issubset(executions.columns)
        or not required_recommendations.issubset(recommendations.columns)
    ):
        return cleaned, stats

    events = executions.copy()
    events["event_id"] = events["event_id"].fillna("").astype(str).str.strip()
    events["recommendation_id"] = (
        events["recommendation_id"].fillna("").astype(str).str.strip()
    )
    events["station_id"] = events["station_id"].map(_normalise_station_id)
    duplicate_ids = events["event_id"].ne("") & events["event_id"].duplicated(keep=False)
    stats["duplicate_event_ids"] = int(duplicate_ids.sum())
    valid_identifiers = (
        events["event_id"].ne("")
        & events["recommendation_id"].ne("")
        & events["station_id"].ne("")
        & ~duplicate_ids
    )
    stats["invalid_identifiers"] = int((~valid_identifiers).sum())
    events = events.loc[valid_identifiers].copy()
    events["_normalized_status"] = (
        events["status"].fillna("").astype(str).str.strip().str.lower()
    )
    completed = events["_normalized_status"].eq("completed")
    stats["excluded_statuses"] = int((~completed).sum())
    events = events.loc[completed].copy()
    if events.empty:
        return cleaned, stats

    recs = recommendations.copy()
    recs["recommendation_id"] = (
        recs["recommendation_id"].fillna("").astype(str).str.strip()
    )
    recs["station_id"] = recs["station_id"].map(_normalise_station_id)
    if "status" in recs.columns:
        recs = recs.loc[
            recs["status"].fillna("").astype(str).str.upper().eq("PUBLISHED")
        ].copy()
    if "provenance_json" not in recs.columns:
        recs["provenance_json"] = "{}"
    if "source_summary" not in recs.columns:
        recs["source_summary"] = ""
    recs["_recommendation_is_estimated"] = recs.get(
        "is_estimated", pd.Series(pd.NA, index=recs.index)
    )
    events["_event_is_estimated"] = events.get(
        "is_estimated", pd.Series(pd.NA, index=events.index)
    )
    duplicate_recommendations = recs["recommendation_id"].duplicated(keep=False)
    recs = recs.loc[
        recs["recommendation_id"].ne("") & ~duplicate_recommendations
    ].copy()
    recs["_recommendation_station_id"] = recs["station_id"]
    events = events.merge(
        recs[
            [
                "recommendation_id",
                "_recommendation_station_id",
                "station_name",
                "reward_points",
                "published_at",
                "valid_until",
                "provenance_json",
                "source_summary",
                "_recommendation_is_estimated",
            ]
        ].drop_duplicates("recommendation_id"),
        on="recommendation_id",
        how="left",
        validate="many_to_one",
    )
    unmatched = events["_recommendation_station_id"].isna()
    stats["unmatched_recommendations"] = int(unmatched.sum())
    station_mismatch = (
        ~unmatched
        & events["station_id"].ne(events["_recommendation_station_id"])
    )
    stats["station_mismatches"] = int(station_mismatch.sum())
    events = events.loc[~unmatched & ~station_mismatch].copy()
    if events.empty:
        return cleaned, stats

    verified = (
        events["reconciliation_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("verified")
    )
    stats["unverified_records"] = int((~verified).sum())
    events = events.loc[verified].copy()
    if events.empty:
        return cleaned, stats

    events["_executed_at"] = pd.to_datetime(
        events["executed_at"], errors="coerce", utc=True
    )
    events["_published_at"] = pd.to_datetime(
        events["published_at"], errors="coerce", utc=True
    )
    events["_valid_until"] = pd.to_datetime(
        events["valid_until"], errors="coerce", utc=True
    )
    events["_bikes_returned"] = pd.to_numeric(
        events["bikes_returned"], errors="coerce"
    )
    events["_reward_points"] = pd.to_numeric(events["reward_points"], errors="coerce")
    numeric_valid = (
        np.isfinite(events["_bikes_returned"])
        & events["_bikes_returned"].gt(0)
        & events["_bikes_returned"].mod(1).eq(0)
        & events["_bikes_returned"].lt(2**63)
        & np.isfinite(events["_reward_points"])
        & events["_reward_points"].gt(0)
        & events["_reward_points"].mod(1).eq(0)
        & events["_reward_points"].lt(2**63)
    )
    time_valid = (
        events["_executed_at"].notna()
        & events["_published_at"].notna()
        & events["_valid_until"].notna()
        & events["_executed_at"].ge(events["_published_at"])
        & events["_executed_at"].le(events["_valid_until"])
    )
    stats["outside_validity_window"] = int((~time_valid).sum())
    stats["invalid_execution_values"] = int((~numeric_valid).sum())
    events = events.loc[time_valid & numeric_valid].copy()
    if events.empty:
        return cleaned, stats

    def normalized_flags(values: pd.Series) -> list[object]:
        normalized_values: list[object] = []
        for value in values:
            if value is None or pd.isna(value):
                normalized_values.append(pd.NA)
            elif isinstance(value, str):
                normalized = value.strip().lower()
                normalized_values.append(
                    True if normalized in {"true", "1", "yes", "y", "có"}
                    else False if normalized in {"false", "0", "no", "n", "không"}
                    else pd.NA
                )
            else:
                normalized_values.append(bool(value))
        return normalized_values

    event_estimated = normalized_flags(events["_event_is_estimated"])
    recommendation_estimated = normalized_flags(
        events["_recommendation_is_estimated"]
    )
    estimated_values: list[object] = []
    for event_flag, recommendation_flag in zip(
        event_estimated, recommendation_estimated
    ):
        if event_flag is True or recommendation_flag is True:
            estimated_values.append(True)
        elif pd.isna(event_flag) or pd.isna(recommendation_flag):
            estimated_values.append(pd.NA)
        else:
            estimated_values.append(False)

    cleaned = pd.DataFrame(
        {
            "event_id": events["event_id"].astype(str),
            "recommendation_id": events["recommendation_id"].astype(str),
            "station_id": events["station_id"].astype(str),
            "station_name": events["station_name"].fillna("").astype(str),
            "status": "completed",
            "reconciliation_status": "verified",
            "executed_at": events["_executed_at"],
            "bikes_returned": events["_bikes_returned"].astype("int64"),
            "reward_points": events["_reward_points"],
            "source": events.get("source", pd.Series("", index=events.index)).fillna("").astype(str),
            "source_summary": events["source_summary"].fillna("").astype(str),
            "event_is_estimated": event_estimated,
            "recommendation_is_estimated": recommendation_estimated,
            "is_estimated": estimated_values,
            "updated_at": events.get("updated_at", pd.Series("", index=events.index)).fillna("").astype(str),
            "imported_at": events.get("imported_at", pd.Series("", index=events.index)).fillna("").astype(str),
            "provenance_json": events.get(
                "provenance_json", pd.Series("{}", index=events.index)
            ).fillna("{}").astype(str),
        },
        columns=_CLEAN_INCENTIVE_EXECUTION_COLUMNS,
    ).reset_index(drop=True)
    stats["included_records"] = len(cleaned)
    return cleaned, stats
