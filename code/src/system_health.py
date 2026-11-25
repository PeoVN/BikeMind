"""Bounded health checks for processed artifacts, models, and SQLite state."""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from .gbfs_collector import DEFAULT_MAX_AGE_SECONDS, MAX_FUTURE_SKEW_SECONDS
from .model_trainer import load_model


_HEALTH_COLUMNS = [
    "component",
    "status",
    "detail",
    "checked_at",
    "updated_at",
    "estimated_flag",
    "path",
]
_ESTIMATE_COLUMNS = {
    "is_estimated",
    "projection_is_estimated",
    "inventory_is_estimated",
    "capacity_is_estimated",
    "uses_estimated_data",
}
_REQUIRED_TABLES = {
    "station_status": (
        {"timestamp", "station_id", "bikes_available", "capacity"},
        "ERROR",
    ),
    "decision_log": (
        {"created_at", "action", "payload"},
        "ERROR",
    ),
    "incentive_recommendations": (
        {
            "recommendation_id",
            "station_id",
            "published_at",
            "valid_until",
            "is_estimated",
            "status",
        },
        "WARNING",
    ),
    "incentive_executions": (
        {
            "event_id",
            "recommendation_id",
            "station_id",
            "status",
            "reconciliation_status",
            "imported_at",
            "is_estimated",
        },
        "WARNING",
    ),
}


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
        timespec="seconds"
    )


def _health_row(
    component: str,
    status: str,
    detail: str,
    checked_at: str,
    path: Path | None = None,
    updated_at: str = "",
    estimated_flag: str = "Không áp dụng",
) -> dict[str, str]:
    return {
        "component": component,
        "status": status,
        "detail": detail,
        "checked_at": checked_at,
        "updated_at": updated_at,
        "estimated_flag": estimated_flag,
        "path": str(path) if path is not None else "",
    }


def _estimate_flag_values(value: object) -> list[tuple[str, object]]:
    flags: list[tuple[str, object]] = []
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if str(key).lower() in _ESTIMATE_COLUMNS:
                    flags.append((str(key), item))
                if isinstance(item, (dict, list)):
                    pending.append(item)
        elif isinstance(current, list):
            pending.extend(current)
    return flags


def _flag_value(value: object) -> bool | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "có"}:
            return True
        if normalized in {"false", "0", "no", "n", "không", ""}:
            return False
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isfinite(numeric) and numeric in {0.0, 1.0}:
        return bool(numeric)
    return None


def _inspect_processed_file(path: Path, checked_at: str) -> dict[str, str]:
    try:
        stat = path.stat()
    except OSError as exc:
        return _health_row(
            path.name,
            "ERROR",
            f"Không đọc được metadata tệp: {exc}",
            checked_at,
            path,
        )

    updated_at = _utc_timestamp(stat.st_mtime)
    sample_columns: set[str] = set()
    sample_size: int | None = None
    sample_data: object = None
    try:
        if path.suffix.lower() == ".parquet":
            with closing(duckdb.connect(database=":memory:")) as connection:
                sample = connection.execute(
                    "SELECT * FROM read_parquet(?) LIMIT 1",
                    [str(path)],
                ).df()
            sample_columns = {str(column).lower() for column in sample.columns}
            sample_size = len(sample)
            sample_data = sample.iloc[0].to_dict() if sample_size else None
        elif path.suffix.lower() == ".csv":
            sample = pd.read_csv(path, nrows=1)
            sample_columns = {str(column).lower() for column in sample.columns}
            sample_size = len(sample)
            sample_data = sample.iloc[0].to_dict() if sample_size else None
        elif path.suffix.lower() == ".json":
            sample_data = json.loads(path.read_text(encoding="utf-8"))
            sample_size = (
                len(sample_data)
                if isinstance(sample_data, (dict, list))
                else 1
            )
        else:
            return _health_row(
                path.name,
                "INFO",
                "Định dạng không thuộc nhóm kiểm tra dữ liệu UC014.",
                checked_at,
                path,
                updated_at,
            )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        duckdb.Error,
        ValueError,
        TypeError,
    ) as exc:
        return _health_row(
            path.name,
            "ERROR",
            f"Tệp tồn tại nhưng không đọc được: {type(exc).__name__}: {exc}",
            checked_at,
            path,
            updated_at,
        )

    estimate_flags = _estimate_flag_values(sample_data)
    flagged_columns = sorted(
        _ESTIMATE_COLUMNS.intersection(sample_columns)
        | {name.lower() for name, _ in estimate_flags}
    )
    if estimate_flags:
        flag_states = [_flag_value(value) for _, value in estimate_flags]
        estimated_flag = (
            f"{', '.join(flagged_columns)}: "
            f"{sum(state is True for state in flag_states):,} true, "
            f"{sum(state is False for state in flag_states):,} false, "
            f"{sum(state is None for state in flag_states):,} chưa rõ"
        )
    elif flagged_columns:
        estimated_flag = (
            "Có cột cờ: " + ", ".join(flagged_columns) + "; giá trị mẫu chưa rõ"
        )
    else:
        estimated_flag = "Không có cờ ước tính trong payload/schema đã kiểm tra"
    status = "WARNING" if sample_size == 0 else "OK"
    if sample_size:
        detail = (
            "Payload JSON đọc được."
            if path.suffix.lower() == ".json"
            else "Đọc được mẫu 1 dòng; không quét toàn bộ dữ liệu."
        )
    else:
        detail = "Tệp đọc được nhưng không có bản ghi."
    detail += f" Dung lượng: {stat.st_size / (1024 * 1024):,.2f} MB."
    return _health_row(
        path.name,
        status,
        detail,
        checked_at,
        path,
        updated_at,
        estimated_flag,
    )


def _inspect_model(path: Path, target: str, checked_at: str) -> dict[str, str]:
    if not path.is_file():
        return _health_row(
            f"Model {target}",
            "WARNING",
            "Chưa có model; forecast sẽ dùng fallback nếu có.",
            checked_at,
            path,
            estimated_flag="Không rõ; chưa có bundle",
        )
    try:
        stat = path.stat()
        bundle = load_model(path)
        if not isinstance(bundle, dict):
            raise ValueError("Bundle model không phải dictionary.")
        model = bundle.get("model")
        features = bundle.get("features")
        if not callable(getattr(model, "predict", None)):
            raise ValueError("Bundle thiếu model có phương thức predict().")
        if not isinstance(features, (list, tuple)) or not features:
            raise ValueError("Bundle thiếu danh sách features.")
        bundle_target = str(bundle.get("target", ""))
        if bundle_target and bundle_target != target:
            raise ValueError(
                f"Target trong bundle là {bundle_target}, cần {target}."
            )
        metrics = bundle.get("metrics", bundle.get("evaluation"))
        metrics_valid = False
        if isinstance(metrics, dict):
            try:
                metrics_valid = all(
                    math.isfinite(float(metrics[name])) and float(metrics[name]) >= 0
                    for name in ("MAE", "WAPE")
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                metrics_valid = False
        metric_state = (
            "MAE/WAPE hold-out hợp lệ"
            if metrics_valid
            else "Thiếu hoặc sai metric MAE/WAPE"
        )
        detail = (
            f"Bundle đọc được; {len(features):,} features; {metric_state}."
        )
        trained_until = bundle.get("trained_until")
        if trained_until:
            detail += f" trained_until={trained_until}."
        estimated = bundle.get("is_estimated")
        estimate_label = (
            "Có"
            if estimated is True
            else "Không"
            if estimated is False
            else "Bundle không gắn cờ"
        )
        status = "OK" if metrics_valid else "WARNING"
        return _health_row(
            f"Model {target}",
            status,
            detail,
            checked_at,
            path,
            _utc_timestamp(stat.st_mtime),
            estimate_label,
        )
    except (
        OSError,
        EOFError,
        pickle.UnpicklingError,
        AttributeError,
        ImportError,
        TypeError,
        ValueError,
        RuntimeError,
        MemoryError,
    ) as exc:
        try:
            updated_at = _utc_timestamp(path.stat().st_mtime)
        except OSError:
            updated_at = ""
        return _health_row(
            f"Model {target}",
            "WARNING",
            f"Model chưa dùng được; forecast cần fallback: {type(exc).__name__}: {exc}",
            checked_at,
            path,
            updated_at,
            "Không thể xác định do bundle lỗi",
        )


def _decision_estimate_summary(connection: sqlite3.Connection) -> str:
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(decision_log)").fetchall()
    }
    if not {"created_at", "payload"}.issubset(columns):
        return "Không có cột provenance để kiểm tra"
    order_column = "updated_at" if "updated_at" in columns else "created_at"
    rows = connection.execute(
        f"""
        SELECT payload
        FROM decision_log
        ORDER BY {order_column} DESC, created_at DESC
        LIMIT 1000
        """
    ).fetchall()
    estimated_records = 0
    unflagged_records = 0
    invalid_payloads = 0
    for (raw_payload,) in rows:
        try:
            payload = json.loads(raw_payload) if raw_payload else {}
        except (TypeError, ValueError):
            invalid_payloads += 1
            continue
        if isinstance(payload, list):
            routes = payload
            summary = {}
        elif isinstance(payload, dict):
            routes = payload.get("routes", [])
            summary = payload.get("summary", {})
        else:
            unflagged_records += 1
            continue
        if isinstance(summary, dict) and "estimated_route_count" in summary:
            try:
                estimated_count = float(summary["estimated_route_count"])
            except (TypeError, ValueError, OverflowError):
                estimated_count = 0
            if math.isfinite(estimated_count) and estimated_count > 0:
                estimated_records += 1
            continue
        if not isinstance(routes, list):
            unflagged_records += 1
            continue
        has_estimate_flag = any(
            isinstance(route, dict) and "uses_estimated_data" in route
            for route in routes
        )
        if not has_estimate_flag:
            unflagged_records += 1
            continue
        if any(
            isinstance(route, dict)
            and str(route.get("uses_estimated_data", "")).strip().lower()
            in {"true", "1", "yes", "y", "có"}
            for route in routes
        ):
            estimated_records += 1
    detail = (
        f"{estimated_records:,} quyết định có tuyến ước tính; "
        f"{unflagged_records:,} không gắn cờ ước tính "
        "trong tối đa 1.000 dòng mới nhất"
    )
    if invalid_payloads:
        detail += f"; {invalid_payloads:,} payload không đọc được"
    return detail


def _inspect_database(path: Path, checked_at: str) -> list[dict[str, str]]:
    try:
        stat = path.stat()
    except OSError as exc:
        return [
            _health_row(
                "SQLite",
                "ERROR",
                f"Không truy cập được database: {exc}",
                checked_at,
                path,
            )
        ]

    updated_at = _utc_timestamp(stat.st_mtime)
    uri = path.as_uri() + "?mode=ro"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=2.0)) as connection:
            integrity_results = [
                str(row[0]) for row in connection.execute("PRAGMA quick_check").fetchall()
            ]
            integrity = "; ".join(integrity_results) or "no result"
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            rows = [
                _health_row(
                    "SQLite",
                    "OK"
                    if integrity_results and all(
                        result.lower() == "ok" for result in integrity_results
                    )
                    else "ERROR",
                    f"quick_check={integrity}; {len(tables):,} bảng.",
                    checked_at,
                    path,
                    updated_at,
                )
            ]
            for table, (required_columns, missing_status) in _REQUIRED_TABLES.items():
                if table not in tables:
                    rows.append(
                        _health_row(
                            f"SQLite · {table}",
                            missing_status,
                            "Chưa có bảng cần thiết; UC014 không tự tạo hoặc sửa schema.",
                            checked_at,
                            path,
                        )
                    )
                    continue
                actual_columns = {
                    str(row[1])
                    for row in connection.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                missing_columns = required_columns.difference(actual_columns)
                if missing_columns:
                    rows.append(
                        _health_row(
                            f"SQLite · {table}",
                            "ERROR",
                            "Thiếu cột: " + ", ".join(sorted(missing_columns)),
                            checked_at,
                            path,
                        )
                    )
                    continue
                count = int(
                    connection.execute(
                        f'SELECT COUNT(*) FROM "{table}"'
                    ).fetchone()[0]
                )
                timestamp_column = {
                    "station_status": "timestamp",
                    "decision_log": (
                        "updated_at" if "updated_at" in actual_columns else "created_at"
                    ),
                    "incentive_recommendations": "published_at",
                    "incentive_executions": "imported_at",
                }[table]
                latest = connection.execute(
                    f'SELECT MAX("{timestamp_column}") FROM "{table}"'
                ).fetchone()[0]
                table_status = "OK"
                if count == 0:
                    table_status = "WARNING" if table == "station_status" else "INFO"
                if count == 0 and table == "station_status":
                    detail = "Chưa có snapshot GBFS; nguồn dữ liệu vận hành chưa phát sinh bản ghi."
                elif count == 0 and table == "decision_log":
                    detail = "Chưa phát sinh bản ghi quyết định/audit; đây không phải lỗi hệ thống."
                elif count == 0:
                    detail = "Chưa phát sinh bản ghi vận hành; đây không phải lỗi hệ thống."
                else:
                    detail = f"{count:,} bản ghi."
                if latest:
                    detail += f" Mốc mới nhất: {latest}."
                estimated = "Không có bản ghi" if count == 0 else "Không có cờ ước tính"
                if table in {"incentive_recommendations", "incentive_executions"}:
                    estimated_count = int(
                        connection.execute(
                            f'SELECT COUNT(*) FROM "{table}" WHERE is_estimated = 1'
                        ).fetchone()[0]
                    )
                    unknown_count = int(
                        connection.execute(
                            f'SELECT COUNT(*) FROM "{table}" WHERE is_estimated IS NULL'
                        ).fetchone()[0]
                    )
                    estimated = (
                        f"{estimated_count:,} có cờ ước tính; "
                        f"{unknown_count:,} chưa rõ cờ"
                    )
                elif table == "decision_log" and count:
                    estimated = _decision_estimate_summary(connection)
                elif table == "station_status" and "is_fresh" in actual_columns:
                    fresh_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM station_status WHERE is_fresh = 1"
                        ).fetchone()[0]
                    )
                    estimated = (
                        f"{fresh_count:,}/{count:,} bản ghi được đánh dấu còn mới; "
                        "không có cờ ước tính"
                    )
                if table == "station_status" and count:
                    latest_timestamp = (
                        pd.to_datetime(latest, errors="coerce", utc=True)
                        if latest
                        else pd.NaT
                    )
                    if pd.isna(latest_timestamp):
                        table_status = "ERROR"
                        detail += " Mốc snapshot bị thiếu hoặc không hợp lệ."
                    else:
                        age_seconds = (
                            datetime.now(timezone.utc)
                            - latest_timestamp.to_pydatetime()
                        ).total_seconds()
                        if age_seconds > DEFAULT_MAX_AGE_SECONDS:
                            table_status = "WARNING"
                            detail += (
                                f" Dữ liệu nguồn GBFS cũ: snapshot quá hạn "
                                f"({age_seconds / 60:.0f} phút; "
                                f"giới hạn {DEFAULT_MAX_AGE_SECONDS / 60:.0f} phút)."
                            )
                        elif age_seconds < -MAX_FUTURE_SKEW_SECONDS:
                            table_status = "ERROR"
                            detail += " Mốc snapshot nằm trong tương lai."
                rows.append(
                    _health_row(
                        f"SQLite · {table}",
                        table_status,
                        detail,
                        checked_at,
                        path,
                        str(latest or updated_at),
                        estimated,
                    )
                )

            dedicated_log_tables = {
                table
                for table in tables
                if any(token in table.lower() for token in ("warning", "error", "system_log"))
            }
            if dedicated_log_tables:
                log_counts: dict[str, int] = {}
                for table in sorted(dedicated_log_tables):
                    quoted_table = table.replace('"', '""')
                    log_counts[table] = int(
                        connection.execute(
                            f'SELECT COUNT(*) FROM "{quoted_table}"'
                        ).fetchone()[0]
                    )
                total_log_records = sum(log_counts.values())
                rows.append(
                    _health_row(
                        "SQLite · nhật ký cảnh báo",
                        "OK" if total_log_records else "INFO",
                        (
                            "Có bản ghi nhật ký cảnh báo: "
                            if total_log_records
                            else "Bảng nhật ký đã cấu hình nhưng chưa phát sinh bản ghi vận hành: "
                        )
                        + ", ".join(
                            f"{table} ({log_counts[table]:,})"
                            for table in sorted(log_counts)
                        ),
                        checked_at,
                        path,
                        updated_at,
                    )
                )
            else:
                rows.append(
                    _health_row(
                        "SQLite · nhật ký cảnh báo",
                        "INFO",
                        "Chưa phát sinh bản ghi nhật ký cảnh báo: chưa cấu hình bảng log riêng; "
                        "decision_log là audit quyết định, không phải nhật ký lỗi hệ thống.",
                        checked_at,
                        path,
                        updated_at,
                    )
                )
            return rows
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        return [
            _health_row(
                "SQLite",
                "ERROR",
                f"Không kiểm tra được database: {type(exc).__name__}: {exc}",
                checked_at,
                path,
                updated_at,
            )
        ]


def collect_system_health(
    processed_directory: str | Path,
    models_directory: str | Path,
    database_path: str | Path,
) -> pd.DataFrame:
    """Inspect required local artifacts with bounded reads and read-only SQLite access."""
    processed = Path(processed_directory).expanduser().resolve()
    models = Path(models_directory).expanduser().resolve()
    database = Path(database_path).expanduser().resolve()
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict[str, str]] = []

    if not processed.is_dir():
        rows.append(
            _health_row(
                "Processed directory",
                "ERROR",
                "Thư mục processed/ không tồn tại hoặc không thể truy cập.",
                checked_at,
                processed,
            )
        )
        entries: list[Path] = []
    else:
        try:
            entries = sorted(
                path
                for path in processed.iterdir()
                if path.is_file() and path.suffix.lower() in {".parquet", ".csv", ".json"}
            )
        except OSError as exc:
            entries = []
            rows.append(
                _health_row(
                    "Processed directory",
                    "ERROR",
                    f"Không thể liệt kê artifact: {exc}",
                    checked_at,
                    processed,
                )
            )

    file_checks: dict[str, str] = {}
    for path in entries:
        result = _inspect_processed_file(path, checked_at)
        rows.append(result)
        file_checks[path.name] = result["status"]

    required_artifacts = {
        "Hourly data": (
            "station_hour_cleaned.parquet",
            "station_hour_cleaned.csv",
        ),
        "Station metadata": (
            "station_metadata.parquet",
            "station_metadata.csv",
        ),
    }
    existing_names = {path.name for path in entries}
    for label, alternatives in required_artifacts.items():
        available = [name for name in alternatives if name in existing_names]
        if not available:
            rows.append(
                _health_row(
                    label,
                    "ERROR",
                    "Không tìm thấy artifact bắt buộc: " + " hoặc ".join(alternatives),
                    checked_at,
                    processed,
            )
        )
        else:
            statuses = [file_checks.get(name, "ERROR") for name in available]
            status = (
                "OK"
                if "OK" in statuses
                else "WARNING"
                if "WARNING" in statuses
                else "ERROR"
            )
            latest_path = processed / available[0]
            latest_mtime = 0.0
            for name in available:
                candidate = processed / name
                try:
                    candidate_mtime = candidate.stat().st_mtime
                except OSError:
                    continue
                if candidate_mtime >= latest_mtime:
                    latest_path = candidate
                    latest_mtime = candidate_mtime
            if latest_mtime:
                updated_at = _utc_timestamp(latest_mtime)
            else:
                status = "ERROR"
                detail = "Artifact biến mất hoặc không thể đọc metadata."
            rows.append(
                _health_row(
                    label,
                    status,
                    "Ứng viên hiện có: " + ", ".join(available),
                    checked_at,
                    latest_path,
                    updated_at,
                )
            )

    if not models.is_dir():
        rows.append(
            _health_row(
                "Models directory",
                "WARNING",
                "Không có thư mục models/; forecast sẽ dùng fallback.",
                checked_at,
                models,
            )
        )
    for target in ("pickups", "returns"):
        rows.append(
            _inspect_model(models / f"lgb_{target}.pkl", target, checked_at)
        )

    rows.extend(_inspect_database(database, checked_at))
    return pd.DataFrame(rows, columns=_HEALTH_COLUMNS)
