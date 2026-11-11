from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist a pipeline table as Parquet and fail with an actionable message."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(destination, index=False)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support is unavailable. Install the project dependencies "
            "(pyarrow) before writing pipeline artifacts."
        ) from exc
    return destination


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_model(model: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def load_model(path: str | Path) -> Any:
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return destination


def load_metrics(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
