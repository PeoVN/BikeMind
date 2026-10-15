from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import pandas as pd


def normalize_station_id(value: object) -> str:
    """Keep station keys identical when CSV parsers return numeric or text IDs."""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


REQUIRED_COLUMNS = {
    "starttime",
    "stoptime",
    "start station id",
    "start station name",
    "start station latitude",
    "start station longitude",
    "end station id",
    "end station name",
    "end station latitude",
    "end station longitude",
}


@dataclass(frozen=True)
class IngestionReport:
    files_scanned: int
    files_loaded: int
    duplicate_files: int
    rows_loaded: int
    duplicate_rows_removed: int
    invalid_files: int


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_csv_files(root: str | Path) -> list[Path]:
    """Recursively return CSV files while excluding macOS metadata folders."""
    root_path = Path(root)
    return sorted(
        path for path in root_path.rglob("*.csv")
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    )


def load_trip_data(root: str | Path) -> tuple[pd.DataFrame, IngestionReport]:
    """Load valid trip CSVs, deduplicate files by MD5 and rows by full record."""
    files = discover_csv_files(root)
    seen_hashes: set[str] = set()
    frames: list[pd.DataFrame] = []
    invalid_files = 0
    duplicate_files = 0

    for path in files:
        file_hash = _md5(path)
        if file_hash in seen_hashes:
            duplicate_files += 1
            continue
        seen_hashes.add(file_hash)
        try:
            frame = pd.read_csv(path, low_memory=False)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError):
            invalid_files += 1
            continue
        if not REQUIRED_COLUMNS.issubset(frame.columns):
            invalid_files += 1
            continue
        frames.append(frame)

    if not frames:
        empty = pd.DataFrame()
        return empty, IngestionReport(len(files), 0, duplicate_files, 0, 0, invalid_files)

    combined = pd.concat(frames, ignore_index=True)
    rows_loaded = len(combined)
    combined = combined.drop_duplicates().copy()
    combined["starttime"] = pd.to_datetime(combined["starttime"], errors="coerce")
    combined["stoptime"] = pd.to_datetime(combined["stoptime"], errors="coerce")
    combined = combined.dropna(subset=["starttime", "stoptime", "start station id", "end station id"])
    report = IngestionReport(
        files_scanned=len(files),
        files_loaded=len(frames),
        duplicate_files=duplicate_files,
        rows_loaded=rows_loaded,
        duplicate_rows_removed=rows_loaded - len(combined),
        invalid_files=invalid_files,
    )
    return combined, report


def build_station_hour_grid(trips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a continuous station-hour demand grid and station metadata."""
    if trips.empty:
        return pd.DataFrame(), pd.DataFrame()
    start = min(trips["starttime"].min(), trips["stoptime"].min()).floor("h")
    end = max(trips["starttime"].max(), trips["stoptime"].max()).ceil("h")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    station_ids = sorted(
        set(trips["start station id"].map(normalize_station_id))
        | set(trips["end station id"].map(normalize_station_id))
    )
    index = pd.MultiIndex.from_product([station_ids, hours], names=["station_id", "timestamp"])
    pickups = trips.assign(
        station_id=trips["start station id"].map(normalize_station_id),
        timestamp=trips["starttime"].dt.floor("h"),
    ).groupby(["station_id", "timestamp"]).size().rename("pickups")
    returns = trips.assign(
        station_id=trips["end station id"].map(normalize_station_id),
        timestamp=trips["stoptime"].dt.floor("h"),
    ).groupby(["station_id", "timestamp"]).size().rename("returns")
    grid = pd.concat([pickups, returns], axis=1).reindex(index, fill_value=0).fillna(0).reset_index()
    grid["pickups"] = grid["pickups"].astype("int32")
    grid["returns"] = grid["returns"].astype("int32")
    starts = trips[[
        "start station id", "start station name", "start station latitude", "start station longitude"
    ]].rename(columns={
        "start station id": "station_id", "start station name": "name",
        "start station latitude": "lat", "start station longitude": "lon",
    })
    ends = trips[[
        "end station id", "end station name", "end station latitude", "end station longitude"
    ]].rename(columns={
        "end station id": "station_id", "end station name": "name",
        "end station latitude": "lat", "end station longitude": "lon",
    })
    metadata = pd.concat([starts, ends], ignore_index=True).drop_duplicates("station_id")
    metadata["station_id"] = metadata["station_id"].map(normalize_station_id)
    return grid, metadata
