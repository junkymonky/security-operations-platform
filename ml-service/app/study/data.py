"""Flight-fare dataset loading (Kaggle "Airfare ML: Predicting Flight Fares", Cleaned_dataset.csv)."""
from __future__ import annotations

import os
import zipfile
from pathlib import Path

import pandas as pd

from ..config import SERVICE_ROOT
from ..errors import NotFound, ValidationFailed

COLUMNS = [
    "Date_of_journey", "Journey_day", "Airline", "Flight_code", "Class", "Source",
    "Departure", "Total_stops", "Arrival", "Destination", "Duration_in_hours", "Days_left", "Fare",
]
CATEGORICAL_ALL = ["Date_of_journey", "Journey_day", "Airline", "Flight_code", "Class", "Source",
                   "Departure", "Total_stops", "Arrival", "Destination"]
# Unit 5 / common pipeline: drop Date_of_journey and Flight_code (report section 4.2)
CATEGORICAL_MODEL = ["Journey_day", "Airline", "Class", "Source", "Departure", "Total_stops", "Arrival", "Destination"]
NUMERIC = ["Duration_in_hours", "Days_left"]
TARGET = "Fare"
SEED = 42

DATA_DIR = SERVICE_ROOT / "data"
CANDIDATES = ["Cleaned_dataset.csv", "Cleaned_dataset.csv.zip", "archive.zip",
              "airfare-ml-predicting-flight-fares.zip"]


def find_dataset(explicit: str | None = None) -> Path:
    """Locate the dataset: explicit path, $FLIGHT_FARE_DATA, then ml-service/data/."""
    for raw in (explicit, os.environ.get("FLIGHT_FARE_DATA")):
        if raw:
            p = Path(raw).expanduser()
            if p.is_file():
                return p
            raise NotFound(f"Flight-fare dataset not found at {p}")
    for name in CANDIDATES:
        p = DATA_DIR / name
        if p.is_file():
            return p
    raise NotFound(
        "Flight-fare dataset not found. Download Cleaned_dataset.csv from "
        "kaggle.com/datasets/yashdharme36/airfare-ml-predicting-flight-fares and place it in "
        f"{DATA_DIR}"
    )


def load(path: Path) -> pd.DataFrame:
    """Read the CSV (directly or from a zip) with memory-efficient dtypes and validate the schema."""
    dtypes = {c: "category" for c in CATEGORICAL_ALL}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise ValidationFailed(f"No CSV file inside {path.name}")
            member = next((m for m in members if Path(m).name == "Cleaned_dataset.csv"), members[0])
            with zf.open(member) as fh:
                df = pd.read_csv(fh, dtype=dtypes)
    else:
        df = pd.read_csv(path, dtype=dtypes)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValidationFailed(f"Dataset is missing column(s): {', '.join(missing)}")
    df = df[COLUMNS]
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    df["Duration_in_hours"] = pd.to_numeric(df["Duration_in_hours"], errors="coerce").astype("float32")
    df["Days_left"] = pd.to_numeric(df["Days_left"], errors="coerce").astype("int16")
    df = df.dropna(subset=["Fare"])
    return df
