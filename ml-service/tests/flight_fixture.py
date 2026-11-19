"""TEST-ONLY stand-in with the same 13 columns as Kaggle's Cleaned_dataset.csv.

Used to exercise the flight-fare study pipeline end to end. Its numbers mean nothing and must
never be presented as results; the real dataset goes in ml-service/data/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AIRLINES = ["Vistara", "Air India", "Indigo", "AirAsia", "GO FIRST", "SpiceJet", "AkasaAir", "AllianceAir", "StarAir"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai", "Ahmedabad"]
SLOTS = ["Before 6 AM", "6 AM - 12 PM", "12 PM - 6 PM", "After 6 PM"]
CLASSES = ["Economy", "Premium Economy", "Business", "First"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def make(n: int = 30000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    airline = rng.choice(AIRLINES, n, p=[0.5, 0.2, 0.12, 0.05, 0.04, 0.04, 0.02, 0.02, 0.01])
    premium = np.isin(airline, ["Vistara", "Air India"])
    cls = np.where(premium, rng.choice(CLASSES, n, p=[0.45, 0.2, 0.3, 0.05]), "Economy")
    src = rng.choice(CITIES, n)
    dst = np.array([rng.choice([c for c in CITIES if c != s]) for s in src])
    stops = rng.choice(["non-stop", "1-stop", "2+-stop"], n, p=[0.12, 0.8, 0.08])
    duration = np.round(np.clip(rng.gamma(3, 4, n) + (stops != "non-stop") * 4, 0.75, 43.5), 2)
    days_left = rng.integers(1, 51, n)
    date_idx = rng.integers(0, 50, n)
    dates = (pd.Timestamp("2023-01-16") + pd.to_timedelta(date_idx, "D"))
    flight = np.array([f"{a[:2].upper()}-{rng.integers(100, 999)}" for a in airline])
    base = {"Economy": 6000, "Premium Economy": 14000, "Business": 45000, "First": 60000}
    fare = (np.vectorize(base.get)(cls) * np.where(premium, 1.3, 1.0) * (1 + 0.6 * np.exp(-days_left / 8))
            + duration * 120 + rng.normal(0, 2500, n))
    df = pd.DataFrame({
        "Date_of_journey": dates.strftime("%Y-%m-%d"), "Journey_day": dates.day_name(), "Airline": airline,
        "Flight_code": flight, "Class": cls, "Source": src, "Departure": rng.choice(SLOTS, n),
        "Total_stops": stops, "Arrival": rng.choice(SLOTS, n), "Destination": dst,
        "Duration_in_hours": duration, "Days_left": days_left, "Fare": np.clip(fare, 1307, None).round().astype(int),
    })
    # A sequence structure: repeat some flights with different days_left, plus a few exact duplicates.
    repeats = df.sample(frac=0.3, random_state=seed).assign(Days_left=lambda d: np.clip(d["Days_left"] - 3, 1, 50))
    return pd.concat([df, repeats, df.head(200)], ignore_index=True)


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "flight_fixture.csv")
    make().to_csv(out, index=False)
    print(f"Wrote {out}")
