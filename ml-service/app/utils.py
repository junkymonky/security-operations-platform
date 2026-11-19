"""Small helpers shared by ML modules."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any, digits: int = 10) -> Any:
    """Recursively convert numpy/pandas values into JSON-safe Python types.

    NaN/inf become None; floats keep `digits` significant digits (not decimal
    places, so tiny probabilities and likelihoods survive) to keep payloads small.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not math.isfinite(f):
            return None
        return float(f"{f:.{digits}g}")
    if isinstance(value, dict):
        return {str(k): to_jsonable(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), digits)
    if isinstance(value, (pd.Series, pd.Index)):
        return to_jsonable(value.tolist(), digits)
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"), digits)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value) if np.isscalar(value) else False:
        return None
    return str(value)


def subsample_indices(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    """Deterministic subsample of row indices for plotting large datasets."""
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))
