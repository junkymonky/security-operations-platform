"""Unit 1 — polynomial curve fitting (Bishop §1.1).

Fits y = Σ w_j x^j by (regularised) least squares, solved in closed form
w = (λI + ΦᵀΦ)⁻¹Φᵀt on a standardised x to keep the Vandermonde matrix well conditioned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..errors import ValidationFailed
from .metrics import regression_metrics


def _design(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, degree + 1, increasing=True)


def _fit(x: np.ndarray, t: np.ndarray, degree: int, lam: float) -> np.ndarray:
    Phi = _design(x, degree)
    reg = lam * np.eye(degree + 1)
    reg[0, 0] = 0.0  # do not penalise the bias term
    return np.linalg.solve(Phi.T @ Phi + reg + 1e-12 * np.eye(degree + 1), Phi.T @ t)


def fit_polynomial(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    degree: int,
    lam: float = 0.0,
    test_size: float = 0.3,
    max_degree_scan: int = 12,
    random_state: int = 42,
) -> dict:
    for c in (x_col, y_col):
        if c not in df.columns:
            raise ValidationFailed(f"Column '{c}' not found.")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValidationFailed(f"Column '{c}' must be numeric.")
    if not 0 <= degree <= 15:
        raise ValidationFailed("degree must be between 0 and 15.")
    if lam < 0:
        raise ValidationFailed("lambda must be non-negative.")
    data = df[[x_col, y_col]].dropna()
    if len(data) < 6:
        raise ValidationFailed("Need at least 6 complete rows.")
    x_raw = data[x_col].to_numpy(float)
    t = data[y_col].to_numpy(float)
    mu, sd = x_raw.mean(), x_raw.std() or 1.0
    x = (x_raw - mu) / sd

    x_tr, x_te, t_tr, t_te = train_test_split(x, t, test_size=test_size, random_state=random_state)
    if degree + 1 > len(x_tr):
        raise ValidationFailed(f"Degree {degree} needs at least {degree + 1} training points.")

    w = _fit(x_tr, t_tr, degree, lam)
    pred_tr, pred_te = _design(x_tr, degree) @ w, _design(x_te, degree) @ w

    grid_std = np.linspace(x.min(), x.max(), 200)
    curve = _design(grid_std, degree) @ w

    # Error vs model complexity: the classic over-fitting curve.
    scan = []
    for m in range(0, min(max_degree_scan, len(x_tr) - 1) + 1):
        wm = _fit(x_tr, t_tr, m, lam)
        e_tr = np.sqrt(np.mean((_design(x_tr, m) @ wm - t_tr) ** 2))
        e_te = np.sqrt(np.mean((_design(x_te, m) @ wm - t_te) ** 2))
        scan.append({"degree": m, "train_rmse": e_tr, "test_rmse": e_te, "weight_norm": float(np.linalg.norm(wm))})

    # Effect of λ at the chosen degree.
    lam_scan = []
    for log_lam in np.arange(-8, 3, 1.0):
        wl = _fit(x_tr, t_tr, degree, float(np.exp(log_lam)))
        lam_scan.append(
            {
                "ln_lambda": log_lam,
                "train_rmse": np.sqrt(np.mean((_design(x_tr, degree) @ wl - t_tr) ** 2)),
                "test_rmse": np.sqrt(np.mean((_design(x_te, degree) @ wl - t_te) ** 2)),
            }
        )

    max_points = 600
    rng = np.random.default_rng(0)
    def pts(xs, ts, split):
        idx = np.arange(len(xs)) if len(xs) <= max_points else rng.choice(len(xs), max_points, replace=False)
        return [{"x": xs[i] * sd + mu, "y": ts[i], "split": split} for i in idx]

    return {
        "x_column": x_col,
        "y_column": y_col,
        "degree": degree,
        "lambda": lam,
        "coefficients_standardized_x": w.tolist(),
        "x_standardization": {"mean": mu, "std": sd},
        "train": regression_metrics(t_tr, pred_tr),
        "test": regression_metrics(t_te, pred_te),
        "points": pts(x_tr, t_tr, "train") + pts(x_te, t_te, "test"),
        "curve": [{"x": g * sd + mu, "y": c} for g, c in zip(grid_std, curve)],
        "degree_scan": scan,
        "lambda_scan": lam_scan,
        "residuals": [{"x": xi * sd + mu, "residual": r} for xi, r in zip(x_te, t_te - pred_te)][:max_points],
    }
