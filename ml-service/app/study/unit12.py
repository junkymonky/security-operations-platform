"""Units 1 and 2: data summary, EDA and regression baselines (report chapter 2)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

from .data import CATEGORICAL_ALL, CATEGORICAL_MODEL, NUMERIC, SEED, TARGET


def regression_scores(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def box_stats(values: pd.Series, max_fliers: int = 300) -> dict:
    """Box-plot statistics (Tukey 1.5 IQR whiskers) plus a capped sample of outliers."""
    v = values.to_numpy(dtype=float)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
    fliers = np.sort(v[(v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)])
    if len(fliers) > max_fliers:
        fliers = fliers[np.linspace(0, len(fliers) - 1, max_fliers).astype(int)]
    return {"q1": q1, "median": med, "q3": q3, "whislo": float(inside.min()), "whishi": float(inside.max()),
            "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "count": int(len(v)),
            "fliers": fliers.tolist()}


# ----------------------------------------------------------------- Unit 1: summary
def describe_dataset(df: pd.DataFrame) -> dict:
    """Tables 2.1 and 2.2: attribute summary and descriptive statistics."""
    attributes = []
    for col in df.columns:
        s = df[col]
        if col in CATEGORICAL_ALL:
            top = s.value_counts().head(1)
            attributes.append({"attribute": col, "type": "Categorical", "distinct": int(s.nunique()),
                               "most_frequent": f"{top.index[0]} ({int(top.iloc[0]):,})"})
        else:
            attributes.append({"attribute": col, "type": "Float" if s.dtype.kind == "f" else "Integer",
                               "distinct": int(s.nunique()),
                               "most_frequent": f"Range {s.min():,.2f} to {s.max():,.2f}"})
    numeric = df[NUMERIC + [TARGET]].astype(float)
    stats = numeric.describe(percentiles=[0.25, 0.5, 0.75]).T
    fare = numeric[TARGET]
    # Share of fare variance explained by each attribute on its own: correlation ratio (eta²) for
    # categoricals, squared Pearson correlation for numerics.
    total_var = float(((fare - fare.mean()) ** 2).sum())
    explained = {}
    for col in CATEGORICAL_ALL:
        if col in ("Date_of_journey", "Flight_code"):
            continue
        g = fare.groupby(df[col], observed=True)
        explained[col] = float((g.count() * (g.mean() - fare.mean()) ** 2).sum() / total_var)
    for col in NUMERIC:
        explained[col] = float(np.corrcoef(numeric[col], fare)[0, 1] ** 2)
    return {
        "fare_skewness": float(fare.skew()),
        "variance_explained_alone": dict(sorted(explained.items(), key=lambda kv: -kv[1])),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "attributes": attributes,
        "numeric_stats": {
            col: {"mean": stats.loc[col, "mean"], "std": stats.loc[col, "std"], "min": stats.loc[col, "min"],
                  "q25": stats.loc[col, "25%"], "median": stats.loc[col, "50%"], "q75": stats.loc[col, "75%"],
                  "max": stats.loc[col, "max"]}
            for col in stats.index
        },
        "share": {
            "airline_top": {k: int(v) for k, v in df["Airline"].value_counts().items()},
            "class": {k: int(v) for k, v in df["Class"].value_counts().items()},
            "stops": {k: int(v) for k, v in df["Total_stops"].value_counts().items()},
        },
    }


def eda_data(df: pd.DataFrame, sample: int = 20000) -> dict:
    """Data behind figures 2.1 – 2.4 (fare histogram, scatter samples, fare by airline)."""
    fares = df[TARGET].to_numpy(dtype=float)
    counts, edges = np.histogram(fares, bins=30)
    pts = df.sample(n=min(sample, len(df)), random_state=SEED)
    by_airline = {str(a): box_stats(g) for a, g in df.groupby("Airline", observed=True)[TARGET]}
    return {
        "fare_hist": {"counts": counts.tolist(), "edges": edges.tolist()},
        "scatter": {"duration": pts["Duration_in_hours"].astype(float).tolist(),
                    "days_left": pts["Days_left"].astype(int).tolist(),
                    "fare": pts[TARGET].astype(float).tolist()},
        "fare_by_airline": by_airline,
    }


# ----------------------------------------------------------------- Unit 2: first experiment
def label_encoded_regression(df: pd.DataFrame) -> dict:
    """Table 2.3: label-encoded features, all records, 80:20 split (random_state 42)."""
    X = pd.DataFrame({c: df[c].cat.codes for c in CATEGORICAL_ALL})
    X[NUMERIC] = df[NUMERIC].astype(float)
    X = X.to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Linear regression": LinearRegression(),
        "Polynomial regression (degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "Ridge regression (alpha = 1)": Ridge(alpha=1.0),
    }
    results = []
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        results.append({"model": name, **regression_scores(y_test, model.predict(X_test)),
                        "train_seconds": elapsed})
    return {"n_train": len(y_train), "n_test": len(y_test), "results": results}


# ----------------------------------------------------------------- common pipeline (Units 2 and 5)
def common_split(df: pd.DataFrame):
    """Duplicate removal + the 80:20 split shared by Units 2, 3 and 5."""
    data = df.drop_duplicates().reset_index(drop=True)
    X = data[CATEGORICAL_MODEL + NUMERIC]
    y = data[TARGET].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    return data, X_train, X_test, y_train, y_test


def one_hot_regression(X_train, X_test, y_train, y_test) -> dict:
    """Table 2.4: one-hot encoded categoricals + numeric attributes."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    cat_train = encoder.fit_transform(X_train[CATEGORICAL_MODEL])
    cat_test = encoder.transform(X_test[CATEGORICAL_MODEL])
    num_train = X_train[NUMERIC].to_numpy(dtype=float)
    num_test = X_test[NUMERIC].to_numpy(dtype=float)

    # Degree-2 expansion of the numeric attributes (squares and their product); categorical
    # indicators stay linear so the design matrix fits in memory on 356k rows.
    poly = PolynomialFeatures(degree=2, include_bias=False)
    design = {
        "linear": (sparse.hstack([cat_train, num_train]).tocsr(), sparse.hstack([cat_test, num_test]).tocsr()),
        "poly": (sparse.hstack([cat_train, poly.fit_transform(num_train)]).tocsr(),
                 sparse.hstack([cat_test, poly.transform(num_test)]).tocsr()),
    }
    models = [
        ("Linear regression", LinearRegression(), "linear"),
        ("Polynomial regression (degree 2)", LinearRegression(), "poly"),
        ("Ridge regression (alpha = 1)", Ridge(alpha=1.0), "linear"),
    ]
    results, sample_pred = [], None
    for name, model, key in models:
        Xtr, Xte = design[key]
        start = time.perf_counter()
        model.fit(Xtr, y_train)
        elapsed = time.perf_counter() - start
        pred = model.predict(Xte)
        results.append({"model": name, **regression_scores(y_test, pred), "train_seconds": elapsed,
                        "train_r2": float(r2_score(y_train, model.predict(Xtr)))})
        if key == "linear" and sample_pred is None:
            sample_pred = pred[:1000].tolist()
    return {"n_encoded_features": int(cat_train.shape[1] + 2), "results": results,
            "linear_sample_pred": sample_pred}
