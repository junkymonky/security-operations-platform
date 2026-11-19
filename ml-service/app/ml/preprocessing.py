"""Feature/target preparation shared by supervised and unsupervised modules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from ..errors import ValidationFailed

MAX_CATEGORIES = 50


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: np.ndarray | None
    features: list[str]
    numeric: list[str]
    categorical: list[str]
    classes: list[str] | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for f in features:
        if pd.api.types.is_numeric_dtype(df[f]) and not pd.api.types.is_bool_dtype(df[f]):
            numeric.append(f)
        else:
            if df[f].nunique(dropna=True) > MAX_CATEGORIES:
                raise ValidationFailed(
                    f"Categorical feature '{f}' has more than {MAX_CATEGORIES} distinct values; "
                    "drop it or encode it first."
                )
            categorical.append(f)
    return numeric, categorical


def make_preprocessor(
    numeric: list[str], categorical: list[str], scale: str = "standard", impute: str = "mean"
) -> ColumnTransformer:
    steps: list = [("impute", SimpleImputer(strategy=impute if impute in ("mean", "median") else "mean"))]
    if scale == "standard":
        steps.append(("scale", StandardScaler()))
    elif scale == "minmax":
        steps.append(("scale", MinMaxScaler()))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(steps), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def default_features(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


def prepare_supervised(
    df: pd.DataFrame,
    features: list[str] | None,
    target: str,
    task: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> PreparedData:
    if target not in df.columns:
        raise ValidationFailed(f"Target column '{target}' not found.")
    features = features or default_features(df, exclude=[target])
    features = [f for f in features if f != target]
    if not features:
        raise ValidationFailed("Select at least one feature column.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValidationFailed(f"Unknown feature column(s): {', '.join(missing)}")

    data = df[features + [target]].dropna(subset=[target])
    notes = []
    if len(data) < len(df):
        notes.append(f"Dropped {len(df) - len(data)} rows with missing target")
    if len(data) < 10:
        raise ValidationFailed("Need at least 10 rows with a non-missing target.")
    numeric, categorical = split_feature_types(data, features)

    classes = None
    if task == "classification":
        y_raw = data[target].astype(str)
        classes = sorted(y_raw.unique().tolist())
        if len(classes) < 2:
            raise ValidationFailed("Classification target needs at least two classes.")
        if len(classes) > 30:
            raise ValidationFailed(
                f"Target '{target}' has {len(classes)} classes; use regression for continuous targets."
            )
        mapping = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(mapping).to_numpy()
    else:
        if not pd.api.types.is_numeric_dtype(data[target]):
            raise ValidationFailed(f"Regression target '{target}' must be numeric.")
        y = data[target].astype(float).to_numpy()

    X = data[features]
    if not 0.05 <= test_size <= 0.5:
        raise ValidationFailed("test_size must be between 0.05 and 0.5.")
    stratify = None
    if task == "classification":
        counts = np.bincount(y)
        if counts.min() >= 2:
            stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return PreparedData(
        X=X,
        y=y,
        features=features,
        numeric=numeric,
        categorical=categorical,
        classes=classes,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        notes=notes,
    )


def prepare_unsupervised(
    df: pd.DataFrame, features: list[str] | None, scale: str = "standard", max_rows: int = 5000
) -> tuple[np.ndarray, list[str], pd.Index, Pipeline]:
    """Return a numeric matrix (scaled, one-hot encoded) for clustering / PCA."""
    features = features or default_features(df)
    if not features:
        raise ValidationFailed("Select at least one feature column.")
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValidationFailed(f"Unknown feature column(s): {', '.join(missing)}")
    data = df[features]
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=0).sort_index()
    numeric, categorical = split_feature_types(data, features)
    prep = make_preprocessor(numeric, categorical, scale=scale)
    X = prep.fit_transform(data)
    names = list(prep.get_feature_names_out())
    if X.shape[0] < 3:
        raise ValidationFailed("Need at least 3 rows.")
    return np.asarray(X, dtype=float), names, data.index, prep
