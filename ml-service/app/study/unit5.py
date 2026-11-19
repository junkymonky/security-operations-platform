"""Unit 5: CART, random forest and XGBoost (report chapter 4) plus extended analysis."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, r2_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .data import CATEGORICAL_ALL, CATEGORICAL_MODEL, NUMERIC, SEED, TARGET
from .unit12 import regression_scores


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_MODEL)],
        remainder="passthrough",
    )


def unit5_models() -> dict:
    """Table 4.1: hyperparameter settings used in the report."""
    return {
        "Decision tree (depth 15)": DecisionTreeRegressor(max_depth=15, random_state=SEED),
        "Random forest": RandomForestRegressor(n_estimators=20, max_depth=8, min_samples_leaf=5,
                                               max_features=0.7, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist",
                                random_state=SEED, n_jobs=-1),
    }


def train_tree_models(X_train, X_test, y_train, y_test) -> dict:
    """Table 4.2 / Figure 4.2, with training scores and feature importance added."""
    results, fitted = [], {}
    for name, model in unit5_models().items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", model)])
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        pred = pipe.predict(X_test)
        results.append({"model": name, **regression_scores(y_test, pred), "train_seconds": elapsed,
                        "train_r2": float(r2_score(y_train, pipe.predict(X_train)))})
        fitted[name] = (pipe, pred)

    xgb_pipe, xgb_pred = fitted["XGBoost"]
    importance = attribute_importance(xgb_pipe)
    rf_importance = attribute_importance(fitted["Random forest"][0])
    return {
        "results": results,
        "n_encoded_columns": int(len(xgb_pipe.named_steps["preprocessor"].get_feature_names_out())),
        "xgb_first_1000": {"actual": y_test[:1000].tolist(), "predicted": xgb_pred[:1000].tolist()},
        "feature_importance": {"XGBoost": importance, "Random forest": rf_importance},
    }


def attribute_importance(pipe: Pipeline) -> list[dict]:
    """Sum one-hot column importances back to their original attribute."""
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    values = pipe.named_steps["model"].feature_importances_
    totals: dict[str, float] = {}
    for name, value in zip(names, values):
        raw = name.split("__", 1)[1]
        attribute = next((c for c in CATEGORICAL_MODEL + NUMERIC if raw == c or raw.startswith(c + "_")), raw)
        totals[attribute] = totals.get(attribute, 0.0) + float(value)
    return sorted([{"attribute": k, "importance": v} for k, v in totals.items()], key=lambda d: -d["importance"])


# ----------------------------------------------------------------- extended analysis
def cross_validation(data: pd.DataFrame, sample: int = 60000) -> list[dict]:
    """5-fold cross-validation on a random sample (report future scope: k-fold evaluation)."""
    sub = data.sample(n=min(sample, len(data)), random_state=SEED)
    X, y = sub[CATEGORICAL_MODEL + NUMERIC], sub[TARGET].to_numpy(dtype=float)
    folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {"Linear regression": LinearRegression(), **unit5_models()}
    out = []
    for name, model in candidates.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=folds, scoring="r2", n_jobs=1)
        out.append({"model": name, "mean_r2": float(scores.mean()), "std_r2": float(scores.std()),
                    "folds": scores.tolist(), "sample_size": int(len(sub))})
    return out


def log_target(X_train, X_test, y_train, y_test) -> list[dict]:
    """Fit on log(fare) and score back in rupees (report future scope: log transformation)."""
    out = []
    for name, model in {"Linear regression": LinearRegression(), "XGBoost": unit5_models()["XGBoost"]}.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", model)])
        pipe.fit(X_train, np.log(y_train))
        pred = np.exp(pipe.predict(X_test))
        out.append({"model": f"{name} on log(fare)", **regression_scores(y_test, pred)})
    return out


def clustering_comparison(data: pd.DataFrame, sample: int = 6000, k: int = 6) -> list[dict]:
    """K-means vs Gaussian mixture vs Ward hierarchical clustering (report future scope)."""
    sub = data.sample(n=min(sample, len(data)), random_state=SEED)
    label = (sub[TARGET] > sub[TARGET].median()).astype(int).to_numpy()
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_ALL),
                             ("num", StandardScaler(), NUMERIC)])
    Z = PCA(n_components=20, random_state=SEED).fit_transform(pre.fit_transform(sub))
    methods = {
        "K-means": KMeans(n_clusters=k, random_state=SEED, n_init=10),
        "Gaussian mixture (EM)": GaussianMixture(n_components=k, covariance_type="diag", random_state=SEED),
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward"),
    }
    out = []
    for name, method in methods.items():
        start = time.perf_counter()
        clusters = method.fit_predict(Z)
        mapped = np.array([round(label[clusters == c].mean()) if (clusters == c).any() else 0 for c in range(k)])
        out.append({"method": name, "silhouette": float(silhouette_score(Z, clusters)),
                    "accuracy": float((mapped[clusters] == label).mean()),
                    "adjusted_rand": float(adjusted_rand_score(label, clusters)),
                    "seconds": time.perf_counter() - start})
    return out
