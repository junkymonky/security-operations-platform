"""Unit 3: K-means clustering with PCA visualisation (report sections 3.2 – 3.6)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_ALL, NUMERIC, SEED, TARGET
from .unit12 import regression_scores

K_VALUES = [2, 3, 4, 5, 6, 8, 10]


# ----------------------------------------------------------------- exploratory K-means
def exploratory_kmeans(df: pd.DataFrame, sample: int = 15000) -> dict:
    """Table 3.1 / Figure 3.1: K = 6 on standardised duration and days left (all records)."""
    features = df[NUMERIC].astype(float)
    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=6, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(scaled)

    summary = (df.assign(Cluster=labels)
                 .groupby("Cluster")
                 .agg(flights=(TARGET, "size"), mean_fare=(TARGET, "mean"),
                      mean_duration=("Duration_in_hours", "mean"), mean_days_left=("Days_left", "mean")))
    idx = np.random.default_rng(SEED).choice(len(df), size=min(sample, len(df)), replace=False)
    return {
        "summary": [{"cluster": int(k), **{c: float(v) for c, v in row.items()}} for k, row in summary.iterrows()],
        "points": {"duration": features.iloc[idx, 0].tolist(), "fare": df[TARGET].iloc[idx].astype(float).tolist(),
                   "cluster": labels[idx].tolist()},
    }


def cluster_label_regression(df: pd.DataFrame) -> dict:
    """Table 3.2 / Figure 3.2: linear regression on duration, days left and a K = 3 cluster label."""
    scaled = StandardScaler().fit_transform(df[NUMERIC].astype(float))
    cluster = KMeans(n_clusters=3, random_state=SEED, n_init=10).fit_predict(scaled)
    X = np.column_stack([df[NUMERIC].to_numpy(dtype=float), cluster])
    y = df[TARGET].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = regression_scores(y_test, pred)
    scores["mse"] = scores["rmse"] ** 2
    idx = np.random.default_rng(SEED).choice(len(y_test), size=min(8000, len(y_test)), replace=False)
    return {**scores, "points": {"actual": y_test[idx].tolist(), "predicted": pred[idx].tolist()}}


# ----------------------------------------------------------------- full pipeline
def map_clusters_to_classes(clusters: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Each cluster takes the majority fare class of its training members."""
    mapping = np.zeros(k, dtype=int)
    for c in range(k):
        members = labels[clusters == c]
        mapping[c] = int(np.round(members.mean())) if len(members) else 0
    return mapping


def kmeans_pipeline(data: pd.DataFrame, progress=lambda msg: None) -> dict:
    """Sections 3.5 – 3.6: one-hot + scaling, K sweep on a validation split, final test evaluation."""
    X = data[CATEGORICAL_ALL + NUMERIC]
    y_fare = data[TARGET].to_numpy(dtype=float)
    X_train, X_test, fare_train, fare_test = train_test_split(X, y_fare, test_size=0.2, random_state=SEED)
    median = float(np.median(fare_train))
    y_train = (fare_train > median).astype(int)  # 1 = Higher fare
    y_test = (fare_test > median).astype(int)

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_ALL),
        ("num", StandardScaler(), NUMERIC),
    ], sparse_threshold=1.0)
    Z_train = preprocess.fit_transform(X_train).tocsr()
    Z_test = preprocess.transform(X_test).tocsr()

    Z_fit, Z_val, y_fit, y_val = train_test_split(Z_train, y_train, test_size=0.2, random_state=SEED)
    rng = np.random.default_rng(SEED)
    sil_idx = rng.choice(Z_val.shape[0], size=min(10000, Z_val.shape[0]), replace=False)

    sweep = []
    for k in K_VALUES:
        start = time.perf_counter()
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z_fit)
        mapping = map_clusters_to_classes(model.labels_, y_fit, k)
        val_clusters = model.predict(Z_val)
        sweep.append({
            "k": k,
            "validation_accuracy": float(accuracy_score(y_val, mapping[val_clusters])),
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(Z_val[sil_idx], val_clusters[sil_idx])),
            "seconds": time.perf_counter() - start,
        })
        progress(f"K-means K={k}: validation accuracy {sweep[-1]['validation_accuracy']:.4f}")

    best_k = max(sweep, key=lambda r: r["validation_accuracy"])["k"]
    final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(Z_train)
    mapping = map_clusters_to_classes(final.labels_, y_train, best_k)
    test_pred = mapping[final.predict(Z_test)]
    report = classification_report(y_test, test_pred, target_names=["Lower Fare", "Higher Fare"], output_dict=True)

    # Table 3.6 is computed on all de-duplicated records for analysis only.
    all_clusters = final.predict(preprocess.transform(X).tocsr())
    profile = (data.assign(Cluster=all_clusters)
                   .groupby("Cluster")
                   .agg(flights=(TARGET, "size"), mean_fare=(TARGET, "mean"),
                        mean_duration=("Duration_in_hours", "mean"), mean_days_left=("Days_left", "mean")))
    clusters_table = [{"cluster": int(c), **{k: float(v) for k, v in row.items()},
                       "mapped_class": "Higher Fare" if mapping[c] else "Lower Fare"} for c, row in profile.iterrows()]

    pca_idx = rng.choice(Z_train.shape[0], size=min(10000, Z_train.shape[0]), replace=False)
    sample = Z_train[pca_idx].toarray()
    coords = PCA(n_components=2, random_state=SEED).fit_transform(sample)
    sil_sample = rng.choice(Z_test.shape[0], size=min(10000, Z_test.shape[0]), replace=False)

    return {
        "n_features": int(Z_train.shape[1]),
        "n_train": int(Z_train.shape[0]), "n_validation": int(Z_val.shape[0]), "n_test": int(Z_test.shape[0]),
        "median_fare": median,
        "sweep": sweep,
        "best_k": best_k,
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_silhouette": float(silhouette_score(Z_test[sil_sample], final.predict(Z_test[sil_sample]))),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "clusters": clusters_table,
        "pca": {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(), "cluster": final.labels_[pca_idx].tolist()},
    }
