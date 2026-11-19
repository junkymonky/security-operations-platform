"""Clustering and dimensionality-reduction orchestration (Unit 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..errors import ValidationFailed
from ..ml import clustering, decomposition
from ..ml.metrics import clustering_metrics
from ..ml.preprocessing import default_features, prepare_unsupervised, split_feature_types
from ..utils import subsample_indices, to_jsonable
from . import datasets, experiments

MAX_POINTS = 1500


def _profiles(df: pd.DataFrame, index: pd.Index, labels: np.ndarray, features: list[str]) -> list[dict]:
    sub = df.loc[index, [f for f in features if pd.api.types.is_numeric_dtype(df[f])]].copy()
    sub["_cluster"] = labels
    out = []
    for k, g in sub.groupby("_cluster"):
        out.append({"cluster": int(k), "size": int(len(g)),
                    "means": {c: float(g[c].mean()) for c in g.columns if c != "_cluster"}})
    return out


def _true_labels(df: pd.DataFrame, index: pd.Index, column: str | None):
    if not column:
        return None
    if column not in df.columns:
        raise ValidationFailed(f"Label column '{column}' not found.")
    return df.loc[index, column].astype(str).to_numpy()


def _points(Z: np.ndarray, labels: np.ndarray, truth, extra: np.ndarray | None = None) -> list[dict]:
    idx = subsample_indices(len(Z), MAX_POINTS)
    pts = []
    for i in idx:
        p = {"x": Z[i, 0], "y": Z[i, 1], "cluster": int(labels[i])}
        if truth is not None:
            p["label"] = truth[i]
        if extra is not None:
            p["confidence"] = float(extra[i].max())
            p["probabilities"] = extra[i].round(4).tolist()
        pts.append(p)
    return pts


def cluster(dataset_id: str, req: dict) -> dict:
    method = req["method"]
    df = datasets.load_df(dataset_id)
    label_col = req.get("label_column")
    features = req.get("features") or default_features(df, exclude=[label_col] if label_col else [])
    if label_col in features:
        features = [f for f in features if f != label_col]
    X, names, index, prep = prepare_unsupervised(df, features, scale=req.get("scale", "standard"))
    truth = _true_labels(df, index, label_col)
    Z, proj_var = clustering.projection(X)
    k = int(req.get("n_clusters", 3))
    seed = int(req.get("random_state", 0))

    if method == "compare":
        return _compare(dataset_id, df, X, index, Z, truth, features, k, req)

    model_file = None
    exp_id = experiments.new_id()
    numeric, _ = split_feature_types(df, features)
    with experiments.timer() as t:
        if method == "kmeans":
            model, info = clustering.kmeans(X, k, req.get("init", "k-means++"), int(req.get("n_init", 10)),
                                            int(req.get("max_iter", 300)), seed)
            labels = model.labels_
            info["centroids_scaled"] = [{"cluster": i, **{n: float(v) for n, v in zip(names, c)}}
                                        for i, c in enumerate(model.cluster_centers_)]
            # PCA is linear, so the projected centroid is the mean of the projected members.
            info["centroids_2d"] = [{"x": Z[labels == i, 0].mean(), "y": Z[labels == i, 1].mean(), "cluster": i}
                                    for i in range(k)]
            pts = _points(Z, labels, truth)
        elif method == "gmm":
            model, info = clustering.gmm(X, k, req.get("covariance_type", "full"), int(req.get("max_iter", 200)),
                                         float(req.get("tol", 1e-4)), seed)
            resp = model.predict_proba(X)
            labels = resp.argmax(1)
            info["mean_max_responsibility"] = float(resp.max(1).mean())
            info["uncertain_points"] = int(np.sum(resp.max(1) < 0.8))
            pts = _points(Z, labels, truth, resp)
        elif method == "hierarchical":
            idx, labels, info = clustering.hierarchical(X, k, req.get("linkage", "ward"), req.get("metric", "euclidean"))
            X, Z, index = X[idx], Z[idx], index[idx]
            truth = truth[idx] if truth is not None else None
            pts = _points(Z, labels, truth)
            model = None
        else:
            raise ValidationFailed("method must be kmeans, gmm, hierarchical or compare.")
        metrics = clustering_metrics(X, labels, truth)
    if model is not None:
        model_file = experiments.save_model(exp_id, {
            "model": Pipeline([("prep", prep), ("model", model)]), "features": features,
            "numeric_features": numeric, "task": "clustering", "classes": None, "algorithm": method})

    result = to_jsonable({
        "method": method,
        "n_clusters": k,
        "features": features,
        "encoded_features": names,
        "n_rows": int(len(X)),
        "projection_variance": proj_var,
        "metrics": metrics,
        "points": pts,
        "cluster_sizes": np.bincount(labels).tolist(),
        "profiles": _profiles(df, index, labels, features),
        **info,
    })
    params = {k2: req.get(k2) for k2 in ("n_clusters", "init", "covariance_type", "linkage", "metric", "scale") if req.get(k2) is not None}
    experiments.record(experiment_id=exp_id, unit=3, task="clustering", algorithm=method, dataset_id=dataset_id,
                       params=params, metrics={k2: v for k2, v in result["metrics"].items()},
                       results={k2: v for k2, v in result.items() if k2 != "points"},
                       features=features, target=label_col, duration_ms=t["ms"], model_file=model_file)
    result["experiment_id"] = exp_id
    return result


def _compare(dataset_id, df, X, index, Z, truth, features, k, req) -> dict:
    rows = []
    group = experiments.new_id()
    for name in ("kmeans", "gmm", "hierarchical"):
        with experiments.timer() as t:
            if name == "kmeans":
                m, _ = clustering.kmeans(X, k, scan_max=1)
                labels, Xe, te = m.labels_, X, truth
            elif name == "gmm":
                m, _ = clustering.gmm(X, k, req.get("covariance_type", "full"))
                labels, Xe, te = m.predict(X), X, truth
            else:
                idx, labels, _ = clustering.hierarchical(X, k, req.get("linkage", "ward"), req.get("metric", "euclidean"))
                Xe, te = X[idx], (truth[idx] if truth is not None else None)
            met = clustering_metrics(Xe, labels, te)
        rows.append({"method": name, "duration_ms": t["ms"], **met})
        experiments.record(unit=3, task="clustering", algorithm=name, dataset_id=dataset_id, group_id=group,
                           params={"n_clusters": k}, metrics=to_jsonable(met), results=None, features=features,
                           duration_ms=t["ms"])
    return to_jsonable({"method": "compare", "n_clusters": k, "comparison": rows, "group_id": group})


def decompose(dataset_id: str, req: dict) -> dict:
    method = req.get("method", "pca")
    df = datasets.load_df(dataset_id)
    color_col = req.get("color_column")
    features = req.get("features") or default_features(df, exclude=[color_col] if color_col else [])
    features = [f for f in features if f != color_col]
    X, names, index, _ = prepare_unsupervised(df, features, scale=req.get("scale", "standard"))
    n = int(req.get("n_components", 2))
    with experiments.timer() as t:
        if method == "pca":
            model, Z, info = decomposition.pca(X, n, names)
        elif method == "factor_analysis":
            rot = req.get("rotation", "varimax")
            model, Z, info = decomposition.factor_analysis(X, n, names, None if rot in (None, "none") else rot)
        else:
            raise ValidationFailed("method must be 'pca' or 'factor_analysis'.")
    color = df.loc[index, color_col].astype(str).to_numpy() if color_col and color_col in df.columns else None
    idx = subsample_indices(len(Z), MAX_POINTS)
    dims = min(3, Z.shape[1])
    points = [{**{f"c{j + 1}": Z[i, j] for j in range(dims)}, **({"label": color[i]} if color is not None else {})}
              for i in idx]
    result = to_jsonable({"method": method, "n_components": n, "features": names, "n_rows": int(len(X)),
                          "points": points, **info})
    metrics = {"retained_ratio": info.get("retained_ratio"),
               "log_likelihood_per_sample": info.get("log_likelihood_per_sample")}
    row = experiments.record(unit=3, task="dimensionality_reduction", algorithm=method, dataset_id=dataset_id,
                             params={"n_components": n}, metrics={k: v for k, v in metrics.items() if v is not None},
                             results={k: v for k, v in result.items() if k != "points"}, features=features,
                             duration_ms=t["ms"])
    result["experiment_id"] = row["id"]
    return result
