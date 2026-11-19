"""Evaluation metrics for classification, regression and clustering."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def classification_metrics(y_true, y_pred, classes: list[str], proba=None) -> dict:
    labels = list(range(len(classes)))
    avg = "binary" if len(classes) == 2 else "macro"
    out = {
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "precision": skm.precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": skm.recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": skm.f1_score(y_true, y_pred, average=avg, zero_division=0),
        "average": avg,
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": classes,
        "per_class": [],
    }
    p, r, f, s = skm.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out["per_class"] = [
        {"class": c, "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
        for i, c in enumerate(classes)
    ]
    if proba is not None:
        try:
            if len(classes) == 2:
                out["roc_auc"] = skm.roc_auc_score(y_true, proba[:, 1])
                fpr, tpr, _ = skm.roc_curve(y_true, proba[:, 1])
                idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 60)).astype(int)
                out["roc_curve"] = [{"fpr": fpr[i], "tpr": tpr[i]} for i in idx]
            else:
                out["roc_auc"] = skm.roc_auc_score(y_true, proba, multi_class="ovr", labels=labels)
            out["log_loss"] = skm.log_loss(y_true, proba, labels=labels)
        except ValueError:
            pass
    return out


def regression_metrics(y_true, y_pred) -> dict:
    mse = skm.mean_squared_error(y_true, y_pred)
    return {
        "mae": skm.mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": skm.r2_score(y_true, y_pred),
    }


def clustering_metrics(X: np.ndarray, labels: np.ndarray, true_labels=None) -> dict:
    out: dict = {"n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0))}
    if 1 < out["n_clusters"] < len(X):
        sample = min(len(X), 3000)
        out["silhouette"] = skm.silhouette_score(X, labels, sample_size=sample, random_state=0)
        out["davies_bouldin"] = skm.davies_bouldin_score(X, labels)
        out["calinski_harabasz"] = skm.calinski_harabasz_score(X, labels)
    # Within-cluster sum of squares (inertia) computed for any partition
    inertia = 0.0
    for k in set(labels):
        pts = X[labels == k]
        inertia += float(((pts - pts.mean(axis=0)) ** 2).sum())
    out["inertia"] = inertia
    if true_labels is not None:
        out["adjusted_rand"] = skm.adjusted_rand_score(true_labels, labels)
        out["normalized_mutual_info"] = skm.normalized_mutual_info_score(true_labels, labels)
        out["homogeneity"] = skm.homogeneity_score(true_labels, labels)
        out["completeness"] = skm.completeness_score(true_labels, labels)
    return out
