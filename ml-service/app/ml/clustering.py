"""Unit 3 — clustering.

* K-means (Lloyd) with elbow / silhouette scan over k.
* Gaussian mixture fitted by Expectation–Maximisation, recording the log-likelihood
  lower bound after every EM iteration; responsibilities γ(z_nk) returned per point.
* Agglomerative hierarchical clustering with selectable dissimilarity measure and
  linkage; dendrogram coordinates computed with SciPy.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from ..errors import ValidationFailed
from ..utils import subsample_indices
from .metrics import clustering_metrics

METRICS = ("euclidean", "cityblock", "cosine", "chebyshev", "correlation")
LINKAGES = ("ward", "complete", "average", "single")
COV_TYPES = ("full", "tied", "diag", "spherical")


def projection(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    if X.shape[1] <= 2:
        Z = X if X.shape[1] == 2 else np.column_stack([X[:, 0], np.zeros(len(X))])
        return Z, [1.0, 0.0]
    p = PCA(n_components=2, random_state=0).fit(X)
    return p.transform(X), p.explained_variance_ratio_.tolist()


def _check_k(k: int, n: int) -> None:
    if not 2 <= k <= min(20, n - 1):
        raise ValidationFailed(f"Number of clusters must be between 2 and {min(20, n - 1)}.")


def kmeans(X: np.ndarray, k: int, init: str = "k-means++", n_init: int = 10, max_iter: int = 300,
           seed: int = 0, scan_max: int = 10) -> tuple[KMeans, dict]:
    _check_k(k, len(X))
    if init not in ("k-means++", "random"):
        raise ValidationFailed("init must be 'k-means++' or 'random'.")
    model = KMeans(n_clusters=k, init=init, n_init=n_init, max_iter=max_iter, random_state=seed).fit(X)
    scan = []
    for kk in range(1, min(scan_max, len(X) - 1) + 1):
        m = KMeans(n_clusters=kk, init=init, n_init=5, random_state=seed).fit(X)
        entry = {"k": kk, "inertia": float(m.inertia_)}
        if kk >= 2:
            entry.update({k2: v for k2, v in clustering_metrics(X, m.labels_).items()
                          if k2 in ("silhouette", "davies_bouldin")})
        scan.append(entry)
    return model, {"n_iter": int(model.n_iter_), "inertia": float(model.inertia_), "k_scan": scan}


def gmm(X: np.ndarray, k: int, covariance_type: str = "full", max_iter: int = 200, tol: float = 1e-4,
        seed: int = 0) -> tuple[GaussianMixture, dict]:
    _check_k(k, len(X))
    if covariance_type not in COV_TYPES:
        raise ValidationFailed(f"covariance_type must be one of {', '.join(COV_TYPES)}.")
    # Run EM one iteration at a time to expose the monotone increase of the bound.
    model = GaussianMixture(n_components=k, covariance_type=covariance_type, max_iter=1, warm_start=True,
                            random_state=seed, reg_covar=1e-5, init_params="kmeans")
    trace = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i in range(max_iter):
            model.fit(X)
            trace.append({"iteration": i + 1, "log_likelihood": float(model.lower_bound_)})
            if i > 0 and abs(trace[-1]["log_likelihood"] - trace[-2]["log_likelihood"]) < tol:
                break
    model_sel = []
    for kk in range(1, min(10, len(X) - 1) + 1):
        m = GaussianMixture(n_components=kk, covariance_type=covariance_type, random_state=seed, reg_covar=1e-5).fit(X)
        model_sel.append({"k": kk, "bic": float(m.bic(X)), "aic": float(m.aic(X))})
    return model, {
        "em_trace": trace,
        "converged_iterations": len(trace),
        "weights": model.weights_.tolist(),
        "log_likelihood_per_sample": float(model.score(X)),
        "bic": float(model.bic(X)),
        "aic": float(model.aic(X)),
        "model_selection": model_sel,
    }


def hierarchical(X: np.ndarray, n_clusters: int, method: str = "ward", metric: str = "euclidean",
                 max_rows: int = 2000) -> tuple[np.ndarray, np.ndarray, dict]:
    if method not in LINKAGES:
        raise ValidationFailed(f"linkage must be one of {', '.join(LINKAGES)}.")
    if metric not in METRICS:
        raise ValidationFailed(f"metric must be one of {', '.join(METRICS)}.")
    if method == "ward" and metric != "euclidean":
        raise ValidationFailed("Ward linkage requires the euclidean metric.")
    idx = subsample_indices(len(X), max_rows)
    Xs = X[idx]
    _check_k(n_clusters, len(Xs))
    D = pdist(Xs, metric=metric)
    if not np.isfinite(D).all():
        raise ValidationFailed(f"The {metric} distance is undefined for some rows (e.g. zero vectors).")
    Z = linkage(D, method=method) if method != "ward" else linkage(Xs, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    coph, _ = cophenet(Z, D)
    dn = dendrogram(Z, truncate_mode="lastp", p=30, no_plot=True, count_sort=True)
    cut = float(Z[-n_clusters + 1, 2]) if n_clusters > 1 else float(Z[-1, 2])
    cut_low = float(Z[-n_clusters, 2]) if len(Z) >= n_clusters else 0.0
    return idx, labels, {
        "dendrogram": {
            "icoord": dn["icoord"],
            "dcoord": dn["dcoord"],
            "leaves": dn["ivl"],
            "cut_height": (cut + cut_low) / 2,
        },
        "cophenetic_correlation": float(coph),
        "merge_heights": Z[-15:, 2][::-1].tolist(),
        "n_rows_used": int(len(Xs)),
    }
