"""Unit 3 — PCA, factor analysis and choosing the number of latent dimensions."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score

from ..errors import ValidationFailed


def pca(X: np.ndarray, n_components: int, feature_names: list[str]) -> tuple[PCA, np.ndarray, dict]:
    d = X.shape[1]
    max_k = min(d, len(X))
    if not 1 <= n_components <= max_k:
        raise ValidationFailed(f"n_components must be between 1 and {max_k}.")
    full = PCA(random_state=0).fit(X)
    model = PCA(n_components=n_components, random_state=0).fit(X)
    Z = model.transform(X)
    evr = full.explained_variance_ratio_
    cum = np.cumsum(evr)
    # Reconstruction error as a function of the number of components
    recon = []
    for k in range(1, min(max_k, 20) + 1):
        p = PCA(n_components=k, random_state=0).fit(X)
        Xr = p.inverse_transform(p.transform(X))
        recon.append({"k": k, "mse": float(np.mean((X - Xr) ** 2))})
    eig = full.explained_variance_
    choice = {
        "components_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "components_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
        "kaiser_rule": int(np.sum(eig > eig.mean())),  # eigenvalues above the average
        "elbow": _elbow(evr),
    }
    # Probabilistic-PCA log-likelihood per k via cross-validation (where cheap enough)
    if len(X) <= 5000 and d <= 50:
        ll = []
        for k in range(1, min(d, 15) + 1):
            try:
                ll.append({"k": k, "cv_log_likelihood": float(np.mean(cross_val_score(PCA(n_components=k), X, cv=3)))})
            except Exception:
                break
        choice["ppca_cv"] = ll
        if ll:
            choice["ppca_best_k"] = max(ll, key=lambda r: r["cv_log_likelihood"])["k"]
    return model, Z, {
        "explained_variance": eig[: max(n_components, min(d, 20))].tolist(),
        "explained_variance_ratio": evr[:20].tolist(),
        "cumulative_ratio": cum[:20].tolist(),
        "retained_ratio": float(cum[n_components - 1]),
        "loadings": [
            {"component": f"PC{i + 1}", **{f: float(v) for f, v in zip(feature_names, row)}}
            for i, row in enumerate(model.components_[: min(n_components, 10)])
        ],
        "reconstruction": recon,
        "dimension_choice": choice,
    }


def _elbow(evr: np.ndarray) -> int:
    """Point of maximum distance to the chord of the scree curve."""
    n = len(evr)
    if n < 3:
        return 1
    pts = np.column_stack([np.arange(n), evr])
    a, b = pts[0], pts[-1]
    line = (b - a) / np.linalg.norm(b - a)
    vec = pts - a
    dist = np.abs(vec[:, 0] * line[1] - vec[:, 1] * line[0])
    return int(np.argmax(dist) + 1)


def factor_analysis(X: np.ndarray, n_factors: int, feature_names: list[str], rotation: str | None = "varimax"):
    d = X.shape[1]
    if not 1 <= n_factors <= d:
        raise ValidationFailed(f"n_factors must be between 1 and {d}.")
    if rotation not in (None, "varimax", "quartimax"):
        raise ValidationFailed("rotation must be varimax, quartimax or none.")
    model = FactorAnalysis(n_components=n_factors, rotation=rotation, random_state=0).fit(X)
    Z = model.transform(X)
    sel = []
    for k in range(1, min(d, 10) + 1):
        try:
            fa = FactorAnalysis(n_components=k, random_state=0)
            sel.append({"k": k, "cv_log_likelihood": float(np.mean(cross_val_score(fa, X, cv=3)))})
        except Exception:
            break
    return model, Z, {
        "loadings": [
            {"factor": f"F{i + 1}", **{f: float(v) for f, v in zip(feature_names, row)}}
            for i, row in enumerate(model.components_)
        ],
        "noise_variance": [{"feature": f, "psi": float(v)} for f, v in zip(feature_names, model.noise_variance_)],
        "communalities": [{"feature": f, "h2": float(v)} for f, v in zip(feature_names, (model.components_ ** 2).sum(0))],
        "log_likelihood_per_sample": float(model.score(X)),
        "model_selection": sel,
        "best_k": max(sel, key=lambda r: r["cv_log_likelihood"])["k"] if sel else None,
    }
