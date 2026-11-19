"""Unit 2 — kernel methods and support vector machines.

Exposes SVC/SVR with linear, polynomial, RBF and sigmoid kernels, and a kernel
(Gram) matrix explorer showing k(x, x') = φ(x)ᵀφ(x') without forming φ (kernel trick).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
from sklearn.svm import SVC, SVR

from ..errors import ValidationFailed

KERNELS = ("linear", "poly", "rbf", "sigmoid")


def _gamma(p: dict):
    g = p.get("gamma", "scale")
    if g in ("scale", "auto"):
        return g
    try:
        g = float(g)
    except (TypeError, ValueError) as exc:
        raise ValidationFailed("gamma must be 'scale', 'auto' or a positive number.") from exc
    if g <= 0:
        raise ValidationFailed("gamma must be positive.")
    return g


def _common(p: dict) -> dict:
    kernel = p.get("kernel", "rbf")
    if kernel not in KERNELS:
        raise ValidationFailed(f"kernel must be one of {', '.join(KERNELS)}.")
    C = float(p.get("C", 1.0))
    if C <= 0:
        raise ValidationFailed("C must be positive.")
    degree = int(p.get("degree", 3))
    if not 1 <= degree <= 10:
        raise ValidationFailed("degree must be between 1 and 10.")
    return {"kernel": kernel, "C": C, "gamma": _gamma(p), "degree": degree, "coef0": float(p.get("coef0", 0.0))}


def build_classifier(p: dict) -> SVC:
    # SVMs are not probabilistic; we report margins/decisions only (no Platt scaling).
    return SVC(**_common(p), random_state=0)


def build_regressor(p: dict) -> SVR:
    return SVR(**_common(p), epsilon=float(p.get("epsilon", 0.1)))


def details(model) -> dict:
    out = {
        "kernel": model.kernel,
        "n_support_vectors": int(len(model.support_)),
        "support_fraction": None,
    }
    if hasattr(model, "n_support_"):
        out["n_support_per_class"] = [int(v) for v in model.n_support_]
    if model.kernel == "linear":
        out["primal_weights"] = np.ravel(model.coef_).tolist()
    return out


def kernel_matrix(X: np.ndarray, p: dict, n: int = 25) -> dict:
    """Gram matrix on the first n samples for the chosen kernel."""
    cfg = _common(p)
    Xs = X[: min(n, len(X))]
    g = cfg["gamma"]
    if g == "scale":
        g = 1.0 / (X.shape[1] * X.var()) if X.var() > 0 else 1.0
    elif g == "auto":
        g = 1.0 / X.shape[1]
    if cfg["kernel"] == "linear":
        K = linear_kernel(Xs)
    elif cfg["kernel"] == "poly":
        K = polynomial_kernel(Xs, degree=cfg["degree"], gamma=g, coef0=cfg["coef0"])
    elif cfg["kernel"] == "rbf":
        K = rbf_kernel(Xs, gamma=g)
    else:
        K = sigmoid_kernel(Xs, gamma=g, coef0=cfg["coef0"])
    eig = np.linalg.eigvalsh((K + K.T) / 2)
    return {
        "kernel": cfg["kernel"],
        "gamma": g,
        "matrix": K.tolist(),
        "min_eigenvalue": float(eig.min()),
        "positive_semidefinite": bool(eig.min() > -1e-8),
    }
