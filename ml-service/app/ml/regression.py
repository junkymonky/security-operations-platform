"""Unit 2 — linear regression family.

* least_squares  : normal equations / lstsq; reports the Gaussian MLE of the noise
                   variance and the log-likelihood (least squares == MLE).
* linear         : scikit-learn LinearRegression (reference implementation).
* ridge          : L2-regularised least squares (MAP with Gaussian prior).
* bayesian_linear: full posterior N(m_N, S_N) with α, β set by evidence maximisation
                   (Bishop §3.3, §3.5); predictive mean and variance.
* huber          : robust regression with Huber loss.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.utils.validation import check_is_fitted

from ..errors import ValidationFailed


class LeastSquaresRegression(BaseEstimator, RegressorMixin):
    """Ordinary least squares solved explicitly, with MLE statistics."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        Phi = np.column_stack([np.ones(len(X)), X])
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        self.intercept_, self.coef_ = float(w[0]), w[1:]
        resid = y - Phi @ w
        n = len(y)
        self.sigma2_mle_ = float(resid @ resid / n)  # β⁻¹ maximum-likelihood estimate
        self.log_likelihood_ = float(-0.5 * n * (np.log(2 * np.pi * self.sigma2_mle_) + 1))
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, float) @ self.coef_ + self.intercept_


class BayesianLinearRegression(BaseEstimator, RegressorMixin):
    """Bayesian linear regression with evidence-approximation hyper-parameters.

    Prior w ~ N(0, α⁻¹I), likelihood t ~ N(wᵀφ, β⁻¹). Posterior
    S_N = (αI + βΦᵀΦ)⁻¹,  m_N = β S_N Φᵀ t.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, fit_hyper: bool = True, max_iter: int = 200):
        self.alpha = alpha
        self.beta = beta
        self.fit_hyper = fit_hyper
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.y_mean_ = float(y.mean())
        t = y - self.y_mean_
        Phi = X
        N, M = Phi.shape
        alpha, beta = float(self.alpha), float(self.beta)
        eig = np.linalg.eigvalsh(Phi.T @ Phi)
        self.evidence_trace_ = []
        for _ in range(self.max_iter if self.fit_hyper else 1):
            S_inv = alpha * np.eye(M) + beta * Phi.T @ Phi
            S = np.linalg.inv(S_inv)
            m = beta * S @ Phi.T @ t
            if not self.fit_hyper:
                break
            lam = beta * eig
            gamma = float(np.sum(lam / (alpha + lam)))
            new_alpha = gamma / max(float(m @ m), 1e-12)
            rss = float(np.sum((t - Phi @ m) ** 2))
            new_beta = (N - gamma) / max(rss, 1e-12)
            self.evidence_trace_.append(self._log_evidence(Phi, t, alpha, beta, m, S_inv))
            converged = abs(new_alpha - alpha) < 1e-6 * alpha and abs(new_beta - beta) < 1e-6 * beta
            alpha, beta = new_alpha, new_beta
            if converged:
                break
        S_inv = alpha * np.eye(M) + beta * Phi.T @ Phi
        self.S_N_ = np.linalg.inv(S_inv)
        self.m_N_ = beta * self.S_N_ @ Phi.T @ t
        self.alpha_, self.beta_ = alpha, beta
        self.log_evidence_ = self._log_evidence(Phi, t, alpha, beta, self.m_N_, S_inv)
        self.coef_ = self.m_N_
        self.coef_std_ = np.sqrt(np.diag(self.S_N_))
        self.intercept_ = self.y_mean_
        self.n_features_in_ = M
        return self

    @staticmethod
    def _log_evidence(Phi, t, alpha, beta, m, S_inv) -> float:
        N, M = Phi.shape
        E = beta / 2 * np.sum((t - Phi @ m) ** 2) + alpha / 2 * m @ m
        _, logdet = np.linalg.slogdet(S_inv)
        return float(M / 2 * np.log(alpha) + N / 2 * np.log(beta) - E - logdet / 2 - N / 2 * np.log(2 * np.pi))

    def predict(self, X, return_std: bool = False):
        check_is_fitted(self, "m_N_")
        X = np.asarray(X, float)
        mean = X @ self.m_N_ + self.y_mean_
        if not return_std:
            return mean
        var = 1.0 / self.beta_ + np.einsum("ij,jk,ik->i", X, self.S_N_, X)
        return mean, np.sqrt(var)


ALGORITHMS = {
    "least_squares": "Least squares (MLE)",
    "linear": "Linear regression",
    "ridge": "Ridge regression",
    "bayesian_linear": "Bayesian linear regression",
    "huber": "Robust regression (Huber)",
}


def build(name: str, p: dict):
    if name == "least_squares":
        return LeastSquaresRegression()
    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=float(p.get("alpha", 1.0)))
    if name == "bayesian_linear":
        return BayesianLinearRegression(
            alpha=float(p.get("prior_alpha", 1.0)),
            beta=float(p.get("noise_beta", 1.0)),
            fit_hyper=bool(p.get("fit_hyper", True)),
        )
    if name == "huber":
        return HuberRegressor(epsilon=float(p.get("epsilon", 1.35)), alpha=float(p.get("alpha", 1e-4)), max_iter=1000)
    raise ValidationFailed(f"Unknown regression algorithm '{name}'.")


def details(name: str, model, feature_names: list[str], X_train, y_train, p: dict) -> dict:
    coef = np.ravel(getattr(model, "coef_", []))
    out: dict = {
        "intercept": float(np.ravel(getattr(model, "intercept_", 0.0))[0]) if hasattr(model, "intercept_") else None,
        "coefficients": [{"feature": f, "value": float(c)} for f, c in zip(feature_names, coef)],
    }
    if name == "least_squares":
        out["mle"] = {
            "noise_variance": model.sigma2_mle_,
            "noise_precision_beta": 1.0 / model.sigma2_mle_ if model.sigma2_mle_ > 0 else None,
            "log_likelihood": model.log_likelihood_,
        }
    if name == "bayesian_linear":
        out["posterior"] = {
            "alpha": model.alpha_,
            "beta": model.beta_,
            "log_evidence": model.log_evidence_,
            "evidence_trace": model.evidence_trace_,
        }
        for row, s in zip(out["coefficients"], model.coef_std_):
            row["std"] = float(s)
    if name == "huber":
        out["outliers_detected"] = int(np.sum(model.outliers_))
        out["scale"] = float(model.scale_)
    if name == "ridge":
        path = []
        for a in np.logspace(-3, 4, 25):
            r = Ridge(alpha=a).fit(X_train, y_train)
            path.append({"alpha": a, **{f: float(c) for f, c in zip(feature_names, np.ravel(r.coef_))}})
        out["ridge_path"] = path
    return out
