"""Unit 2 — linear models for classification.

Discriminative: logistic regression, Bayesian logistic regression (Laplace
approximation, Bishop §4.4–4.5), perceptron (pure discriminant function).
Generative: LDA (shared covariance → linear discriminant), QDA, Gaussian naive Bayes.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted

from ..errors import ValidationFailed


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary Bayesian logistic regression via the Laplace approximation.

    MAP weights found with Newton–Raphson (IRLS) under prior w ~ N(0, α⁻¹I).
    Posterior approximated by N(w_MAP, S_N), S_N⁻¹ = αI + Σ y_n(1−y_n)φ_nφ_nᵀ.
    Predictive probability uses the probit approximation
    p(C₁|φ) ≈ σ(κ(σ_a²) μ_a), κ(s²) = (1 + πs²/8)^(-1/2).
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 100, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def _phi(self, X):
        X = np.asarray(X, float)
        return np.column_stack([np.ones(len(X)), X])

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("BayesianLogisticRegression is binary; wrap it in OneVsRestClassifier.")
        t = (y == self.classes_[1]).astype(float)
        Phi = self._phi(X)
        M = Phi.shape[1]
        prior = self.alpha * np.eye(M)
        prior[0, 0] = 1e-6  # nearly flat prior on the bias
        w = np.zeros(M)
        self.n_iter_ = 0
        for i in range(self.max_iter):
            p = expit(Phi @ w)
            grad = Phi.T @ (p - t) + prior @ w
            H = Phi.T @ (Phi * (p * (1 - p))[:, None]) + prior
            step = np.linalg.solve(H, grad)
            w -= step
            self.n_iter_ = i + 1
            if np.max(np.abs(step)) < self.tol:
                break
        p = expit(Phi @ w)
        H = Phi.T @ (Phi * (p * (1 - p))[:, None]) + prior
        self.w_map_ = w
        self.S_N_ = np.linalg.inv(H)
        self.coef_ = w[1:][None, :]
        self.intercept_ = np.array([w[0]])
        self.coef_std_ = np.sqrt(np.diag(self.S_N_))[1:]
        # Laplace approximation to the log model evidence
        ll = float(np.sum(t * np.log(p + 1e-12) + (1 - t) * np.log(1 - p + 1e-12)))
        log_prior = float(-0.5 * w @ prior @ w + 0.5 * np.linalg.slogdet(prior)[1] - M / 2 * np.log(2 * np.pi))
        self.log_evidence_ = ll + log_prior + M / 2 * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(H)[1]
        self.n_features_in_ = M - 1
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "w_map_")
        Phi = self._phi(X)
        mu_a = Phi @ self.w_map_
        var_a = np.einsum("ij,jk,ik->i", Phi, self.S_N_, Phi)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * var_a / 8.0)
        p1 = expit(kappa * mu_a)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


ALGORITHMS = {
    "logistic": "Logistic regression",
    "bayesian_logistic": "Bayesian logistic (Laplace)",
    "lda": "Linear discriminant (LDA)",
    "qda": "Quadratic discriminant (QDA)",
    "naive_bayes": "Gaussian naive Bayes",
    "perceptron": "Perceptron",
}

FAMILY = {
    "logistic": "discriminative",
    "bayesian_logistic": "discriminative",
    "perceptron": "discriminant function",
    "lda": "generative",
    "qda": "generative",
    "naive_bayes": "generative",
}


def build(name: str, p: dict, n_classes: int):
    if name == "logistic":
        return LogisticRegression(C=float(p.get("C", 1.0)), max_iter=2000)
    if name == "bayesian_logistic":
        base = BayesianLogisticRegression(alpha=float(p.get("prior_alpha", 1.0)))
        return base if n_classes == 2 else OneVsRestClassifier(base)
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "qda":
        return QuadraticDiscriminantAnalysis(reg_param=float(p.get("reg_param", 0.01)))
    if name == "naive_bayes":
        return GaussianNB()
    if name == "perceptron":
        return Perceptron(max_iter=2000, random_state=0)
    raise ValidationFailed(f"Unknown classification algorithm '{name}'.")


def details(name: str, model, feature_names: list[str], classes: list[str]) -> dict:
    out: dict = {"family": FAMILY.get(name)}
    est = model
    if isinstance(model, OneVsRestClassifier):
        ests = model.estimators_
        out["coefficients_by_class"] = [
            {
                "class": classes[i],
                "coefficients": [
                    {"feature": f, "value": float(c), "std": float(s)}
                    for f, c, s in zip(feature_names, e.coef_[0], e.coef_std_)
                ],
                "log_evidence": e.log_evidence_,
            }
            for i, e in enumerate(ests)
        ]
        return out
    if hasattr(est, "coef_") and name != "qda":
        coef = np.atleast_2d(est.coef_)
        labels = [classes[1]] if coef.shape[0] == 1 else classes
        out["coefficients_by_class"] = [
            {
                "class": labels[i],
                "coefficients": [
                    {
                        "feature": f,
                        "value": float(c),
                        **({"std": float(est.coef_std_[j])} if name == "bayesian_logistic" else {}),
                    }
                    for j, (f, c) in enumerate(zip(feature_names, row))
                ],
            }
            for i, row in enumerate(coef)
        ]
    if name == "bayesian_logistic":
        out["laplace"] = {"log_evidence": est.log_evidence_, "newton_iterations": est.n_iter_}
    if name in ("lda", "qda", "naive_bayes"):
        out["class_priors"] = {classes[i]: float(v) for i, v in enumerate(est.priors_ if name != "naive_bayes" else est.class_prior_)}
        means = est.means_ if name != "naive_bayes" else est.theta_
        out["class_means"] = [
            {"class": classes[i], **{f: float(v) for f, v in zip(feature_names, row)}} for i, row in enumerate(means)
        ]
    if name == "lda" and hasattr(est, "explained_variance_ratio_"):
        out["discriminant_variance_ratio"] = est.explained_variance_ratio_.tolist()
    return out
