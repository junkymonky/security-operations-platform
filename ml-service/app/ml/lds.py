"""Unit 4 — linear dynamical systems (Kalman filter and RTS smoother).

State space model  z_t = A z_{t−1} + w_t,  w ~ N(0, Q)
                   x_t = C z_t + v_t,      v ~ N(0, R)
Two standard models are offered: a random walk (local level) and a
constant-velocity (local linear trend) model. Noise variances q, r can be set or
estimated by maximising the innovations log-likelihood.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..errors import ValidationFailed


def _matrices(model: str, q: float, r: float):
    if model == "random_walk":
        A, C = np.eye(1), np.ones((1, 1))
        Q = np.array([[q]])
    elif model == "constant_velocity":
        A = np.array([[1.0, 1.0], [0.0, 1.0]])
        C = np.array([[1.0, 0.0]])
        Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])  # discretised white-noise acceleration
    else:
        raise ValidationFailed("model must be 'random_walk' or 'constant_velocity'.")
    return A, C, Q, np.array([[r]])


def kalman(y: np.ndarray, model: str, q: float, r: float):
    A, C, Q, R = _matrices(model, q, r)
    n, d = len(y), A.shape[0]
    mu = np.zeros((n, d))
    P = np.zeros((n, d, d))
    mu_pred = np.zeros((n, d))
    P_pred = np.zeros((n, d, d))
    m0 = np.zeros(d)
    m0[0] = y[0]
    P0 = np.eye(d) * (np.var(y) + 1.0)
    ll = 0.0
    innovations = np.zeros(n)
    for t in range(n):
        mp = m0 if t == 0 else A @ mu[t - 1]
        Pp = P0 if t == 0 else A @ P[t - 1] @ A.T + Q
        S = C @ Pp @ C.T + R
        K = Pp @ C.T @ np.linalg.inv(S)
        e = y[t] - (C @ mp)[0]
        mu[t] = mp + (K * e).ravel()
        P[t] = (np.eye(d) - K @ C) @ Pp
        mu_pred[t], P_pred[t] = mp, Pp
        innovations[t] = e
        s = S[0, 0]
        ll += -0.5 * (np.log(2 * np.pi * s) + e * e / s)
    # Rauch–Tung–Striebel smoother
    mu_s, P_s = mu.copy(), P.copy()
    for t in range(n - 2, -1, -1):
        J = P[t] @ A.T @ np.linalg.inv(P_pred[t + 1])
        mu_s[t] = mu[t] + J @ (mu_s[t + 1] - mu_pred[t + 1])
        P_s[t] = P[t] + J @ (P_s[t + 1] - P_pred[t + 1]) @ J.T
    return {"filtered": mu, "filtered_var": P, "smoothed": mu_s, "smoothed_var": P_s,
            "predicted": mu_pred, "predicted_var": P_pred, "log_likelihood": float(ll), "innovations": innovations}


def run(y: np.ndarray, model: str = "random_walk", q: float | None = None, r: float | None = None,
        estimate: bool = True, forecast: int = 10) -> dict:
    y = np.asarray(y, float)
    if len(y) < 5:
        raise ValidationFailed("Need at least 5 observations.")
    scale = np.var(np.diff(y)) or 1.0
    q = float(q) if q is not None else scale * 0.1
    r = float(r) if r is not None else scale * 0.5
    if q <= 0 or r <= 0:
        raise ValidationFailed("Noise variances must be positive.")
    estimated = False
    if estimate:
        obj = lambda th: -kalman(y, model, float(np.exp(th[0])), float(np.exp(th[1])))["log_likelihood"]
        res = minimize(obj, x0=np.log([q, r]), method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-4})
        if res.success or res.fun < obj(np.log([q, r])):
            q, r = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
            estimated = True
    out = kalman(y, model, q, r)
    A, C, Q, R = _matrices(model, q, r)
    m, P = out["filtered"][-1], out["filtered_var"][-1]
    fc = []
    for h in range(1, forecast + 1):
        m = A @ m
        P = A @ P @ A.T + Q
        fc.append({"t": len(y) - 1 + h, "mean": float((C @ m)[0]), "std": float(np.sqrt((C @ P @ C.T + R)[0, 0]))})
    series = [{
        "t": t, "observed": float(y[t]),
        "filtered": float(out["filtered"][t, 0]), "filtered_std": float(np.sqrt(out["filtered_var"][t, 0, 0])),
        "smoothed": float(out["smoothed"][t, 0]), "smoothed_std": float(np.sqrt(out["smoothed_var"][t, 0, 0])),
        **({"velocity": float(out["smoothed"][t, 1])} if model == "constant_velocity" else {}),
    } for t in range(len(y))]
    resid = y - out["filtered"][:, 0]
    return {
        "model": model, "q": q, "r": r, "noise_estimated": estimated,
        "log_likelihood": out["log_likelihood"],
        "A": A.tolist(), "C": C.tolist(),
        "series": series, "forecast": fc,
        "filter_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "innovation_std": float(np.std(out["innovations"][1:])),
    }
