"""Unit 4 — Markov chains and discrete-emission Hidden Markov Models (NumPy).

Implements, following Bishop ch. 13:
* MLE of a first-order Markov chain (transition counts), stationary distribution.
* Forward algorithm (unscaled, to show underflow) and scaled forward pass with
  scaling factors c_t, giving ln p(X) = Σ ln c_t.
* Backward algorithm with the same scaling factors.
* Posterior marginals γ(z_t) and pairwise ξ(z_{t−1}, z_t).
* Sum-product message passing on the HMM factor graph (independent derivation
  of the marginals, checked against forward–backward).
* Viterbi (max-sum in log space) for the most probable hidden path.
* Baum–Welch EM for maximum-likelihood parameter learning over several sequences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..errors import ValidationFailed

EPS = 1e-300


def _check_stochastic(M: np.ndarray, name: str, axis: int = -1) -> np.ndarray:
    M = np.asarray(M, float)
    if (M < 0).any():
        raise ValidationFailed(f"{name} must not contain negative probabilities.")
    sums = M.sum(axis=axis)
    if not np.allclose(sums, 1.0, atol=1e-3):
        raise ValidationFailed(f"Each row of {name} must sum to 1 (got {np.round(sums, 4).tolist()}).")
    return M / M.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------ Markov chain
def markov_chain(sequences: list[np.ndarray], n_symbols: int, smoothing: float = 0.0) -> dict:
    counts = np.zeros((n_symbols, n_symbols))
    starts = np.zeros(n_symbols)
    for seq in sequences:
        if len(seq) == 0:
            continue
        starts[seq[0]] += 1
        np.add.at(counts, (seq[:-1], seq[1:]), 1)
    counts_s = counts + smoothing
    row = counts_s.sum(1, keepdims=True)
    A = np.divide(counts_s, row, out=np.full_like(counts_s, 1.0 / n_symbols), where=row > 0)
    pi = (starts + smoothing) / max((starts + smoothing).sum(), EPS)
    # Stationary distribution: left eigenvector of A with eigenvalue 1
    w, v = np.linalg.eig(A.T)
    stat = np.real(v[:, np.argmin(np.abs(w - 1))])
    stat = stat / stat.sum() if stat.sum() != 0 else np.full(n_symbols, 1 / n_symbols)
    ll = 0.0
    for seq in sequences:
        if len(seq) > 1:
            ll += float(np.sum(np.log(A[seq[:-1], seq[1:]] + EPS)))
    return {"counts": counts, "A": A, "pi": pi, "stationary": np.abs(stat), "log_likelihood": ll}


def simulate_chain(A: np.ndarray, pi: np.ndarray, length: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = np.empty(length, int)
    s[0] = rng.choice(len(pi), p=pi)
    for t in range(1, length):
        s[t] = rng.choice(len(pi), p=A[s[t - 1]])
    return s


# ------------------------------------------------------------------ HMM
@dataclass
class HMM:
    pi: np.ndarray  # (K,)
    A: np.ndarray  # (K, K)
    B: np.ndarray  # (K, M)

    @classmethod
    def validated(cls, pi, A, B) -> "HMM":
        pi = np.asarray(pi, float)
        A = np.asarray(A, float)
        B = np.asarray(B, float)
        K = len(pi)
        if A.shape != (K, K):
            raise ValidationFailed(f"Transition matrix must be {K}x{K}.")
        if B.ndim != 2 or B.shape[0] != K:
            raise ValidationFailed(f"Emission matrix must have {K} rows.")
        return cls(_check_stochastic(pi, "initial distribution"), _check_stochastic(A, "transition matrix"),
                   _check_stochastic(B, "emission matrix"))

    @property
    def K(self) -> int:
        return len(self.pi)

    # ---- inference
    def forward_unscaled(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        alpha = np.zeros((len(x), self.K))
        alpha[0] = self.pi * self.B[:, x[0]]
        for t in range(1, len(x)):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, x[t]]
        return alpha, float(alpha[-1].sum())

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scaled forward pass: returns α̂ (rows sum to 1) and scaling factors c."""
        T = len(x)
        alpha = np.zeros((T, self.K))
        c = np.zeros(T)
        a = self.pi * self.B[:, x[0]]
        c[0] = a.sum() + EPS
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * self.B[:, x[t]]
            c[t] = a.sum() + EPS
            alpha[t] = a / c[t]
        return alpha, c

    def backward(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        T = len(x)
        beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, x[t + 1]] * beta[t + 1])) / c[t + 1]
        return beta

    def posteriors(self, x: np.ndarray):
        alpha, c = self.forward(x)
        beta = self.backward(x, c)
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        xi = np.zeros((max(len(x) - 1, 0), self.K, self.K))
        for t in range(1, len(x)):
            xi[t - 1] = (alpha[t - 1][:, None] * self.A * (self.B[:, x[t]] * beta[t])[None, :]) / c[t]
        return alpha, beta, c, gamma, xi

    def log_likelihood(self, x: np.ndarray) -> float:
        _, c = self.forward(x)
        return float(np.sum(np.log(c)))

    def sum_product(self, x: np.ndarray) -> np.ndarray:
        """Marginals p(z_t | X) from sum-product on the chain factor graph.

        Factors: h(z_1)=π(z_1)p(x_1|z_1); f_t(z_{t−1}, z_t)=A(z_{t−1},z_t)p(x_t|z_t).
        Messages are normalised at every step (equivalent to the scaling factors).
        """
        T, K = len(x), self.K
        fwd = np.zeros((T, K))  # μ_{f_t → z_t}
        bwd = np.ones((T, K))  # μ_{f_{t+1} → z_t}
        m = self.pi * self.B[:, x[0]]
        fwd[0] = m / m.sum()
        for t in range(1, T):
            factor = self.A * self.B[:, x[t]][None, :]
            m = fwd[t - 1] @ factor  # sum over z_{t−1}
            fwd[t] = m / m.sum()
        for t in range(T - 2, -1, -1):
            factor = self.A * self.B[:, x[t + 1]][None, :]
            m = factor @ bwd[t + 1]  # sum over z_{t+1}
            bwd[t] = m / m.sum()
        marg = fwd * bwd
        return marg / marg.sum(1, keepdims=True)

    def viterbi(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        T, K = len(x), self.K
        logA, logB = np.log(self.A + EPS), np.log(self.B + EPS)
        omega = np.zeros((T, K))
        back = np.zeros((T, K), int)
        omega[0] = np.log(self.pi + EPS) + logB[:, x[0]]
        for t in range(1, T):
            scores = omega[t - 1][:, None] + logA
            back[t] = scores.argmax(0)
            omega[t] = scores.max(0) + logB[:, x[t]]
        path = np.zeros(T, int)
        path[-1] = omega[-1].argmax()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path, float(omega[-1].max())

    def sample(self, T: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        z = np.zeros(T, int)
        x = np.zeros(T, int)
        z[0] = rng.choice(self.K, p=self.pi)
        x[0] = rng.choice(self.B.shape[1], p=self.B[z[0]])
        for t in range(1, T):
            z[t] = rng.choice(self.K, p=self.A[z[t - 1]])
            x[t] = rng.choice(self.B.shape[1], p=self.B[z[t]])
        return z, x


def baum_welch(sequences: list[np.ndarray], K: int, M: int, n_iter: int = 100, tol: float = 1e-4,
               seed: int = 0, init: HMM | None = None) -> tuple[HMM, list[float]]:
    """Maximum-likelihood HMM parameters by EM (Baum–Welch) over multiple sequences."""
    rng = np.random.default_rng(seed)
    if init is None:
        pi = rng.dirichlet(np.ones(K))
        A = rng.dirichlet(np.ones(K) * 2, size=K) + np.eye(K) * 0.5
        A /= A.sum(1, keepdims=True)
        B = rng.dirichlet(np.ones(M), size=K)
        model = HMM(pi, A, B)
    else:
        model = init
    trace: list[float] = []
    for _ in range(n_iter):
        pi_acc = np.zeros(K)
        A_num = np.zeros((K, K))
        B_num = np.zeros((K, M))
        ll = 0.0
        for x in sequences:
            if len(x) < 2:
                continue
            _, _, c, gamma, xi = model.posteriors(x)
            ll += float(np.sum(np.log(c)))
            pi_acc += gamma[0]
            A_num += xi.sum(0)
            for m in range(M):
                B_num[:, m] += gamma[x == m].sum(0)
        trace.append(ll)
        model = HMM(
            pi_acc / pi_acc.sum(),
            (A_num + 1e-10) / (A_num + 1e-10).sum(1, keepdims=True),
            (B_num + 1e-10) / (B_num + 1e-10).sum(1, keepdims=True),
        )
        if len(trace) > 1 and abs(trace[-1] - trace[-2]) < tol * max(1.0, abs(trace[-2])):
            break
    return model, trace


def align_states(pred: np.ndarray, truth: np.ndarray, K: int, n_true: int):
    """Map learned states onto labelled states (Hungarian matching) and score accuracy.

    Returns (accuracy, confusion[K x n_true], mapping {learned_state: true_state_index}).
    """
    size = max(K, n_true)
    conf = np.zeros((size, size), int)
    np.add.at(conf, (pred, truth), 1)
    rows, cols = linear_sum_assignment(-conf)
    mapping = {int(r): int(c) for r, c in zip(rows, cols) if r < K and c < n_true}
    mapped = np.array([mapping.get(int(p), -1) for p in pred])
    return float(np.mean(mapped == truth)), conf[:K, :n_true].tolist(), mapping
