"""Unit 4: fare-state sequences, Markov chain and HMM inference (report sections 3.7 – 3.11)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..ml.hmm import HMM, baum_welch
from .data import SEED, TARGET
from .unit12 import box_stats

STATES = ["Low", "Medium", "High"]

# Table 3.10: hand-specified HMM used for the demonstration in the report.
REPORT_HMM = {
    "initial": [0.5, 0.3, 0.2],
    "transition": [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]],
    "emission": [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]],
    "observations": [0, 1, 2, 1],  # Low, Medium, High, Medium
}


def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """One sequence per flight, journey date and route, ordered by days left (earliest booking first)."""
    keys = ["Flight_code", "Date_of_journey", "Source", "Destination"]
    seq = df[keys + ["Days_left", TARGET]].copy()
    seq["Sequence_ID"] = seq.groupby(keys, observed=True, sort=False).ngroup()
    seq["Fare_State"] = pd.qcut(seq[TARGET], q=3, labels=STATES)
    return seq.sort_values(["Sequence_ID", "Days_left"], ascending=[True, False], kind="stable")


def sequence_summary(seq: pd.DataFrame) -> dict:
    """Table 3.7."""
    sizes = seq.groupby("Sequence_ID").size()
    return {
        "total_sequences": int(len(sizes)),
        "single_observation": int((sizes == 1).sum()),
        "multiple_observations": int((sizes > 1).sum()),
        "by_length": {str(n): int((sizes == n).sum()) for n in (2, 3, 4, 5)},
        "longest": int(sizes.max()),
        "average_length": float(len(seq) / len(sizes)),
    }


def fare_states(seq: pd.DataFrame) -> dict:
    """Table 3.8 and the data for figures 3.9 – 3.10."""
    rows = [{"state": str(state), "records": int(len(g)), **box_stats(g)}
            for state, g in seq.groupby("Fare_State", observed=True)[TARGET]]
    return {"states": rows}


def transition_matrix(seq: pd.DataFrame) -> dict:
    """Table 3.9: count consecutive state pairs inside each sequence and normalise each row."""
    codes = seq["Fare_State"].cat.codes.to_numpy()
    ids = seq["Sequence_ID"].to_numpy()
    same = ids[1:] == ids[:-1]
    counts = np.zeros((3, 3), dtype=np.int64)
    np.add.at(counts, (codes[:-1][same], codes[1:][same]), 1)
    matrix = counts / counts.sum(axis=1, keepdims=True)
    return {"states": STATES, "counts": counts.tolist(), "matrix": matrix.tolist(),
            "transitions": int(counts.sum())}


def forward_algorithm(obs, initial, transition, emission) -> float:
    alpha = initial * emission[:, obs[0]]
    for o in obs[1:]:
        alpha = (alpha @ transition) * emission[:, o]
    return float(alpha.sum())


def backward_algorithm(obs, initial, transition, emission) -> float:
    beta = np.ones(len(initial))
    for o in reversed(obs[1:]):
        beta = transition @ (emission[:, o] * beta)
    return float(np.sum(initial * emission[:, obs[0]] * beta))


def hmm_demonstration() -> dict:
    """Table 3.11: forward, backward, scaling factors, sum-product and Viterbi on the report's HMM."""
    pi, A, B = (np.array(REPORT_HMM[k], dtype=float) for k in ("initial", "transition", "emission"))
    obs = np.array(REPORT_HMM["observations"])
    model = HMM(pi, A, B)
    alpha_hat, scales = model.forward(obs)
    posterior = model.sum_product(obs)
    path, log_delta = model.viterbi(obs)
    return {
        "parameters": REPORT_HMM,
        "forward_probability": forward_algorithm(obs, pi, A, B),
        "backward_probability": backward_algorithm(obs, pi, A, B),
        "scaling_factors": scales.tolist(),
        "scaled_log_likelihood": float(np.log(scales).sum()),
        "scaled_alpha": alpha_hat.tolist(),
        "posterior": posterior.tolist(),
        "viterbi_path": path.tolist(),
        "viterbi_log_probability": log_delta,
    }


def learned_hmm(seq: pd.DataFrame, n_sequences: int = 4000, n_iter: int = 40) -> dict:
    """Extension (report future scope): estimate the HMM from data with Baum–Welch."""
    codes = seq["Fare_State"].cat.codes.to_numpy()
    ids = seq["Sequence_ID"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    all_seqs = [codes[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2]
    rng = np.random.default_rng(SEED)
    if len(all_seqs) < 10:
        raise ValueError("Too few multi-observation sequences to train and evaluate an HMM.")
    pick = rng.permutation(len(all_seqs))
    n_train = min(n_sequences, int(0.8 * len(all_seqs)))  # always leave a held-out set
    train = [all_seqs[i] for i in pick[:n_train]]
    held = [all_seqs[i] for i in pick[n_train:n_train + 1000]]
    init = HMM(np.array(REPORT_HMM["initial"]), np.array(REPORT_HMM["transition"]), np.array(REPORT_HMM["emission"]))
    model, trace = baum_welch(train, K=3, M=3, n_iter=n_iter, init=init)
    held_ll = sum(model.log_likelihood(s) for s in held)
    report_ll = sum(init.log_likelihood(s) for s in held)
    n_obs = sum(len(s) for s in held)
    return {
        "train_sequences": len(train), "held_out_sequences": len(held),
        "initial": model.pi.tolist(), "transition": model.A.tolist(), "emission": model.B.tolist(),
        "trace": trace,
        "held_out_log_likelihood_per_obs": held_ll / n_obs,
        "report_hmm_log_likelihood_per_obs": report_ll / n_obs,
    }
