"""Markov models, HMMs and linear dynamical systems over dataset sequences (Unit 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import ValidationFailed
from ..ml import hmm as H
from ..ml import lds
from ..utils import to_jsonable
from . import datasets, experiments

MAX_SYMBOLS = 30
SHOW_STEPS = 400  # steps returned for visualisation


def _sequences(df: pd.DataFrame, column: str, group: str | None, order: str | None):
    if column not in df.columns:
        raise ValidationFailed(f"Column '{column}' not found.")
    data = df.dropna(subset=[column])
    if order:
        if order not in df.columns:
            raise ValidationFailed(f"Order column '{order}' not found.")
        data = data.sort_values(order, kind="stable")
    symbols = sorted(data[column].astype(str).unique().tolist())
    if len(symbols) < 2:
        raise ValidationFailed("The sequence column needs at least two distinct symbols.")
    if len(symbols) > MAX_SYMBOLS:
        raise ValidationFailed(f"The sequence column has {len(symbols)} symbols; max is {MAX_SYMBOLS}.")
    enc = {s: i for i, s in enumerate(symbols)}
    if group:
        if group not in df.columns:
            raise ValidationFailed(f"Group column '{group}' not found.")
        frames = [g for _, g in data.groupby(group, sort=False)]
    else:
        frames = [data]
    seqs = [g[column].astype(str).map(enc).to_numpy() for g in frames]
    return seqs, symbols, frames


def markov(dataset_id: str, req: dict) -> dict:
    df = datasets.load_df(dataset_id)
    seqs, symbols, _ = _sequences(df, req["column"], req.get("group_column"), req.get("order_column"))
    smoothing = float(req.get("smoothing", 0.0))
    if smoothing < 0:
        raise ValidationFailed("smoothing must be non-negative.")
    res = H.markov_chain(seqs, len(symbols), smoothing)
    steps = int(req.get("n_steps", 3))
    if not 1 <= steps <= 50:
        raise ValidationFailed("n_steps must be between 1 and 50.")
    length = int(req.get("simulate_length", 50))
    sim = H.simulate_chain(res["A"], res["pi"] if res["pi"].sum() > 0 else res["stationary"], max(2, min(length, 500)),
                           int(req.get("random_state", 0)))
    observed = np.concatenate(seqs)[:SHOW_STEPS]
    out = to_jsonable({
        "symbols": symbols, "n_sequences": len(seqs), "n_transitions": int(res["counts"].sum()),
        "counts": res["counts"], "transition_matrix": res["A"], "initial": res["pi"],
        "stationary": res["stationary"], "n_step_matrix": np.linalg.matrix_power(res["A"], steps),
        "n_steps": steps, "log_likelihood": res["log_likelihood"],
        "observed": [symbols[i] for i in observed], "simulated": [symbols[i] for i in sim],
    })
    row = experiments.record(unit=4, task="sequence", algorithm="markov_chain", dataset_id=dataset_id,
                             params={"column": req["column"], "smoothing": smoothing},
                             metrics={"log_likelihood": out["log_likelihood"]}, results=out)
    out["experiment_id"] = row["id"]
    return out


def _inference_payload(model: H.HMM, x: np.ndarray, state_names: list[str], symbols: list[str]) -> dict:
    alpha, beta, c, gamma, xi = model.posteriors(x)
    sp = model.sum_product(x)
    path, logp = model.viterbi(x)
    _, unscaled = model.forward_unscaled(x)
    n = min(len(x), SHOW_STEPS)
    return {
        "log_likelihood": float(np.sum(np.log(c))),
        "unscaled_likelihood": unscaled,
        "unscaled_underflow": bool(unscaled == 0.0),
        "viterbi_path": [state_names[s] for s in path[:n]],
        "viterbi_log_prob": logp,
        "posterior": [{"t": t, **{state_names[k]: gamma[t, k] for k in range(model.K)}} for t in range(n)],
        "posterior_decoding": [state_names[s] for s in gamma.argmax(1)[:n]],
        "scaling_factors": c[:n],
        "alpha_hat": alpha[: min(n, 60)],
        "beta_hat": beta[: min(n, 60)],
        "sum_product_max_abs_diff": float(np.max(np.abs(sp - gamma))),
        "expected_transitions": xi.sum(0) if len(xi) else np.zeros((model.K, model.K)),
        "observations": [symbols[o] for o in x[:n]],
        "length": int(len(x)),
    }


def hmm_fit(dataset_id: str, req: dict) -> dict:
    df = datasets.load_df(dataset_id)
    seqs, symbols, frames = _sequences(df, req["column"], req.get("group_column"), req.get("order_column"))
    K = int(req.get("n_states", 2))
    if not 2 <= K <= 10:
        raise ValidationFailed("n_states must be between 2 and 10.")
    n_iter, n_init = int(req.get("n_iter", 100)), int(req.get("n_init", 3))
    if not 1 <= n_iter <= 500 or not 1 <= n_init <= 10:
        raise ValidationFailed("n_iter must be 1–500 and n_init 1–10.")
    seed = int(req.get("random_state", 0))
    with experiments.timer() as t:
        best, best_trace = None, None
        for i in range(n_init):
            m, trace = H.baum_welch(seqs, K, len(symbols), n_iter, float(req.get("tol", 1e-4)), seed + i)
            if best is None or trace[-1] > best_trace[-1]:
                best, best_trace = m, trace
    state_names = [f"S{k}" for k in range(K)]
    x = seqs[0]
    payload = _inference_payload(best, x, state_names, symbols)
    evaluation = None
    hidden_col = req.get("hidden_column")
    if hidden_col:
        if hidden_col not in df.columns:
            raise ValidationFailed(f"Hidden-state column '{hidden_col}' not found.")
        truth_raw = frames[0][hidden_col].astype(str).to_numpy()
        true_names = sorted(set(truth_raw))
        truth = np.array([true_names.index(v) for v in truth_raw])
        path, _ = best.viterbi(x)
        acc, conf, mapping = H.align_states(path, truth, K, len(true_names))
        post_acc, _, _ = H.align_states(best.posteriors(x)[3].argmax(1), truth, K, len(true_names))
        name_map = {state_names[s]: true_names[j] for s, j in mapping.items()}
        evaluation = {"hidden_column": hidden_col, "true_states": true_names, "viterbi_accuracy": acc,
                      "posterior_decoding_accuracy": post_acc, "confusion": conf, "state_mapping": name_map,
                      "true_path": truth_raw[:SHOW_STEPS].tolist()}
    out = to_jsonable({
        "symbols": symbols, "states": state_names, "n_sequences": len(seqs), "total_length": int(sum(map(len, seqs))),
        "initial": best.pi, "transition_matrix": best.A, "emission_matrix": best.B,
        "em_trace": [{"iteration": i + 1, "log_likelihood": v} for i, v in enumerate(best_trace)],
        "evaluation": evaluation, **payload,
    })
    metrics = {"log_likelihood": out["log_likelihood"], "em_iterations": len(best_trace)}
    if evaluation:
        metrics["viterbi_accuracy"] = out["evaluation"]["viterbi_accuracy"]
    row = experiments.record(unit=4, task="sequence", algorithm="hmm_baum_welch", dataset_id=dataset_id,
                             params={"column": req["column"], "n_states": K, "n_iter": n_iter, "n_init": n_init},
                             metrics=metrics, results={k: v for k, v in out.items() if k not in ("alpha_hat", "beta_hat")},
                             duration_ms=t["ms"])
    out["experiment_id"] = row["id"]
    return out


def hmm_decode(req: dict) -> dict:
    states, symbols = req["states"], req["symbols"]
    if len(states) < 2 or len(symbols) < 2:
        raise ValidationFailed("Define at least two hidden states and two observation symbols.")
    if len(set(states)) != len(states) or len(set(symbols)) != len(symbols):
        raise ValidationFailed("State and symbol names must be unique.")
    model = H.HMM.validated(req["initial"], req["transition"], req["emission"])
    if model.B.shape[1] != len(symbols):
        raise ValidationFailed(f"Emission matrix needs {len(symbols)} columns (one per symbol).")
    true_path = None
    if req.get("observations"):
        enc = {s: i for i, s in enumerate(symbols)}
        unknown = sorted({o for o in req["observations"] if o not in enc})
        if unknown:
            raise ValidationFailed(f"Unknown observation symbol(s): {', '.join(unknown)}")
        x = np.array([enc[o] for o in req["observations"]])
    else:
        T = int(req.get("sample_length", 0))
        if not 2 <= T <= 5000:
            raise ValidationFailed("Provide observations or a sample_length between 2 and 5000.")
        z, x = model.sample(T, int(req.get("random_state", 0)))
        true_path = [states[s] for s in z[:SHOW_STEPS]]
    with experiments.timer() as t:
        out = _inference_payload(model, x, states, symbols)
    out["true_path"] = true_path
    if true_path is not None:
        out["viterbi_accuracy"] = float(np.mean(np.array(out["viterbi_path"]) == np.array(true_path)))
    out = to_jsonable({"states": states, "symbols": symbols, "initial": model.pi, "transition_matrix": model.A,
                       "emission_matrix": model.B, **out})
    metrics = {"log_likelihood": out["log_likelihood"], "viterbi_log_prob": out["viterbi_log_prob"]}
    if "viterbi_accuracy" in out:
        metrics["viterbi_accuracy"] = out["viterbi_accuracy"]
    row = experiments.record(unit=4, task="sequence", algorithm="hmm_inference", dataset_id=None,
                             params={"states": states, "symbols": symbols, "length": out["length"]},
                             metrics=metrics, results={k: v for k, v in out.items() if k not in ("alpha_hat", "beta_hat")},
                             duration_ms=t["ms"])
    out["experiment_id"] = row["id"]
    return out


def lds_run(dataset_id: str, req: dict) -> dict:
    df = datasets.load_df(dataset_id)
    col = req["column"]
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        raise ValidationFailed(f"Column '{col}' must exist and be numeric.")
    data = df
    if req.get("order_column"):
        if req["order_column"] not in df.columns:
            raise ValidationFailed(f"Order column '{req['order_column']}' not found.")
        data = df.sort_values(req["order_column"], kind="stable")
    y = data[col].dropna().to_numpy(float)[: int(req.get("max_points", 1000))]
    with experiments.timer() as t:
        out = lds.run(y, req.get("model", "random_walk"), req.get("q"), req.get("r"),
                      bool(req.get("estimate_noise", True)), int(req.get("forecast", 10)))
    out = to_jsonable(out)
    row = experiments.record(unit=4, task="sequence", algorithm=f"lds_{out['model']}", dataset_id=dataset_id,
                             params={"column": col, "q": out["q"], "r": out["r"]},
                             metrics={"log_likelihood": out["log_likelihood"], "filter_rmse": out["filter_rmse"]},
                             results={k: v for k, v in out.items() if k != "series"}, duration_ms=t["ms"])
    out["experiment_id"] = row["id"]
    return out
