"""Network-attack study on KDD Cup 1999 (Units 1–5), mirroring the structure of the project report.

Dataset: data/kdd_attacks_full.csv (built by `python -m sample_data.make_kdd`): 145,586 de-duplicated
connections, 40 features, labels attack_type / attack_category (normal, DoS, Probe, R2L, U2R) / is_attack.
The raw, time-ordered 10% stream (scikit-learn cache) is used for the sequential models of Unit 4.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, adjusted_rand_score, classification_report, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..config import SERVICE_ROOT
from ..errors import NotFound
from ..ml.hmm import baum_welch

SEED = 42
DATA_FILE = SERVICE_ROOT / "data" / "kdd_attacks_full.csv"
CATEGORIES = ["normal", "DoS", "Probe", "R2L", "U2R"]
CATEGORICAL = ["protocol_type", "service", "flag"]
LABELS = ["attack_type", "attack_category", "is_attack"]
# Heavy-tailed counts are log-transformed before scaling.
LOG_FEATURES = ["duration", "src_bytes", "dst_bytes", "hot", "num_compromised", "num_root", "num_file_creations",
                "num_access_files", "count", "srv_count", "dst_host_count", "dst_host_srv_count"]
KEY_FEATURES = ["src_bytes", "dst_bytes", "count", "srv_count", "serror_rate", "rerror_rate", "same_srv_rate",
                "diff_srv_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_serror_rate"]


def find_dataset() -> Path:
    if not DATA_FILE.is_file():
        raise NotFound("Attack dataset not found. Build it with: cd ml-service && "
                       ".venv\\Scripts\\python -m sample_data.make_kdd")
    return DATA_FILE


def load() -> pd.DataFrame:
    return pd.read_csv(find_dataset())


def load_raw_stream() -> pd.DataFrame:
    """Original (time-ordered) 10% KDD stream, before de-duplication."""
    from sklearn.datasets import fetch_kddcup99
    from sample_data.make_kdd import CATEGORY
    raw = fetch_kddcup99(percent10=True, as_frame=True, data_home=str(SERVICE_ROOT / "data" / "kddcup99_cache")).frame
    out = pd.DataFrame({
        "protocol_type": raw["protocol_type"].map(lambda v: v.decode() if isinstance(v, bytes) else str(v)),
        "flag": raw["flag"].map(lambda v: v.decode() if isinstance(v, bytes) else str(v)),
        "attack_type": raw["labels"].map(lambda v: (v.decode() if isinstance(v, bytes) else str(v)).rstrip(".")),
    })
    out["attack_category"] = out["attack_type"].map(CATEGORY)
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in CATEGORICAL + LABELS]


def make_preprocessor(df: pd.DataFrame, sparse: bool = False) -> ColumnTransformer:
    numeric = numeric_columns(df)
    logged = [c for c in numeric if c in LOG_FEATURES]
    plain = [c for c in numeric if c not in LOG_FEATURES]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse), CATEGORICAL),
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]), logged),
        ("num", StandardScaler(), plain),
    ])


def split(df: pd.DataFrame):
    """Stratified 80:20 split on the attack category (random_state 42)."""
    X = df.drop(columns=LABELS)
    y = df["attack_category"].to_numpy()
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


# ----------------------------------------------------------------- Unit 1
def describe(df: pd.DataFrame, raw_rows: int) -> dict:
    """Dataset summary, class balance and descriptive statistics of key features."""
    num = df[KEY_FEATURES]
    return {
        "rows": int(len(df)), "raw_rows": int(raw_rows), "duplicates_removed": int(raw_rows - len(df)),
        "features": int(df.shape[1] - len(LABELS)), "categorical": CATEGORICAL,
        "missing_values": int(df.isna().sum().sum()),
        "category_counts": {c: int((df["attack_category"] == c).sum()) for c in CATEGORIES},
        "attack_type_counts": {k: int(v) for k, v in df["attack_type"].value_counts().items()},
        "protocol_by_category": pd.crosstab(df["protocol_type"], df["attack_category"])[CATEGORIES].to_dict("index"),
        "key_stats": {c: {"mean": float(num[c].mean()), "std": float(num[c].std()), "median": float(num[c].median()),
                          "max": float(num[c].max())} for c in KEY_FEATURES},
        "correlation": {"features": KEY_FEATURES, "matrix": num.corr().round(4).to_numpy().tolist()},
    }


def feature_distributions(df: pd.DataFrame) -> dict:
    """Data for the histogram (log bytes by class) and box plot (connection count by category)."""
    bins = np.linspace(0, 10, 41)
    hist = {}
    for name, mask in (("normal", df["is_attack"] == "normal"), ("attack", df["is_attack"] == "attack")):
        counts, _ = np.histogram(np.log10(df.loc[mask, "src_bytes"] + 1), bins=bins)
        hist[name] = counts.tolist()
    from .unit12 import box_stats
    box = {c: box_stats(df.loc[df["attack_category"] == c, "count"]) for c in CATEGORIES}
    return {"log_src_bytes_hist": {"edges": bins.tolist(), **hist}, "count_box": box}


def bayes_evidence(df: pd.DataFrame) -> list[dict]:
    """Bayes' rule P(attack | evidence) for several observable connection properties."""
    attack = (df["is_attack"] == "attack").to_numpy()
    evidence = {
        "SYN error rate ≥ 0.5": df["serror_rate"] >= 0.5,
        "flag = S0 (no SYN-ACK)": df["flag"] == "S0",
        "service = private": df["service"] == "private",
        "≥ 1 failed login": df["num_failed_logins"] >= 1,
        "logged in": df["logged_in"] == 1,
        "same service rate < 0.1": df["same_srv_rate"] < 0.1,
    }
    prior = float(attack.mean())
    out = []
    for name, mask in evidence.items():
        e = mask.to_numpy()
        p_e = float(e.mean())
        if p_e == 0:
            continue
        likelihood = float(e[attack].mean())
        posterior = likelihood * prior / p_e  # Bayes' rule
        chi2, p_value, _, _ = stats.chi2_contingency(pd.crosstab(e, attack).to_numpy())
        out.append({"evidence": name, "prior": prior, "likelihood": likelihood, "evidence_prob": p_e,
                    "posterior": posterior, "posterior_direct": float(attack[e].mean()),
                    "lift": posterior / prior, "chi2": float(chi2), "p_value": float(p_value)})
    return out


def polynomial_attack_curve(df: pd.DataFrame, max_degree: int = 10) -> dict:
    """Polynomial curve fitting: attack rate as a function of log(connection count), degree 0..10."""
    x = np.log1p(df["count"].to_numpy(float))
    y = (df["is_attack"] == "attack").to_numpy(float)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=SEED)
    edges = np.quantile(x_tr, np.linspace(0, 1, 41))
    edges = np.unique(edges)

    def binned(xs, ys):
        idx = np.clip(np.digitize(xs, edges[1:-1]), 0, len(edges) - 2)
        centres = np.array([xs[idx == i].mean() for i in range(len(edges) - 1) if (idx == i).any()])
        rates = np.array([ys[idx == i].mean() for i in range(len(edges) - 1) if (idx == i).any()])
        return centres, rates

    cx, cy = binned(x_tr, y_tr)
    tx, ty = binned(x_te, y_te)
    scan, fits = [], {}
    for d in range(0, max_degree + 1):
        coef = np.polyfit(cx, cy, d)
        scan.append({"degree": d, "train_rmse": float(np.sqrt(np.mean((np.polyval(coef, cx) - cy) ** 2))),
                     "test_rmse": float(np.sqrt(np.mean((np.polyval(coef, tx) - ty) ** 2)))})
        fits[d] = coef
    grid = np.linspace(cx.min(), cx.max(), 200)
    return {"train_points": {"x": cx.tolist(), "y": cy.tolist()}, "test_points": {"x": tx.tolist(), "y": ty.tolist()},
            "scan": scan, "curves": {d: np.clip(np.polyval(fits[d], grid), -0.1, 1.1).tolist() for d in (1, 3, 9)},
            "grid": grid.tolist()}


# ----------------------------------------------------------------- Unit 2
def unit2_models() -> dict:
    return {
        "Logistic regression": LogisticRegression(max_iter=2000, C=1.0),
        "Linear discriminant (LDA)": LinearDiscriminantAnalysis(),
        "Gaussian naive Bayes": GaussianNB(),
        "SVM (RBF kernel)": SVC(kernel="rbf", C=10, gamma="scale", probability=False, random_state=SEED),
    }


def evaluate(name, y_test, pred, seconds, y_train=None, pred_train=None, attack_score=None, note=None) -> dict:
    rep = classification_report(y_test, pred, labels=CATEGORIES, output_dict=True, zero_division=0)
    out = {"model": name, "accuracy": float(accuracy_score(y_test, pred)),
           "macro_f1": float(f1_score(y_test, pred, labels=CATEGORIES, average="macro", zero_division=0)),
           "weighted_f1": float(f1_score(y_test, pred, labels=CATEGORIES, average="weighted", zero_division=0)),
           "recall": {c: float(rep[c]["recall"]) for c in CATEGORIES},
           "precision": {c: float(rep[c]["precision"]) for c in CATEGORIES},
           "confusion_matrix": confusion_matrix(y_test, pred, labels=CATEGORIES).tolist(),
           "train_seconds": seconds, "note": note}
    if pred_train is not None:
        out["train_accuracy"] = float(accuracy_score(y_train, pred_train))
    if attack_score is not None:
        is_attack = (y_test != "normal").astype(int)
        fpr, tpr, _ = roc_curve(is_attack, attack_score)
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 150)).astype(int)
        out["roc_auc"] = float(roc_auc_score(is_attack, attack_score))
        out["roc"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
    return out


def train_unit2(df, X_train, X_test, y_train, y_test, svm_sample: int = 20000) -> list[dict]:
    """Linear, generative and kernel classifiers for the 5 attack categories."""
    results = []
    for name, model in unit2_models().items():
        Xtr, ytr, note = X_train, y_train, None
        if name.startswith("SVM") and len(X_train) > svm_sample:
            Xtr, _, ytr, _ = train_test_split(X_train, y_train, train_size=svm_sample, random_state=SEED, stratify=y_train)
            note = f"trained on a stratified {svm_sample:,}-row sample (kernel SVM is O(n²))"
        pipe = Pipeline([("pre", make_preprocessor(df)), ("model", model)])
        start = time.perf_counter()
        pipe.fit(Xtr, ytr)
        seconds = time.perf_counter() - start
        score = None
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)
            score = 1 - proba[:, list(pipe.classes_).index("normal")]
        results.append(evaluate(name, y_test, pipe.predict(X_test), seconds, ytr, pipe.predict(Xtr), score, note))
    return results


# ----------------------------------------------------------------- Unit 3
K_VALUES = [2, 3, 4, 5, 6, 8, 10]


def kmeans_study(df, X_train, X_test, y_train, y_test, progress=lambda m: None) -> dict:
    """K sweep with majority-class mapping (attack vs normal), then test evaluation and PCA."""
    pre = make_preprocessor(df)
    Z_train = pre.fit_transform(X_train)
    Z_test = pre.transform(X_test)
    b_train = (y_train != "normal").astype(int)
    b_test = (y_test != "normal").astype(int)
    Z_fit, Z_val, b_fit, b_val = train_test_split(Z_train, b_train, test_size=0.2, random_state=SEED, stratify=b_train)
    rng = np.random.default_rng(SEED)
    sil_idx = rng.choice(len(Z_val), size=min(8000, len(Z_val)), replace=False)

    def mapping(labels, truth, k):
        return np.array([round(truth[labels == c].mean()) if (labels == c).any() else 0 for c in range(k)])

    sweep = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Z_fit)
        m = mapping(model.labels_, b_fit, k)
        val = model.predict(Z_val)
        sweep.append({"k": k, "validation_accuracy": float((m[val] == b_val).mean()), "inertia": float(model.inertia_),
                      "silhouette": float(silhouette_score(Z_val[sil_idx], val[sil_idx]))})
        progress(f"K-means K={k}: validation accuracy {sweep[-1]['validation_accuracy']:.4f}")
    best_k = max(sweep, key=lambda r: r["validation_accuracy"])["k"]
    final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10).fit(Z_train)
    m = mapping(final.labels_, b_train, best_k)
    test_clusters = final.predict(Z_test)
    pred = m[test_clusters]
    profile = []
    train_frame = X_train.assign(category=y_train, cluster=final.labels_)
    for c in range(best_k):
        g = train_frame[train_frame["cluster"] == c]
        profile.append({"cluster": c, "connections": int(len(g)), "attack_share": float((g["category"] != "normal").mean()),
                        "dominant_category": g["category"].mode().iloc[0], "top_service": g["service"].mode().iloc[0],
                        "mean_count": float(g["count"].mean()), "mean_serror_rate": float(g["serror_rate"].mean()),
                        "mapped": "attack" if m[c] else "normal"})
    idx = rng.choice(len(Z_train), size=min(8000, len(Z_train)), replace=False)
    coords = PCA(n_components=2, random_state=SEED).fit_transform(Z_train[idx])
    sil = rng.choice(len(Z_test), size=min(8000, len(Z_test)), replace=False)
    return {
        "n_features": int(Z_train.shape[1]), "sweep": sweep, "best_k": best_k,
        "test_accuracy": float((pred == b_test).mean()),
        "test_silhouette": float(silhouette_score(Z_test[sil], test_clusters[sil])),
        "adjusted_rand_category": float(adjusted_rand_score(y_test, test_clusters)),
        "confusion_matrix": confusion_matrix(b_test, pred).tolist(),
        "profile": profile,
        "pca": {"x": coords[:, 0].tolist(), "y": coords[:, 1].tolist(),
                "cluster": final.labels_[idx].tolist(), "category": y_train[idx].tolist()},
    }


def clustering_comparison(df, X_train, y_train, k: int = 5, sample: int = 6000) -> list[dict]:
    """K-means vs Gaussian mixture (EM) vs Ward hierarchical clustering on a stratified sample."""
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=sample, random_state=SEED, stratify=y_train)
    Z = PCA(n_components=15, random_state=SEED).fit_transform(make_preprocessor(df).fit_transform(Xs))
    out = []
    for name, method in {"K-means": KMeans(n_clusters=k, random_state=SEED, n_init=10),
                         "Gaussian mixture (EM)": GaussianMixture(n_components=k, covariance_type="full", random_state=SEED),
                         "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=k, linkage="ward")}.items():
        start = time.perf_counter()
        labels = method.fit_predict(Z)
        out.append({"method": name, "silhouette": float(silhouette_score(Z, labels)),
                    "adjusted_rand": float(adjusted_rand_score(ys, labels)), "seconds": time.perf_counter() - start})
    return out


# ----------------------------------------------------------------- Unit 4
SYMBOLS = ["icmp", "tcp:SF", "tcp:S0", "tcp:REJ", "tcp:other", "udp"]


def markov_categories(stream: pd.DataFrame) -> dict:
    """First-order Markov chain over the attack category of consecutive connections in the raw stream."""
    codes = pd.Categorical(stream["attack_category"], categories=CATEGORIES).codes
    counts = np.zeros((5, 5), dtype=np.int64)
    np.add.at(counts, (codes[:-1], codes[1:]), 1)
    rows = counts.sum(1, keepdims=True)
    matrix = np.divide(counts, rows, out=np.zeros_like(counts, dtype=float), where=rows > 0)
    w, v = np.linalg.eig(matrix.T)
    stationary = np.abs(np.real(v[:, np.argmin(np.abs(w - 1))]))
    runs = np.diff(np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True]))
    return {"states": CATEGORIES, "counts": counts.tolist(), "matrix": matrix.tolist(),
            "stationary": (stationary / stationary.sum()).tolist(), "transitions": int(counts.sum()),
            "mean_run_length": float(runs.mean()), "longest_run": int(runs.max())}


def observation_symbols(stream: pd.DataFrame) -> np.ndarray:
    """Observable symbol of each connection: protocol, plus the TCP flag for TCP connections."""
    flag = stream["flag"].where(stream["flag"].isin(["SF", "S0", "REJ"]), "other")
    sym = np.where(stream["protocol_type"] == "tcp", "tcp:" + flag, stream["protocol_type"])
    return pd.Categorical(sym, categories=SYMBOLS).codes


def hmm_attack_phases(stream: pd.DataFrame, window: int = 200, n_train: int = 400, n_states: int = 4,
                      n_iter: int = 30) -> dict:
    """HMM: hidden attack phase → protocol/flag symbol. Baum–Welch, then Viterbi decoding.

    Each learned hidden state is mapped to the majority true label of the *training* windows it decodes
    (like the K-means cluster mapping); accuracy is measured on held-out windows only.
    """
    obs = observation_symbols(stream)
    truth = (stream["attack_category"] != "normal").to_numpy().astype(int)
    n_windows = len(obs) // window
    starts = np.random.default_rng(SEED).permutation(n_windows) * window
    train_starts, test_starts = starts[:n_train], starts[n_train:n_train + 200]
    model, trace = baum_welch([obs[s:s + window] for s in train_starts], K=n_states, M=len(SYMBOLS),
                              n_iter=n_iter, seed=SEED)

    def decode(windows):
        return (np.concatenate([model.viterbi(obs[s:s + window])[0] for s in windows]),
                np.concatenate([truth[s:s + window] for s in windows]))

    train_path, train_truth = decode(train_starts)
    mapping = np.array([round(train_truth[train_path == k].mean()) if (train_path == k).any() else 0
                        for k in range(n_states)])
    test_path, test_truth = decode(test_starts)
    pred = mapping[test_path]
    majority = float(max(test_truth.mean(), 1 - test_truth.mean()))  # always predicting the commoner class
    # Illustrate with a held-out window containing both phases: the one with median decoding accuracy
    # (neither the best nor the worst case).
    # Prefer windows where both phases are substantial (20–80% attack); fall back to any mixed window.
    mixed = ([s for s in test_starts if 0.2 <= truth[s:s + window].mean() <= 0.8]
             or [s for s in test_starts if 0 < truth[s:s + window].mean() < 1] or list(test_starts))
    window_acc = [float((mapping[model.viterbi(obs[s:s + window])[0]] == truth[s:s + window]).mean()) for s in mixed]
    order = np.argsort(window_acc)
    demo_start = int(mixed[order[len(order) // 2]])
    demo_obs = obs[demo_start:demo_start + window]
    posterior = model.sum_product(demo_obs)
    _, scales = model.forward(demo_obs)
    return {
        "symbols": SYMBOLS, "n_states": n_states, "train_sequences": len(train_starts), "window": window,
        "initial": model.pi.tolist(), "transition": model.A.tolist(), "emission": model.B.tolist(), "trace": trace,
        "state_meaning": {str(k): ("attack" if m else "normal") for k, m in enumerate(mapping)},
        "viterbi_accuracy": float((pred == test_truth).mean()), "majority_baseline": majority,
        "viterbi_confusion": confusion_matrix(test_truth, pred, labels=[0, 1]).tolist(),
        "demo": {"window_accuracy": window_acc[order[len(order) // 2]], "mixed_windows": len(mixed),
                 "observations": demo_obs.tolist(), "truth": truth[demo_start:demo_start + window].tolist(),
                 "viterbi_attack": mapping[model.viterbi(demo_obs)[0]].tolist(),
                 "posterior_attack": posterior[:, mapping == 1].sum(axis=1).tolist(),
                 "scaling_factors": scales.tolist(), "log_likelihood": float(np.log(scales).sum()),
                 "forward_unscaled": model.forward_unscaled(demo_obs[:20])[1]},
    }


# ----------------------------------------------------------------- Unit 5
def unit5_models(n_classes: int) -> dict:
    return {
        "Decision tree (CART)": DecisionTreeClassifier(max_depth=12, random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1,
                                                random_state=SEED, n_jobs=-1),
        "AdaBoost (stumps)": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100,
                                                random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                                 colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1,
                                 objective="multi:softprob", num_class=n_classes),
    }


def train_unit5(df, X_train, X_test, y_train, y_test) -> dict:
    enc = LabelEncoder().fit(CATEGORIES)
    results, fitted = [], {}
    for name, model in unit5_models(len(CATEGORIES)).items():
        pipe = Pipeline([("pre", make_preprocessor(df)), ("model", model)])
        start = time.perf_counter()
        pipe.fit(X_train, enc.transform(y_train))
        seconds = time.perf_counter() - start
        pred = enc.inverse_transform(pipe.predict(X_test))
        proba = pipe.predict_proba(X_test)
        score = 1 - proba[:, list(enc.classes_).index("normal")]
        results.append(evaluate(name, y_test, pred, seconds, y_train, enc.inverse_transform(pipe.predict(X_train)), score))
        fitted[name] = pipe
    tree = fitted["Decision tree (CART)"]
    return {"results": results,
            "feature_importance": {n: attribute_importance(fitted[n]) for n in ("Random forest", "XGBoost")},
            "tree": {"model": tree.named_steps["model"], "feature_names": list(tree.named_steps["pre"].get_feature_names_out()),
                     "class_names": list(enc.classes_)}}


def attribute_importance(pipe: Pipeline, top: int = 15) -> list[dict]:
    """Sum one-hot column importances back to their original attribute."""
    names = pipe.named_steps["pre"].get_feature_names_out()
    totals: dict[str, float] = {}
    for name, value in zip(names, pipe.named_steps["model"].feature_importances_):
        raw = name.split("__", 1)[1]
        attribute = next((c for c in CATEGORICAL if raw.startswith(c + "_")), raw)
        totals[attribute] = totals.get(attribute, 0.0) + float(value)
    return sorted([{"attribute": k, "importance": v} for k, v in totals.items()], key=lambda d: -d["importance"])[:top]


def cross_validation(df, X_train, y_train, sample: int = 30000) -> list[dict]:
    """5-fold stratified cross-validated macro-F1 on a stratified sample."""
    Xs, _, ys, _ = train_test_split(X_train, y_train, train_size=sample, random_state=SEED, stratify=y_train)
    enc = LabelEncoder().fit(CATEGORIES)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {"Logistic regression": LogisticRegression(max_iter=2000), **unit5_models(len(CATEGORIES))}
    candidates.pop("AdaBoost (stumps)")
    out = []
    for name, model in candidates.items():
        pipe = Pipeline([("pre", make_preprocessor(df)), ("model", model)])
        scores = cross_val_score(pipe, Xs, enc.transform(ys), cv=folds, scoring="f1_macro")
        out.append({"model": name, "mean_macro_f1": float(scores.mean()), "std": float(scores.std()),
                    "folds": scores.tolist(), "sample_size": int(len(Xs))})
    return out
