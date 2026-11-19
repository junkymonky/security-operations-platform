"""Unit 5 — adaptive basis-function models and ensembles.

* cart              : Classification and Regression Trees (structure exported).
* random_forest     : bagged decorrelated trees.
* bagging           : bootstrap aggregation of full trees.
* adaboost          : boosting of decision stumps (weak learners), staged accuracy.
* gradient_boosting : functional gradient boosting of shallow trees.
* voting            : heterogeneous ensemble (soft voting / averaging).
* gam               : generalized additive model, f(x) = α + Σ_j f_j(x_j) with
                      cubic B-spline bases and an L2 penalty (penalised regression splines).
* Bayesian model averaging over linear sub-models using BIC weights (bma()).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.utils.validation import check_is_fitted

from ..errors import ValidationFailed
from .metrics import classification_metrics, regression_metrics

ALGORITHMS = {
    "cart": "Decision tree (CART)",
    "random_forest": "Random forest",
    "bagging": "Bagging",
    "adaboost": "AdaBoost",
    "gradient_boosting": "Gradient boosting",
    "voting": "Voting ensemble",
    "gam": "Generalized additive model",
}


def _int(p, key, default, lo, hi):
    v = p.get(key, default)
    if v is None:
        return None
    v = int(v)
    if not lo <= v <= hi:
        raise ValidationFailed(f"{key} must be between {lo} and {hi}.")
    return v


# ------------------------------------------------------------------- GAM
class GeneralizedAdditiveModel(BaseEstimator):
    """Additive model with one cubic B-spline basis per (non-binary) feature."""

    def __init__(self, task: str = "regression", n_knots: int = 6, lam: float = 1.0):
        self.task = task
        self.n_knots = n_knots
        self.lam = lam

    def _basis(self, X, fit: bool):
        X = np.asarray(X, float)
        blocks = []
        for j in range(X.shape[1]):
            col = X[:, [j]]
            if fit:
                if len(np.unique(col)) > 3:
                    st = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False, extrapolation="linear")
                    st.fit(col)
                    self.splines_.append(st)
                else:
                    self.splines_.append(None)
            st = self.splines_[j]
            blocks.append(st.transform(col) if st is not None else col)
        sizes = [b.shape[1] for b in blocks]
        return np.hstack(blocks), sizes

    def fit(self, X, y):
        self.splines_ = []
        B, sizes = self._basis(X, fit=True)
        self.slices_ = np.cumsum([0] + sizes)
        if self.task == "classification":
            self.model_ = LogisticRegression(C=1.0 / max(self.lam, 1e-6), max_iter=3000).fit(B, y)
            self.classes_ = self.model_.classes_
        else:
            self.model_ = Ridge(alpha=self.lam).fit(B, y)
        self.n_features_in_ = np.asarray(X).shape[1]
        self.X_range_ = (np.min(X, axis=0), np.max(X, axis=0))
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        B, _ = self._basis(X, fit=False)
        return self.model_.predict(B)

    def predict_proba(self, X):
        B, _ = self._basis(X, fit=False)
        return self.model_.predict_proba(B)

    def shape_functions(self, feature_names: list[str], points: int = 40) -> list[dict]:
        """Centred partial function f_j(x_j) for each feature (first class for multinomial)."""
        coef = np.atleast_2d(self.model_.coef_)[0]
        out = []
        lo, hi = self.X_range_
        for j, name in enumerate(feature_names):
            s, e = self.slices_[j], self.slices_[j + 1]
            grid = np.linspace(lo[j], hi[j], points)[:, None]
            st = self.splines_[j]
            basis = st.transform(grid) if st is not None else grid
            f = basis @ coef[s:e]
            f = f - f.mean()
            out.append({"feature": name, "points": [{"x": float(g), "f": float(v)} for g, v in zip(grid[:, 0], f)],
                        "range": float(f.max() - f.min())})
        out.sort(key=lambda d: -d["range"])
        return out


class GAMClassifier(GeneralizedAdditiveModel, ClassifierMixin):
    pass


class GAMRegressor(GeneralizedAdditiveModel, RegressorMixin):
    pass


# ------------------------------------------------------------------- builders
def build(name: str, p: dict, task: str):
    clf = task == "classification"
    seed = 0
    if name == "cart":
        criterion = p.get("criterion") or ("gini" if clf else "squared_error")
        allowed = ("gini", "entropy", "log_loss") if clf else ("squared_error", "absolute_error", "friedman_mse")
        if criterion not in allowed:
            raise ValidationFailed(f"criterion must be one of {', '.join(allowed)}.")
        kw = dict(
            criterion=criterion,
            max_depth=_int(p, "max_depth", 5, 1, 30),
            min_samples_split=_int(p, "min_samples_split", 2, 2, 1000),
            min_samples_leaf=_int(p, "min_samples_leaf", 1, 1, 1000),
            ccp_alpha=float(p.get("ccp_alpha", 0.0)),
            random_state=seed,
        )
        return DecisionTreeClassifier(**kw) if clf else DecisionTreeRegressor(**kw)
    n_est = _int(p, "n_estimators", 100, 1, 1000)
    if name == "random_forest":
        mf = p.get("max_features", "sqrt")
        mf = None if mf in (None, "all", "none") else (float(mf) if isinstance(mf, (int, float)) else mf)
        cls = RandomForestClassifier if clf else RandomForestRegressor
        return cls(n_estimators=n_est, max_depth=_int(p, "max_depth", None, 1, 50), max_features=mf,
                   oob_score=True, random_state=seed, n_jobs=-1)
    if name == "bagging":
        base = DecisionTreeClassifier(random_state=seed) if clf else DecisionTreeRegressor(random_state=seed)
        cls = BaggingClassifier if clf else BaggingRegressor
        # Single process: bagging's process pool is re-spawned on every predict() call on Windows, which made
        # permutation importance ~10x slower than the model itself. Small trees train fast serially.
        return cls(estimator=base, n_estimators=n_est, oob_score=True, random_state=seed, n_jobs=1)
    if name == "adaboost":
        depth = _int(p, "base_depth", 1, 1, 10)
        lr = float(p.get("learning_rate", 1.0))
        if clf:
            return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=n_est,
                                      learning_rate=lr, random_state=seed)
        return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=depth or 3), n_estimators=n_est,
                                 learning_rate=lr, random_state=seed)
    if name == "gradient_boosting":
        # Histogram-based gradient boosting: same stage-wise algorithm, multi-threaded and far faster than the
        # classic implementation (which fits one tree per class per stage on a single core).
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(max_iter=n_est, learning_rate=float(p.get("learning_rate", 0.1)),
                   max_depth=_int(p, "max_depth", 3, 1, 20), early_stopping=False, random_state=seed)
    if name == "voting":
        if clf:
            return VotingClassifier(
                [("logistic", LogisticRegression(max_iter=2000)), ("tree", DecisionTreeClassifier(max_depth=5, random_state=seed)),
                 ("naive_bayes", GaussianNB())], voting="soft")
        return VotingRegressor(
            [("linear", LinearRegression()), ("tree", DecisionTreeRegressor(max_depth=5, random_state=seed)),
             ("ridge", Ridge(alpha=1.0))])
    if name == "gam":
        kw = dict(task=task, n_knots=_int(p, "n_knots", 6, 3, 20), lam=float(p.get("lam", 1.0)))
        return GAMClassifier(**kw) if clf else GAMRegressor(**kw)
    raise ValidationFailed(f"Unknown ensemble algorithm '{name}'.")


# ------------------------------------------------------------------- inspection
def tree_structure(tree, feature_names: list[str], classes: list[str] | None, max_depth: int = 6) -> dict:
    t = tree.tree_

    def node(i: int, depth: int) -> dict:
        value = t.value[i][0]
        n = {"id": int(i), "samples": int(t.n_node_samples[i]), "impurity": float(t.impurity[i])}
        if classes is not None:
            dist = value / value.sum() if value.sum() else value
            n["class"] = classes[int(np.argmax(value))]
            n["distribution"] = [float(v) for v in dist]
        else:
            n["value"] = float(value[0])
        if t.children_left[i] != -1:
            n["feature"] = feature_names[t.feature[i]]
            n["threshold"] = float(t.threshold[i])
            if depth < max_depth:
                n["children"] = [node(t.children_left[i], depth + 1), node(t.children_right[i], depth + 1)]
            else:
                n["truncated"] = True
        return n

    return node(0, 0)


def _score(task, y, pred):
    return accuracy_score(y, pred) if task == "classification" else float(np.sqrt(mean_squared_error(y, pred)))


def details(name, model, feature_names, task, X_train, y_train, X_test, y_test, classes) -> dict:
    out: dict = {"score_name": "accuracy" if task == "classification" else "rmse"}
    if hasattr(model, "feature_importances_"):
        out["feature_importance"] = sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(feature_names, model.feature_importances_)],
            key=lambda d: -d["importance"])
    try:
        # Permutation importance re-scores the model n_features × n_repeats times; a 1,500-row random sample of
        # the test set gives near-identical estimates at a fraction of the cost.
        Xp, yp = X_test, y_test
        if len(y_test) > 1500:
            idx = np.random.default_rng(0).choice(len(y_test), 1500, replace=False)
            Xp, yp = X_test[idx], np.asarray(y_test)[idx]
        out["permutation_sample"] = int(len(yp))
        perm = permutation_importance(model, Xp, yp, n_repeats=3, random_state=0, n_jobs=1)
        out["permutation_importance"] = sorted(
            [{"feature": f, "importance": float(m), "std": float(s)}
             for f, m, s in zip(feature_names, perm.importances_mean, perm.importances_std)],
            key=lambda d: -d["importance"])[:25]
    except Exception:
        pass
    if hasattr(model, "oob_score_"):
        out["oob_score"] = float(model.oob_score_)

    if name == "cart":
        out["tree"] = tree_structure(model, feature_names, classes)
        out["depth"] = int(model.get_depth())
        out["n_leaves"] = int(model.get_n_leaves())
        out["text"] = export_text(model, feature_names=list(feature_names), max_depth=4)[:6000]
        path = model.cost_complexity_pruning_path(X_train, y_train)
        alphas = np.unique(path.ccp_alphas)
        if len(alphas) > 20:
            alphas = alphas[np.linspace(0, len(alphas) - 1, 20).astype(int)]
        prune = []
        for a in alphas:
            m = clone(model).set_params(ccp_alpha=float(a), max_depth=None).fit(X_train, y_train)
            prune.append({"ccp_alpha": float(a), "leaves": int(m.get_n_leaves()),
                          "train": _score(task, y_train, m.predict(X_train)),
                          "test": _score(task, y_test, m.predict(X_test))})
        out["pruning_path"] = prune

    if name in ("adaboost", "gradient_boosting"):
        staged = []
        for i, pred in enumerate(model.staged_predict(X_test), start=1):
            staged.append({"n_estimators": i, "test": _score(task, y_test, pred)})
        step = max(1, len(staged) // 100)
        out["staged"] = staged[::step] + ([staged[-1]] if staged and len(staged) % step else [])
    if name == "adaboost":
        learners = []
        for i, (est, w, err) in enumerate(zip(model.estimators_, model.estimator_weights_, model.estimator_errors_)):
            if i >= 50:
                break
            learners.append({"index": i + 1, "weight": float(w), "weighted_error": float(err),
                             "test_score_alone": _score(task, y_test, est.predict(X_test))})
        out["weak_learners"] = learners
    if name in ("random_forest", "bagging"):
        member_scores = []
        for est, feats in zip(model.estimators_[:100], getattr(model, "estimators_features_", [None] * len(model.estimators_))):
            Xt = X_test if feats is None else X_test[:, feats]
            pred = est.predict(Xt)
            if task == "classification":
                pred = pred.astype(int)
            member_scores.append(_score(task, y_test, pred))
        out["member_scores"] = {"mean": float(np.mean(member_scores)), "min": float(np.min(member_scores)),
                                "max": float(np.max(member_scores)), "values": member_scores}
    if name == "voting":
        out["members"] = [{"name": n, "test": _score(task, y_test, est.predict(X_test))}
                          for n, est in zip(model.named_estimators_.keys(), model.estimators_)]
    if name == "gam":
        out["shape_functions"] = model.shape_functions(feature_names)[:12]
    return out


# ------------------------------------------------------------------- BMA
def bma(X_train, y_train, X_test, y_test, feature_names, task, classes, max_features: int = 8) -> dict:
    """Bayesian model averaging over all feature subsets of a linear / logistic model.

    The marginal likelihood p(D|M) is approximated with BIC:  ln p(D|M) ≈ ln L̂ − (k/2) ln N,
    so posterior model weights are  w_M ∝ exp(−BIC_M / 2)  under a uniform model prior.
    """
    n, d = X_train.shape
    if d > max_features:
        # Keep the features most correlated with the target to bound the 2^d model space.
        y_num = y_train.astype(float)
        corr = np.nan_to_num([abs(np.corrcoef(X_train[:, j], y_num)[0, 1]) for j in range(d)])
        keep = np.argsort(corr)[::-1][:max_features]
    else:
        keep = np.arange(d)
    keep = sorted(keep.tolist())
    models = []
    for r in range(0, len(keep) + 1):
        for subset in combinations(keep, r):
            cols = list(subset)
            if task == "classification":
                if cols:
                    m = LogisticRegression(C=1e4, max_iter=3000).fit(X_train[:, cols], y_train)
                    proba_tr = m.predict_proba(X_train[:, cols])
                    proba_te = m.predict_proba(X_test[:, cols])
                else:
                    prior = np.bincount(y_train, minlength=len(classes)) / n
                    proba_tr = np.tile(prior, (n, 1))
                    proba_te = np.tile(prior, (len(X_test), 1))
                ll = float(np.sum(np.log(proba_tr[np.arange(n), y_train] + 1e-12)))
                k = (len(cols) + 1) * (len(classes) - 1)
                pred = proba_te
            else:
                if cols:
                    m = LinearRegression().fit(X_train[:, cols], y_train)
                    res = y_train - m.predict(X_train[:, cols])
                    pred = m.predict(X_test[:, cols])
                else:
                    res = y_train - y_train.mean()
                    pred = np.full(len(X_test), y_train.mean())
                s2 = max(float(res @ res / n), 1e-12)
                ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1)
                k = len(cols) + 2
            bic = -2 * ll + k * np.log(n)
            models.append({"features": [feature_names[c] for c in cols], "cols": cols, "bic": bic, "pred": pred})
    bics = np.array([m["bic"] for m in models])
    w = np.exp(-(bics - bics.min()) / 2)
    w /= w.sum()
    order = np.argsort(-w)

    if task == "classification":
        avg = sum(wi * m["pred"] for wi, m in zip(w, models))
        bma_metrics = classification_metrics(y_test, avg.argmax(1), classes, avg)
        best = models[order[0]]
        best_metrics = classification_metrics(y_test, best["pred"].argmax(1), classes, best["pred"])
        full = next(m for m in models if len(m["cols"]) == len(keep))
        full_metrics = classification_metrics(y_test, full["pred"].argmax(1), classes, full["pred"])
    else:
        avg = sum(wi * m["pred"] for wi, m in zip(w, models))
        bma_metrics = regression_metrics(y_test, avg)
        best = models[order[0]]
        best_metrics = regression_metrics(y_test, best["pred"])
        full = next(m for m in models if len(m["cols"]) == len(keep))
        full_metrics = regression_metrics(y_test, full["pred"])

    inclusion = {feature_names[c]: float(sum(wi for wi, m in zip(w, models) if c in m["cols"])) for c in keep}
    strip = lambda m: {k: v for k, v in m.items() if k not in ("confusion_matrix", "per_class", "roc_curve", "labels")}
    return {
        "n_models": len(models),
        "candidate_features": [feature_names[c] for c in keep],
        "top_models": [{"rank": i + 1, "features": models[j]["features"], "bic": float(models[j]["bic"]),
                        "posterior": float(w[j])} for i, j in enumerate(order[:15])],
        "inclusion_probability": sorted([{"feature": f, "probability": p} for f, p in inclusion.items()],
                                        key=lambda d: -d["probability"]),
        "comparison": [
            {"model": "Bayesian model average", **strip(bma_metrics)},
            {"model": "Best single model (MAP)", **strip(best_metrics)},
            {"model": "Full model (all candidates)", **strip(full_metrics)},
        ],
    }
