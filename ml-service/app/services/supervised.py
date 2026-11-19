"""Train / evaluate / persist supervised models (Units 2 and 5).

Every algorithm is wrapped in a Pipeline(preprocessor → estimator) so the saved
model accepts raw feature records at prediction time.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ..errors import ValidationFailed
from ..ml import classification, ensemble, regression, svm
from ..ml.metrics import classification_metrics, regression_metrics
from ..ml.preprocessing import make_preprocessor, prepare_supervised
from ..utils import subsample_indices, to_jsonable
from . import datasets, experiments

UNIT2_REGRESSION = set(regression.ALGORITHMS) | {"svm"}
UNIT2_CLASSIFICATION = set(classification.ALGORITHMS) | {"svm"}
UNIT5 = set(ensemble.ALGORITHMS)

LABELS = {**regression.ALGORITHMS, **classification.ALGORITHMS, **ensemble.ALGORITHMS, "svm": "Support vector machine"}


def catalog() -> dict:
    return {
        "regression": {k: LABELS[k] for k in [*regression.ALGORITHMS, "svm", *ensemble.ALGORITHMS]},
        "classification": {k: LABELS[k] for k in [*classification.ALGORITHMS, "svm", *ensemble.ALGORITHMS]},
    }


def _build(name: str, params: dict, task: str, n_classes: int):
    if name == "svm":
        return svm.build_classifier(params) if task == "classification" else svm.build_regressor(params)
    if name in ensemble.ALGORITHMS:
        return ensemble.build(name, params, task)
    if task == "regression":
        return regression.build(name, params)
    return classification.build(name, params, n_classes)


def _unit(name: str) -> int:
    return 5 if name in UNIT5 else 2


def _decision_surface(model, Z_train, y_train, Z_test, y_test, grid_n: int = 45) -> dict | None:
    """Re-fit the same estimator on a 2-D PCA projection to draw its decision regions.

    The plot is illustrative, so the refit uses at most 3,000 training points to keep it fast.
    """
    if len(Z_train) > 3000:
        idx = np.random.default_rng(0).choice(len(Z_train), 3000, replace=False)
        Z_train, y_train = Z_train[idx], y_train[idx]
    try:
        m = clone(model).fit(Z_train, y_train)
    except Exception:
        return None
    lo = np.minimum(Z_train.min(0), Z_test.min(0))
    hi = np.maximum(Z_train.max(0), Z_test.max(0))
    pad = (hi - lo) * 0.05
    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], grid_n)
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], grid_n)
    gx, gy = np.meshgrid(xs, ys)
    pred = m.predict(np.column_stack([gx.ravel(), gy.ravel()])).reshape(grid_n, grid_n)
    idx = subsample_indices(len(Z_test), 400)
    return {
        "x": xs.tolist(),
        "y": ys.tolist(),
        "grid": pred.astype(int).tolist(),
        "points": [{"x": Z_test[i, 0], "y": Z_test[i, 1], "label": int(y_test[i])} for i in idx],
        "accuracy_2d": float(np.mean(m.predict(Z_test) == y_test)),
    }


def train(dataset_id: str, req: dict) -> dict:
    task = req["task"]
    if task not in ("regression", "classification"):
        raise ValidationFailed("task must be 'regression' or 'classification'.")
    algos: list[dict] = req.get("algorithms") or []
    if not algos:
        raise ValidationFailed("Select at least one algorithm.")
    if len(algos) > 10:
        raise ValidationFailed("At most 10 algorithms per comparison.")
    allowed = (UNIT2_REGRESSION if task == "regression" else UNIT2_CLASSIFICATION) | UNIT5
    for a in algos:
        if a["name"] not in allowed:
            raise ValidationFailed(f"Algorithm '{a['name']}' is not available for {task}.")

    datasets.get_meta(dataset_id)
    df = datasets.load_df(dataset_id)
    data = prepare_supervised(
        df, req.get("features"), req["target"], task, float(req.get("test_size", 0.25)), int(req.get("random_state", 42))
    )
    scale, impute = req.get("scale", "standard"), req.get("impute", "mean")
    n_classes = len(data.classes) if data.classes else 0
    cv_folds = int(req.get("cv_folds", 0) or 0)
    if cv_folds and not 2 <= cv_folds <= 10:
        raise ValidationFailed("cv_folds must be between 2 and 10 (or 0 to disable).")

    # Shared 2-D projection for decision boundaries
    prep_for_plot = make_preprocessor(data.numeric, data.categorical, scale="standard", impute=impute)
    Xt_tr = prep_for_plot.fit_transform(data.X_train)
    Xt_te = prep_for_plot.transform(data.X_test)
    Z_tr = Z_te = None
    if task == "classification" and Xt_tr.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=0).fit(Xt_tr)
        Z_tr, Z_te = pca.transform(Xt_tr), pca.transform(Xt_te)

    group_id = experiments.new_id() if len(algos) > 1 else None
    results = []
    for spec in algos:
        name, params = spec["name"], spec.get("params") or {}
        est = _build(name, params, task, n_classes)
        prep = make_preprocessor(data.numeric, data.categorical, scale=scale, impute=impute)
        pipe = Pipeline([("prep", prep), ("model", est)])
        with experiments.timer() as t:
            try:
                pipe.fit(data.X_train, data.y_train)
            except ValueError as exc:
                raise ValidationFailed(f"{LABELS[name]} failed to train: {exc}") from exc
        model = pipe.named_steps["model"]
        feat_names = list(pipe.named_steps["prep"].get_feature_names_out())
        Xtr = pipe.named_steps["prep"].transform(data.X_train)
        Xte = pipe.named_steps["prep"].transform(data.X_test)
        pred_te = pipe.predict(data.X_test)
        pred_tr = pipe.predict(data.X_train)

        payload: dict = {}
        if task == "classification":
            proba = pipe.predict_proba(data.X_test) if hasattr(pipe, "predict_proba") else None
            metrics = classification_metrics(data.y_test, pred_te, data.classes, proba)
            metrics["train_accuracy"] = float(np.mean(pred_tr == data.y_train))
        else:
            metrics = regression_metrics(data.y_test, pred_te)
            metrics["train_rmse"] = regression_metrics(data.y_train, pred_tr)["rmse"]
            idx = subsample_indices(len(pred_te), 400)
            std = None
            if name == "bayesian_linear":
                _, std = model.predict(Xte, return_std=True)
            payload["predicted_vs_actual"] = [
                {"actual": data.y_test[i], "predicted": pred_te[i], **({"std": std[i]} if std is not None else {})}
                for i in idx
            ]

        if cv_folds:
            splitter = (StratifiedKFold(cv_folds, shuffle=True, random_state=0) if task == "classification"
                        else KFold(cv_folds, shuffle=True, random_state=0))
            scoring = "accuracy" if task == "classification" else "neg_root_mean_squared_error"
            scores = cross_val_score(clone(pipe), data.X, data.y, cv=splitter, scoring=scoring)
            scores = scores if task == "classification" else -scores
            metrics["cv"] = {"metric": "accuracy" if task == "classification" else "rmse",
                             "mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}

        if name == "svm":
            payload["details"] = svm.details(model)
            payload["details"]["support_fraction"] = len(model.support_) / len(data.y_train)
            payload["details"]["kernel_matrix"] = svm.kernel_matrix(Xtr, params, n=12)
        elif name in ensemble.ALGORITHMS:
            payload["details"] = ensemble.details(name, model, feat_names, task, Xtr, data.y_train, Xte, data.y_test,
                                                  data.classes)
        elif task == "regression":
            payload["details"] = regression.details(name, model, feat_names, Xtr, data.y_train, params)
        else:
            payload["details"] = classification.details(name, model, feat_names, data.classes)

        if Z_tr is not None and name not in ("voting",):
            payload["decision_surface"] = _decision_surface(est, Z_tr, data.y_train, Z_te, data.y_test)

        exp_id = experiments.new_id()
        model_file = experiments.save_model(exp_id, {
            "model": pipe, "features": data.features, "numeric_features": data.numeric,
            "task": task, "classes": data.classes, "target": req["target"], "algorithm": name,
        })
        result = to_jsonable({
            "algorithm": name,
            "label": LABELS[name],
            "params": params,
            "metrics": metrics,
            "duration_ms": t["ms"],
            "n_train": len(data.y_train),
            "n_test": len(data.y_test),
            "feature_names": feat_names,
            **payload,
        })
        summary_metrics = {k: v for k, v in result["metrics"].items() if not isinstance(v, (list, dict))}
        if "cv" in result["metrics"]:
            summary_metrics["cv_mean"] = result["metrics"]["cv"]["mean"]
        row = experiments.record(
            experiment_id=exp_id, unit=_unit(name), task=task, algorithm=name, dataset_id=dataset_id,
            params={**params, "scale": scale, "impute": impute, "test_size": req.get("test_size", 0.25)},
            metrics=summary_metrics, results=result, features=data.features, target=req["target"],
            duration_ms=t["ms"], group_id=group_id, model_file=model_file,
        )
        result["experiment_id"] = row["id"]
        results.append(result)

    sample = data.X_test.head(1).to_dict(orient="records")[0]
    return {
        "task": task,
        "target": req["target"],
        "features": data.features,
        "classes": data.classes,
        "group_id": group_id,
        "notes": data.notes,
        "sample_record": to_jsonable(sample),
        "results": results,
    }


def bma(dataset_id: str, req: dict) -> dict:
    task = req["task"]
    df = datasets.load_df(dataset_id)
    data = prepare_supervised(df, req.get("features"), req["target"], task, float(req.get("test_size", 0.25)),
                              int(req.get("random_state", 42)))
    prep = make_preprocessor(data.numeric, data.categorical, scale="standard")
    Xtr = prep.fit_transform(data.X_train)
    Xte = prep.transform(data.X_test)
    names = list(prep.get_feature_names_out())
    max_f = int(req.get("max_features", 8))
    if not 1 <= max_f <= 10:
        raise ValidationFailed("max_features must be between 1 and 10.")
    with experiments.timer() as t:
        out = ensemble.bma(Xtr, data.y_train, Xte, data.y_test, names, task, data.classes, max_features=max_f)
    out = to_jsonable(out)
    bma_row = out["comparison"][0]
    metrics = {k: v for k, v in bma_row.items() if k != "model" and not isinstance(v, (list, dict))}
    row = experiments.record(unit=5, task=task, algorithm="bayesian_model_averaging", dataset_id=dataset_id,
                             params={"max_features": max_f}, metrics=metrics, results=out, features=data.features,
                             target=req["target"], duration_ms=t["ms"])
    out["experiment_id"] = row["id"]
    return out
