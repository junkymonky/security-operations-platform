"""Experiment history, model persistence and prediction logging."""
from __future__ import annotations

import re
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd

from .. import db
from ..config import settings
from ..errors import NotFound, ValidationFailed
from ..utils import to_jsonable

_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@contextmanager
def timer() -> Iterator[dict]:
    box = {"ms": 0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = int((time.perf_counter() - start) * 1000)


def new_id() -> str:
    return uuid.uuid4().hex


def save_model(experiment_id: str, bundle: dict) -> str:
    """Persist a fitted model bundle: {'model', 'features', 'task', 'classes', ...}."""
    path = settings.models_dir / f"{experiment_id}.joblib"
    joblib.dump(bundle, path)
    return path.name


def record(
    *,
    unit: int,
    task: str,
    algorithm: str,
    dataset_id: str | None,
    params: dict,
    metrics: dict,
    results: Any,
    features: list[str] | None = None,
    target: str | None = None,
    duration_ms: int | None = None,
    group_id: str | None = None,
    experiment_id: str | None = None,
    model_file: str | None = None,
) -> dict:
    experiment_id = experiment_id or new_id()
    row = db.fetch_one(
        """INSERT INTO ml_experiments
           (id, group_id, dataset_id, unit, task, algorithm, params, features, target,
            metrics, results, model_path, duration_ms)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           RETURNING id, group_id, dataset_id, unit, task, algorithm, created_at""",
        (
            experiment_id,
            group_id,
            dataset_id,
            unit,
            task,
            algorithm,
            db.jsonb(to_jsonable(params)),
            db.jsonb(features) if features is not None else None,
            target,
            db.jsonb(to_jsonable(metrics)),
            db.jsonb(to_jsonable(results)),
            model_file,
            duration_ms,
        ),
    )
    return row


def list_experiments(
    dataset_id: str | None = None, unit: int | None = None, limit: int = 100
) -> list[dict]:
    clauses, params = [], []
    if dataset_id:
        clauses.append("e.dataset_id = %s")
        params.append(dataset_id)
    if unit:
        clauses.append("e.unit = %s")
        params.append(unit)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    return db.fetch_all(
        f"""SELECT e.id, e.group_id, e.dataset_id, d.name AS dataset_name, e.unit, e.task,
                   e.algorithm, e.params, e.features, e.target, e.metrics, e.duration_ms,
                   (e.model_path IS NOT NULL) AS has_model, e.created_at,
                   (SELECT count(*) FROM ml_predictions p WHERE p.experiment_id = e.id) AS prediction_count
            FROM ml_experiments e LEFT JOIN ml_datasets d ON d.id = e.dataset_id
            {where} ORDER BY e.created_at DESC LIMIT %s""",
        tuple(params),
    )


def get_experiment(experiment_id: str) -> dict:
    if not _ID_RE.match(experiment_id or ""):
        raise NotFound("Experiment not found.")
    row = db.fetch_one(
        """SELECT e.*, d.name AS dataset_name FROM ml_experiments e
           LEFT JOIN ml_datasets d ON d.id = e.dataset_id WHERE e.id = %s""",
        (experiment_id,),
    )
    if not row:
        raise NotFound("Experiment not found.")
    row["has_model"] = row.pop("model_path") is not None
    return row


def delete_experiment(experiment_id: str) -> None:
    exp = get_experiment(experiment_id)
    db.execute("DELETE FROM ml_experiments WHERE id = %s", (experiment_id,))
    if exp["has_model"]:
        (settings.models_dir / f"{experiment_id}.joblib").unlink(missing_ok=True)


def _load_bundle(experiment_id: str) -> dict:
    exp = get_experiment(experiment_id)
    path = settings.models_dir / f"{experiment_id}.joblib"
    if not exp["has_model"] or not path.is_file():
        raise ValidationFailed("This experiment has no saved model to predict with.")
    return joblib.load(path)


def predict(experiment_id: str, records: list[dict]) -> dict:
    if not records:
        raise ValidationFailed("Provide at least one record to predict.")
    if len(records) > 1000:
        raise ValidationFailed("At most 1000 records per prediction request.")
    bundle = _load_bundle(experiment_id)
    features: list[str] = bundle["features"]
    missing = sorted({f for r in records for f in features if f not in r})
    if missing:
        raise ValidationFailed(f"Records are missing feature(s): {', '.join(missing)}")
    X = pd.DataFrame([{f: r.get(f) for f in features} for r in records])
    for col in bundle.get("numeric_features", []):
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].isna().any():
            raise ValidationFailed(f"Feature '{col}' must be numeric.")
    model = bundle["model"]
    try:
        raw = model.predict(X)
    except Exception as exc:  # surface model errors as validation problems
        raise ValidationFailed(f"Prediction failed: {exc}") from exc

    outputs: list[dict] = []
    classes = bundle.get("classes")
    proba = None
    if bundle["task"] == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
        except Exception:
            proba = None
    for i, value in enumerate(raw):
        out: dict[str, Any] = {}
        if bundle["task"] == "classification" and classes is not None:
            out["prediction"] = classes[int(value)]
            if proba is not None:
                out["probabilities"] = {str(c): float(p) for c, p in zip(classes, proba[i])}
        elif bundle["task"] == "clustering":
            out["cluster"] = int(value)
            if hasattr(model, "predict_proba"):  # GMM responsibilities
                out["probabilities"] = model.predict_proba(X.iloc[[i]])[0].tolist()
        else:
            out["prediction"] = float(value)
        outputs.append(out)
    outputs = to_jsonable(outputs)
    with db.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ml_predictions (experiment_id, inputs, outputs) VALUES (%s,%s,%s)",
                [(experiment_id, db.jsonb(to_jsonable(r)), db.jsonb(o)) for r, o in zip(records, outputs)],
            )
    return {"experiment_id": experiment_id, "task": bundle["task"], "predictions": outputs}


def list_predictions(experiment_id: str, limit: int = 100) -> list[dict]:
    get_experiment(experiment_id)
    return db.fetch_all(
        "SELECT id, inputs, outputs, created_at FROM ml_predictions WHERE experiment_id = %s "
        "ORDER BY created_at DESC LIMIT %s",
        (experiment_id, limit),
    )


def numeric_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)
