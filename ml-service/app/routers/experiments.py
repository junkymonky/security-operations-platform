"""Experiment history and prediction endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Query

from ..schemas import PredictRequest
from ..services import experiments as svc
from ..utils import to_jsonable

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("")
def list_experiments(dataset_id: str | None = None, unit: int | None = Query(None, ge=1, le=5),
                     limit: int = Query(100, ge=1, le=500)):
    return to_jsonable(svc.list_experiments(dataset_id, unit, limit))


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str):
    return to_jsonable(svc.get_experiment(experiment_id))


@router.delete("/{experiment_id}", status_code=204)
def delete_experiment(experiment_id: str):
    svc.delete_experiment(experiment_id)


@router.post("/{experiment_id}/predict")
def predict(experiment_id: str, body: PredictRequest):
    return svc.predict(experiment_id, body.records)


@router.get("/{experiment_id}/predictions")
def predictions(experiment_id: str, limit: int = Query(100, ge=1, le=500)):
    return to_jsonable(svc.list_predictions(experiment_id, limit))
