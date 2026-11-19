"""Unit 1 endpoints: descriptive statistics, probability, Bayes, polynomial fitting."""
from __future__ import annotations

from fastapi import APIRouter

from ..ml import polynomial, statistics
from ..schemas import BayesRequest, CovarianceRequest, DescribeRequest, DistributionRequest, PolynomialRequest
from ..services import datasets, experiments
from ..utils import to_jsonable

router = APIRouter(prefix="/datasets/{dataset_id}/stats", tags=["unit 1 - statistics"])


@router.post("/describe")
def describe(dataset_id: str, body: DescribeRequest):
    return to_jsonable(statistics.describe(datasets.load_df(dataset_id), body.columns))


@router.post("/covariance")
def covariance(dataset_id: str, body: CovarianceRequest):
    return to_jsonable(statistics.covariance(datasets.load_df(dataset_id), body.columns, body.method))


@router.post("/distribution")
def distribution(dataset_id: str, body: DistributionRequest):
    return to_jsonable(statistics.distribution(datasets.load_df(dataset_id), body.column, body.bins, body.lower, body.upper))


@router.post("/bayes")
def bayes(dataset_id: str, body: BayesRequest):
    out = to_jsonable(statistics.bayes(datasets.load_df(dataset_id), body.model_dump()))
    row = experiments.record(unit=1, task="probability", algorithm="bayes_rule", dataset_id=dataset_id,
                             params=body.model_dump(exclude_none=True),
                             metrics={"prior": out["prior"], "posterior": out["posterior"]}, results=out)
    out["experiment_id"] = row["id"]
    return out


@router.post("/polynomial")
def polynomial_fit(dataset_id: str, body: PolynomialRequest):
    with experiments.timer() as t:
        out = to_jsonable(polynomial.fit_polynomial(
            datasets.load_df(dataset_id), body.x_column, body.y_column, body.degree, body.lam, body.test_size,
            random_state=body.random_state))
    row = experiments.record(
        unit=1, task="regression", algorithm="polynomial_curve_fitting", dataset_id=dataset_id,
        params=body.model_dump(), features=[body.x_column], target=body.y_column,
        metrics={"train_rmse": out["train"]["rmse"], "test_rmse": out["test"]["rmse"], "r2": out["test"]["r2"]},
        results={k: v for k, v in out.items() if k not in ("points", "curve", "residuals")}, duration_ms=t["ms"])
    out["experiment_id"] = row["id"]
    return out
