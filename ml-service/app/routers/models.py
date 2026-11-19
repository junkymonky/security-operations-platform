"""Supervised (Units 2 & 5), unsupervised (Unit 3) and sequential (Unit 4) endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from ..schemas import (
    BMARequest,
    ClusterRequest,
    DecomposeRequest,
    HMMDecodeRequest,
    HMMFitRequest,
    LDSRequest,
    MarkovRequest,
    TrainRequest,
)
from ..services import sequential, supervised, unsupervised

router = APIRouter(tags=["models"])


@router.get("/algorithms")
def algorithms():
    return supervised.catalog()


# Units 2 & 5 -------------------------------------------------------------
@router.post("/datasets/{dataset_id}/train", tags=["unit 2 / unit 5 - supervised"])
def train(dataset_id: str, body: TrainRequest):
    return supervised.train(dataset_id, body.model_dump())


@router.post("/datasets/{dataset_id}/bma", tags=["unit 5 - ensembles"])
def bayesian_model_averaging(dataset_id: str, body: BMARequest):
    return supervised.bma(dataset_id, body.model_dump())


# Unit 3 ------------------------------------------------------------------
@router.post("/datasets/{dataset_id}/cluster", tags=["unit 3 - clustering"])
def cluster(dataset_id: str, body: ClusterRequest):
    return unsupervised.cluster(dataset_id, body.model_dump())


@router.post("/datasets/{dataset_id}/decompose", tags=["unit 3 - dimensionality reduction"])
def decompose(dataset_id: str, body: DecomposeRequest):
    return unsupervised.decompose(dataset_id, body.model_dump())


# Unit 4 ------------------------------------------------------------------
@router.post("/datasets/{dataset_id}/markov", tags=["unit 4 - sequential"])
def markov(dataset_id: str, body: MarkovRequest):
    return sequential.markov(dataset_id, body.model_dump())


@router.post("/datasets/{dataset_id}/hmm", tags=["unit 4 - sequential"])
def hmm_fit(dataset_id: str, body: HMMFitRequest):
    return sequential.hmm_fit(dataset_id, body.model_dump())


@router.post("/hmm/decode", tags=["unit 4 - sequential"])
def hmm_decode(body: HMMDecodeRequest):
    return sequential.hmm_decode(body.model_dump())


@router.post("/datasets/{dataset_id}/lds", tags=["unit 4 - sequential"])
def lds(dataset_id: str, body: LDSRequest):
    return sequential.lds_run(dataset_id, body.model_dump())
