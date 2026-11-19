"""Request models; FastAPI rejects malformed bodies with 422 before services run."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Column = str
Scale = Literal["standard", "minmax", "none"]


class PreprocessRequest(BaseModel):
    name: str | None = Field(None, max_length=120)
    drop_columns: list[Column] = []
    drop_duplicates: bool = False
    missing: Literal["none", "drop", "mean", "median"] = "none"
    outlier_zscore: float | None = Field(None, gt=0, le=10)
    outlier_columns: list[Column] | None = None
    scale: Scale = "none"
    scale_columns: list[Column] | None = None
    one_hot: list[Column] = []


# ---------------------------------------------------------------- Unit 1
class DescribeRequest(BaseModel):
    columns: list[Column] | None = None


class CovarianceRequest(BaseModel):
    columns: list[Column] | None = None
    method: Literal["pearson", "spearman", "kendall"] = "pearson"


class DistributionRequest(BaseModel):
    column: Column
    bins: int = Field(30, ge=5, le=100)
    lower: float | None = None
    upper: float | None = None


class BayesRequest(BaseModel):
    hypothesis_column: Column
    hypothesis_value: str | None = None
    hypothesis_threshold: float | None = None
    hypothesis_op: Literal[">", ">=", "<", "<="] = ">="
    evidence_column: Column
    evidence_value: str | None = None
    evidence_threshold: float | None = None
    evidence_op: Literal[">", ">=", "<", "<="] = ">="
    condition_column: Column | None = None


class PolynomialRequest(BaseModel):
    x_column: Column
    y_column: Column
    degree: int = Field(3, ge=0, le=15)
    lam: float = Field(0.0, ge=0)
    test_size: float = Field(0.3, ge=0.1, le=0.5)
    random_state: int = 42


# ---------------------------------------------------------------- Units 2 & 5
class AlgorithmSpec(BaseModel):
    name: str
    params: dict[str, Any] = {}


class TrainRequest(BaseModel):
    task: Literal["regression", "classification"]
    target: Column
    features: list[Column] | None = None
    algorithms: list[AlgorithmSpec] = Field(..., min_length=1, max_length=10)
    test_size: float = Field(0.25, ge=0.05, le=0.5)
    scale: Scale = "standard"
    impute: Literal["mean", "median"] = "mean"
    cv_folds: int = Field(0, ge=0, le=10)
    random_state: int = 42


class BMARequest(BaseModel):
    task: Literal["regression", "classification"]
    target: Column
    features: list[Column] | None = None
    max_features: int = Field(6, ge=1, le=10)
    test_size: float = Field(0.25, ge=0.05, le=0.5)
    random_state: int = 42


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)


# ---------------------------------------------------------------- Unit 3
class ClusterRequest(BaseModel):
    method: Literal["kmeans", "gmm", "hierarchical", "compare"]
    features: list[Column] | None = None
    n_clusters: int = Field(3, ge=2, le=20)
    scale: Scale = "standard"
    label_column: Column | None = None
    init: Literal["k-means++", "random"] = "k-means++"
    n_init: int = Field(10, ge=1, le=50)
    max_iter: int = Field(300, ge=1, le=1000)
    covariance_type: Literal["full", "tied", "diag", "spherical"] = "full"
    tol: float = Field(1e-4, gt=0)
    linkage: Literal["ward", "complete", "average", "single"] = "ward"
    metric: Literal["euclidean", "cityblock", "cosine", "chebyshev", "correlation"] = "euclidean"
    random_state: int = 0


class DecomposeRequest(BaseModel):
    method: Literal["pca", "factor_analysis"] = "pca"
    features: list[Column] | None = None
    n_components: int = Field(2, ge=1, le=50)
    scale: Scale = "standard"
    color_column: Column | None = None
    rotation: Literal["varimax", "quartimax", "none"] = "varimax"


# ---------------------------------------------------------------- Unit 4
class MarkovRequest(BaseModel):
    column: Column
    group_column: Column | None = None
    order_column: Column | None = None
    smoothing: float = Field(0.0, ge=0, le=10)
    n_steps: int = Field(3, ge=1, le=50)
    simulate_length: int = Field(50, ge=2, le=500)
    random_state: int = 0


class HMMFitRequest(BaseModel):
    column: Column
    group_column: Column | None = None
    order_column: Column | None = None
    hidden_column: Column | None = None
    n_states: int = Field(2, ge=2, le=10)
    n_iter: int = Field(100, ge=1, le=500)
    n_init: int = Field(3, ge=1, le=10)
    tol: float = Field(1e-4, gt=0)
    random_state: int = 0


class HMMDecodeRequest(BaseModel):
    states: list[str] = Field(..., min_length=2, max_length=10)
    symbols: list[str] = Field(..., min_length=2, max_length=30)
    initial: list[float]
    transition: list[list[float]]
    emission: list[list[float]]
    observations: list[str] | None = Field(None, max_length=5000)
    sample_length: int | None = Field(None, ge=2, le=5000)
    random_state: int = 0


class LDSRequest(BaseModel):
    column: Column
    order_column: Column | None = None
    model: Literal["random_walk", "constant_velocity"] = "random_walk"
    q: float | None = Field(None, gt=0)
    r: float | None = Field(None, gt=0)
    estimate_noise: bool = True
    forecast: int = Field(10, ge=0, le=100)
    max_points: int = Field(1000, ge=5, le=5000)
