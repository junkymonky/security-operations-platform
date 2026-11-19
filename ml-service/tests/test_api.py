"""End-to-end API tests against the configured database.

Run from ml-service/:  .venv/Scripts/python -m pytest -q
All rows created here are deleted in teardown.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

created_experiments: list[str] = []


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def datasets(client):
    ids = {}
    for key in ("network_intrusion", "auth_sequences"):
        r = client.post(f"/datasets/samples/{key}")
        assert r.status_code == 201, r.text
        ids[key] = r.json()["id"]
    yield ids
    for exp in created_experiments:
        client.delete(f"/experiments/{exp}")
    for ds in ids.values():
        for e in client.get("/experiments", params={"dataset_id": ds, "limit": 500}).json():
            client.delete(f"/experiments/{e['id']}")
        client.delete(f"/datasets/{ds}")


def track(body: dict) -> dict:
    if "experiment_id" in body:
        created_experiments.append(body["experiment_id"])
    return body


# ------------------------------------------------------------------ infra
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["database"] == "ok"


def test_upload_validation(client):
    r = client.post("/datasets", files={"file": ("x.exe", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422
    r = client.post("/datasets", files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 422
    r = client.post("/datasets", files={"file": ("bin.csv", b"\x00\x01\x02", "text/csv")})
    assert r.status_code == 422


def test_upload_preview_preprocess(client, datasets):
    csv = b"a;b;c\n1;2;x\n3;;y\n5;6;x\n5;6;x\n"
    r = client.post("/datasets", files={"file": ("../../evil name.csv", csv, "text/csv")})
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["n_rows"] == 4 and ds["n_columns"] == 3 and "/" not in ds["name"]
    try:
        p = client.get(f"/datasets/{ds['id']}/preview").json()
        assert p["columns"] == ["a", "b", "c"] and p["total"] == 4
        d = client.get(f"/datasets/{ds['id']}").json()
        assert d["summary"]["numeric"]["a"]["mean"] == pytest.approx(3.5)
        r = client.post(f"/datasets/{ds['id']}/preprocess",
                        json={"drop_duplicates": True, "missing": "mean", "one_hot": ["c"]})
        assert r.status_code == 201, r.text
        child = r.json()["dataset"]
        assert child["n_rows"] == 3 and child["parent_id"] == ds["id"]
        client.delete(f"/datasets/{child['id']}")
    finally:
        client.delete(f"/datasets/{ds['id']}")


def test_not_found(client):
    assert client.get("/datasets/" + "0" * 32).status_code == 404
    assert client.get("/datasets/not-an-id").status_code == 404


# ------------------------------------------------------------------ Unit 1
def test_unit1_statistics(client, datasets):
    ds = datasets["network_intrusion"]
    d = client.post(f"/datasets/{ds}/stats/describe", json={"columns": ["conn_rate", "protocol"]}).json()
    assert d["continuous"][0]["column"] == "conn_rate"
    assert abs(sum(x["p"] for x in d["discrete"][0]["pmf"]) - 1) < 1e-6
    c = client.post(f"/datasets/{ds}/stats/covariance", json={"columns": ["packets", "src_bytes", "conn_rate"]}).json()
    chk = c["identity_check"]
    assert chk["cov_population"] == pytest.approx(chk["cov_population_numpy"], rel=1e-6)
    dist = client.post(f"/datasets/{ds}/stats/distribution", json={"column": "payload_entropy", "bins": 25}).json()
    width = dist["histogram"][0]["x1"] - dist["histogram"][0]["x0"]
    assert sum(h["density"] * width for h in dist["histogram"]) == pytest.approx(1.0, abs=1e-3)
    assert dist["fits"]


def test_unit1_bayes(client, datasets):
    ds = datasets["network_intrusion"]
    b = track(client.post(f"/datasets/{ds}/stats/bayes", json={
        "hypothesis_column": "verdict", "hypothesis_value": "malicious",
        "evidence_column": "failed_logins", "evidence_threshold": 5, "evidence_op": ">=",
        "condition_column": "protocol"}).json())
    assert b["posterior"] == pytest.approx(b["posterior_direct"], abs=1e-9)  # Bayes == direct count
    assert b["evidence_sum_rule"] == pytest.approx(b["evidence_prob"], abs=1e-9)  # sum rule
    assert b["posterior"] > b["prior"]
    assert "conditional_independence" in b


def test_unit1_polynomial(client, datasets):
    ds = datasets["auth_sequences"]
    r = track(client.post(f"/datasets/{ds}/stats/polynomial",
                          json={"x_column": "t", "y_column": "bytes_out_kb", "degree": 4}).json())
    scan = r["degree_scan"]
    assert scan[4]["train_rmse"] <= scan[1]["train_rmse"] + 1e-9  # higher degree never worse on train
    assert len(r["curve"]) == 200


# ------------------------------------------------------------------ Unit 2
def test_unit2_regression_and_predict(client, datasets):
    ds = datasets["network_intrusion"]
    feats = ["failed_logins", "distinct_ports", "conn_rate", "payload_entropy", "hour", "packets"]
    algos = [{"name": n} for n in ("least_squares", "linear", "ridge", "bayesian_linear", "huber")]
    algos.append({"name": "svm", "params": {"kernel": "rbf", "C": 10}})
    r = client.post(f"/datasets/{ds}/train", json={"task": "regression", "target": "risk_score", "features": feats,
                                                  "algorithms": algos, "cv_folds": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    res = {x["algorithm"]: x for x in body["results"]}
    for x in res.values():
        created_experiments.append(x["experiment_id"])
        assert {"mae", "mse", "rmse", "r2"} <= x["metrics"].keys()
    # Explicit least squares must agree with sklearn's LinearRegression
    assert res["least_squares"]["metrics"]["rmse"] == pytest.approx(res["linear"]["metrics"]["rmse"], rel=1e-6)
    assert res["least_squares"]["details"]["mle"]["noise_variance"] > 0
    assert res["bayesian_linear"]["details"]["posterior"]["alpha"] > 0
    assert "std" in res["bayesian_linear"]["predicted_vs_actual"][0]
    assert res["huber"]["details"]["outliers_detected"] > 0
    assert res["ridge"]["details"]["ridge_path"]
    # Prediction from the persisted model
    exp = res["ridge"]["experiment_id"]
    p = client.post(f"/experiments/{exp}/predict", json={"records": [body["sample_record"]]})
    assert p.status_code == 200, p.text
    assert isinstance(p.json()["predictions"][0]["prediction"], float)
    assert len(client.get(f"/experiments/{exp}/predictions").json()) == 1
    bad = client.post(f"/experiments/{exp}/predict", json={"records": [{"failed_logins": 1}]})
    assert bad.status_code == 422


def test_unit2_classification(client, datasets):
    ds = datasets["network_intrusion"]
    algos = [{"name": n} for n in ("logistic", "bayesian_logistic", "lda", "qda", "naive_bayes", "perceptron")]
    algos += [{"name": "svm", "params": {"kernel": k}} for k in ("linear", "rbf", "poly")]
    r = client.post(f"/datasets/{ds}/train", json={"task": "classification", "target": "verdict",
                                                  "features": ["failed_logins", "distinct_ports", "conn_rate",
                                                               "payload_entropy", "packets", "protocol"],
                                                  "algorithms": algos})
    assert r.status_code == 200, r.text
    for x in r.json()["results"]:
        created_experiments.append(x["experiment_id"])
        m = x["metrics"]
        assert np.array(m["confusion_matrix"]).sum() == x["n_test"]
        assert m["accuracy"] > 0.8, (x["algorithm"], m["accuracy"])
        assert x["decision_surface"]["grid"]
    blr = next(x for x in r.json()["results"] if x["algorithm"] == "bayesian_logistic")
    assert "std" in blr["details"]["coefficients_by_class"][0]["coefficients"][0]
    # multiclass Bayesian logistic (one-vs-rest)
    r = client.post(f"/datasets/{ds}/train", json={"task": "classification", "target": "attack_type",
                                                  "algorithms": [{"name": "bayesian_logistic"}]})
    assert r.status_code == 200, r.text
    created_experiments.append(r.json()["results"][0]["experiment_id"])
    exp = r.json()["results"][0]["experiment_id"]
    p = client.post(f"/experiments/{exp}/predict", json={"records": [r.json()["sample_record"]]}).json()
    assert p["predictions"][0]["prediction"] in r.json()["classes"]
    assert sum(p["predictions"][0]["probabilities"].values()) == pytest.approx(1, abs=0.05)


def test_unit2_bad_input(client, datasets):
    ds = datasets["network_intrusion"]
    r = client.post(f"/datasets/{ds}/train", json={"task": "regression", "target": "protocol",
                                                  "algorithms": [{"name": "linear"}]})
    assert r.status_code == 422
    r = client.post(f"/datasets/{ds}/train", json={"task": "classification", "target": "verdict",
                                                  "algorithms": [{"name": "svm", "params": {"kernel": "magic"}}]})
    assert r.status_code == 422


# ------------------------------------------------------------------ Unit 3
@pytest.mark.parametrize("method", ["kmeans", "gmm", "hierarchical"])
def test_unit3_clustering(client, datasets, method):
    ds = datasets["network_intrusion"]
    body = {"method": method, "n_clusters": 5, "label_column": "attack_type",
            "features": ["failed_logins", "distinct_ports", "conn_rate", "payload_entropy", "packets", "duration_s"]}
    if method == "hierarchical":
        body.update({"linkage": "average", "metric": "cosine"})
    r = client.post(f"/datasets/{ds}/cluster", json=body)
    assert r.status_code == 200, r.text
    out = track(r.json())
    m = out["metrics"]
    assert -1 <= m["silhouette"] <= 1 and m["davies_bouldin"] > 0 and m["inertia"] > 0
    assert m["adjusted_rand"] > 0.3, m  # recovers the planted attack classes
    if method == "gmm":
        lls = [e["log_likelihood"] for e in out["em_trace"]]
        assert all(b >= a - 1e-6 for a, b in zip(lls, lls[1:]))  # EM is monotone
        assert abs(sum(out["points"][0]["probabilities"]) - 1) < 1e-3
    if method == "hierarchical":
        assert out["dendrogram"]["icoord"]
    if method == "kmeans":
        assert len(out["centroids_2d"]) == 5
        p = client.post(f"/experiments/{out['experiment_id']}/predict",
                        json={"records": [{f: 1.0 for f in body["features"]}]})
        assert p.status_code == 200 and "cluster" in p.json()["predictions"][0]


def test_unit3_compare_and_ward_validation(client, datasets):
    ds = datasets["network_intrusion"]
    r = client.post(f"/datasets/{ds}/cluster", json={"method": "compare", "n_clusters": 4})
    assert r.status_code == 200 and len(r.json()["comparison"]) == 3
    r = client.post(f"/datasets/{ds}/cluster", json={"method": "hierarchical", "linkage": "ward", "metric": "cosine"})
    assert r.status_code == 422


def test_unit3_pca_fa(client, datasets):
    ds = datasets["network_intrusion"]
    r = track(client.post(f"/datasets/{ds}/decompose", json={"method": "pca", "n_components": 3,
                                                              "color_column": "attack_type"}).json())
    assert r["cumulative_ratio"][-1] == pytest.approx(1.0, abs=1e-6)
    assert "c3" in r["points"][0] and r["dimension_choice"]["components_for_90pct"] >= 1
    r = track(client.post(f"/datasets/{ds}/decompose", json={"method": "factor_analysis", "n_components": 2}).json())
    assert r["loadings"] and r["noise_variance"]


# ------------------------------------------------------------------ Unit 4
def test_unit4_markov(client, datasets):
    ds = datasets["auth_sequences"]
    r = track(client.post(f"/datasets/{ds}/markov", json={"column": "event", "group_column": "session_id"}).json())
    A = np.array(r["transition_matrix"])
    assert np.allclose(A.sum(1), 1) and np.isclose(sum(r["stationary"]), 1)
    assert np.allclose(np.array(r["stationary"]) @ A, r["stationary"], atol=1e-6)


def test_unit4_hmm_fit(client, datasets):
    ds = datasets["auth_sequences"]
    r = client.post(f"/datasets/{ds}/hmm", json={"column": "event", "group_column": "session_id",
                                                 "hidden_column": "phase", "n_states": 3, "n_init": 3})
    assert r.status_code == 200, r.text
    out = track(r.json())
    lls = [e["log_likelihood"] for e in out["em_trace"]]
    assert all(b >= a - 1e-6 for a, b in zip(lls, lls[1:]))  # Baum-Welch monotone
    assert out["sum_product_max_abs_diff"] < 1e-8  # sum-product == forward-backward
    assert out["evaluation"]["viterbi_accuracy"] > 0.6


def test_unit4_hmm_decode_and_scaling(client):
    body = {"states": ["normal", "attack"], "symbols": ["ok", "fail"], "initial": [0.9, 0.1],
            "transition": [[0.95, 0.05], [0.1, 0.9]], "emission": [[0.9, 0.1], [0.2, 0.8]], "sample_length": 1500}
    out = track(client.post("/hmm/decode", json=body).json())
    assert out["unscaled_underflow"] is True and np.isfinite(out["log_likelihood"])  # scaling matters
    assert out["viterbi_accuracy"] > 0.8
    short = track(client.post("/hmm/decode", json={**body, "sample_length": None,
                                                   "observations": ["ok", "fail", "fail", "ok"]}).json())
    assert np.log(short["unscaled_likelihood"]) == pytest.approx(short["log_likelihood"], rel=1e-9)
    bad = client.post("/hmm/decode", json={**body, "transition": [[0.5, 0.2], [0.1, 0.9]]})
    assert bad.status_code == 422


def test_unit4_lds(client, datasets):
    ds = datasets["auth_sequences"]
    out = track(client.post(f"/datasets/{ds}/lds", json={"column": "bytes_out_kb", "order_column": "t",
                                                         "model": "random_walk", "max_points": 400}).json())
    assert out["noise_estimated"] and out["q"] > 0 and out["r"] > 0
    obs = np.array([s["observed"] for s in out["series"]])
    smooth = np.array([s["smoothed"] for s in out["series"]])
    assert np.std(np.diff(smooth)) < np.std(np.diff(obs))  # smoother than the raw signal


# ------------------------------------------------------------------ Unit 5
def test_unit5_ensembles(client, datasets):
    ds = datasets["network_intrusion"]
    feats = ["failed_logins", "distinct_ports", "conn_rate", "payload_entropy", "packets", "duration_s", "hour"]
    algos = [{"name": "cart", "params": {"max_depth": 4}}, {"name": "random_forest", "params": {"n_estimators": 60}},
             {"name": "bagging", "params": {"n_estimators": 30}}, {"name": "adaboost", "params": {"n_estimators": 40}},
             {"name": "gradient_boosting", "params": {"n_estimators": 60}}, {"name": "voting"}, {"name": "gam"}]
    r = client.post(f"/datasets/{ds}/train", json={"task": "classification", "target": "attack_type",
                                                  "features": feats, "algorithms": algos})
    assert r.status_code == 200, r.text
    res = {x["algorithm"]: x for x in r.json()["results"]}
    for x in res.values():
        created_experiments.append(x["experiment_id"])
    assert res["cart"]["details"]["tree"]["children"] and res["cart"]["details"]["pruning_path"]
    ada = res["adaboost"]["details"]
    assert ada["staged"][-1]["test"] > max(w["test_score_alone"] for w in ada["weak_learners"])  # boosting helps
    rf = res["random_forest"]
    assert rf["metrics"]["accuracy"] >= rf["details"]["member_scores"]["mean"]  # ensemble >= average member
    assert rf["details"]["feature_importance"]
    assert res["gam"]["details"]["shape_functions"]
    # Regression trees / ensembles
    r = client.post(f"/datasets/{ds}/train", json={"task": "regression", "target": "risk_score", "features": feats,
                                                  "algorithms": [{"name": "cart"}, {"name": "gradient_boosting"},
                                                                 {"name": "gam"}]})
    assert r.status_code == 200, r.text
    for x in r.json()["results"]:
        created_experiments.append(x["experiment_id"])
        assert x["metrics"]["rmse"] > 0


def test_unit5_bma(client, datasets):
    ds = datasets["network_intrusion"]
    for task, target in (("regression", "risk_score"), ("classification", "verdict")):
        r = client.post(f"/datasets/{ds}/bma", json={"task": task, "target": target, "max_features": 5,
                                                     "features": ["failed_logins", "distinct_ports", "conn_rate",
                                                                  "payload_entropy", "packets", "hour"]})
        assert r.status_code == 200, r.text
        out = track(r.json())
        assert out["n_models"] == 32
        assert sum(m["posterior"] for m in out["top_models"]) <= 1 + 1e-9


# ------------------------------------------------------------------ history
def test_experiment_history(client, datasets):
    rows = client.get("/experiments", params={"dataset_id": datasets["network_intrusion"]}).json()
    assert rows and {r["unit"] for r in rows} >= {1, 2, 3, 5}
    one = client.get(f"/experiments/{rows[0]['id']}").json()
    assert one["results"] is not None or one["metrics"]
