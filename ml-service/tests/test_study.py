"""End-to-end test of the flight-fare study on a small TEST-ONLY stand-in (never written to report_assets)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from app.study import runner, unit4
from tests.flight_fixture import make


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    folder = tmp_path_factory.mktemp("study")
    data = folder / "Cleaned_dataset.csv"
    make(n=6000).to_csv(data, index=False)
    result = runner.run_study(str(data), extended=True, progress=lambda m: None, assets=folder / "assets")
    return folder / "assets", result


def test_all_artifacts_written(study):
    assets, r = study
    assert (assets / "results.json").is_file() and (assets / "summary.md").is_file()
    figs = {f["file"] for f in r["figures"]}
    for number in ["2_1", "2_4", "2_6", "3_4", "3_5", "3_6", "3_8", "3_12", "4_2", "4_3"]:
        assert any(f.startswith(f"fig_{number}_") for f in figs), number
    assert all((assets / "figures" / f).stat().st_size > 5000 for f in figs)
    assert len(r["code_images"]) >= 10 and all((assets / "code" / c["file"]).is_file() for c in r["code_images"])
    json.loads((assets / "results.json").read_text(encoding="utf-8"))  # valid JSON (no NaN)


def test_report_hmm_values_reproduced(study):
    """Table 3.11 of the report: these exact numbers verify the algorithms."""
    h = study[1]["unit4"]["hmm"]
    assert h["forward_probability"] == pytest.approx(0.0119528, abs=1e-7)
    assert h["backward_probability"] == pytest.approx(h["forward_probability"], rel=1e-12)
    assert h["scaled_log_likelihood"] == pytest.approx(-4.4268, abs=1e-4)
    assert h["scaling_factors"] == pytest.approx([0.43, 0.3042, 0.2593, 0.3524], abs=1e-4)
    assert h["viterbi_path"] == [0, 1, 1, 1]
    assert np.allclose(np.sum(h["posterior"], axis=1), 1)


def test_results_are_consistent(study):
    r = study[1]
    assert np.allclose(np.sum(r["unit4"]["transition"]["matrix"], axis=1), 1)
    ks = [s["k"] for s in r["unit3"]["pipeline"]["sweep"]]
    assert ks == [2, 3, 4, 5, 6, 8, 10] and r["unit3"]["pipeline"]["best_k"] in ks
    assert sum(sum(row) for row in r["unit3"]["pipeline"]["confusion_matrix"]) == r["unit3"]["pipeline"]["n_test"]
    assert [m["r2"] for m in r["comparison"]] == sorted((m["r2"] for m in r["comparison"]), reverse=True)
    assert {m["model"] for m in r["unit5"]["results"]} == {"Decision tree (depth 15)", "Random forest", "XGBoost"}
    trace = r["unit4"]["learned_hmm"]["trace"]
    assert all(b >= a - 1e-6 for a, b in zip(trace, trace[1:]))  # Baum–Welch is monotone
    assert all(s["points"] for s in r["summary"])


def test_forward_backward_helpers_match_scaled():
    rng = np.random.default_rng(1)
    pi = rng.dirichlet(np.ones(3)); A = rng.dirichlet(np.ones(3), 3); B = rng.dirichlet(np.ones(3), 3)
    obs = rng.integers(0, 3, 12)
    fwd = unit4.forward_algorithm(obs, pi, A, B)
    assert unit4.backward_algorithm(obs, pi, A, B) == pytest.approx(fwd, rel=1e-10)
    from app.ml.hmm import HMM
    assert np.log(fwd) == pytest.approx(HMM(pi, A, B).log_likelihood(obs), rel=1e-10)


def test_missing_dataset_message(monkeypatch, tmp_path):
    from app.errors import NotFound
    from app.study import data
    monkeypatch.delenv("FLIGHT_FARE_DATA", raising=False)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)
    with pytest.raises(NotFound, match="kaggle"):
        data.find_dataset()
