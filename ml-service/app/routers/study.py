"""Project-report study endpoints: /study/attacks (KDD Cup 1999) and /study/flight-fare (Kaggle airfare)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..errors import NotFound, ServiceError
from ..study import attack_report, attacks, data, runner

router = APIRouter(prefix="/study", tags=["project report"])
_PNG = re.compile(r"^[a-z0-9_]+\.png$")

STUDIES = {
    "attacks": {"title": "Network attacks (KDD Cup 1999)", "assets": attack_report.ASSETS, "job": attack_report.JOB,
                "find": attacks.find_dataset,
                "missing_hint": "Build it with: cd ml-service && .venv\\Scripts\\python -m sample_data.make_kdd"},
    "flight-fare": {"title": "Flight fares (Kaggle airfare)", "assets": runner.ASSETS, "job": runner.JOB,
                    "find": data.find_dataset,
                    "missing_hint": "Download Cleaned_dataset.csv from kaggle.com/datasets/yashdharme36/"
                                    "airfare-ml-predicting-flight-fares into ml-service/data/"},
}
# Point arrays only needed to draw PNGs are dropped from the API payload.
_HEAVY = {
    "attacks": [("unit3", "pca"), ("unit4", "hmm", "demo"), ("unit1", "polynomial"), ("unit1", "distributions")],
    "flight-fare": [("unit1", "eda", "scatter"), ("unit3", "exploratory", "points"), ("unit3", "cluster_regression", "points"),
                    ("unit3", "pipeline", "pca"), ("unit5", "xgb_first_1000"), ("unit2", "one_hot", "linear_sample_pred")],
}


class RunRequest(BaseModel):
    extended: bool = True


def _study(key: str) -> dict:
    if key not in STUDIES:
        raise NotFound("Unknown study.")
    return STUDIES[key]


def _dataset_status(s: dict) -> dict:
    try:
        path: Path = s["find"]()
        return {"found": True, "path": str(path), "size_mb": round(path.stat().st_size / 1e6, 1)}
    except ServiceError as exc:
        return {"found": False, "message": exc.message, "hint": s["missing_hint"]}


@router.get("")
def list_studies():
    return [{"key": k, "title": s["title"], "has_results": (s["assets"] / "results.json").is_file(),
             "status": s["job"].state["status"]} for k, s in STUDIES.items()]


@router.get("/{key}")
def status(key: str):
    s = _study(key)
    results = None
    file = s["assets"] / "results.json"
    if file.is_file():
        results = json.loads(file.read_text(encoding="utf-8"))
        for path in _HEAVY[key]:
            node = results
            for part in path[:-1]:
                node = node.get(part, {}) if isinstance(node, dict) else {}
            if isinstance(node, dict):
                node.pop(path[-1], None)
        if key == "attacks":  # ROC point lists are only used by the PNG figure
            for m in results.get("comparison", []) + results.get("unit2", []) + results.get("unit5", {}).get("results", []):
                m.pop("roc", None)
    return {"key": key, "title": s["title"], "dataset": _dataset_status(s), "job": s["job"].state,
            "assets_folder": str(s["assets"]), "results": results}


@router.post("/{key}/run", status_code=202)
def run(key: str, body: RunRequest):
    return _study(key)["job"].start(extended=body.extended)


def _file(key: str, kind: str, name: str) -> FileResponse:
    s = _study(key)
    if not _PNG.match(name):
        raise NotFound("Image not found.")
    path = s["assets"] / kind / name
    if not path.is_file():
        raise NotFound("Image not found. Run the study first.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get("/{key}/figures/{name}")
def figure(key: str, name: str):
    return _file(key, "figures", name)


@router.get("/{key}/code/{name}")
def code_image(key: str, name: str):
    return _file(key, "code", name)
