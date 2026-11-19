"""Dataset upload, listing, preview, summary and preprocessing endpoints."""
from __future__ import annotations

from fastapi import APIRouter, File, Query, UploadFile

from ..config import settings
from ..errors import NotFound, PayloadTooLarge, ValidationFailed
from ..schemas import PreprocessRequest
from ..services import datasets as svc
from ..utils import to_jsonable

router = APIRouter(prefix="/datasets", tags=["datasets"])

SAMPLES = {
    "kdd_attacks": ("kdd_attacks.csv", "Real network attacks: KDD Cup 1999 (DARPA) connections, de-duplicated — "
                    "normal vs DoS / Probe / R2L / U2R, 22 attack types, 40 features"),
    "network_intrusion": ("network_intrusion.csv", "Synthetic network flows labelled benign / attack type"),
    "auth_sequences": ("auth_sequences.csv", "Authentication event sequences from a known HMM, with hidden phases"),
}


@router.get("")
def list_datasets():
    return to_jsonable(svc.list_datasets())


@router.get("/samples")
def list_samples():
    return [{"key": k, "filename": f, "description": d} for k, (f, d) in SAMPLES.items()]


@router.post("/samples/{key}", status_code=201)
def import_sample(key: str):
    if key not in SAMPLES:
        raise NotFound("Unknown sample dataset.")
    filename = SAMPLES[key][0]
    path = settings.samples_dir / filename
    if not path.is_file():
        raise NotFound(f"Sample file {filename} is missing; run `python -m sample_data.generate`.")
    df = svc.parse_csv(path.read_bytes())
    return to_jsonable(svc.save_dataset(df, filename, filename, source="sample"))


@router.post("", status_code=201)
async def upload(file: UploadFile = File(...)):
    name = file.filename or "dataset.csv"
    if not any(name.lower().endswith(ext) for ext in (".csv", ".txt")):
        raise ValidationFailed("Only .csv files are accepted.")
    limit = int(settings.max_upload_mb * 1024 * 1024)
    raw = await file.read(limit + 1)
    if len(raw) > limit:
        raise PayloadTooLarge(f"File exceeds the {settings.max_upload_mb:g} MB upload limit.")
    df = svc.parse_csv(raw)
    return to_jsonable(svc.save_dataset(df, name, name))


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str):
    meta = svc.get_meta(dataset_id)
    df = svc.load_df(dataset_id)
    return to_jsonable({**meta, "summary": svc.summary(df)})


@router.get("/{dataset_id}/preview")
def preview(dataset_id: str, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return svc.preview(dataset_id, limit, offset)


@router.post("/{dataset_id}/preprocess", status_code=201)
def preprocess(dataset_id: str, body: PreprocessRequest):
    return to_jsonable(svc.preprocess(dataset_id, body.model_dump()))


@router.delete("/{dataset_id}", status_code=204)
def delete(dataset_id: str):
    svc.delete_dataset(dataset_id)
