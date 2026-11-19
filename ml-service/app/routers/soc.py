"""SOC integration: open investigations from ML predictions."""
from __future__ import annotations

from fastapi import APIRouter

from ..services import soc_detection

router = APIRouter(prefix="/soc", tags=["soc"])


@router.post("/detect")
def detect():
    """Score the network-intrusion sample and (re)create ML-generated incidents in the SOC tables."""
    return soc_detection.generate_incidents()
