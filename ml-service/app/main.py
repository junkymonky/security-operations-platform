"""FastAPI entry point for the Grep Analyst ML service.

Run:  uvicorn app.main:app --reload --port 8000   (from the ml-service directory)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import db
from .errors import ServiceError
from .routers import datasets, experiments, models, soc, statistics, study

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ml-service")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield
    db.close_db()


app = FastAPI(title="Grep Analyst ML Service", version="1.0.0", lifespan=lifespan)

from .config import settings  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(ServiceError)
async def service_error_handler(_: Request, exc: ServiceError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error while processing the request."})


@app.get("/health", tags=["health"])
def health():
    try:
        db.fetch_one("SELECT 1 AS ok")
        database = "ok"
    except Exception as exc:  # report, don't crash the probe
        database = f"error: {exc.__class__.__name__}"
    return {"status": "ok" if database == "ok" else "degraded", "database": database}


app.include_router(datasets.router)
app.include_router(statistics.router)
app.include_router(models.router)
app.include_router(experiments.router)
app.include_router(study.router)
app.include_router(soc.router)
