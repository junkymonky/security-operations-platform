"""PostgreSQL persistence (shares the Neon database with the Next.js app).

Tables are prefixed ``ml_`` so they never collide with the SOC schema.
"""
from __future__ import annotations

import json
import logging
import os
import socket
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from .config import settings
from .errors import ServiceUnavailable
from .neon_http import NeonHttpConnection, NeonHttpError

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ml_datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'upload',
    parent_id TEXT REFERENCES ml_datasets(id) ON DELETE SET NULL,
    n_rows INTEGER NOT NULL,
    n_columns INTEGER NOT NULL,
    columns JSONB NOT NULL,
    size_bytes BIGINT NOT NULL,
    sha256 TEXT NOT NULL,
    preprocessing JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_experiments (
    id TEXT PRIMARY KEY,
    group_id TEXT,
    dataset_id TEXT REFERENCES ml_datasets(id) ON DELETE SET NULL,
    unit SMALLINT NOT NULL,
    task TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    features JSONB,
    target TEXT,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    results JSONB,
    model_path TEXT,
    duration_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ml_experiments_created_idx ON ml_experiments (created_at DESC);
CREATE INDEX IF NOT EXISTS ml_experiments_dataset_idx ON ml_experiments (dataset_id);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id BIGSERIAL PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES ml_experiments(id) ON DELETE CASCADE,
    inputs JSONB NOT NULL,
    outputs JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ml_predictions_experiment_idx ON ml_predictions (experiment_id);
"""

_pool: ConnectionPool | None = None
_http: NeonHttpConnection | None = None


def _tcp_reachable(dsn: str, timeout: float = 3.0) -> bool:
    info = psycopg.conninfo.conninfo_to_dict(dsn)
    host = (info.get("host") or "localhost").split(",")[0]
    port = int(str(info.get("port") or 5432).split(",")[0])
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _choose_driver(dsn: str) -> str:
    driver = os.environ.get("DB_DRIVER", "auto").strip().lower()
    if driver in ("tcp", "http"):
        return driver
    if _tcp_reachable(dsn):
        return "tcp"
    host = psycopg.conninfo.conninfo_to_dict(dsn).get("host") or ""
    if host.endswith(".neon.tech"):
        log.warning("Postgres port unreachable from this network; using Neon SQL-over-HTTPS (port 443)")
        return "http"
    return "tcp"


def init_db() -> None:
    """Open the DB connection and create tables. Raises if DATABASE_URL is missing."""
    global _pool, _http
    if not settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set it in ml-service/.env or the project's .env.local."
        )
    if _choose_driver(settings.database_url) == "http":
        _http = NeonHttpConnection(settings.database_url)
    else:
        _pool = ConnectionPool(
            settings.database_url,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row, "autocommit": True, "connect_timeout": 10},
            check=ConnectionPool.check_connection,
            # Serverless Postgres (Neon) drops idle sessions; recycle ours first.
            max_idle=120,
            max_lifetime=900,
            timeout=20,
            open=True,
        )
    with connection() as conn:
        conn.execute(SCHEMA)
    log.info("Database schema ready (%s)", "neon-http" if _http else "tcp")


def close_db() -> None:
    global _pool, _http
    if _pool is not None:
        _pool.close()
        _pool = None
    _http = None


@contextmanager
def connection() -> Iterator[Any]:
    if _http is not None:
        try:
            yield _http
        except (NeonHttpError, OSError) as exc:
            log.error("Database unavailable: %s", exc)
            raise ServiceUnavailable("Database is temporarily unavailable. Please retry.") from exc
        return
    if _pool is None:
        raise ServiceUnavailable("Database is not initialised")
    try:
        with _pool.connection() as conn:
            yield conn
    except (PoolTimeout, psycopg.OperationalError) as exc:
        log.error("Database unavailable: %s", exc)
        raise ServiceUnavailable("Database is temporarily unavailable. Please retry.") from exc


def jsonb(value: Any) -> Jsonb:
    # Values are pre-sanitised (see utils.to_jsonable), so plain dumps is safe.
    return Jsonb(value, dumps=lambda v: json.dumps(v, allow_nan=False))


def fetch_all(query: str, params: tuple | dict = ()) -> list[dict]:
    with connection() as conn:
        return list(conn.execute(query, params).fetchall())


def fetch_one(query: str, params: tuple | dict = ()) -> dict | None:
    with connection() as conn:
        return conn.execute(query, params).fetchone()


def execute(query: str, params: tuple | dict = ()) -> int:
    with connection() as conn:
        return conn.execute(query, params).rowcount
