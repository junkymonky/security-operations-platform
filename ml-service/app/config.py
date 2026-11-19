"""Service configuration, read from environment variables.

DATABASE_URL falls back to the Next.js app's ``.env.local`` so both halves of the
project share one database without duplicating the secret.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SERVICE_ROOT.parent


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _load_env_files() -> None:
    """Populate os.environ from ml-service/.env then ../.env.local (env vars win)."""
    for path in (SERVICE_ROOT / ".env", PROJECT_ROOT / ".env.local"):
        for key, value in _read_env_file(path).items():
            os.environ.setdefault(key, value)


_load_env_files()


@dataclass(frozen=True)
class Settings:
    database_url: str | None = field(default_factory=lambda: os.environ.get("DATABASE_URL"))
    storage_dir: Path = field(
        default_factory=lambda: (SERVICE_ROOT / os.environ.get("ML_STORAGE_DIR", "storage")).resolve()
    )
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("MAX_UPLOAD_MB", "10")))
    max_rows: int = field(default_factory=lambda: int(os.environ.get("MAX_ROWS", "200000")))
    max_columns: int = field(default_factory=lambda: int(os.environ.get("MAX_COLUMNS", "200")))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()
        ]
    )

    @property
    def datasets_dir(self) -> Path:
        return self.storage_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.storage_dir / "models"

    @property
    def samples_dir(self) -> Path:
        return SERVICE_ROOT / "sample_data"


settings = Settings()
settings.datasets_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
