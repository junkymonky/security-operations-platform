"""Dataset ingestion, validation, storage, profiling and preprocessing (Unit 1)."""
from __future__ import annotations

import csv
import hashlib
import io
import re
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import db
from ..config import settings
from ..errors import NotFound, PayloadTooLarge, ValidationFailed
from ..utils import to_jsonable

_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_ALLOWED_EXT = {".csv", ".txt"}


# ---------------------------------------------------------------- validation
def _clean_name(name: str, max_len: int = 120) -> str:
    name = Path(name or "dataset.csv").name  # strip any directory components
    name = re.sub(r"[^\w.\- ]+", "_", name).strip() or "dataset.csv"
    return name[:max_len]


def _dedupe_columns(columns: list) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, col in enumerate(columns):
        base = str(col).strip() or f"column_{i + 1}"
        base = re.sub(r"\s+", " ", base)[:80]
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        out.append(base)
    return out


def parse_csv(raw: bytes) -> pd.DataFrame:
    """Parse and validate CSV bytes into a DataFrame."""
    if not raw:
        raise ValidationFailed("The uploaded file is empty.")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise PayloadTooLarge(f"File exceeds the {settings.max_upload_mb:g} MB upload limit.")
    if b"\x00" in raw[:4096]:
        raise ValidationFailed("File looks binary. Upload a plain-text CSV file.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    try:
        dialect = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|")
        sep = dialect.delimiter
    except csv.Error:
        sep = ","
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValidationFailed(f"Could not parse CSV: {exc}") from exc
    if df.shape[1] < 1 or df.shape[0] < 1:
        raise ValidationFailed("CSV must contain a header row and at least one data row.")
    if df.shape[0] > settings.max_rows:
        raise PayloadTooLarge(f"CSV has {df.shape[0]} rows; the limit is {settings.max_rows}.")
    if df.shape[1] > settings.max_columns:
        raise PayloadTooLarge(f"CSV has {df.shape[1]} columns; the limit is {settings.max_columns}.")
    df.columns = _dedupe_columns(list(df.columns))
    # Coerce object columns that are actually numeric (e.g. "1,000" left as text is kept as text).
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            non_null = df[col].notna().sum()
            if non_null and converted.notna().sum() == non_null:
                df[col] = converted
    return df


# ---------------------------------------------------------------- profiling
def column_info(df: pd.DataFrame) -> list[dict]:
    info = []
    for col in df.columns:
        s = df[col]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        info.append(
            {
                "name": col,
                "dtype": str(s.dtype),
                "kind": "numeric" if numeric else "categorical",
                "missing": int(s.isna().sum()),
                "unique": int(s.nunique(dropna=True)),
            }
        )
    return info


def summary(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    stats = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        if s.empty:
            continue
        mode = s.mode()
        stats[col] = {
            "count": int(s.count()),
            "mean": s.mean(),
            "median": s.median(),
            "mode": mode.iloc[0] if not mode.empty else None,
            "std": s.std(ddof=1) if len(s) > 1 else 0.0,
            "variance": s.var(ddof=1) if len(s) > 1 else 0.0,
            "min": s.min(),
            "q25": s.quantile(0.25),
            "q75": s.quantile(0.75),
            "max": s.max(),
        }
    categorical = {}
    for col in df.columns.difference(numeric.columns):
        counts = df[col].astype("string").value_counts(dropna=True).head(10)
        categorical[col] = [{"value": str(k), "count": int(v)} for k, v in counts.items()]
    return to_jsonable({"numeric": stats, "categorical": categorical})


# ---------------------------------------------------------------- storage
def _path_for(dataset_id: str) -> Path:
    if not _ID_RE.match(dataset_id or ""):
        raise NotFound("Dataset not found.")
    return settings.datasets_dir / f"{dataset_id}.csv"


def save_dataset(
    df: pd.DataFrame,
    name: str,
    original_filename: str,
    source: str = "upload",
    parent_id: str | None = None,
    preprocessing: dict | None = None,
) -> dict:
    dataset_id = uuid.uuid4().hex
    path = _path_for(dataset_id)
    payload = df.to_csv(index=False).encode("utf-8")
    path.write_bytes(payload)
    cols = column_info(df)
    try:
        row = db.fetch_one(
            """INSERT INTO ml_datasets
               (id, name, original_filename, source, parent_id, n_rows, n_columns, columns,
                size_bytes, sha256, preprocessing)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (
                dataset_id,
                _clean_name(name),
                _clean_name(original_filename),
                source,
                parent_id,
                int(df.shape[0]),
                int(df.shape[1]),
                db.jsonb(cols),
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                db.jsonb(to_jsonable(preprocessing)) if preprocessing else None,
            ),
        )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return row


def list_datasets() -> list[dict]:
    return db.fetch_all(
        "SELECT id, name, original_filename, source, parent_id, n_rows, n_columns, size_bytes, created_at "
        "FROM ml_datasets ORDER BY created_at DESC"
    )


def get_meta(dataset_id: str) -> dict:
    _path_for(dataset_id)
    row = db.fetch_one("SELECT * FROM ml_datasets WHERE id = %s", (dataset_id,))
    if not row:
        raise NotFound("Dataset not found.")
    return row


@lru_cache(maxsize=16)
def _read_cached(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_df(dataset_id: str) -> pd.DataFrame:
    path = _path_for(dataset_id)
    if not path.is_file():
        # Metadata may exist without the file (e.g. storage wiped); surface clearly.
        get_meta(dataset_id)
        raise NotFound("Dataset file is missing from storage. Re-upload the dataset.")
    return _read_cached(str(path), path.stat().st_mtime).copy()


def delete_dataset(dataset_id: str) -> None:
    path = _path_for(dataset_id)
    if db.execute("DELETE FROM ml_datasets WHERE id = %s", (dataset_id,)) == 0:
        raise NotFound("Dataset not found.")
    path.unlink(missing_ok=True)


def preview(dataset_id: str, limit: int = 50, offset: int = 0) -> dict:
    df = load_df(dataset_id)
    page = df.iloc[offset : offset + limit]
    return {
        "columns": list(df.columns),
        "rows": to_jsonable(page.astype(object).where(page.notna(), None).values.tolist()),
        "total": int(len(df)),
        "offset": offset,
    }


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValidationFailed(f"Unknown column(s): {', '.join(missing)}")


# ---------------------------------------------------------------- preprocessing
def preprocess(dataset_id: str, opts: dict) -> dict:
    """Create a derived dataset by applying cleaning steps; the original is kept."""
    df = load_df(dataset_id)
    meta = get_meta(dataset_id)
    log: list[str] = []
    n0 = len(df)

    drop = [c for c in opts.get("drop_columns") or [] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
        log.append(f"Dropped columns: {', '.join(drop)}")

    if opts.get("drop_duplicates"):
        before = len(df)
        df = df.drop_duplicates()
        log.append(f"Removed {before - len(df)} duplicate rows")

    missing = opts.get("missing", "none")
    numeric_cols = df.select_dtypes(include="number").columns
    if missing == "drop":
        before = len(df)
        df = df.dropna()
        log.append(f"Dropped {before - len(df)} rows with missing values")
    elif missing in ("mean", "median"):
        fill = df[numeric_cols].mean() if missing == "mean" else df[numeric_cols].median()
        n_missing = int(df[numeric_cols].isna().sum().sum())
        df[numeric_cols] = df[numeric_cols].fillna(fill)
        cat_cols = df.columns.difference(numeric_cols)
        for c in cat_cols:
            if df[c].isna().any():
                mode = df[c].mode()
                if not mode.empty:
                    df[c] = df[c].fillna(mode.iloc[0])
        log.append(f"Imputed {n_missing} numeric missing values with column {missing}; categorical with mode")

    z = opts.get("outlier_zscore")
    if z:
        cols = [c for c in (opts.get("outlier_columns") or numeric_cols) if c in numeric_cols]
        if cols:
            sub = df[cols]
            std = sub.std(ddof=0).replace(0, np.nan)
            zs = ((sub - sub.mean()) / std).abs()
            mask = (zs.fillna(0) <= float(z)).all(axis=1)
            log.append(f"Removed {int((~mask).sum())} outlier rows with |z| > {z} on {len(cols)} column(s)")
            df = df[mask]

    scale = opts.get("scale", "none")
    scale_cols = [c for c in (opts.get("scale_columns") or numeric_cols) if c in numeric_cols]
    if scale in ("standard", "minmax") and scale_cols:
        sub = df[scale_cols].astype(float)
        if scale == "standard":
            std = sub.std(ddof=0).replace(0, 1.0)
            df[scale_cols] = (sub - sub.mean()) / std
        else:
            rng = (sub.max() - sub.min()).replace(0, 1.0)
            df[scale_cols] = (sub - sub.min()) / rng
        log.append(f"Applied {scale} scaling to {len(scale_cols)} column(s)")

    if opts.get("one_hot"):
        cols = [c for c in opts["one_hot"] if c in df.columns]
        if cols:
            df = pd.get_dummies(df, columns=cols, dtype=int)
            log.append(f"One-hot encoded: {', '.join(cols)}")

    if df.empty:
        raise ValidationFailed("Preprocessing removed every row; relax the options.")
    if not log:
        raise ValidationFailed("No preprocessing steps selected.")

    name = opts.get("name") or f"{Path(meta['name']).stem}_clean.csv"
    if not name.endswith(".csv"):
        name += ".csv"
    record = save_dataset(
        df.reset_index(drop=True),
        name=name,
        original_filename=meta["original_filename"],
        source="preprocessed",
        parent_id=dataset_id,
        preprocessing={"steps": log, "options": opts, "rows_before": n0, "rows_after": len(df)},
    )
    return {"dataset": record, "steps": log}
