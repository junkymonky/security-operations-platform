"""Minimal psycopg-compatible connection over Neon's SQL-over-HTTPS API (port 443).

Used when outbound TCP 5432 is blocked (common on campus/corporate networks). It
implements only what this service uses: ``execute`` -> result with
``fetchone``/``fetchall``/iteration/``rowcount``, ``cursor().executemany`` and
``transaction()``. Inside a transaction, reads run immediately and writes are
buffered and sent as one atomic batch on exit.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterator
from urllib.parse import urlparse

from psycopg.types.json import Jsonb

_READ_RE = re.compile(r"^\s*(SELECT|WITH|SHOW|EXPLAIN|VALUES)\b", re.IGNORECASE)
_TZ_RE = re.compile(r"([+-]\d{2})$")


class NeonHttpError(Exception):
    pass


def _to_dollar_params(query: str) -> str:
    out, n, i = [], 0, 0
    while i < len(query):
        if query.startswith("%%", i):
            out.append("%")
            i += 2
        elif query.startswith("%s", i):
            n += 1
            out.append(f"${n}")
            i += 2
        else:
            out.append(query[i])
            i += 1
    return "".join(out)


def _array_literal(items: Any) -> str:
    parts = []
    for v in items:
        if v is None:
            parts.append("NULL")
        else:
            s = _param(v)
            parts.append('"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"')
    return "{" + ",".join(parts) + "}"


def _param(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, Jsonb):
        dumps = v.dumps or json.dumps
        return dumps(v.obj)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return _array_literal(v)
    if isinstance(v, dict):
        return json.dumps(v)
    if isinstance(v, (bytes, bytearray)):
        return "\\x" + bytes(v).hex()
    return str(v)


def _parse_ts(s: str) -> Any:
    try:
        return datetime.fromisoformat(_TZ_RE.sub(r"\1:00", s))
    except ValueError:
        return s


def _parse_array(s: str, elem) -> Any:
    inner = s[1:-1]
    if not inner:
        return []
    items, buf, quoted, esc, was_quoted = [], [], False, False, False
    for ch in inner:
        if esc:
            buf.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == '"':
            quoted = not quoted
            was_quoted = True
        elif ch == "," and not quoted:
            tok = "".join(buf)
            items.append(None if (tok == "NULL" and not was_quoted) else elem(tok))
            buf, was_quoted = [], False
        else:
            buf.append(ch)
    tok = "".join(buf)
    items.append(None if (tok == "NULL" and not was_quoted) else elem(tok))
    return items


_SCALAR = {
    16: lambda s: s == "t",
    20: int, 21: int, 23: int, 26: int,
    700: float, 701: float, 1700: Decimal,
    114: json.loads, 3802: json.loads,
    1114: _parse_ts, 1184: _parse_ts,
    1082: date.fromisoformat,
}
_ARRAY = {1000: 16, 1005: 21, 1007: 23, 1016: 20, 1021: 700, 1022: 701, 1231: 1700, 199: 114, 3807: 3802,
          1009: 25, 1015: 25}


def _converter(type_id: int):
    if type_id in _SCALAR:
        return _SCALAR[type_id]
    if type_id in _ARRAY:
        elem = _SCALAR.get(_ARRAY[type_id], str)
        return lambda s: _parse_array(s, elem)
    return str


class Result:
    def __init__(self, payload: dict | None):
        payload = payload or {}
        fields = payload.get("fields") or []
        convs = {f["name"]: _converter(f.get("dataTypeID", 25)) for f in fields}
        self._rows = [
            {k: (None if v is None else convs.get(k, str)(v)) for k, v in row.items()}
            for row in payload.get("rows") or []
        ]
        self.rowcount = payload.get("rowCount") if payload.get("rowCount") is not None else len(self._rows)
        self._i = 0

    def fetchone(self) -> dict | None:
        if self._i >= len(self._rows):
            return None
        row = self._rows[self._i]
        self._i += 1
        return row

    def fetchall(self) -> list[dict]:
        rows = self._rows[self._i:]
        self._i = len(self._rows)
        return rows

    def __iter__(self) -> Iterator[dict]:
        return iter(self.fetchall())


class _Cursor:
    def __init__(self, conn: "NeonHttpConnection"):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query: str, params: Any = ()) -> Result:
        return self._conn.execute(query, params)

    def executemany(self, query: str, params_seq) -> None:
        stmts = [self._conn._stmt(query, p) for p in params_seq]
        if not stmts:
            return
        if self._conn._buffer is not None:
            self._conn._buffer.extend(stmts)
        else:
            self._conn._post_batch(stmts)


class NeonHttpConnection:
    def __init__(self, dsn: str, timeout: float = 30):
        self._dsn = dsn
        self._url = f"https://{urlparse(dsn).hostname}/sql"
        self._timeout = timeout
        self._buffer: list[dict] | None = None

    # -- transport -------------------------------------------------------
    def _post(self, body: dict, extra_headers: dict | None = None) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Neon-Connection-String": self._dsn,
            "Neon-Raw-Text-Output": "true",
            "Neon-Array-Mode": "false",
            **(extra_headers or {}),
        }
        req = urllib.request.Request(self._url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("message", detail)
            except Exception:
                pass
            raise NeonHttpError(f"Neon HTTP {exc.code}: {detail}") from None

    def _post_batch(self, stmts: list[dict]) -> list[dict]:
        out = self._post(
            {"queries": stmts},
            {"Neon-Batch-Isolation-Level": "ReadCommitted", "Neon-Batch-Read-Only": "false"},
        )
        return out.get("results", [])

    @staticmethod
    def _stmt(query: str, params: Any) -> dict:
        return {"query": _to_dollar_params(query) if params else query.replace("%%", "%"),
                "params": [_param(p) for p in (params or ())]}

    # -- psycopg-like API ------------------------------------------------
    def execute(self, query: str, params: Any = ()) -> Result:
        stmt = self._stmt(query, params)
        if self._buffer is not None and not _READ_RE.match(query):
            # Buffered write: results (incl. RETURNING) are not available until commit.
            self._buffer.append(stmt)
            return Result({"rowCount": None})
        if ";" in query.strip().rstrip(";"):
            # Multi-statement script (e.g. schema DDL): run as one batch.
            parts = [p.strip() for p in query.split(";") if p.strip()]
            self._post_batch([{"query": p, "params": []} for p in parts])
            return Result(None)
        return Result(self._post(stmt))

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    @contextmanager
    def transaction(self):
        if self._buffer is not None:  # nested: fold into outer transaction
            yield self
            return
        self._buffer = []
        try:
            yield self
            pending, self._buffer = self._buffer, None
            if pending:
                self._post_batch(pending)
        finally:
            self._buffer = None

    def close(self) -> None:
        pass
