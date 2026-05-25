"""Shared helpers: env loading, Postgres connection, ingest_log writer, sha256, universe loader."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import psycopg2
import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

BRONZE_ROOT = REPO_ROOT / "bronze"
BRONZE_ROOT.mkdir(exist_ok=True)

def pg_conn(dbname: str = "advdatafinal") -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=os.getenv("PG_HOST", "127.0.0.1"),
        port=os.getenv("PG_PORT", "5432"),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD"),
        database=dbname,
    )

def sha256_of_file(path: Path) -> str:
    """Hash a single file. If `path` is a directory, hash a manifest of (filename, size) tuples."""
    p = Path(path)
    h = hashlib.sha256()
    if p.is_dir():
        manifest = sorted((str(f.relative_to(p)), f.stat().st_size) for f in p.rglob("*") if f.is_file())
        for name, size in manifest:
            h.update(f"{name}:{size}\n".encode())
        return h.hexdigest()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def log_ingest(
    source: str,
    symbol: str | None,
    window_key: str,
    file_path: Path,
    row_count: int,
) -> None:
    """Idempotent upsert into raw.ingest_log."""
    sha = sha256_of_file(file_path) if Path(file_path).exists() else "0" * 64
    with pg_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw.ingest_log
              (source, symbol, window_key, file_path, row_count, sha256, ingested_at)
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (source, symbol, window_key) DO UPDATE SET
              file_path   = EXCLUDED.file_path,
              row_count   = EXCLUDED.row_count,
              sha256      = EXCLUDED.sha256,
              ingested_at = now();
            """,
            (source, symbol, window_key, str(file_path), row_count, sha),
        )

def load_universe() -> dict[str, list[str]]:
    """Return {sector_name: [ticker, ...]} from config/universe.yaml."""
    cfg = yaml.safe_load(open(REPO_ROOT / "config" / "universe.yaml"))
    return cfg["universe"]["tickers"]

def all_tickers() -> list[str]:
    return [t for tickers in load_universe().values() for t in tickers]

def sector_for(symbol: str) -> str:
    for sector, tickers in load_universe().items():
        if symbol in tickers:
            return sector
    return "Unknown"
