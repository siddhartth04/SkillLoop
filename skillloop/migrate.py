"""Schema versioning + backup/restore. A production store must survive upgrades and be recoverable.

`user_version` PRAGMA tracks the schema. Migrations are idempotent and run automatically on Store init.
Backup uses SQLite's online backup API (safe while the DB is in use); restore is a guarded file swap.
"""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

SCHEMA_VERSION = 1  # bump when a migration is added


def current_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Run any pending migrations. Returns the list applied. Idempotent."""
    applied = []
    v = current_version(conn)
    # v0 -> v1: columns added over v0.3 development that older DBs may lack. ALTER is a no-op-safe add.
    if v < 1:
        cols = {
            "skills": [],  # skills are stored as a JSON blob, so no column migration needed
        }
        # ensure the newer tables exist (older stores predate principles/metrics)
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS principles (id TEXT PRIMARY KEY, ts REAL, json TEXT);
        CREATE TABLE IF NOT EXISTS metrics (ts REAL, key TEXT, value REAL);
        """)
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        applied.append("v0->v1: ensure principles/metrics tables")
    return applied


def backup(db_path: str, dest_dir: str | None = None) -> str:
    """Online backup (safe while in use). Returns the backup file path."""
    src = Path(db_path)
    dest_dir = Path(dest_dir or src.parent / "backups")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{src.stem}-{time.strftime('%Y%m%d-%H%M%S')}.db"
    with sqlite3.connect(str(src)) as s, sqlite3.connect(str(dest)) as d:
        s.backup(d)
    return str(dest)


def restore(backup_path: str, db_path: str) -> None:
    """Replace the live DB with a backup. The caller must ensure no Store is writing."""
    src, dst = Path(backup_path), Path(db_path)
    if not src.exists():
        raise FileNotFoundError(backup_path)
    # keep the WAL/SHM from colliding with the restored file
    for suffix in ("-wal", "-shm"):
        p = Path(str(dst) + suffix)
        if p.exists():
            p.unlink()
    shutil.copy2(src, dst)
