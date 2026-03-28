"""
db.py -- SQLite persistence for agent runs and steps.

Schema:
  steps(id, run_id, task, step, screenshot, vision, action, result, ts)

Thread safety: WAL mode allows concurrent readers. Writes are serialised by
SQLite's internal locking. One connection is reused per thread (threading.local)
to avoid fd exhaustion from opening a new connection on every call.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path("runs.db")

_local = threading.local()


def _connect() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return _local.conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")  # concurrent read/write -- no stutter
        conn.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT    NOT NULL,
                task       TEXT    NOT NULL,
                step       INTEGER NOT NULL,
                screenshot TEXT    NOT NULL,
                vision     TEXT    NOT NULL,
                action     TEXT    NOT NULL,
                result     TEXT    NOT NULL,
                ts         REAL    NOT NULL
            )
        """)
        conn.commit()


def log_step(
    run_id: str,
    task: str,
    step: int,
    screenshot: str,
    vision: dict,
    action: str,
    result: str,
) -> None:
    """Persist one agent step to SQLite."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO steps (run_id, task, step, screenshot, vision, action, result, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, task, step, screenshot, json.dumps(vision), action, result, time.time()),
        )
        conn.commit()


def get_actions(run_id: str) -> list:
    """Return all steps for a run as [{"step": n, "action": s, "result": r}, ...]."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT step, action, result FROM steps WHERE run_id = ? ORDER BY step",
            (run_id,),
        ).fetchall()
    return [{"step": row["step"], "action": row["action"], "result": row["result"]}
            for row in rows]
