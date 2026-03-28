"""Tests for db.py -- SQLite persistence layer."""

import os
import tempfile
import pytest

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect DB_PATH to a temp file so tests don't touch runs.db."""
    import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test_runs.db")
    db.init_db()
    return db


def test_init_db_idempotent(tmp_db):
    """init_db can be called multiple times without error."""
    tmp_db.init_db()
    tmp_db.init_db()


def test_log_step_and_get_actions(tmp_db):
    run_id = "test-run-001"
    tmp_db.log_step(run_id, "open terminal", 0, "screenshots/s.png", {}, "CLICK 100 200", "OK")
    tmp_db.log_step(run_id, "open terminal", 1, "screenshots/s2.png", {}, "DONE", "done")

    actions = tmp_db.get_actions(run_id)
    assert len(actions) == 2
    assert actions[0]["step"] == 0
    assert actions[0]["action"] == "CLICK 100 200"
    assert actions[0]["result"] == "OK"
    assert actions[1]["step"] == 1
    assert actions[1]["action"] == "DONE"


def test_get_actions_empty_for_unknown_run(tmp_db):
    actions = tmp_db.get_actions("no-such-run")
    assert actions == []


def test_get_actions_ordered_by_step(tmp_db):
    run_id = "test-order"
    # Insert out of order
    tmp_db.log_step(run_id, "task", 2, "s.png", {}, "PRESS enter", "OK")
    tmp_db.log_step(run_id, "task", 0, "s.png", {}, "CLICK 10 20", "OK")
    tmp_db.log_step(run_id, "task", 1, "s.png", {}, "TYPING hello", "OK")

    actions = tmp_db.get_actions(run_id)
    assert [a["step"] for a in actions] == [0, 1, 2]
