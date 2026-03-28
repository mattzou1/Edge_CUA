"""Tests for server.py -- FastAPI endpoints."""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub out heavy imports before server loads
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.LLM", MagicMock())

# Patch get_llm and init_db so server import doesn't load GPU or touch disk
with patch("ecua2_agent.planner_module.planner.get_llm", return_value=MagicMock()), \
     patch("db.init_db"):
    from fastapi.testclient import TestClient
    import server
    client = TestClient(server.app)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset in-place on the module server.py already captured at import time."""
    import state as s
    with s._lock:
        s._agent_running = False
        s._cancel_requested = False
        s._shared.update({
            "run_id": None, "task": None, "step": 0, "max_steps": 5,
            "status": "idle", "screenshot_b64": None, "actions": [],
        })
    yield
    with s._lock:
        s._agent_running = False
        s._cancel_requested = False


def _reset():
    import state as s
    with s._lock:
        s._agent_running = False
        s._cancel_requested = False
        s._shared["status"] = "idle"


def test_get_status_idle():
    _reset()
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["status"] == "idle"


def test_post_run_starts_task():
    _reset()
    with patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start = MagicMock()
        r = client.post("/run", json={"task": "open terminal", "max_steps": 2})
    assert r.status_code == 200
    assert r.json()["status"] == "started"
    assert "run_id" in r.json()


def test_post_run_409_when_already_running():
    _reset()
    import state as s
    s._agent_running = True
    r = client.post("/run", json={"task": "open terminal"})
    assert r.status_code == 409


def test_post_run_422_empty_task():
    _reset()
    r = client.post("/run", json={"task": "", "max_steps": 5})
    assert r.status_code == 422


def test_post_run_422_max_steps_out_of_range():
    _reset()
    r = client.post("/run", json={"task": "do something", "max_steps": 99})
    assert r.status_code == 422


def test_post_cancel_400_when_idle():
    _reset()
    r = client.post("/cancel")
    assert r.status_code == 400


def test_post_cancel_ok_when_running():
    _reset()
    import state as s
    with s._lock:
        s._agent_running = True
        s._shared["status"] = "running"
    r = client.post("/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelling"
