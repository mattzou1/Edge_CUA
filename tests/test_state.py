"""Tests for state.py -- shared agent state module."""

import pytest
import state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state in-place before each test.

    Using del sys.modules["state"] + reimport creates a NEW module object, which
    breaks any code (agent_loop, server) that holds a reference to the original
    module -- they would see a stale object while sys.modules points somewhere new.
    In-place reset avoids this: one module object, all references stay valid.
    """
    with state._lock:
        state._agent_running = False
        state._cancel_requested = False
        state._shared.update({
            "run_id": None, "task": None, "step": 0, "max_steps": 5,
            "status": "idle", "screenshot_b64": None, "actions": [],
        })
    yield
    with state._lock:
        state._agent_running = False
        state._cancel_requested = False


def test_try_start_run_succeeds_when_idle():
    assert state.try_start_run() is True


def test_try_start_run_fails_when_running():
    state.try_start_run()
    assert state.try_start_run() is False


def test_end_run_allows_second_start():
    state.try_start_run()
    state.end_run()
    assert state.try_start_run() is True


def test_update_and_get():
    state.try_start_run()
    state.update(task="click the button", step=2, status="running")
    s = state.get()
    assert s["task"] == "click the button"
    assert s["step"] == 2
    assert s["status"] == "running"


def test_encode_png_missing_file_raises():
    with pytest.raises(OSError):
        state.encode_png("/tmp/definitely_does_not_exist_ecua.png")


def test_cancel_flag():
    state.try_start_run()
    assert state.is_cancel_requested() is False
    state.request_cancel()
    assert state.is_cancel_requested() is True
    state.clear_cancel()
    assert state.is_cancel_requested() is False


def test_try_start_run_clears_stale_cancel():
    state.try_start_run()
    state.request_cancel()
    state.end_run()
    state.try_start_run()  # should clear the cancel
    assert state.is_cancel_requested() is False
