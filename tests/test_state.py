"""Tests for state.py -- shared agent state module."""

import importlib
import sys
import pytest


def fresh_state():
    """Reload state module so each test starts with a clean slate."""
    if "state" in sys.modules:
        del sys.modules["state"]
    import state
    return state


def test_try_start_run_succeeds_when_idle():
    state = fresh_state()
    assert state.try_start_run() is True


def test_try_start_run_fails_when_running():
    state = fresh_state()
    state.try_start_run()
    assert state.try_start_run() is False


def test_end_run_allows_second_start():
    state = fresh_state()
    state.try_start_run()
    state.end_run()
    assert state.try_start_run() is True


def test_update_and_get():
    state = fresh_state()
    state.try_start_run()
    state.update(task="click the button", step=2, status="running")
    s = state.get()
    assert s["task"] == "click the button"
    assert s["step"] == 2
    assert s["status"] == "running"


def test_encode_png_missing_file_raises():
    state = fresh_state()
    with pytest.raises(OSError):
        state.encode_png("/tmp/definitely_does_not_exist_ecua.png")


def test_cancel_flag():
    state = fresh_state()
    state.try_start_run()
    assert state.is_cancel_requested() is False
    state.request_cancel()
    assert state.is_cancel_requested() is True
    state.clear_cancel()
    assert state.is_cancel_requested() is False


def test_try_start_run_clears_stale_cancel():
    state = fresh_state()
    state.try_start_run()
    state.request_cancel()
    state.end_run()
    state.try_start_run()  # should clear the cancel
    assert state.is_cancel_requested() is False
