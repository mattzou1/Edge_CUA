"""Tests for agent_loop.py -- observe-act loop."""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Stub GPU, display, and VNC deps before any project import
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("pyautogui", MagicMock())
vnc_mock = MagicMock()
sys.modules.setdefault("vncdotool", vnc_mock)
sys.modules.setdefault("vncdotool.api", vnc_mock)

import agent_loop
import state as s


def _reset_state():
    import importlib
    if "state" in sys.modules:
        del sys.modules["state"]
    import state
    return state


@pytest.fixture(autouse=True)
def clean_state():
    # Reset in-place on the module agent_loop already captured at import time.
    # del sys.modules + reimport would create a new object agent_loop never sees.
    import state
    with state._lock:
        state._agent_running = False
        state._cancel_requested = False
        state._shared.update({
            "run_id": None, "task": None, "step": 0, "max_steps": 5,
            "status": "idle", "screenshot_b64": None, "actions": [],
        })
    yield state
    with state._lock:
        state._agent_running = False
        state._cancel_requested = False


def _make_run(task="click terminal", max_steps=3, run_id="run-test-001"):
    return task, max_steps, run_id


def _mock_client(screenshot_side_effect=None):
    client = MagicMock()
    if screenshot_side_effect:
        client.captureScreen.side_effect = screenshot_side_effect
    return client


def test_done_terminates_early():
    """Agent stops after DONE without exhausting max_steps."""
    with patch("agent_loop.wait_for_vnc"), \
         patch("vncdotool.api.connect", return_value=_mock_client()), \
         patch("agent_loop.run_vision_subprocess", return_value={"elements": []}), \
         patch("agent_loop._run_with_timeout", side_effect=[
             {"elements": []},               # vision step 0
             ("DONE", None),                 # planner step 0
         ]), \
         patch("db.log_step"), \
         patch("db.get_actions", return_value=[]):
        import state as st
        st.try_start_run()
        agent_loop.run_task("open terminal", max_steps=5, run_id="r1", llm=MagicMock())
        assert st.get()["status"] == "done"


def test_fail_terminates_early():
    """Agent stops after FAIL."""
    with patch("agent_loop.wait_for_vnc"), \
         patch("vncdotool.api.connect", return_value=_mock_client()), \
         patch("agent_loop._run_with_timeout", side_effect=[
             {"elements": []},
             ("FAIL", None),
         ]), \
         patch("db.log_step"), \
         patch("db.get_actions", return_value=[]):
        import state as st
        st.try_start_run()
        agent_loop.run_task("do task", max_steps=5, run_id="r2", llm=MagicMock())
        assert st.get()["status"] == "fail"


def test_max_steps_exhausted_sets_fail():
    """When all steps complete without DONE/FAIL, status=fail."""
    with patch("agent_loop.wait_for_vnc"), \
         patch("vncdotool.api.connect", return_value=_mock_client()), \
         patch("agent_loop._run_with_timeout", side_effect=[
             # step 0: vision, planner
             {"elements": []}, ("CLICK 100 200", "CLICK 100 200"),
             # step 1: vision, planner
             {"elements": []}, ("CLICK 300 400", "CLICK 300 400"),
         ]), \
         patch("db.log_step"), \
         patch("db.get_actions", return_value=[]):
        import state as st
        st.try_start_run()
        agent_loop.run_task("do task", max_steps=2, run_id="r3", llm=MagicMock())
        assert st.get()["status"] == "fail"


def test_cancel_stops_loop():
    """Cancel flag causes loop to stop with status=cancelled."""
    with patch("agent_loop.wait_for_vnc"), \
         patch("vncdotool.api.connect", return_value=_mock_client()):
        import state as st
        st.try_start_run()
        st.request_cancel()
        agent_loop.run_task("do task", max_steps=5, run_id="r4", llm=MagicMock())
        assert st.get()["status"] == "cancelled"


def test_end_run_called_in_finally():
    """state.end_run() is always called regardless of outcome."""
    with patch("agent_loop.wait_for_vnc", side_effect=TimeoutError("VNC timeout")):
        import state as st
        st.try_start_run()
        agent_loop.run_task("do task", max_steps=5, run_id="r5", llm=MagicMock())
        # After end_run, a new run should be startable
        assert st.try_start_run() is True


def test_db_log_step_called_per_action():
    """db.log_step is called once per executed action."""
    with patch("agent_loop.wait_for_vnc"), \
         patch("vncdotool.api.connect", return_value=_mock_client()), \
         patch("agent_loop._run_with_timeout", side_effect=[
             {"elements": []}, ("CLICK 10 20", "CLICK 10 20"),
             {"elements": []}, ("DONE", None),
         ]), \
         patch("db.log_step") as mock_log, \
         patch("db.get_actions", return_value=[]):
        import state as st
        st.try_start_run()
        agent_loop.run_task("task", max_steps=5, run_id="r6", llm=MagicMock())
        assert mock_log.call_count == 2  # one action + DONE
