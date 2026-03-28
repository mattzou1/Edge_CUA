"""
state.py -- shared in-memory state between server.py and agent_loop.py.

Neither server.py nor agent_loop.py imports the other; both import this module.
All access is protected by a single threading.Lock.
"""

import threading
import base64

_lock = threading.Lock()
_agent_running = False
_cancel_requested = False

_shared: dict = {
    "run_id": None,
    "task": None,
    "step": 0,
    "max_steps": 5,
    "status": "idle",       # idle | running | done | fail | cancelled
    "screenshot_b64": None,
    "actions": [],
}


def update(**kwargs) -> None:
    with _lock:
        _shared.update(kwargs)


def get() -> dict:
    with _lock:
        return {**_shared, "actions": list(_shared["actions"])}


def try_start_run() -> bool:
    """Atomically claim the agent slot. Returns True if successful, False if already running."""
    global _agent_running, _cancel_requested
    with _lock:
        if _agent_running:
            return False
        _agent_running = True
        _cancel_requested = False  # clear any stale cancel from the previous run
        return True


def end_run() -> None:
    global _agent_running
    with _lock:
        _agent_running = False


def request_cancel() -> bool:
    """Set cancel flag only if an agent is currently running. Returns True if set."""
    global _cancel_requested
    with _lock:
        if not _agent_running:
            return False
        _cancel_requested = True
        return True


def is_cancel_requested() -> bool:
    with _lock:
        return _cancel_requested


def clear_cancel() -> None:
    global _cancel_requested
    with _lock:
        _cancel_requested = False


def encode_png(path: str) -> str:
    """Read a PNG file and return base64-encoded string for JSON transport."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
