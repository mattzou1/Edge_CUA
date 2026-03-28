"""
agent_loop.py -- observe-act loop running in a dedicated daemon thread.

Owned by server.py, which spawns one thread per POST /run.
Shares state with server.py exclusively through state.py (no circular imports).
"""

import json
import os
import socket
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import state
import db
from ecua2_agent.planner_module import planner
from ecua2_agent.controller_module.controller import Controller

VNC_HOST = os.environ.get("VNC_HOST", "localhost")
VNC_PORT = int(os.environ.get("VNC_PORT", "5900"))

_ROOT = Path(__file__).parent
(_ROOT / "screenshots").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# VNC readiness check
# ---------------------------------------------------------------------------

def wait_for_vnc(host: str = VNC_HOST, port: int = VNC_PORT, timeout: int = 120) -> None:
    """Block until VNC port is accepting connections (or raise TimeoutError)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=1).close()
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"VNC not ready at {host}:{port} after {timeout}s")


# ---------------------------------------------------------------------------
# Vision subprocess (runs vision_CPU.py against a pre-captured screenshot)
# ---------------------------------------------------------------------------

def run_vision_subprocess(screenshot_path: str) -> dict:
    """
    Run vision_CPU.py on a saved screenshot and return the parsed vision dict.
    Uses --img flag so it never touches the host display.
    """
    cmd = [
        "python3",
        str(_ROOT / "ecua2_agent/vision_module/vision_CPU.py"),
        "--img", screenshot_path,
        "--bbox", "0,0,1280,800",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=55)
    except subprocess.TimeoutExpired:
        raise RuntimeError("vision_CPU.py timed out after 55s (subprocess killed)")

    if result.returncode != 0:
        raise RuntimeError(
            f"vision_CPU.py failed (rc={result.returncode}): {result.stderr[:500]}"
        )

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(f"vision_CPU.py produced no output. stderr: {result.stderr[:500]}")

    # Try parsing the whole stdout, then fall back to the last JSON line
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        data = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if data is None:
            raise RuntimeError(f"Could not parse JSON from vision_CPU.py. stdout: {stdout[:300]}")

    return data.get("vision", data)


# ---------------------------------------------------------------------------
# Per-step timeout helpers (work from daemon threads, unlike signal.alarm)
# ---------------------------------------------------------------------------

class _StepTimeout(Exception):
    pass


# Bounded single-worker executors prevent stale threads from accumulating.
# Vision and planner each get their own executor so a slow LLM doesn't block
# a vision call and vice versa. max_workers=1 means at most one inflight call
# per type; a timed-out call won't start a second one until the worker finishes.
_vision_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision")
_planner_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner")


def _run_with_timeout(fn, args=(), timeout_secs: int = 60):
    # Pick the matching executor so we don't accidentally share workers.
    executor = _planner_executor if fn.__name__ == "generate_step" else _vision_executor
    future = executor.submit(fn, *args)
    try:
        return future.result(timeout=timeout_secs)
    except FutureTimeout:
        future.cancel()  # no-op if already running, but signals intent
        raise _StepTimeout(f"Step timed out after {timeout_secs}s")


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_task(task: str, max_steps: int = 5, run_id: str = None, llm=None) -> None:
    """
    Run the observe-act loop for `task` inside the Docker VNC VM.

    Args:
        task:      Plain-English task string.
        max_steps: Maximum steps before giving up.
        run_id:    UUID generated in server.py (single source of truth).
        llm:       Pre-loaded vllm.LLM singleton (loaded on main thread).
    """
    import vncdotool.api as vnc_api

    status = "fail"
    client = None

    try:
        # Wait for both VNC (5900) and websockify/noVNC (6080)
        wait_for_vnc(host=VNC_HOST, port=VNC_PORT)
        wait_for_vnc(host=VNC_HOST, port=6080)

        client = vnc_api.connect(VNC_HOST, port=VNC_PORT)
        ctrl = Controller(vnc_client=client)
        previous_actions = []

        for step in range(max_steps):
            # --- cancel check ---
            if state.is_cancel_requested():
                state.clear_cancel()
                status = "cancelled"
                state.update(step=step, status="cancelled")
                print(f"[AGENT] Run {run_id} cancelled at step {step}")
                return

            screenshot_path = str(_ROOT / f"screenshots/{run_id}_{step:03d}.png")

            # --- capture screenshot ---
            try:
                client.captureScreen(screenshot_path)
            except Exception as e:
                print(f"[AGENT] captureScreen failed at step {step}: {e}")
                status = "fail"
                state.update(step=step, status="fail",
                             actions=db.get_actions(run_id))
                return

            # --- run vision subprocess ---
            try:
                vision = _run_with_timeout(
                    run_vision_subprocess, args=(screenshot_path,), timeout_secs=60
                )
            except (_StepTimeout, RuntimeError) as e:
                print(f"[AGENT] Vision failed at step {step}: {e}")
                db.log_step(run_id, task, step, screenshot_path, {}, "VISION_ERROR", str(e))
                state.update(step=step, status="running",
                             screenshot_b64=_safe_encode(screenshot_path),
                             actions=db.get_actions(run_id))
                continue  # skip this step, try again

            # --- generate action (with timeout) ---
            try:
                raw_action, coord_action = _run_with_timeout(
                    planner.generate_step,
                    args=(llm, task, vision, previous_actions),
                    timeout_secs=90,  # LLM can be slow
                )
            except (_StepTimeout, Exception) as e:
                print(f"[AGENT] Planner failed at step {step}: {e}")
                db.log_step(run_id, task, step, screenshot_path, vision, "PLANNER_ERROR", str(e))
                state.update(step=step, status="running",
                             screenshot_b64=_safe_encode(screenshot_path),
                             actions=db.get_actions(run_id))
                continue

            print(f"[STEP {step}] raw={raw_action!r} exec={coord_action!r}")

            # --- terminal check (DONE / FAIL) ---
            token = raw_action.strip().upper().split()[0] if raw_action.strip() else "FAIL"
            if token in ("DONE", "FAIL"):
                status = token.lower()
                db.log_step(run_id, task, step, screenshot_path, vision, raw_action, status)
                state.update(
                    step=step,
                    status=status,
                    screenshot_b64=_safe_encode(screenshot_path),
                    actions=db.get_actions(run_id),
                )
                return

            # --- execute action via VNC ---
            executable = coord_action if coord_action is not None else raw_action
            result = ctrl.execute_action_vnc(executable)

            # Feed the result back so the LLM knows if the action failed.
            if result.startswith("ERROR"):
                print(f"[AGENT] Action error at step {step}: {result}")
                previous_actions.append(f"{raw_action} [FAILED: {result}]")
            else:
                previous_actions.append(raw_action)
            db.log_step(run_id, task, step, screenshot_path, vision, raw_action, result)
            state.update(
                step=step,
                status="running",
                screenshot_b64=_safe_encode(screenshot_path),
                actions=db.get_actions(run_id),
            )

        # Exhausted max_steps without DONE/FAIL
        status = "fail"
        state.update(status="fail")

    except TimeoutError as e:
        print(f"[AGENT] VNC connection timed out: {e}")
        state.update(status="fail")
    except Exception as e:
        print(f"[AGENT] Unexpected error: {e}")
        state.update(status="fail")
    finally:
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass
        state.end_run()
        print(f"[AGENT] Run {run_id} finished with status={status}")


def _safe_encode(path: str) -> str | None:
    """Encode a PNG to base64, returning None if the file doesn't exist."""
    try:
        return state.encode_png(path)
    except OSError:
        return None
