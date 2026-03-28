"""
server.py -- FastAPI server for ECUA.

Endpoints:
  POST /run     -- start a task (returns 409 if already running)
  POST /cancel  -- stop the running task
  GET  /status  -- current agent state (polling endpoint)
  GET  /        -- serve web/index.html

Start with:
  VNC_HOST=localhost uvicorn server:app --port 8000
"""

import os
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import db
import state
import agent_loop
from ecua2_agent.planner_module import planner

MODEL_PATH = "ecua2_agent/planner_module/models/llama-3.2-1b"

# Load LLM at module import time (main thread) -- prevents CUDA daemon-thread error.
# vllm's CUDA context must be initialized on the main thread before FastAPI starts.
_LLM = planner.get_llm(MODEL_PATH)

app = FastAPI(title="ECUA Agent")
db.init_db()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=500,
                      description="Plain English task for the agent")
    max_steps: int = Field(default=5, ge=1, le=10,
                           description="Maximum agent steps (1-10)")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/run")
async def run_endpoint(req: TaskRequest):
    """Start a new task. Returns 409 if an agent is already running."""
    if not state.try_start_run():
        raise HTTPException(status_code=409, detail="Agent already running")

    run_id = str(uuid.uuid4())
    state.update(
        run_id=run_id,
        task=req.task,
        step=0,
        max_steps=req.max_steps,
        status="running",
        screenshot_b64=None,
        actions=[],
    )

    t = threading.Thread(
        target=agent_loop.run_task,
        args=(req.task, req.max_steps, run_id, _LLM),
        daemon=True,
    )
    try:
        t.start()
    except Exception as e:
        state.end_run()
        raise HTTPException(status_code=500, detail=f"Failed to start agent thread: {e}")

    return {"status": "started", "run_id": run_id}


@app.post("/cancel")
async def cancel_endpoint():
    """Cancel the currently running task."""
    if not state.request_cancel():
        raise HTTPException(status_code=400, detail="No task is currently running")
    return {"status": "cancelling"}


@app.get("/status")
def status_endpoint():
    """
    Returns the current agent state.

    status: idle | running | done | fail | cancelled
    """
    return state.get()


@app.get("/")
async def index():
    return FileResponse("web/index.html")
