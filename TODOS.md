# ECUA TODOs

## Build Order (do in this order)

- [x] **1. Fix planner.py LLM singleton** (30 min)
  - Add `get_llm(model_path)` singleton
  - Change `generate_step` signature: `(llm, task, vision_data, previous_actions) -> tuple[str, str | None]`
  - Delete dead functions: `generate_plan`, `build_prompt`, `load_vision_json`

- [x] **2. Write state.py** -- shared state + lock + encode_png

- [x] **3. Write db.py** -- `init_db()`, `log_step()`, `get_actions()`

- [x] **4. Write agent_loop.py**
  - Move `run_vision_subprocess` from run_agent.py, change to use `--img` flag
  - Write `run_task(task, max_steps, run_id, llm)`
  - Unpack tuple: `raw_action, coord_action = planner.generate_step(...)`

- [x] **5. Wire controller.py VNC mode**
  - Add `execute_action_vnc(action)` dispatch
  - NOTE: VNC_HOST env guard not added to controller (agent_loop defaults to localhost)

- [x] **6. Docker setup**
  - Write `docker/Dockerfile` (extends dorowu/ubuntu-desktop-lxde-vnc, installs LibreOffice + gnome-terminal, pins RESOLUTION=1280x800)
  - Write `docker/docker-compose.yml`
  - `docker-compose up -d` + confirm `vncdotool -s localhost captureScreen test.png` works

- [x] **7. Write server.py**
  - Load LLM at module level (main thread): `_LLM = planner.get_llm(MODEL_PATH)`
  - POST /run with 409 guard, Pydantic validators (min_length=1, max_length=500, ge=1, le=10)
  - POST /cancel -- 200 if running, 400 if not; calls state.request_cancel()
  - GET /status (status enum: idle | running | done | fail | cancelled)
  - GET / serves web/index.html

- [x] **2b. state.py: add cancel support**
  - _cancel_requested flag, request_cancel(), is_cancel_requested(), clear_cancel()
  - clear_cancel() called inside try_start_run() to reset stale cancel

- [x] **4b. agent_loop.py: add cancel + timeout**
  - Check is_cancel_requested() at start of each step
  - Per-step timeout uses threading.Thread.join(timeout=N) -- signal.SIGALRM doesn't work from daemon threads

- [x] **3b. db.py: WAL mode**
  - `conn.execute("PRAGMA journal_mode=WAL")` in init_db()

- [x] **6b. noVNC -- expose websockify port**
  - Add `"6080:6080"` to docker-compose.yml ports
  - Add `wait_for_vnc(host=VNC_HOST, port=6080)` in agent_loop.py startup

- [x] **8. Write web/index.html**
  - Submit form + 1500ms polling + screenshot display (fallback)
  - noVNC iframe: `src="http://localhost:6080/vnc.html?autoconnect=true&resize=scale"`
  - Cancel button (visible only when status=running)
  - "Waiting for first screenshot..." placeholder
  - Error display if POST /run returns non-200
  - Character count indicator on task input (500 max)

- [x] **9. Write tests**
  - `tests/test_state.py` (7 tests)
  - `tests/test_db.py` (4 tests)
  - `tests/test_server.py` (7 tests)
  - `tests/test_agent_loop.py` (6 tests)
  - `tests/test_controller.py` (9 tests)

- [x] **10. Delete OSWorld files**
  - `ecua_run.py`, `ecua_lib_run_single.py`, `ecua_score.py`, `run_agent.py`

- [ ] **11. Run cliOpenTerminalAndType end-to-end** -- record 60 seconds

## Design Doc

Full design: `~/.gstack/projects/mattzou1-Edge_CUA/mattzou-main-design-20260327-114203.md`

## Deferred (from CEO Review)

- **Cloud deployment**: Deploy to RunPod/Lambda Labs GPU instance. Public URL. Do this after localhost demo is stable and impressive. Effort: L (human: ~3 days / CC: ~2 hrs). Why: turns a portfolio project into a live product interviewers can visit.

- **Failure replay UI**: Add `/runs` (list all runs) and `/runs/{run_id}/replay` (step-by-step timeline). SQLite already logs all data. Effort: M (human: ~2 days / CC: ~45 min). Why: makes failed runs a feature, not a bug.

- ~~**noVNC browser streaming**~~ -- moved to main build order (step 6b)

- **Auto-record video**: Move ScreenRecorder from run_agent.py to agent_loop.py. Record .mp4 per task run. Effort: S (human: ~2 hrs / CC: ~20 min). Why: shareable artifact for recruiters.

- **Run history in web UI**: GET /runs endpoint + list view. Reads from SQLite. Effort: S (human: ~1 hr / CC: ~15 min). Why: shows off the logging, makes the demo feel more like a product.

## Notes

- VNC_HOST must be set: `VNC_HOST=localhost uvicorn server:app --port 8000`
- If vncdotool Twisted "reactor already running" error: swap to `python-vnc` or `pyVNC`
- Confirm `captureScreen` over TCP VNC before building the loop (step 6 smoke test)
