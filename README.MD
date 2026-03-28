![ECUA](./images/AI_Agent.png)

# ECUA — Edge Computer Use Agent

An AI agent that controls a desktop VM. You give it a task in plain English; it uses a vision model (OmniParser) to read the screen, a local LLM (Llama 3.2-1B via vllm) to decide what to do, and vncdotool to click/type/press keys — all running inside a Docker container.

## Hardware Requirements

- **GPU:** NVIDIA GeForce RTX 3050 or similar (for vllm inference)
- **RAM:** 16 GB or more
- **OS:** Ubuntu Linux 22.04 (host)

## How it works

1. Docker container runs Ubuntu LXDE desktop + x11vnc + websockify (noVNC)
2. Agent captures a screenshot via VNC, runs OmniParser to get UI elements
3. Llama 3.2-1B picks the next action (click, type, hotkey, etc.)
4. vncdotool executes the action inside the container
5. Repeat until DONE or max steps reached
6. Web UI at `localhost:8000` shows live desktop stream + action log

## Setup

### 1. Install prerequisites

- Docker + Docker Compose
- Python 3.11+ with [uv](https://github.com/astral-sh/uv)

### 2. Clone and install dependencies

```bash
git clone https://github.com/mattzou1/Edge_CUA.git
cd Edge_CUA
uv pip install -r requirements-uv.txt
```

### 3. Download the model

Place Llama 3.2-1B-Instruct weights at:
```
ecua2_agent/planner_module/models/llama-3.2-1b/
```

### 4. Start the Docker desktop

```bash
cd docker
docker compose up -d
cd ..
```

Verify VNC is ready:
```bash
python3 -c "import socket; socket.create_connection(('localhost', 5900), timeout=5).close(); print('VNC OK')"
```

### 5. Start the server

```bash
VNC_HOST=localhost uvicorn server:app --port 8000
```

First startup takes ~60 seconds while the LLM loads.

### 6. Open the web UI

Visit `http://localhost:8000`. Type a task and click Run.

Example tasks:
- `open terminal and type top`
- `open LibreOffice Writer and type Hello World`

## Running tests

```bash
uvx --with fastapi --with httpx pytest tests/ -v
```

## Project structure

```
server.py            FastAPI server (POST /run, POST /cancel, GET /status, GET /)
agent_loop.py        Observe-act loop (VNC screenshot -> vision -> LLM -> action)
state.py             Shared in-memory state (thread-safe)
db.py                SQLite persistence for run history
web/index.html       Single-page UI (noVNC iframe + action log)
docker/              Dockerfile + docker-compose.yml for the desktop VM
ecua2_agent/
  planner_module/    LLM singleton + generate_step()
  vision_module/     OmniParser (YOLOv8 + OCR) -> UI element list
  controller_module/ vncdotool action dispatch
tests/               pytest test suite (27 tests)
```
