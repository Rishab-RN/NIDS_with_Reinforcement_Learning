"""
app.py — FastAPI server exposing OpenEnv HTTP endpoints.

Endpoints:
  POST /reset   — start new episode
  POST /step    — submit action
  GET  /state   — get current state
  GET  /health  — liveness check
  GET  /tasks   — list available tasks & graders
  POST /grade   — run grader for a completed episode
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from models import NIDSAction, NIDSObservation, NIDSState
from server.nids_environment import NIDSEnvironment, TASKS, grade_task

app = FastAPI(
    title="NIDS OpenEnv",
    description="Network Intrusion Detection System — OpenEnv hackathon submission",
    version="1.0.0",
)

# Single shared environment instance (SUPPORTS_CONCURRENT_SESSIONS=False)
_env: Optional[NIDSEnvironment] = None


# ---------------------------------------------------------------------------
# Request / Response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "easy_classification"


class StepResponse(BaseModel):
    observation: dict
    reward:      float
    done:        bool
    state:       dict
    info:        dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "env": "nids_env",
        "title": "NIDS OpenEnv — Network Intrusion Detection System",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state", "/grade", "/docs"],
        "status": "running",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "env": "nids_env"}


@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "name":        name,
                "difficulty":  cfg["difficulty"],
                "max_steps":   cfg["max_steps"],
                "description": cfg["description"],
            }
            for name, cfg in TASKS.items()
        ]
    }


@app.post("/reset")
async def reset(req: ResetRequest = ResetRequest()):
    global _env
    if req.task_name not in TASKS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown task '{req.task_name}'. "
                                   f"Valid: {list(TASKS.keys())}")
    _env = NIDSEnvironment(task_name=req.task_name)
    obs  = _env.reset()
    st   = _env.state()
    return {
        "observation": obs.model_dump(),
        "reward":      0.0,
        "done":        False,
        "state":       st.model_dump(),
        "info":        {"task": req.task_name},
    }


@app.post("/step")
async def step(action: NIDSAction):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")

    obs = _env.step(action)
    st  = _env.state()

    # Compute partial reward at every step
    from server.nids_environment import _compute_reward, TASKS as _TASKS
    cfg    = _TASKS[_env.task_name]
    reward = _compute_reward(
        _env.true_positives, _env.false_positives,
        _env.missed_threats, _env.escalations,
        cfg["max_steps"], cfg["difficulty"],
    )

    return {
        "observation": obs.model_dump(),
        "reward":      reward,
        "done":        obs.episode_done,
        "state":       st.model_dump(),
        "info":        {},
    }


@app.get("/state")
async def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.state().model_dump()


@app.post("/grade")
async def grade(task_name: str | None = None):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="No active episode to grade.")
    # Always grade against the environment's actual task — ignore any stale param
    effective_task = _env.task_name
    result = grade_task(effective_task, _env.get_episode_data())
    return result


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
